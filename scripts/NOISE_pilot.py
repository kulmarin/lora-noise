#!/usr/bin/env python3
"""


Usage:
    python scripts/02_pilot_experiment.py
    python scripts/02_pilot_experiment.py --rank 4 --seed 42 --epochs 5
    python scripts/02_pilot_experiment.py --snli-size 50000
    python scripts/02_pilot_experiment.py --synthetic  # synthetic data
"""

from __future__ import annotations

import argparse
import json
import os
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
from accelerate.test_utils.examples import clean_lines
from huggingface_hub.utils.tqdm import progress_bar_states
from sympy.parsing.sympy_parser import EvaluateFalseTransformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.annotation_entropy import compute_annotation_entropy_from_distribution
from src.training.temporal_tracker import TemporalTracker
from src.utils.seed import set_seed

from src.utils.mydataloader import *


# --------------------------------------------------------------------------- #
# Model creation
# --------------------------------------------------------------------------- #

def create_lora_wrapped_base_model(
    base_model_name: str = "roberta-base",
    num_labels: int = 3,
    rank: int = 4,
    lora_alpha: Optional[int] = None,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Create RoBERTa + LoRA model for sequence classification.

    Uses the PEFT library to apply LoRA adapters. Alpha defaults to 2*rank
    per the experimental protocol (scaling that preserves effective learning
    rate across ranks).

    Args:
        base_model_name: HuggingFace model name.
        num_labels: Number of output classes (3 for NLI).
        rank: LoRA rank r.
        lora_alpha: LoRA scaling. Defaults to 2 * rank.
        lora_dropout: Dropout in LoRA layers.
        target_modules: Which attention matrices to adapt.

    Returns:
        PEFT-wrapped model.
    """
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    if lora_alpha is None:
        lora_alpha = 2 * rank

    if target_modules is None:
        target_modules = ["query", "value"]

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=num_labels,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=["classifier"],  # Keep classification head trainable
    )

    model = get_peft_model(base_model, lora_config)

    # Report parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA rank={rank}, alpha={lora_alpha}")
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def train_peft_wrapped_model(
    peft_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    noisy_or_clean_label: str,
    n_epochs: int = 5,
    learning_rate: float = 2e-5,
    device: str = "cpu",
    max_grad_norm: float = 1.0,

    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Train LoRA + base model.

    The training loop uses the full dataset (NOISY AG)
    for gradient updates.

    Args:
        peft_model: PEFT model to train.
        train_loader: Training data loader .
        val_loader: Validation data loader.
        n_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        eval_every_n_steps: Record per-example losses every N steps.
        device: Device string ("mps", "cuda", "cpu").
        max_grad_norm: Gradient clipping norm.
        class_weights: Pre-computed class weights for weighted CE loss.

    Returns:
        Dictionary with training history: per-step metrics, final metrics.
    """

    peft_model = peft_model.to( device)
    optimizer = torch.optim.AdamW(
        [p for p in peft_model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    # Cosine annealing with warmup (6% warmup, min LR = 10% of peak)
    total_steps = n_epochs * len(train_loader)
    warmup_steps = int(0.06 * total_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Class-weighted loss for training (handles label imbalance)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"  Class weights: {class_weights.tolist()}")

    loss_fn = nn.CrossEntropyLoss(reduction="none")  # per-example losses (unweighted, for tracking)
    loss_fn_mean = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rates": [],
    }

    global_step = 0

    print(f"  Total training steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Training examples: {len(train_loader.dataset)}")


    # Initial tracking pass (step 0, before any training)
    print("  Recording initial per-example losses (step 0)...")

    peft_model.eval()
    peft_model.train()

    for epoch in range(n_epochs):
        peft_model.train()
        epoch_losses = []

        progress_bar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1}/{n_epochs}",
            leave=False,
        )

        for batch in progress_bar:
            #print(batch)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch[noisy_or_clean_label].to(device)

            outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn_mean(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())
            global_step += 1

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")


        #------------------------------------#
        # Evaluate
        #------------------------------------#

        train_loss = np.mean(epoch_losses)
        val_loss, val_acc = _evaluate(peft_model, val_loader, loss_fn_mean, device)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))
        history["learning_rates"].append(float(scheduler.get_last_lr()[0]))

        print(
            f"  Epoch {epoch+1}/{n_epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )


    return history



@torch.no_grad()
def _evaluate(
    peft_model: nn.Module,
    val_data_loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate model on validation set.

    Returns:
        (mean_loss, accuracy).
    """
    peft_model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in val_data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        clean_labels = batch["clean_labels"].to(device)

        outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, clean_labels)

        total_loss += loss.item() * clean_labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == clean_labels).sum().item()
        total += clean_labels.size(0)

    peft_model.train()
    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# --------------------------------------------------------------------------- #
# Analysis functions
# --------------------------------------------------------------------------- #



def compute_spearman_correlation(
    learning_times: np.ndarray,
    entropies: np.ndarray,
) -> Tuple[float, float]:
    """Compute Spearman correlation between learning time and entropy.

    Filters out examples that were never learned (inf) or have no entropy.

    Returns:
        (rho, p_value).
    """
    valid = np.isfinite(learning_times) & np.isfinite(entropies)
    if valid.sum() < 3:
        return 0.0, 1.0

    rho, p = stats.spearmanr(learning_times[valid], entropies[valid])
    return float(rho), float(p)





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Pilot experiment (single rank, single seed)."
    )
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank (default: 4).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5).")
    parser.add_argument("--eval-every-n-steps", type=int, default=100, help="Tracking interval in training steps.")
    parser.add_argument("--batch-size", type=int, default=32, help="Train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Eval batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate (default: 2e-5).")
    parser.add_argument("--loss-threshold", type=float, default=0.693, help="Learning time threshold (default: -log(0.5) for confident prediction).")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length.")
    parser.add_argument("--model-name", type=str, default="roberta-base", help="Base model.")
    parser.add_argument(
        "--snli-size", type=int, default=20000,
        help="Number of SNLI training examples to subsample (default: 20000).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (auto-detected if not specified).",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to processed ChaosNLI data JSON (from 01_prepare_data.py).",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data for development.",
    )
    parser.add_argument(
        "--n-synthetic", type=int, default=800,
        help="Number of synthetic examples.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for tracker files.",
    )
    parser.add_argument(
        "--figure-dir", type=str, default=None,
        help="Directory for figures.",
    )
    return parser.parse_args()


def detect_device(requested: Optional[str] = None) -> str:
    """Detect the best available device."""
    if requested is not None:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:

    # ------------------------------------------------------------------ #
    # Step 1: Create dataloader
    # ------------------------------------------------------------------ #

    device = detect_device()

    print("Step 1: Loading data...")

    train_dataloader = create_dataloader(
        file_path="../data/samples400trainMed.csv",
        tokenizer_path="bert-base-uncased",
        batch_size=32,
        max_length=256,
        shuffle=True
    )
    print(train_dataloader.dataset.__len__())

    val_dataloader = create_dataloader(
        file_path="../data/samples100valMed.csv",
        tokenizer_path="bert-base-uncased",
        batch_size=32,
        max_length=256,
        shuffle=True
    )
    print(val_dataloader.dataset.__len__())

    test_dataloader = create_dataloader(
        file_path="../data/samplesTest.csv",
        tokenizer_path="bert-base-uncased",
        batch_size=32,
        max_length=256,
        shuffle=True
    )

    print(test_dataloader.dataset.__len__())

    # ------------------------------------------------------------------ #
    # Step 2: Create model
    # ------------------------------------------------------------------ #
    print("\nStep 2: Creating LoRA model...")

    wrapped_model = create_lora_wrapped_base_model(
        base_model_name="bert-base-uncased" ,
        num_labels=4,
        rank=4,
        lora_alpha=8,
        lora_dropout=0.05,
    )

    # ------------------------------------------------------------------ #
    # Step 3: Class weights...
    # ------------------------------------------------------------------ #
    print("\nStep 3: Computing class weights...")

    combined_groups=train_dataloader.dataset.groups

    all_train_groups = torch.tensor(combined_groups, dtype=torch.long)
    label_counts = torch.bincount(all_train_groups, minlength=3).float()
    class_weights = (1.0 / label_counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"  Label distribution: {label_counts.tolist()}")
    print(f"  Class weights: {class_weights.tolist()}")
    # ------------------------------------------------------------------ #
    # Step 4: Train
    # ------------------------------------------------------------------ #
    print("\nStep 4: Training on noisy labels")

    history = train_peft_wrapped_model(
        peft_model=wrapped_model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        noisy_or_clean_label="noise_label",
        device=device,
        n_epochs=2,
        learning_rate=2.0e-5,
        max_grad_norm=1.0,
        class_weights=class_weights,
    )
    print(history)



    print("\nStep 5: Training on clean labels")

    wrapped_model = create_lora_wrapped_base_model(
        base_model_name="bert-base-uncased",
        num_labels=4,
        rank=4,
        lora_alpha=8,
        lora_dropout=0.05,
    )
    history = train_peft_wrapped_model(
        peft_model=wrapped_model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        noisy_or_clean_label="clean_label",
        device=device,
        n_epochs=2,
        learning_rate=2.0e-5,
        max_grad_norm=1.0,
        class_weights=class_weights,
    )
    print(history)




if __name__ == "__main__":
    main()
