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

def create_lora_model(
    model_name: str = "roberta-base",
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
        model_name: HuggingFace model name.
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
        model_name, num_labels=num_labels,
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


# --------------------------------------------------------------------------- #
# Training with per-example tracking
# --------------------------------------------------------------------------- #

def train_peft_wrapped_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 5,
    learning_rate: float = 2e-5,
    eval_every_n_steps: int = 100,
    max_grad_norm: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Train model while recording per-example losses at regular intervals.

    The training loop uses the full dataset (NOISY AG)
    for gradient updates.

    Args:
        model: PEFT model to train.
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

    model = model.to("cpu")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
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
        class_weights = class_weights.to("cpu")
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

    model.eval()
    model.train()

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []

        pbar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1}/{n_epochs}",
            leave=False,
        )

        for batch in pbar:
            #print(batch)
            input_ids = batch["input_ids"].to("cpu")
            attention_mask = batch["attention_mask"].to("cpu")
            labels = batch["noisy_label"].to("cpu")

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn_mean(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")


        # Record epoch-level metrics
        train_loss = np.mean(epoch_losses)
        val_loss, val_acc = _evaluate(model, val_loader, loss_fn_mean, "cpu")
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
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate model on validation set.

    Returns:
        (mean_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["noisy_label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    model.train()
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



# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

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


def main() -> None:

    print("Step 1: Loading data...")

    train_dataloader, noise_dataset = create_dataloader(
        file_path="../data/samples400trainMed.csv",
        tokenizer_path="bert-base-uncased",
        batch_size=32,
        max_length=256,
        shuffle=True
    )
    print(noise_dataset.__len__())

    eval_dataloader, noise_dataset = create_dataloader(
        file_path="../data/samples100valMed.csv",
        tokenizer_path="bert-base-uncased",
        batch_size=32,
        max_length=256,
        shuffle=True
    )
    print(noise_dataset.__len__())

    test_dataloader, noise_dataset = create_dataloader(
        file_path="../data/samplesTest.csv",
        tokenizer_path="bert-base-uncased",
        batch_size=32,
        max_length=256,
        shuffle=True
    )

    print(noise_dataset.__len__())






    # ------------------------------------------------------------------ #
    # Step 3: Create model
    # ------------------------------------------------------------------ #
    print("\nStep 3: Creating LoRA model...")

    model = create_lora_model(
        model_name="bert-base-uncased" ,
        num_labels=4,
        rank=4,
        lora_alpha=8,
        lora_dropout=0.05,
    )

    trainable = 0
    total = 0

    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    print("Trainable:", trainable)
    print("Total:", total)
    print("Percentage:", 100 * trainable / total)

    print("CUDA available:", torch.cuda.is_available())
    print("Model device:", next(model.parameters()).device)
    print("Number of batches:", len(train_dataloader))

    # ------------------------------------------------------------------ #
    # Step 6: Train with tracking
    # ------------------------------------------------------------------ #
    print("\nStep 6: Training with per-example tracking...")

    history = train_peft_wrapped_model(
        model=model,
        train_loader=train_dataloader,

        val_loader=eval_dataloader,

        n_epochs=2,
        learning_rate=2.0e-5,
        eval_every_n_steps=100,

        max_grad_norm=1.0,
        #class_weights=class_weights,
    )
    print(history)

    # # ------------------------------------------------------------------ #
    # # Step 7: Compute correlations (three metrics)
    # # ------------------------------------------------------------------ #
    # print("\nStep 7: Computing entropy correlations...")
    #
    # # 7a. AULC (primary metric -- continuous, uses full trajectory)
    # _, aulc_arr, aulc_ent = compute_aulc(tracker)
    # valid_aulc = np.isfinite(aulc_arr) & np.isfinite(aulc_ent)
    # if valid_aulc.sum() >= 3:
    #     rho_aulc, p_aulc = stats.spearmanr(aulc_arr[valid_aulc], aulc_ent[valid_aulc])
    # else:
    #     rho_aulc, p_aulc = 0.0, 1.0
    # print(f"  [AULC]       Spearman rho = {rho_aulc:+.4f}, p = {p_aulc:.2e}  (n={valid_aulc.sum()})")
    #
    # # 7b. Final loss (sanity check -- do clean examples end with lower loss?)
    # _, final_arr, final_ent = compute_final_loss(tracker)
    # valid_final = np.isfinite(final_arr) & np.isfinite(final_ent)
    # if valid_final.sum() >= 3:
    #     rho_final, p_final = stats.spearmanr(final_arr[valid_final], final_ent[valid_final])
    # else:
    #     rho_final, p_final = 0.0, 1.0
    # print(f"  [Final loss] Spearman rho = {rho_final:+.4f}, p = {p_final:.2e}  (n={valid_final.sum()})")
    #
    # # 7c. Threshold crossing (legacy metric -- for comparison)
    # ids_arr, times_arr, entropies_arr = compute_learning_times(
    #     tracker, threshold=args.loss_threshold,
    # )
    # rho, p_value = compute_spearman_correlation(times_arr, entropies_arr)
    # n_learned = np.isfinite(times_arr).sum()
    # n_unlearned = (~np.isfinite(times_arr)).sum()
    # print(f"  [Threshold]  Spearman rho = {rho:+.4f}, p = {p_value:.2e}  (learned={n_learned}/{len(times_arr)})")
    #
    # # Use AULC as the primary gate metric
    # primary_rho = rho_aulc
    # primary_p = p_aulc
    #
    # # ------------------------------------------------------------------ #
    # # Step 8: Save tracker
    # # ------------------------------------------------------------------ #
    # print("\nStep 8: Saving tracker and results...")
    #
    # tracker_path = output_dir / f"pilot_r{args.rank}_s{args.seed}.json"
    # tracker.save(tracker_path)
    # print(f"  Saved tracker to {tracker_path}")
    #
    # # Save pilot results summary
    # pilot_results = {
    #     "rank": args.rank,
    #     "seed": args.seed,
    #     "epochs": args.epochs,
    #     "learning_rate": args.learning_rate,
    #     "loss_threshold": args.loss_threshold,
    #     "eval_every_n_steps": args.eval_every_n_steps,
    #     "snli_size": args.snli_size,
    #     "n_train_combined": len(combined_premises),
    #     "n_train_chaosnli": len(tracking_premises),
    #     "n_val_chaosnli": len(val_premises),
    #     "spearman_aulc_rho": rho_aulc,
    #     "spearman_aulc_p": p_aulc,
    #     "spearman_final_loss_rho": rho_final,
    #     "spearman_final_loss_p": p_final,
    #     "spearman_threshold_rho": rho,
    #     "spearman_threshold_p": p_value,
    #     "n_learned": int(n_learned),
    #     "n_unlearned": int(n_unlearned),
    #     "n_total": len(times_arr),
    #     "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
    #     "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
    #     "final_val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else None,
    #     "tracking_steps": history["tracking_steps"],
    #     "train_loss_history": history["train_loss"],
    #     "val_loss_history": history["val_loss"],
    #     "val_accuracy_history": history["val_accuracy"],
    # }
    #
    # results_path = output_dir / f"pilot_results_r{args.rank}_s{args.seed}.json"
    # with open(results_path, "w") as f:
    #     json.dump(pilot_results, f, indent=2)
    # print(f"  Saved results to {results_path}")
    #
    # # ------------------------------------------------------------------ #
    # # Step 9: Generate hero figure
    # # ------------------------------------------------------------------ #
    # print("\nStep 9: Generating hero figure...")
    #
    # plot_hero_figure(
    #     tracker=tracker,
    #     category_names=["clean", "ambiguous", "contested"],
    #     tracking_steps=history["tracking_steps"],
    #     output_path=figure_dir / f"hero_loss_curves_r{args.rank}_s{args.seed}.png",
    #     title_suffix=f" (rank={args.rank}, seed={args.seed})",
    #     loss_threshold=args.loss_threshold,
    # )
    #
    # # ------------------------------------------------------------------ #
    # # Gate check (uses AULC as primary metric)
    # # ------------------------------------------------------------------ #
    # elapsed = time.time() - t0
    # print(f"\n{'=' * 70}")
    # print(f"Phase 1 complete ({elapsed:.1f}s)")
    #
    # # For AULC: positive rho means higher entropy -> higher mean loss (slower learning)
    # # For final loss: positive rho means higher entropy -> higher final loss
    # # Both are the predicted direction.
    # best_val_acc = max(history["val_accuracy"]) if history["val_accuracy"] else 0.0
    # print(f"\n  Best val accuracy: {best_val_acc:.4f}")
    # print(f"  Final val accuracy: {history['val_accuracy'][-1]:.4f}" if history["val_accuracy"] else "")
    #
    # if primary_rho > 0.10 and primary_p < 0.05:
    #     print(f"\nPHASE 1 GATE PASSED: AULC Spearman rho = {primary_rho:.3f} (p = {primary_p:.3e})")
    #     print("  Positive correlation: higher entropy -> higher mean loss (slower learning).")
    #     print("  This confirms the temporal separation hypothesis.")
    #     print("  Proceed to Phase 2 (rank sweep).")
    # else:
    #     print(f"\nPHASE 1 GATE FAILED: AULC Spearman rho = {primary_rho:.3f} (p = {primary_p:.3e})")
    #     print("  Diagnostics:")
    #     print(f"    AULC rho > 0.10?     {'YES' if primary_rho > 0.10 else 'NO'} (rho = {primary_rho:+.4f})")
    #     print(f"    AULC p < 0.05?       {'YES' if primary_p < 0.05 else 'NO'} (p = {primary_p:.2e})")
    #     print(f"    Final-loss rho:      {rho_final:+.4f} (p = {p_final:.2e})")
    #     print(f"    Threshold rho:       {rho:+.4f} (p = {p_value:.2e})")
    #     print(f"    Best val accuracy:   {best_val_acc:.4f}")
    #     if best_val_acc < 0.55:
    #         print("    Model may not be learning the task. Check training config.")
    #     if primary_rho < 0:
    #         print("    Negative correlation: contested examples have LOWER mean loss.")
    #         print("    This contradicts the hypothesis.")
    #     print("  Do NOT proceed to Phase 2 without diagnosing the failure.")
    #
    # print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
