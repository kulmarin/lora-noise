#!/usr/bin/env python3


from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader
import argparse
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.temporal_tracker import TemporalTracker
from src.utils.seed import set_seed


# --------------------------------------------------------------------------- #
# Import shared functions from pilot script
# --------------------------------------------------------------------------- #

def _import_pilot():
    """Import functions from the pilot experiment script."""
    spec = importlib.util.spec_from_file_location(
        "pilot", str(PROJECT_ROOT / "scripts" / "NOISE_pilot.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -----------------------------------------------------------------------------
# 3. Model Creation (Full Model, Unfrozen)
# -----------------------------------------------------------------------------
def build_full_base_model(
        model_name: str, num_labels:int
) -> nn.Module:

    # Load raw BERT sequence classification model
    model = AutoModelForSequenceClassification.from_pretrained(
       model_name,
        num_labels=num_labels
    )

    # Calculate total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Full Fine-Tuning active. Total trainable parameters: {total_params:,}")

    return model




# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    pilot = _import_pilot()
    args = pilot.parse_args()


    device = pilot.detect_device(args.device)

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    print("Loading data...")

    train_dataloader = pilot.create_dataloader(
        file_path=args.train_data_path,
        tokenizer_path="bert-base-uncased",
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True
    )
    print(train_dataloader.dataset.__len__())

    val_dataloader = pilot.create_dataloader(
        file_path=args.val_data_path,
        tokenizer_path="bert-base-uncased",
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False
    )
    print(val_dataloader.dataset.__len__())

    test_dataloader = pilot.create_dataloader(
        file_path=args.test_data_path,
        tokenizer_path="bert-base-uncased",
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False
    )

    print(test_dataloader.dataset.__len__())

    # ------------------------------------------------------------------ #
    # Step 3: Class weights...
    # ------------------------------------------------------------------ #
    print("\nStep 3: Computing class weights...")


    combined_labels = getattr(train_dataloader.dataset, args.target_label_col + "s")
    all_train_labels = torch.tensor(combined_labels, dtype=torch.long)
    label_counts = torch.bincount(all_train_labels, minlength= args.num_labels).float()
    class_weights = (1.0 / label_counts.clamp(min=1))
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    print(f"  Label distribution: {label_counts.tolist()}")
    print(f"  Class weights: {class_weights.tolist()}")

    # ------------------------------------------------------------------ #
    # Create base-model
    # ------------------------------------------------------------------ #

    model = build_full_base_model(args.model_name, args.num_labels)


    # ------------------------------------------------------------------ #
    # Run full-tuning
    # ------------------------------------------------------------------ #

    history= pilot.train_any_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        noisy_or_clean_label=args.target_label_col,
        device=device,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        class_weights=class_weights,
    )
    print(history)



if __name__ == "__main__":
    main()
