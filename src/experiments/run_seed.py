# run_seeds.py
import csv
import os
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from src.utils import importer


def append_result_to_csv(filepath: str, row_data: dict):
    """Appends a single experiment's result as a row in a CSV file."""
    file_exists = os.path.isfile(filepath)

    # Define the table column headers
    headers = ["seed", "target_label_col", "test_accuracy"]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)

        # Write headers if the CSV file is newly created
        if not file_exists:
            writer.writeheader()

        writer.writerow(row_data)

def main():
    SEEDS = [ 42, 43, 44]
    results = []
    csv_file = "results/multi_seed_lora_results.csv"
    lora = importer._import_lora()

    for seed in SEEDS:
        cli_args = ["NOISE-lora.py",
                    "--seed", str(seed),
                    "--target-label-col", "noisy_label"]
        with patch.object(sys, "argv", cli_args):
            print(f"\n--- Running Seed {seed} ---")
            history = lora.pseudotraining()
            test_acc = history["test_accuracy"][-1] if isinstance(history["test_accuracy"], list) else history[
                "test_accuracy"]
            results.append(history["test_accuracy"])
            row = {
                "seed": seed,
                "target_label_col": "noisy_label",
                "test_accuracy": round(float(test_acc), 4),
            }
            append_result_to_csv(csv_file, row)

    print(f"\nSaved all results to: {csv_file}")
    #print(f"Final Multi-Seed Result: {np.mean(results):.4f} ± {np.std(results):.4f}")
    print(f"\nFinal Multi-Seed Result: {np.mean(results):.4f} ± {np.std(results):.4f}")


if __name__ == "__main__":
        main()
