# run_seeds.py
import numpy as np
from unittest.mock import patch
import sys
from src.utils import importer


def main():
    SEEDS = [42, 43, 44]
    results = []

    lora = importer._import_lora()

    for seed in SEEDS:
        cli_args = ["NOISE-lora.py", "--seed", str(seed), "--target-label-col", "noisy_label"]
        with patch.object(sys, "argv", cli_args):
            print(f"\n--- Running Seed {seed} ---")
            history = lora.whole_training()
            results.append(history["test_accuracy"])

    print(f"\nFinal Multi-Seed Result: {np.mean(results):.4f} ± {np.std(results):.4f}")


if __name__ == "__main__":
        main()
