import numpy as np
from unittest.mock import patch
import sys

from src.utils import importer


def main():
    noise_levels = [ "Worst", "Med", "Best"]
    seeds = [43,44,45]
    sample_size="00"

    results = []
    lora = importer._import_lora()

    for noise in noise_levels:
        accuracies=[]
        for seed in seeds:
            cli_args = ["NOISE-lora.py",
                    "--seed", str(seed),
                    "--train-data-path", "../../data/samples4" + sample_size + "train" + noise + ".csv",
                     "--val-data-path", "../../data/samples1" + sample_size + "val" + noise+ ".csv",
                    "--test-data-path",
                    f"../../data/samples1{sample_size}test{noise}.csv",
                    "--target-label-col", "noisy_label"]
            with patch.object(sys, "argv", cli_args):
                print(f"\n--- Running noise level {noise}, seed {seed}---")
                history = lora.whole_training()


                accuracies.append(history["test_accuracy"])

                results.append({
                    "noise_level": noise,
                    "seed": seed,
                     "test_accuracy": history["test_accuracy"],
                })     #history["test_accuracy"])

        print(
        f"{noise}: "
        f"{np.mean(accuracies):.4f} ± "
        f"{np.std(accuracies, ddof=1):.4f}"
    )



if __name__ == "__main__":
        main()
