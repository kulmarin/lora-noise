
import numpy as np
from unittest.mock import patch
import sys

from src.utils import importer

def main() -> None:
    results_v=[]
    results_t=[]

    lora = importer._import_lora()

    rank_configs = [
        {"rank": 2, "alpha": 4},
        {"rank": 4, "alpha": 8},
        {"rank": 8, "alpha": 16},
        {"rank": 16, "alpha": 32},
    ]

    print("Loading train, validation, and test data...")

    for config in rank_configs:
        cli_args = ["NOISE-lora.py", "--rank", str(config["rank"]), "--lora-alpha", str(config["alpha"]),  "--target-label-col", "noisy_label"]
        with patch.object(sys, "argv", cli_args):
            history = lora.whole_training()
            results_v.append(history["val_accuracy"][-1])
            results_t.append(history["test_accuracy"])


    print ("Valiadion accuracies for each config ", results_v)
    print ("Test accuracies for each config ", results_t)

    max_index = max(range(len(results_t)), key=results_t.__getitem__)
    print("index of highest test accuracy ",max_index)
    print ("the config, that leads to highest test accuracy ", rank_configs[max_index])

    max_index1 = max(range(len(results_v)), key=results_v.__getitem__)
    print("index of highes validation accuracy ", max_index1)
    print("the config, that leads to highest validation accuracy ", rank_configs[max_index1])

    dropouts = [0, 0.1]
    results = []

    for dropout in dropouts:
        cli_args = ["NOISE-lora.py", "--lora-dropout", str(dropout), "--rank", str(rank_configs[max_index]["rank"]),
                    "--lora-alpha", str(rank_configs[max_index]["alpha"]),"--target-label-col", "noisy_label"]
        with patch.object(sys, "argv", cli_args):
            print(f"\n--- Running dropout {dropout} ---")
            history = lora.whole_training()
            results.append(history["test_accuracy"])

    max_index_d = max(range(len(results)), key=results.__getitem__)
    print("index of highest test accuracy with varied dropouts", max_index_d)





if __name__ == "__main__":
        main()

