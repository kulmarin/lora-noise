
from unittest.mock import patch
import csv
from statistics import mean, stdev
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import importer


def find_best_lora_config(lora) -> dict[str, int | float]:
    rank_configs = [
        {"rank": 2, "alpha": 4},
        {"rank": 4, "alpha": 8},
        {"rank": 8, "alpha": 16},
        {"rank": 16, "alpha": 32},
    ]
    dropouts = [0.0, 0.05, 0.1]

    best_config = {}
    best_val_acc = 0

    for dropout in dropouts:
        for config in rank_configs:

            rank = config["rank"]
            alpha = config["alpha"]

            cli_args = ["NOISE-lora.py",
                        "--rank", str(rank),
                        "--lora-alpha", str(alpha),
                        "--lora-dropout", str(dropout),
                        ]
            with patch.object(sys, "argv", cli_args):
                print(
                    f"\n--- Rank={rank}, Alpha={alpha}, "
                    f"Dropout={dropout} ---"
                )
                history = lora.run_or_load_experiment()

                val_acc = history["val_accuracy"][-1]
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_config = {"rank": rank, "alpha": alpha, "dropout": dropout}
    return best_config


if __name__ == "__main__":
    main()

