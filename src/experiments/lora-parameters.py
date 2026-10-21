
import numpy as np
from unittest.mock import patch
import csv
import sys
from collections import defaultdict
from statistics import mean, stdev
from src.utils import importer

def main() -> None:
    #results_v=[]
    #results_t=[]

    seeds= [42,43,44]
    rank_configs = [
        {"rank": 2, "alpha": 4},
        {"rank": 4, "alpha": 8},
        {"rank": 8, "alpha": 16},
        {"rank": 16, "alpha": 32},
    ]
    dropouts=[0.0,0.05,0.1]

    lora = importer._import_lora()
    individual_results = []
    print("Loading model and utilities...")

    def run_single_config (experiment:str, rank:int, alpha:int, dropout:float) -> list:
        """Run one configuration for every seed."""
        run_results = []
        #for config in rank_configs:
        for seed in seeds:
            cli_args = ["NOISE-lora.py",
                        "--seed", str(seed),
                        "--rank", str(rank),
                        "--lora-alpha", str(alpha),
                        "--lora-dropout", str(dropout),
                        "--target-label-col", "noisy_label"]
            with patch.object(sys, "argv", cli_args):
                print(
                    f"\n--- [{experiment}] Rank={rank}, Alpha={alpha}, "
                    f"Dropout={dropout}, Seed={seed} ---"
                )
                history = lora.pseudotraining()

            val_acc = (history["val_accuracy"][-1]
                            if isinstance(history["val_accuracy"], list)
                            else history["val_accuracy"]
            )
            test_acc = (history["test_accuracy"][-1]
                        if isinstance(history["test_accuracy"], list)
                        else history["test_accuracy"]
            )
            row = {
                "experiment": experiment,
                "rank": rank,
                "alpha": alpha,
                "dropout": dropout,
                "seed": seed,
                "val_accuracy": val_acc,
                "test_accuracy": test_acc,
            }

            individual_results.append(row)
            run_results.append(row)

        return run_results

            #results_v.append(history["val_accuracy"][-1])
            #results_t.append(history["test_accuracy"])
    # Rank sweep
    rank_results = []

    for config in rank_configs:
        rows = run_single_config(
            experiment="rank_sweep",
            rank=config["rank"],
            alpha=config["alpha"],
            dropout=0.05,
        )
        rank_results.extend(rows)

    # Select rank by mean validation accuracy
    val_means = {}

    for config in rank_configs:
        rank = config["rank"]
        accuracies = [
            row["val_accuracy"]
            for row in rank_results
            if row["rank"] == rank
        ]
        val_means[rank] = mean(accuracies)

    best_config = max(
        rank_configs,
        key=lambda config: val_means[config["rank"]],
    )

    print("\nSelected rank based on mean validation accuracy:")
    print(f"  Best Rank: {best_config['rank']} (Alpha: {best_config['alpha']})")
    print(f"  Mean Val Accuracy: {val_means[best_config['rank']]:.4f}")
    '''
    print(best_config)
    print(f"Mean validation accuracy: {val_means[best_config['rank']]:.4f}")

    print ("Valiadion accuracies for each config ", results_v)
    print ("Test accuracies for each config ", results_t)

    max_index = max(range(len(results_t)), key=results_t.__getitem__)
    print("index of highest test accuracy ",max_index)
    print ("the config, that leads to highest test accuracy ", rank_configs[max_index])

    max_index1 = max(range(len(results_v)), key=results_v.__getitem__)
    print("index of highes validation accuracy ", max_index1)
    print("the config, that leads to highest validation accuracy ", rank_configs[max_index1])
    '''
    #dropouts = [0, 0.1]
    #results = []
    #Dropout Sweep (Using Best Rank)
    for dropout in dropouts:
        run_single_config(
            experiment="dropout_sweep",
            rank=best_config["rank"],
            alpha=best_config["alpha"],
            dropout=dropout,
        )

        # Save individual results
        individual_fields = [
            "experiment", "rank", "alpha", "dropout", "seed",
            "val_accuracy", "test_accuracy",
        ]

        with open(
                "lora_parameters_individual.csv",
                "w",
                newline="",
                encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=individual_fields)
            writer.writeheader()
            writer.writerows(individual_results)

            # Aggregate mean and standard deviation across seeds
            grouped = defaultdict(list)

            for row in individual_results:
                key = (
                    row["experiment"],
                    row["rank"],
                    row["alpha"],
                    row["dropout"],
                )
                grouped[key].append(row)

            summary = []

            for key, rows in grouped.items():
                val_values = [row["val_accuracy"] for row in rows]
                test_values = [row["test_accuracy"] for row in rows]

                summary.append({
                    "experiment": key[0],
                    "rank": key[1],
                    "alpha": key[2],
                    "dropout": key[3],
                    "n_seeds": len(rows),
                    "val_accuracy_mean": mean(val_values),
                    "val_accuracy_std": stdev(val_values) if len(rows) > 1 else 0.0,
                    "test_accuracy_mean": mean(test_values),
                    "test_accuracy_std": stdev(test_values) if len(rows) > 1 else 0.0,
                })

            summary_fields = [
                "experiment", "rank", "alpha", "dropout", "n_seeds",
                "val_accuracy_mean", "val_accuracy_std",
                "test_accuracy_mean", "test_accuracy_std",
            ]

            with open(
                    "lora_parameters_summary.csv",
                    "w",
                    newline="",
                    encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(file, fieldnames=summary_fields)
                writer.writeheader()
                writer.writerows(summary)

            # Print summary
            print("\n--- LoRA parameter summary ---")
            print(
                f"{'Experiment':<16} {'Rank':>5} {'Alpha':>6} "
                f"{'Dropout':>8} {'Val. mean ± SD':>21} "
                f"{'Test mean ± SD':>21}"
            )

            for row in summary:
                val_text = (
                    f"{row['val_accuracy_mean']:.4f} "
                    f"± {row['val_accuracy_std']:.4f}"
                )
                test_text = (
                    f"{row['test_accuracy_mean']:.4f} "
                    f"± {row['test_accuracy_std']:.4f}"
                )

                print(
                    f"{row['experiment']:<16} "
                    f"{row['rank']:>5} "
                    f"{row['alpha']:>6} "
                    f"{row['dropout']:>8.2f} "
                    f"{val_text:>21} "
                    f"{test_text:>21}"
                )

            print("\nSuccessfully saved:")
            print("- lora_parameters_individual.csv")
            print("- lora_parameters_summary.csv")


        '''
        cli_args = ["NOISE-lora.py", "--lora-dropout", str(dropout), "--rank", str(rank_configs[max_index]["rank"]),
                    "--lora-alpha", str(rank_configs[max_index]["alpha"]),"--target-label-col", "noisy_label"]
        with patch.object(sys, "argv", cli_args):
            print(f"\n--- Running dropout {dropout} ---")
            history = lora.whole_training()
            results.append(history["test_accuracy"])

    max_index_d = max(range(len(results)), key=results.__getitem__)
    print("index of highest test accuracy with varied dropouts", max_index_d)
    '''




if __name__ == "__main__":
    main()

