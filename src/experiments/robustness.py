import csv
from unittest.mock import patch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.utils import importer
from statistics import mean, stdev
from lora_parameters import find_best_lora_config

def main():
    lora = importer._import_lora()

    seeds = [42, 43, 44]
    noise_levels = ["Worst", "Med", "Best"]
    sample_size = "0000"
    noisy_test_acc=[]
    clean_test_acc= []
    noisy_val_acc =[]
    clean_val_acc = []

    best_config = find_best_lora_config(lora)
    print("Best lora config is: ", best_config)

    path = Path("base_table.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)  # , fieldnames=fieldnames)

        # First header row
        writer.writerow([
            "Noise level",
            "","","",
            "Test accuracy:", "","","",
            "Noisy/ Dev accuracy:"
        ])
        # First header row
        writer.writerow([
            "", "", "test acc: train clean", "", "test acc: train clean – test acc: train noisy", "", "clean val acc: train clean","",
            "noisy val acc: train clean – noisy val acc: train noisy"
        ])
        writer.writerow([
            "", "mean","stddev","mean","stddev","mean","stddev","mean","stddev"
        ])

    for noise in noise_levels:
        for seed in seeds:
            cli_args = ["NOISE-lora.py",
                       "--seed", str(seed),
                        "--rank", str(best_config["rank"]),
                        "--lora-alpha", str(best_config["alpha"]),
                        "--lora-dropout", str(best_config["dropout"]),
                        "--train-data-path", "../../data/samples4" + sample_size + "train" + noise + ".csv",
                        "--val-data-path", "../../data/samples1" + sample_size + "val" + noise + ".csv",
                        "--test-data-path", "../../data/samplesTest.csv"
                       ]
            with patch.object(sys, "argv", cli_args):
                print(
                    f"\n--- Seed {seed}, noisy: ---"
                )
                # Train on noisy labels
                history_lora_noisy = lora.whole_training()
            cli_args = ["NOISE-lora.py",
                        "--seed", str(seed),
                        "--rank", str(best_config["rank"]),
                        "--lora-alpha", str(best_config["alpha"]),
                        "--lora-dropout", str(best_config["dropout"]),
                        "--train-data-path", "../../data/samples4" + sample_size + "train" + noise + ".csv",
                        "--val-data-path", "../../data/samples1" + sample_size + "val" + noise+ ".csv",
                        "--test-data-path", "../../data/samplesTest.csv",
                        "--target-label-col", "clean_label"
                        ]
            with patch.object(sys, "argv", cli_args):
                print(
                    f"\n--- Seed {seed}, clean: ---"
                )
                #Train on clean labels
                history_lora_clean = lora.whole_training()

            #Extract final accuracy
            noisy_test_acc.append( history_lora_noisy["test_accuracy"])
            clean_test_acc.append ( history_lora_clean["test_accuracy"])
            noisy_val_acc.append (history_lora_noisy["val_accuracy"][-1])
            clean_val_acc.append(history_lora_clean["val_accuracy"][-1])

        mean_noisy_test_acc = mean(noisy_test_acc)
        mean_clean_test_acc = mean(clean_test_acc)
        mean_noisy_val_acc = mean(noisy_val_acc)
        mean_clean_val_acc = mean(clean_val_acc)
        standdev_nt = stdev(noisy_test_acc)
        standdev_ct = stdev(clean_test_acc)
        standdev_nv = stdev(noisy_val_acc)
        standdev_cv = stdev(clean_val_acc)

        clean_minus_noisy_test = [a - b for a, b in zip(clean_test_acc, noisy_test_acc)]
        mean_clean_minus_noisy_test=mean(clean_minus_noisy_test)
        standdev_c_minus_n_test=stdev(clean_minus_noisy_test)

        clean_minus_noisy_val = [a - b for a, b in zip(clean_val_acc, noisy_val_acc)]
        mean_clean_minus_noisy_val = mean(clean_minus_noisy_val)
        standdev_c_minus_n_val = stdev(clean_minus_noisy_val)


        with open(path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)#, fieldnames=fieldnames)
            writer.writerow([ noise, mean_clean_test_acc,  standdev_ct,
                              mean_clean_minus_noisy_test, standdev_c_minus_n_test,
                              mean_clean_val_acc, standdev_cv ,
                              mean_clean_minus_noisy_val, standdev_c_minus_n_val
                              ])

    print(f"Results successfully saved to CSV: base_table.csv")


if __name__ == "__main__":
    main()
