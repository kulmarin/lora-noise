import csv
from pathlib import Path
import sys
from src.utils import importer

def save_robustness_to_csv(filepath: str, clean_acc: float, noisy_acc: float, robustness: float):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    fieldnames = ["clean_accuracy", "noisy_accuracy", "robustness_gap"]

    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "clean_accuracy": round(clean_acc, 4),
            "noisy_accuracy": round(noisy_acc, 4),
            "robustness_gap": round(robustness, 4),
        })
    print(f"Results successfully saved to CSV: {filepath}")

def main():
    lora = importer._import_lora()
    # Train on noisy labels
    history_lora_noisy = lora.pseudotraining()
    #Train on clean labels
    sys.argv.extend(["--target-label-col", "clean_label"])
    history_lora_clean = lora.pseudotraining()
    #Extract final accuracy

    noisy_acc = (
        history_lora_noisy["test_accuracy"][-1]
        if isinstance(history_lora_noisy["test_accuracy"], list)
        else history_lora_noisy["test_accuracy"]
    )
    clean_acc = (
        history_lora_clean["test_accuracy"][-1]
        if isinstance(history_lora_clean["test_accuracy"], list)
        else history_lora_clean["test_accuracy"]
    )

    # Calculate robustness
    robustness = clean_acc - noisy_acc
    print("Robustness in execution with LoRA between noise and clean data:", robustness)

    # Save main results to csv
    save_robustness_to_csv(
        filepath="results/robustness_results.csv",
        clean_acc=clean_acc,
        noisy_acc=noisy_acc,
        robustness=robustness,
    )


   # robustness = history_lora_clean["test_accuracy"]-history_lora_noisy["test_accuracy"]
    #print ("robustness in execution  with LoRA between noise and clean data: ",robustness)



if __name__ == "__main__":
    main()
