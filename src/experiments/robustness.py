import sys
from src.utils import importer


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    lora = importer._import_lora()

    history_lora_noisy = lora.whole_training()

    sys.argv.extend(["--target-label-col", "clean_label"])
    history_lora_clean = lora.whole_training()

    robustness = history_lora_noisy["test_accuracy"][-1] - history_lora_clean["test_accuracy"][-1]
    print ("robustness in execution  with LoRA between noise and clean data: ",robustness)



if __name__ == "__main__":
    main()
