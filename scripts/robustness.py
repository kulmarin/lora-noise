import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# Import shared functions from pilot script
# --------------------------------------------------------------------------- #

def _import_bert():
    spec = importlib.util.spec_from_file_location(
        "bert", str(PROJECT_ROOT / "scripts" / "NOISE_bert.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    bert = _import_bert()
    lora = bert._import_lora()

    history_lora_noisy = lora.whole_training()

    sys.argv.extend(["--target-label-col", "clean_label"])
    history_lora_clean = lora.whole_training()

   # history = {
  #      "train_loss": [],
   #     "val_loss": [],
   #     "val_accuracy": [],
   #     "learning_rates": [],
   # }

    robustness = history_lora_noisy["val_accuracy"][-1] - history_lora_clean["val_accuracy"][-1]
    print ("robustness in execution  with LoRA between noise and clean data: ",robustness)

if __name__ == "__main__":
    main()
