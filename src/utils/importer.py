import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- #
# Import shared functions from NOISE-bert script
# --------------------------------------------------------------------------- #

def _import_bert():
    spec = importlib.util.spec_from_file_location(
        "bert", str(PROJECT_ROOT / "NOISE" / "NOISE_bert.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# --------------------------------------------------------------------------- #
# Import shared functions from NOISE-lora script
# --------------------------------------------------------------------------- #

def _import_lora():
    spec = importlib.util.spec_from_file_location(
        "lora", str(PROJECT_ROOT / "NOISE" / "NOISE_lora.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod