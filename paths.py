from pathlib import Path
import os


PROJECT_ROOT=Path(__file__).resolve().parent

DATA_DIR=Path(os.getenv("DATA_DIR",PROJECT_ROOT / "data"))

DATASET_PATH=DATA_DIR / "my_iitb_dataset"

CHECKPOINT_DIR=Path(os.getenv("CHECKPOINT_DIR",PROJECT_ROOT / "checkpoints"))

BEST_CHECKPOINT_PATH=(CHECKPOINT_DIR / "best_translation_model.zip")

LATEST_CHECKPOINT_PATH=(CHECKPOINT_DIR / "latest_translation_model.zip")

TOKENIZER_PATH=Path(os.getenv("TOKENIZER_PATH",PROJECT_ROOT / "shared_tokenizer.json"))

# Create the checkpoint directory if it doesn't exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)