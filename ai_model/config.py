"""Central AI configuration. Dataset paths and model settings will be added here."""
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = AI_ROOT / "dataset"
HAM10000_ROOT = DATASET_ROOT / "HAM10000"
ISIC2024_ROOT = DATASET_ROOT / "ISIC2024"
MODEL_ROOT = AI_ROOT / "saved_models"

HAM10000_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMAGE_SIZE = 224
