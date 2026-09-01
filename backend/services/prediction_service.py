from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ai_model.classification.model import create_model, CLASS_NAMES, NUM_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "ai_model" / "models" / "dermavision_efficientnet_b0_best.pth"

DISPLAY_NAMES = {
    "akiec": "Actinic Keratosis / Bowen Disease",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}

# HAM10000 classes that are treated as cancer/suspicious for the UI.
# This is a screening grouping, not a clinical diagnosis.
SUSPICIOUS_CLASSES = {"akiec", "bcc", "mel"}

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _apply_denoising(image: np.ndarray) -> np.ndarray:
    image = cv2.medianBlur(image, 3)
    return cv2.GaussianBlur(image, (3, 3), 0)


def _apply_clahe(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a valid image.")

    # Match the current HAM10000 validation preprocessing used during training.
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = _apply_denoising(image)
    image = _apply_clahe(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image)
    return EVAL_TRANSFORM(pil).unsqueeze(0)


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = create_model(pretrained=False, dropout=0.30)
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=_DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint

    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(_DEVICE)
    model.eval()
    _MODEL = model
    return _MODEL


def predict_image(image_bytes: bytes) -> dict:
    model = _load_model()
    tensor = _preprocess(image_bytes).to(_DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)

    class_name = CLASS_NAMES[int(index.item())]
    confidence_pct = float(confidence.item() * 100.0)

    if class_name in SUSPICIOUS_CLASSES:
        status = "Cancer / suspicious"
        risk = "High" if confidence_pct >= 50 else "Medium"
    else:
        status = "Non-cancer / benign-class"
        risk = "Low" if confidence_pct >= 70 else "Medium"

    return {
        "prediction": DISPLAY_NAMES[class_name],
        "predicted_class": class_name,
        "condition": DISPLAY_NAMES[class_name],
        "confidence": confidence_pct,
        "risk": risk,
        "cancer_status": status,
        "probabilities": {
            DISPLAY_NAMES[name]: float(probabilities[i].item() * 100.0)
            for i, name in enumerate(CLASS_NAMES)
        },
        "model": "DermaVision EfficientNet-B0",
        "device": str(_DEVICE),
    }
