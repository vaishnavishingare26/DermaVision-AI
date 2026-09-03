from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import torch

from ai_model.classification.inference import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    DEFAULT_DROPOUT,
    generate_gradcam,
    get_device,
    load_model,
    predict_image,
)

from backend.services.preprocessing_service import (
    preprocess_image,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "ai_model"
    / "models"
    / "dermavision_efficientnet_b0_best.pth"
)


# ============================================================
# DEVICE
# ============================================================

_DEVICE = get_device("auto")


# ============================================================
# LOAD MODEL ONCE
# ============================================================

_MODEL = load_model(
    model_path=MODEL_PATH,
    device=_DEVICE,
)


# ============================================================
# HELPER - FILE TO BASE64
# ============================================================

def file_to_data_url(
    file_path: Path,
) -> str:
    """
    Convert an image file into a base64 data URL.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Grad-CAM image not found: {file_path}"
        )

    image_bytes = file_path.read_bytes()

    encoded = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_uploaded_image(
    image_bytes: bytes,
) -> dict:
    """
    Complete AI pipeline for an uploaded image.

    Pipeline:

        Uploaded Image
              ↓
        EfficientNet-B0
              ↓
        Prediction
              ↓
        Real Grad-CAM
              ↓
        Base64 Grad-CAM
    """

    if not image_bytes:
        raise ValueError(
            "Uploaded image is empty."
        )

    # --------------------------------------------------------
    # CREATE TEMPORARY IMAGE
    # --------------------------------------------------------

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False,
        ) as temp_file:

            temp_file.write(
                image_bytes
            )

            temp_path = Path(
                temp_file.name
            )

        # ----------------------------------------------------
        # NORMAL PREDICTION
        # ----------------------------------------------------

        prediction = predict_image(
            model=_MODEL,
            image_path=temp_path,
            device=_DEVICE,
            top_k=3,
        )

        # ----------------------------------------------------
        # PREPROCESSING IMAGE
        # ----------------------------------------------------

        preprocessing = preprocess_image(
            image_bytes
        )

        # ----------------------------------------------------
        # GRAD-CAM OUTPUT PATH
        # ----------------------------------------------------

        gradcam_dir = (
            PROJECT_ROOT
            / "ai_model"
            / "gradcam"
        )

        gradcam_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        gradcam_path = (
            gradcam_dir
            / "latest_gradcam.jpg"
        )

        # ----------------------------------------------------
        # REAL GRAD-CAM
        # ----------------------------------------------------

        gradcam_result = generate_gradcam(
            model=_MODEL,
            image_path=temp_path,
            device=_DEVICE,
            output_path=gradcam_path,
        )

        # ----------------------------------------------------
        # CONVERT GRAD-CAM TO BASE64
        # ----------------------------------------------------

        gradcam_image = file_to_data_url(
            Path(
                gradcam_result[
                    "heatmap_path"
                ]
            )
        )

        # ----------------------------------------------------
        # FINAL API RESPONSE
        # ----------------------------------------------------

        result = {
            "success": True,

            # Prediction
            "prediction": prediction[
                "condition"
            ],

            "predicted_class": prediction[
                "class"
            ],

            "condition": prediction[
                "condition"
            ],

            "confidence": prediction[
                "confidence"
            ],

            "risk": prediction[
                "risk"
            ],

            "suspicious": prediction[
                "suspicious"
            ],

            # Model
            "model": prediction[
                "model"
            ],

            "dataset": prediction[
                "dataset"
            ],

            "device": prediction[
                "device"
            ],

            # Predictions
            "top_predictions": prediction[
                "top_predictions"
            ],

            "probabilities": prediction[
                "probabilities"
            ],

            # Preprocessing
            "preprocessed_image":
                preprocessing["image"],

            "preprocessing_steps":
                preprocessing["processing"],

            # Grad-CAM
            "gradcam": {
                "available": True,

                "class":
                    gradcam_result["class"],

                "condition":
                    gradcam_result["condition"],

                "image":
                    gradcam_image,
            },

            # Backward-compatible direct field
            "gradcam_image":
                gradcam_image,
        }

        return result

    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY FILE
        # ----------------------------------------------------

        if (
            temp_path is not None
            and temp_path.exists()
        ):

            try:
                temp_path.unlink()
            except Exception:
                pass