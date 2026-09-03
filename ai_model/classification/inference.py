"""
DermaVision AI - Skin Lesion Inference + Grad-CAM

Runs inference using the trained EfficientNet-B0 model
and generates a real Grad-CAM explanation heatmap.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ai_model.classification.model import (
    DermaVisionEfficientNet,
    NUM_CLASSES,
    CLASS_NAMES,
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

DEFAULT_DROPOUT = 0.30

DEFAULT_MODEL = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "dermavision_efficientnet_b0_best.pth"
)


# ============================================================
# DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {
    "akiec": "Actinic Keratosis / Bowen Disease",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion",
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("dermavision")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE to the luminance channel.

    Input:
        BGR uint8 image

    Output:
        BGR uint8 image
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel,
        )
    )

    return cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )


def apply_denoising(image: np.ndarray) -> np.ndarray:
    """
    Apply the same light denoising used during training.
    """

    image = cv2.medianBlur(
        image,
        3,
    )

    image = cv2.GaussianBlur(
        image,
        (3, 3),
        0,
    )

    return image


# ============================================================
# TRANSFORM
# ============================================================

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# DEVICE
# ============================================================

def get_device(
    requested_device: str = "auto",
) -> torch.device:
    """
    Select CPU or CUDA.
    """

    if requested_device == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    if requested_device == "cpu":
        return torch.device("cpu")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model_path: Path,
    device: torch.device,
):
    """
    Load trained EfficientNet-B0 checkpoint.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    logger.info(
        "Loading model: %s",
        model_path,
    )

    model = DermaVisionEfficientNet(
        num_classes=NUM_CLASSES,
        dropout=DEFAULT_DROPOUT,
        pretrained=False,
    )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model = model.to(device)

    model.eval()

    logger.info(
        "Model loaded successfully."
    )

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image_path: Path,
) -> torch.Tensor:
    """
    Apply the same preprocessing pipeline used
    during validation/test.

    Pipeline:

        Read image
        ↓
        Resize 224x224
        ↓
        Denoising
        ↓
        CLAHE
        ↓
        BGR -> RGB
        ↓
        PIL
        ↓
        Tensor
        ↓
        ImageNet Normalize
    """

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    logger.info(
        "Reading image: %s",
        image_path,
    )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Unable to read image:\n{image_path}"
        )

    image = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    image = apply_denoising(image)

    image = apply_clahe(image)

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    image = Image.fromarray(
        image
    )

    tensor = INFERENCE_TRANSFORM(
        image
    )

    tensor = tensor.unsqueeze(0)

    return tensor


# ============================================================
# RISK LEVEL
# ============================================================

def get_risk_level(
    predicted_class: str,
) -> str:
    """
    Give a project-level risk category.

    NOTE:
    This is NOT a medical diagnosis.
    """

    if predicted_class == "mel":
        return "High"

    if predicted_class in {
        "bcc",
        "akiec",
    }:
        return "High"

    if predicted_class in {
        "bkl",
        "df",
        "nv",
        "vasc",
    }:
        return "Low / Moderate"

    return "Unknown"


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:
    """
    Real Grad-CAM implementation for EfficientNet-B0.

    Uses the last convolutional feature layer:

        model.backbone.features[-1]
    """

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self._backward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):
        self.activations = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):
        self.gradients = grad_output[0]

    def generate(
        self,
        image_tensor,
        target_class=None,
    ):
        """
        Generate Grad-CAM heatmap.

        Returns:
            heatmap: numpy array 0-255
            target_class: selected class index
        """

        self.model.zero_grad(set_to_none=True)

        image_tensor = image_tensor.clone().detach()
        image_tensor.requires_grad_(True)

        logits = self.model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        if target_class is None:

            target_class = int(
                torch.argmax(
                    probabilities,
                    dim=1,
                ).item()
            )

        target_score = logits[
            0,
            target_class,
        ]

        target_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        activations = self.activations.detach()

        gradients = self.gradients.detach()

        # Global average pooling of gradients
        weights = torch.mean(
            gradients,
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted feature maps
        cam = torch.sum(
            weights * activations,
            dim=1,
        )

        # ReLU
        cam = torch.relu(cam)

        cam = cam[0].cpu().numpy()

        # Normalize 0-1
        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(
                cam,
                dtype=np.float32,
            )

        # Resize to original model input size
        cam = cv2.resize(
            cam,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        heatmap = np.uint8(
            255 * cam
        )

        return heatmap, target_class

    def close(self):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ============================================================
# GRAD-CAM IMAGE OVERLAY
# ============================================================

def create_gradcam_overlay(
    original_image_path: Path,
    heatmap: np.ndarray,
    output_path: Path,
):
    """
    Create a real Grad-CAM heatmap overlay
    on the original image.
    """

    original = cv2.imread(
        str(original_image_path),
        cv2.IMREAD_COLOR,
    )

    if original is None:
        raise RuntimeError(
            "Unable to read original image for Grad-CAM."
        )

    original = cv2.resize(
        original,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    # OpenCV JET is only used to visualize
    # the computed Grad-CAM intensity.
    color_map = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        color_map,
        0.45,
        0,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(output_path),
        overlay,
    )

    if not success:
        raise RuntimeError(
            f"Unable to save Grad-CAM image:\n{output_path}"
        )

    return output_path


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image_path: Path,
    device: torch.device,
    output_path: Path | None = None,
):
    """
    Generate real Grad-CAM for the predicted class.

    Returns:
        dictionary containing Grad-CAM information.
    """

    image_tensor = preprocess_image(
        image_path
    ).to(device)

    # EfficientNet-B0 final convolutional block
    target_layer = model.backbone.features[-1]

    gradcam = GradCAM(
        model=model,
        target_layer=target_layer,
    )

    try:

        heatmap, target_class = (
            gradcam.generate(
                image_tensor
            )
        )

    finally:

        gradcam.close()

    predicted_class = CLASS_NAMES[
        target_class
    ]

    condition = DISPLAY_NAMES.get(
        predicted_class,
        predicted_class,
    )

    if output_path is None:

        output_path = (
            Path(__file__).resolve().parents[1]
            / "gradcam"
            / "gradcam_result.jpg"
        )

    saved_path = create_gradcam_overlay(
        original_image_path=image_path,
        heatmap=heatmap,
        output_path=output_path,
    )

    logger.info(
        "Grad-CAM generated for: %s",
        condition,
    )

    logger.info(
        "Grad-CAM saved to: %s",
        saved_path,
    )

    return {
        "success": True,
        "class": predicted_class,
        "condition": condition,
        "heatmap_path": str(saved_path),
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_image(
    model,
    image_path: Path,
    device: torch.device,
    top_k: int = 3,
) -> dict:
    """
    Predict HAM10000 skin lesion class.

    Returns:
        dictionary containing prediction,
        confidence and top predictions.
    """

    image_tensor = preprocess_image(
        image_path
    )

    image_tensor = image_tensor.to(
        device
    )

    with torch.inference_mode():

        logits = model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    top_k = min(
        top_k,
        NUM_CLASSES,
    )

    top_probabilities, top_indices = (
        torch.topk(
            probabilities,
            k=top_k,
        )
    )

    top_predictions = []

    for probability, index in zip(
        top_probabilities,
        top_indices,
    ):

        class_index = int(
            index.item()
        )

        class_name = CLASS_NAMES[
            class_index
        ]

        confidence = float(
            probability.item() * 100
        )

        top_predictions.append(
            {
                "class": class_name,
                "condition": DISPLAY_NAMES.get(
                    class_name,
                    class_name,
                ),
                "confidence": round(
                    confidence,
                    2,
                ),
            }
        )

    predicted_index = int(
        torch.argmax(
            probabilities
        ).item()
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[
            predicted_index
        ].item()
        * 100
    )

    condition = DISPLAY_NAMES.get(
        predicted_class,
        predicted_class,
    )

    risk = get_risk_level(
        predicted_class
    )

    all_probabilities = {}

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        all_probabilities[
            class_name
        ] = round(
            float(
                probabilities[index].item()
                * 100
            ),
            2,
        )

    result = {
        "success": True,

        "class": predicted_class,

        "condition": condition,

        "confidence": round(
            confidence,
            2,
        ),

        "risk": risk,

        "suspicious": predicted_class
        in {
            "mel",
            "bcc",
            "akiec",
        },

        "model": "EfficientNet-B0",

        "dataset": "HAM10000",

        "device": str(device),

        "top_predictions":
            top_predictions,

        "probabilities":
            all_probabilities,
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "DermaVision AI - "
            "EfficientNet-B0 Inference + Grad-CAM"
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to trained model checkpoint",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        help="Inference device",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions",
    )

    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate Grad-CAM visualization",
    )

    parser.add_argument(
        "--gradcam-output",
        type=Path,
        default=None,
        help="Path for Grad-CAM output image",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = get_device(
        args.device
    )

    logger.info(
        "Device: %s",
        device,
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_model(
        model_path=args.model,
        device=device,
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    result = predict_image(
        model=model,
        image_path=args.image,
        device=device,
        top_k=args.top_k,
    )

    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DERMAVISION AI - PREDICTION")
    print("=" * 60)

    print(
        f"Image       : {args.image}"
    )

    print(
        f"Model       : {result['model']}"
    )

    print(
        f"Prediction  : {result['condition']}"
    )

    print(
        f"Class       : {result['class']}"
    )

    print(
        f"Confidence  : {result['confidence']:.2f}%"
    )

    print(
        f"Risk        : {result['risk']}"
    )

    print(
        f"Suspicious  : {result['suspicious']}"
    )

    print()
    print("TOP PREDICTIONS")
    print("-" * 60)

    for item in result[
        "top_predictions"
    ]:

        print(
            f"{item['condition']:<40}"
            f"{item['confidence']:>7.2f}%"
        )

    print()
    print("ALL CLASS PROBABILITIES")
    print("-" * 60)

    for class_name, probability in (
        result["probabilities"].items()
    ):

        print(
            f"{class_name:<10}"
            f"{DISPLAY_NAMES[class_name]:<40}"
            f"{probability:>7.2f}%"
        )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    if args.gradcam:

        print()
        print("GENERATING REAL GRAD-CAM...")
        print("-" * 60)

        gradcam_result = generate_gradcam(
            model=model,
            image_path=args.image,
            device=device,
            output_path=args.gradcam_output,
        )

        print(
            f"Grad-CAM Class : "
            f"{gradcam_result['condition']}"
        )

        print(
            f"Grad-CAM Image : "
            f"{gradcam_result['heatmap_path']}"
        )

    print()
    print("=" * 60)

    print()
    print(
        "NOTE: This is an AI research prediction, "
        "not a medical diagnosis."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()