"""
DermaVision AI - Skin Cancer Classification Model

EfficientNet-B0 based classifier for HAM10000.

Classes:
    0 - akiec
    1 - bcc
    2 - bkl
    3 - df
    4 - mel
    5 - nv
    6 - vasc
"""

import argparse
import logging

import torch
import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 7

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]

DEFAULT_DROPOUT = 0.30


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
# MODEL
# ============================================================

class DermaVisionEfficientNet(nn.Module):
    """
    EfficientNet-B0 classifier for HAM10000.

    Uses ImageNet pretrained weights by default.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = DEFAULT_DROPOUT,
        pretrained: bool = True,
    ):
        super().__init__()

        # ----------------------------------------------------
        # LOAD BACKBONE
        # ----------------------------------------------------

        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None

        self.backbone = efficientnet_b0(
            weights=weights
        )

        # ----------------------------------------------------
        # GET CLASSIFIER INPUT SIZE
        # ----------------------------------------------------

        classifier_input_features = (
            self.backbone.classifier[1].in_features
        )

        # ----------------------------------------------------
        # REPLACE ORIGINAL CLASSIFIER
        # ----------------------------------------------------

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(
                p=dropout
            ),

            nn.Linear(
                classifier_input_features,
                num_classes,
            ),
        )

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.backbone(x)


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    num_classes: int = NUM_CLASSES,
    dropout: float = DEFAULT_DROPOUT,
    pretrained: bool = True,
) -> DermaVisionEfficientNet:
    """
    Create the DermaVision classification model.
    """

    if num_classes < 2:
        raise ValueError(
            "num_classes must be >= 2"
        )

    if not 0.0 <= dropout < 1.0:
        raise ValueError(
            "dropout must be between 0 and 1"
        )

    model = DermaVisionEfficientNet(
        num_classes=num_classes,
        dropout=dropout,
        pretrained=pretrained,
    )

    return model


# ============================================================
# MODEL INFORMATION
# ============================================================

def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    """
    Return total and trainable parameter counts.
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


# ============================================================
# TEST MODEL
# ============================================================

def test_model(
    model: nn.Module,
    device: torch.device,
) -> None:
    """
    Perform a dummy forward pass.
    """

    model = model.to(device)

    # HAM10000 input:
    # batch = 2
    # channels = 3
    # height = 224
    # width = 224

    dummy_input = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    model.eval()

    with torch.no_grad():

        output = model(
            dummy_input
        )

    logger.info(
        "Input shape  : %s",
        tuple(dummy_input.shape),
    )

    logger.info(
        "Output shape : %s",
        tuple(output.shape),
    )

    logger.info(
        "Expected     : (2, %d)",
        NUM_CLASSES,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Test DermaVision AI "
            "EfficientNet-B0 classifier"
        )
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help=(
            "Do not use ImageNet pretrained weights"
        ),
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
        help="Dropout probability",
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
        help="Execution device",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if args.device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was requested but is not available."
            )

        device = torch.device(
            "cuda"
        )

    elif args.device == "cpu":

        device = torch.device(
            "cpu"
        )

    else:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    logger.info(
        "Device: %s",
        device,
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = create_model(
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
    )

    # --------------------------------------------------------
    # PARAMETER COUNT
    # --------------------------------------------------------

    total_params, trainable_params = (
        count_parameters(model)
    )

    logger.info(
        "Total parameters    : %s",
        f"{total_params:,}",
    )

    logger.info(
        "Trainable parameters: %s",
        f"{trainable_params:,}",
    )

    # --------------------------------------------------------
    # TEST FORWARD PASS
    # --------------------------------------------------------

    test_model(
        model,
        device,
    )

    logger.info(
        "Model test completed successfully."
    )