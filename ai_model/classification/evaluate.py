"""
DermaVision AI - Model Evaluation

Evaluates the trained EfficientNet-B0 model on the HAM10000
test dataset.

Outputs:
    - Test Loss
    - Test Accuracy
    - Weighted Precision / Recall / F1
    - Macro Precision / Recall / F1
    - Melanoma Recall
    - Confusion Matrix
    - Classification Report
    - Optional JSON report

Run from project root:

python -m ai_model.classification.evaluate ^
    --dataset ".\ai_model\database\HAM1000" ^
    --metadata ".\ai_model\database\HAM1000\HAM10000_metadata.csv" ^
    --model ".\ai_model\models\dermavision_efficientnet_b0_best.pth" ^
    --batch-size 8 ^
    --device cpu ^
    --report-output ".\ai_model\models\evaluation_results.json"
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from ai_model.classification.model import (
    DermaVisionEfficientNet,
    NUM_CLASSES,
    CLASS_NAMES,
)

from ai_model.classification.dataset import create_dataloaders


# ============================================================
# CONFIGURATION
# ============================================================

MEL_INDEX = CLASS_NAMES.index("mel")


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
# LOAD TRAINED MODEL
# ============================================================

def load_trained_model(
    model_path: Path,
    device: torch.device,
) -> nn.Module:
    """
    Load trained DermaVision EfficientNet-B0 model.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{model_path}"
        )

    logger.info(
        "Loading model: %s",
        model_path,
    )

    # Create architecture
    model = DermaVisionEfficientNet(
        num_classes=NUM_CLASSES,
        dropout=0.30,
        pretrained=False,
    )

    # Load checkpoint
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove DataParallel "module." prefix if present
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

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
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
    report_output: Path | None = None,
) -> dict:
    """
    Evaluate model on test dataset.

    Returns:
        Dictionary containing all evaluation metrics.
    """

    logger.info("")
    logger.info("=" * 60)
    logger.info("DERMAVISION AI - MODEL EVALUATION")
    logger.info("=" * 60)

    criterion = nn.CrossEntropyLoss()

    all_predictions = []
    all_labels = []

    total_loss = 0.0
    total_samples = 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(
            test_loader,
            start=1,
        ):

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            # Progress
            if batch_index % 50 == 0:

                logger.info(
                    "Evaluated %d test batches...",
                    batch_index,
                )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if total_samples == 0:

        raise RuntimeError(
            "Test dataset contains zero samples."
        )

    # --------------------------------------------------------
    # Test loss
    # --------------------------------------------------------

    test_loss = (
        total_loss / total_samples
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    # --------------------------------------------------------
    # Weighted metrics
    # --------------------------------------------------------

    precision_weighted = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    recall_weighted = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Macro metrics
    # --------------------------------------------------------

    precision_macro = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Melanoma recall
    # --------------------------------------------------------

    melanoma_recall = recall_score(
        all_labels,
        all_predictions,
        labels=[MEL_INDEX],
        average="macro",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report_dict = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    report_text = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)

    logger.info(
        "Test Loss              : %.4f",
        test_loss,
    )

    logger.info(
        "Test Accuracy          : %.4f (%.2f%%)",
        accuracy,
        accuracy * 100,
    )

    logger.info(
        "Weighted Precision     : %.4f",
        precision_weighted,
    )

    logger.info(
        "Weighted Recall        : %.4f",
        recall_weighted,
    )

    logger.info(
        "Weighted F1            : %.4f",
        f1_weighted,
    )

    logger.info(
        "Macro Precision        : %.4f",
        precision_macro,
    )

    logger.info(
        "Macro Recall           : %.4f",
        recall_macro,
    )

    logger.info(
        "Macro F1               : %.4f",
        f1_macro,
    )

    logger.info(
        "Melanoma Recall (mel)  : %.4f",
        melanoma_recall,
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 60)

    for line in report_text.splitlines():

        logger.info(
            "%s",
            line,
        )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("CONFUSION MATRIX")
    logger.info("=" * 60)

    logger.info(
        "Rows = Actual | Columns = Predicted"
    )

    header = "          " + " ".join(
        f"{name:>7}"
        for name in CLASS_NAMES
    )

    logger.info(
        "%s",
        header,
    )

    for index, row in enumerate(cm):

        row_text = " ".join(
            f"{value:7d}"
            for value in row
        )

        logger.info(
            "%-8s %s",
            CLASS_NAMES[index],
            row_text,
        )

    # ========================================================
    # CREATE RESULTS DICTIONARY
    # ========================================================

    results = {

        "timestamp": datetime.now().isoformat(),

        "model": str(report_output)
        if report_output is not None
        else "DermaVision EfficientNet-B0",

        "dataset": "HAM10000",

        "num_test_samples": total_samples,

        "classes": CLASS_NAMES,

        "test_loss": float(test_loss),

        "test_accuracy": float(accuracy),

        "test_accuracy_percent": float(
            accuracy * 100
        ),

        "weighted_precision": float(
            precision_weighted
        ),

        "weighted_recall": float(
            recall_weighted
        ),

        "weighted_f1": float(
            f1_weighted
        ),

        "macro_precision": float(
            precision_macro
        ),

        "macro_recall": float(
            recall_macro
        ),

        "macro_f1": float(
            f1_macro
        ),

        "melanoma_recall": float(
            melanoma_recall
        ),

        "confusion_matrix": cm.tolist(),

        "classification_report": report_dict,
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    if report_output is not None:

        report_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            report_output,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                results,
                file,
                indent=4,
            )

        logger.info("")
        logger.info(
            "Evaluation report saved:"
        )

        logger.info(
            "%s",
            report_output,
        )

    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETED")
    logger.info("=" * 60)

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DermaVision AI "
            "EfficientNet-B0 on HAM10000"
        )
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="HAM10000 dataset directory",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="HAM10000 metadata CSV",
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Trained .pth model/checkpoint",
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Optional JSON evaluation report",
    )

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size",
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
        help="Evaluation device",
    )

    args = parser.parse_args()

    # ========================================================
    # DEVICE
    # ========================================================

    if args.device == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but CUDA is not available."
            )

        device = torch.device("cuda")

    elif args.device == "cpu":

        device = torch.device("cpu")

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

    # ========================================================
    # VALIDATE PATHS
    # ========================================================

    if not args.dataset.exists():

        raise FileNotFoundError(
            f"Dataset directory not found:\n{args.dataset}"
        )

    if not args.metadata.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n{args.metadata}"
        )

    if not args.model.exists():

        raise FileNotFoundError(
            f"Model file not found:\n{args.model}"
        )

    # ========================================================
    # CREATE DATA LOADERS
    # ========================================================

    logger.info(
        "Creating data loaders..."
    )

    # IMPORTANT:
    #
    # DO NOT pass random_seed here.
    #
    # Your current dataset.py defines:
    #
    # create_dataloaders(
    #     dataset_dir,
    #     metadata_path,
    #     batch_size,
    #     num_workers,
    #     use_denoising,
    #     use_clahe
    # )
    #
    # Therefore random_seed causes:
    #
    # TypeError:
    # unexpected keyword argument 'random_seed'

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        dataset_dir=args.dataset,
        metadata_path=args.metadata,
        batch_size=args.batch_size,
        num_workers=0,
        use_denoising=True,
        use_clahe=True,
    )

    logger.info(
        "Train batches      : %d",
        len(train_loader),
    )

    logger.info(
        "Validation batches : %d",
        len(val_loader),
    )

    logger.info(
        "Test batches       : %d",
        len(test_loader),
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_trained_model(
        model_path=args.model,
        device=device,
    )

    # ========================================================
    # EVALUATE
    # ========================================================

    evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        report_output=args.report_output,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()