"""
DermaVision AI - EfficientNet-B0 Training

HAM10000 Skin Lesion Classification

Classes:
0 - akiec
1 - bcc
2 - bkl
3 - df
4 - mel
5 - nv
6 - vasc

Features:
- EfficientNet-B0
- ImageNet pretrained weights
- Class-weighted CrossEntropyLoss
- Training / validation
- Early stopping
- ReduceLROnPlateau scheduler
- Best model checkpoint
- Training history
- Accuracy, precision, recall and F1-score
- Confusion matrix
- CPU / CUDA support
"""

import argparse
import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from model import create_model, CLASS_NAMES
from dataset import create_dataloaders


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 7

DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4

DEFAULT_PATIENCE = 5

RANDOM_SEED = 42

DEFAULT_DATASET = Path(
    "ai_model/database/HAM1000"
)

DEFAULT_METADATA = Path(
    "ai_model/database/HAM1000/HAM10000_metadata.csv"
)

DEFAULT_OUTPUT = Path(
    "ai_model/models"
)


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
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    logger.info(
        "Random seed: %d",
        seed,
    )


# ============================================================
# DEVICE
# ============================================================

def get_device(
    requested_device: str,
) -> torch.device:
    """
    Select CPU or CUDA device.
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
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    train_loader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Calculate inverse-frequency class weights
    from the training dataset.
    """

    counts = np.zeros(
        num_classes,
        dtype=np.float64,
    )

    dataset = train_loader.dataset

    # Try to obtain labels from the dataset.
    if hasattr(dataset, "dataframe"):

        labels = dataset.dataframe["label"].to_numpy()

    elif hasattr(dataset, "df"):

        labels = dataset.df["label"].to_numpy()

    else:

        # Safe fallback:
        labels = []

        for _, batch_labels in train_loader:

            labels.extend(
                batch_labels.cpu().numpy().tolist()
            )

    for label in labels:

        label = int(label)

        if 0 <= label < num_classes:

            counts[label] += 1

    if np.sum(counts) == 0:

        raise RuntimeError(
            "Unable to calculate class frequencies."
        )

    logger.info("")
    logger.info("Training class distribution:")

    for index, count in enumerate(counts):

        logger.info(
            "  %-5s : %d",
            CLASS_NAMES[index],
            int(count),
        )

    # Inverse frequency weighting
    weights = np.zeros_like(counts)

    total = np.sum(counts)

    for index in range(num_classes):

        if counts[index] > 0:

            weights[index] = (
                total /
                (num_classes * counts[index])
            )

    weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    logger.info("")
    logger.info("Class weights:")

    for index, weight in enumerate(weights):

        logger.info(
            "  %-5s : %.4f",
            CLASS_NAMES[index],
            weight.item(),
        )

    return weights


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
) -> tuple[float, float]:

    model.train()

    running_loss = 0.0

    all_predictions = []
    all_labels = []

    total_samples = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        running_loss += (
            loss.item() *
            batch_size
        )

        total_samples += batch_size

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    epoch_loss = (
        running_loss /
        max(total_samples, 1)
    )

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# ============================================================
# VALIDATE
# ============================================================

def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]:

    model.eval()

    running_loss = 0.0

    total_samples = 0

    all_predictions = []

    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

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

            batch_size = images.size(0)

            running_loss += (
                loss.item() *
                batch_size
            )

            total_samples += batch_size

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_predictions.extend(
                predictions.cpu()
                .numpy()
            )

            all_labels.extend(
                labels.cpu()
                .numpy()
            )

    validation_loss = (
        running_loss /
        max(total_samples, 1)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )
    )

    return (
        validation_loss,
        accuracy,
        precision,
        recall,
        f1,
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_f1,
    history,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_f1": best_f1,
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "history": history,
    }

    torch.save(
        checkpoint,
        output_path,
    )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    dataset_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
) -> None:

    logger.info("=" * 60)
    logger.info(
        "DERMAVISION AI - MODEL TRAINING"
    )
    logger.info("=" * 60)

    logger.info(
        "Dataset  : %s",
        dataset_dir,
    )

    logger.info(
        "Metadata : %s",
        metadata_path,
    )

    logger.info(
        "Device   : %s",
        device,
    )

    logger.info(
        "Epochs   : %d",
        epochs,
    )

    logger.info(
        "Batch    : %d",
        batch_size,
    )

    logger.info(
        "LR       : %.6f",
        learning_rate,
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    logger.info("")
    logger.info("Creating data loaders...")

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        dataset_dir=dataset_dir,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=0,
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

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    logger.info("")
    logger.info(
        "Creating EfficientNet-B0..."
    )

    model = create_model(
        num_classes=NUM_CLASSES,
        dropout=0.30,
        pretrained=True,
    )

    model = model.to(device)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    logger.info(
        "Total parameters    : %s",
        f"{total_parameters:,}",
    )

    logger.info(
        "Trainable parameters: %s",
        f"{trainable_parameters:,}",
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_loader,
        NUM_CLASSES,
        device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model_path = (
        output_dir /
        "dermavision_efficientnet_b0_best.pth"
    )

    final_model_path = (
        output_dir /
        "dermavision_efficientnet_b0_final.pth"
    )

    history_path = (
        output_dir /
        "training_history.json"
    )

    # --------------------------------------------------------
    # TRAINING VARIABLES
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "learning_rate": [],
    }

    best_f1 = -1.0

    best_state = None

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # TRAIN LOOP
    # --------------------------------------------------------

    for epoch in range(1, epochs + 1):

        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "EPOCH %d/%d",
            epoch,
            epochs,
        )
        logger.info("=" * 60)

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Learning rate: %.8f",
            current_lr,
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        (
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
        ) = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step(
            val_f1
        )

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        history["val_precision"].append(
            val_precision
        )

        history["val_recall"].append(
            val_recall
        )

        history["val_f1"].append(
            val_f1
        )

        history["learning_rate"].append(
            current_lr
        )

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        logger.info("")
        logger.info(
            "Train Loss      : %.4f",
            train_loss,
        )

        logger.info(
            "Train Accuracy  : %.4f",
            train_accuracy,
        )

        logger.info(
            "Val Loss        : %.4f",
            val_loss,
        )

        logger.info(
            "Val Accuracy    : %.4f",
            val_accuracy,
        )

        logger.info(
            "Val Precision   : %.4f",
            val_precision,
        )

        logger.info(
            "Val Recall      : %.4f",
            val_recall,
        )

        logger.info(
            "Val F1          : %.4f",
            val_f1,
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_f1 > best_f1:

            best_f1 = val_f1

            epochs_without_improvement = 0

            best_state = copy.deepcopy(
                model.state_dict()
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_f1=best_f1,
                history=history,
                output_path=best_model_path,
            )

            logger.info("")
            logger.info(
                "NEW BEST MODEL!"
            )

            logger.info(
                "Best validation F1: %.4f",
                best_f1,
            )

            logger.info(
                "Saved: %s",
                best_model_path,
            )

        else:

            epochs_without_improvement += 1

            logger.info(
                "No improvement: %d/%d",
                epochs_without_improvement,
                patience,
            )

        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        with open(
            history_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                history,
                file,
                indent=4,
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            logger.info("")
            logger.info(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "class_names":
                CLASS_NAMES,

            "num_classes":
                NUM_CLASSES,

            "best_validation_f1":
                best_f1,

            "history":
                history,
        },
        final_model_path,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "TRAINING COMPLETED"
    )
    logger.info("=" * 60)

    logger.info(
        "Best validation F1 : %.4f",
        best_f1,
    )

    logger.info(
        "Best model         : %s",
        best_model_path,
    )

    logger.info(
        "Final model        : %s",
        final_model_path,
    )

    logger.info(
        "History            : %s",
        history_path,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Train DermaVision AI "
            "EfficientNet-B0 on HAM10000"
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="HAM10000 dataset directory",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="HAM10000 metadata CSV",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Model output directory",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Initial learning rate",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="AdamW weight decay",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early stopping patience",
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
        help="Training device",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if args.epochs < 1:

        raise ValueError(
            "epochs must be >= 1"
        )

    if args.batch_size < 1:

        raise ValueError(
            "batch-size must be >= 1"
        )

    if args.learning_rate <= 0:

        raise ValueError(
            "learning-rate must be > 0"
        )

    if args.patience < 1:

        raise ValueError(
            "patience must be >= 1"
        )

    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    set_seed(
        args.seed
    )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = get_device(
        args.device
    )

    logger.info(
        "Using device: %s",
        device,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_model(
        dataset_dir=args.dataset,
        metadata_path=args.metadata,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
    )