"""
DermaVision AI - HAM10000 Dataset Loader

Storage-efficient dataset pipeline.

Features:
    - Reads HAM10000 metadata
    - Searches images from both image_part folders
    - No duplicate processed images are stored on disk
    - Resize to 224x224 on-the-fly
    - Optional denoising
    - Optional CLAHE enhancement
    - Training augmentation
    - Stratified train/validation/test split
    - Class-weighted sampling for class imbalance
    - PyTorch DataLoaders

HAM10000 Classes:
    akiec
    bcc
    bkl
    df
    mel
    nv
    vasc
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "ai_model" / "dataset" / "HAM10000"
)

DEFAULT_METADATA = (
    DEFAULT_DATASET_DIR / "HAM10000_metadata.csv"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

NUM_CLASSES = len(CLASS_NAMES)

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0

DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15

RANDOM_SEED = 42


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

    Image format:
        BGR uint8
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
    Apply light median + Gaussian filtering.

    Processing is performed in memory.
    No image is written to disk.
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
# FIND IMAGE FILES
# ============================================================

def build_image_index(
    dataset_dir: Path,
) -> Dict[str, Path]:
    """
    Search both HAM10000 image folders.

    Expected structure:

        HAM10000/
            HAM10000_images_part_1/
            HAM10000_images_part_2/
            HAM10000_metadata.csv

    Returns:

        {
            "ISIC_0027419": Path(...),
            ...
        }
    """

    image_index: Dict[str, Path] = {}

    image_folders = [
        dataset_dir / "HAM10000_images_part_1",
        dataset_dir / "HAM10000_images_part_2",
        dataset_dir / "images_part_1",
        dataset_dir / "images_part_2",
    ]

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    folders_found = 0

    for folder in image_folders:

        if not folder.exists():
            continue

        folders_found += 1

        logger.info(
            "Scanning image folder: %s",
            folder,
        )

        for image_path in folder.rglob("*"):

            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in supported_extensions:
                continue

            image_id = image_path.stem

            image_index[image_id] = image_path

    logger.info(
        "Image folders found: %d",
        folders_found,
    )

    logger.info(
        "Images indexed: %d",
        len(image_index),
    )

    return image_index


# ============================================================
# READ METADATA
# ============================================================

def load_metadata(
    metadata_path: Path,
) -> pd.DataFrame:
    """
    Load HAM10000 metadata CSV.
    """

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"Metadata file not found:\n{metadata_path}"
        )

    dataframe = pd.read_csv(
        metadata_path
    )

    required_columns = {
        "image_id",
        "dx",
    }

    missing = required_columns - set(
        dataframe.columns
    )

    if missing:

        raise ValueError(
            "Missing required metadata columns: "
            f"{sorted(missing)}"
        )

    dataframe = dataframe[
        [
            "image_id",
            "dx",
        ]
    ].copy()

    dataframe["label"] = dataframe["dx"].map(
        CLASS_TO_INDEX
    )

    dataframe = dataframe.dropna(
        subset=["label"]
    )

    dataframe["label"] = dataframe[
        "label"
    ].astype(int)

    return dataframe


# ============================================================
# PREPARE DATAFRAME
# ============================================================

def prepare_dataframe(
    metadata_path: Path,
    dataset_dir: Path,
) -> pd.DataFrame:
    """
    Combine metadata with actual image paths.
    """

    logger.info("=" * 60)
    logger.info("DERMAVISION AI - DATASET PREPARATION")
    logger.info("=" * 60)

    logger.info(
        "Dataset directory: %s",
        dataset_dir,
    )

    logger.info(
        "Metadata: %s",
        metadata_path,
    )

    metadata = load_metadata(
        metadata_path
    )

    logger.info(
        "Metadata records: %d",
        len(metadata),
    )

    image_index = build_image_index(
        dataset_dir
    )

    metadata["image_path"] = metadata[
        "image_id"
    ].map(image_index)

    before = len(metadata)

    metadata = metadata.dropna(
        subset=["image_path"]
    ).reset_index(drop=True)

    missing_images = before - len(metadata)

    if missing_images > 0:

        logger.warning(
            "Metadata entries without images: %d",
            missing_images,
        )

    if len(metadata) == 0:

        raise RuntimeError(
            "No matching HAM10000 images were found."
        )

    logger.info(
        "Usable images: %d",
        len(metadata),
    )

    return metadata


# ============================================================
# DATASET CLASS
# ============================================================

class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000.

    All preprocessing happens in RAM.

    No processed image files are created.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
        use_denoising: bool = True,
        use_clahe: bool = True,
    ):
        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

        self.use_denoising = use_denoising
        self.use_clahe = use_clahe

    def __len__(self) -> int:

        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, int]:

        row = self.dataframe.iloc[index]

        image_path = Path(
            row["image_path"]
        )

        label = int(
            row["label"]
        )

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise RuntimeError(
                f"Unable to read image: {image_path}"
            )

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            interpolation=cv2.INTER_AREA,
        )

        # ----------------------------------------------------
        # DENOISING
        # ----------------------------------------------------

        if self.use_denoising:

            image = apply_denoising(
                image
            )

        # ----------------------------------------------------
        # CLAHE
        # ----------------------------------------------------

        if self.use_clahe:

            image = apply_clahe(
                image
            )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(
            image
        )

        # ----------------------------------------------------
        # TRANSFORMS
        # ----------------------------------------------------

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

def get_train_transform():
    """
    Training transformations.

    Augmentation improves model generalization.
    """

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomVerticalFlip(
                p=0.2
            ),

            transforms.RandomRotation(
                degrees=15
            ),

            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),

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


def get_validation_transform():
    """
    Validation/test transformations.
    """

    return transforms.Compose(
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
# STRATIFIED SPLIT
# ============================================================

def create_splits(
    dataframe: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_seed: int = RANDOM_SEED,
):
    """
    Create stratified train / validation / test splits.

    Default:

        70% train
        15% validation
        15% test
    """

    if test_size <= 0 or test_size >= 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if val_size <= 0 or val_size >= 1:
        raise ValueError(
            "val_size must be between 0 and 1."
        )

    if test_size + val_size >= 1:
        raise ValueError(
            "test_size + val_size must be < 1."
        )

    train_df, temp_df = train_test_split(
        dataframe,
        test_size=test_size + val_size,
        stratify=dataframe["label"],
        random_state=random_seed,
    )

    relative_test_size = (
        test_size /
        (test_size + val_size)
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=random_seed,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(
    dataframe: pd.DataFrame,
    name: str,
) -> None:

    logger.info(
        "%s distribution:",
        name,
    )

    counts = dataframe[
        "dx"
    ].value_counts()

    for class_name in CLASS_NAMES:

        count = int(
            counts.get(
                class_name,
                0,
            )
        )

        logger.info(
            "  %-6s : %d",
            class_name,
            count,
        )


# ============================================================
# WEIGHTED SAMPLER
# ============================================================

def create_weighted_sampler(
    dataframe: pd.DataFrame,
) -> WeightedRandomSampler:
    """
    Create a weighted sampler to reduce class imbalance.

    Minority classes receive higher sampling probability.
    """

    class_counts = (
        dataframe["label"]
        .value_counts()
        .sort_index()
    )

    class_weights = {}

    for class_index in range(
        NUM_CLASSES
    ):

        count = int(
            class_counts.get(
                class_index,
                1,
            )
        )

        class_weights[
            class_index
        ] = 1.0 / count

    sample_weights = dataframe[
        "label"
    ].map(
        class_weights
    ).values

    sample_weights = torch.DoubleTensor(
        sample_weights
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(
            sample_weights
        ),
        replacement=True,
    )

    return sampler


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    metadata_path: Path = DEFAULT_METADATA,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    use_denoising: bool = True,
    use_clahe: bool = True,
):
    """
    Create train, validation and test DataLoaders.
    """

    dataframe = prepare_dataframe(
        metadata_path=metadata_path,
        dataset_dir=dataset_dir,
    )

    train_df, val_df, test_df = create_splits(
        dataframe
    )

    logger.info("")
    logger.info(
        "Train images      : %d",
        len(train_df),
    )

    logger.info(
        "Validation images : %d",
        len(val_df),
    )

    logger.info(
        "Test images       : %d",
        len(test_df),
    )

    logger.info("")

    print_class_distribution(
        train_df,
        "Training",
    )

    logger.info("")

    print_class_distribution(
        val_df,
        "Validation",
    )

    logger.info("")

    print_class_distribution(
        test_df,
        "Test",
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = HAM10000Dataset(
        train_df,
        transform=get_train_transform(),
        use_denoising=use_denoising,
        use_clahe=use_clahe,
    )

    val_dataset = HAM10000Dataset(
        val_df,
        transform=get_validation_transform(),
        use_denoising=use_denoising,
        use_clahe=use_clahe,
    )

    test_dataset = HAM10000Dataset(
        test_df,
        transform=get_validation_transform(),
        use_denoising=use_denoising,
        use_clahe=use_clahe,
    )

    # --------------------------------------------------------
    # SAMPLER
    # --------------------------------------------------------

    train_sampler = create_weighted_sampler(
        train_df
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ============================================================
# TEST DATASET
# ============================================================

def test_dataset(
    dataset_dir: Path,
    metadata_path: Path,
    batch_size: int = 4,
) -> None:
    """
    Test dataset and DataLoader.
    """

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        dataset_dir=dataset_dir,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=0,
        use_denoising=True,
        use_clahe=True,
    )

    # --------------------------------------------------------
    # GET ONE BATCH
    # --------------------------------------------------------

    images, labels = next(
        iter(train_loader)
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET TEST")
    logger.info("=" * 60)

    logger.info(
        "Batch image shape : %s",
        tuple(images.shape),
    )

    logger.info(
        "Batch label shape : %s",
        tuple(labels.shape),
    )

    logger.info(
        "Labels             : %s",
        labels.tolist(),
    )

    logger.info(
        "Pixel minimum      : %.4f",
        images.min().item(),
    )

    logger.info(
        "Pixel maximum      : %.4f",
        images.max().item(),
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

    logger.info("=" * 60)

    logger.info(
        "Dataset test completed successfully."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Test DermaVision AI HAM10000 "
            "storage-efficient dataset pipeline"
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="HAM10000 dataset directory",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="HAM10000 metadata CSV",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for testing",
    )

    args = parser.parse_args()

    test_dataset(
        dataset_dir=args.dataset,
        metadata_path=args.metadata,
        batch_size=args.batch_size,
    )