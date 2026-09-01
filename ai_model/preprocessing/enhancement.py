"""
DermaVision AI - CLAHE Image Enhancement

Enhances local contrast in dermoscopic images using
Contrast Limited Adaptive Histogram Equalization (CLAHE).

Processing:
    Denoised Image
          ↓
        CLAHE
          ↓
    Enhanced Image
"""

import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

OUTPUT_FORMAT = ".jpg"
JPEG_QUALITY = 95

# CLAHE parameters
DEFAULT_CLIP_LIMIT = 2.0
DEFAULT_TILE_GRID_SIZE = 8


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
# VALIDATION
# ============================================================

def validate_clahe_parameters(
    clip_limit: float,
    tile_grid_size: int,
) -> None:
    """Validate CLAHE parameters."""

    if clip_limit <= 0:
        raise ValueError(
            "clip_limit must be greater than 0"
        )

    if tile_grid_size < 1:
        raise ValueError(
            "tile_grid_size must be >= 1"
        )


# ============================================================
# ENHANCE SINGLE IMAGE
# ============================================================

def enhance_image(
    input_path: Path,
    output_path: Path,
    clip_limit: float,
    tile_grid_size: int,
    overwrite: bool = False,
) -> tuple[Path, bool, str]:
    """
    Apply CLAHE enhancement to one image.

    CLAHE is applied to the L channel in LAB color space
    so local luminance/contrast can be enhanced while
    preserving the original color information.

    Returns:
        (input_path, success, message)
    """

    # --------------------------------------------------------
    # SKIP EXISTING OUTPUT
    # --------------------------------------------------------

    if output_path.exists() and not overwrite:
        return (
            input_path,
            True,
            "skipped - already exists",
        )

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(
            str(input_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            return (
                input_path,
                False,
                "unable to read image",
            )

        # ----------------------------------------------------
        # VALIDATE IMAGE DIMENSIONS
        # ----------------------------------------------------

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            return (
                input_path,
                False,
                "invalid image dimensions",
            )

        # ----------------------------------------------------
        # BGR → LAB
        # ----------------------------------------------------

        lab_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2LAB,
        )

        # ----------------------------------------------------
        # SPLIT LAB CHANNELS
        # ----------------------------------------------------

        l_channel, a_channel, b_channel = cv2.split(
            lab_image
        )

        # ----------------------------------------------------
        # CREATE CLAHE
        # ----------------------------------------------------

        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(
                tile_grid_size,
                tile_grid_size,
            ),
        )

        # ----------------------------------------------------
        # APPLY CLAHE ONLY TO L CHANNEL
        # ----------------------------------------------------

        enhanced_l = clahe.apply(
            l_channel
        )

        # ----------------------------------------------------
        # MERGE CHANNELS
        # ----------------------------------------------------

        enhanced_lab = cv2.merge(
            (
                enhanced_l,
                a_channel,
                b_channel,
            )
        )

        # ----------------------------------------------------
        # LAB → BGR
        # ----------------------------------------------------

        enhanced_image = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2BGR,
        )

        # ----------------------------------------------------
        # CREATE OUTPUT DIRECTORY
        # ----------------------------------------------------

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # SAVE IMAGE
        # ----------------------------------------------------

        saved = cv2.imwrite(
            str(output_path),
            enhanced_image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY,
            ],
        )

        if not saved:
            return (
                input_path,
                False,
                "failed to save image",
            )

        return (
            input_path,
            True,
            "OK",
        )

    except Exception as error:

        return (
            input_path,
            False,
            str(error),
        )


# ============================================================
# COLLECT IMAGE PATHS
# ============================================================

def collect_image_paths(
    input_folder: Path,
) -> list[Path]:
    """Recursively collect supported image files."""

    return [
        path
        for path in input_folder.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_FORMATS
        )
    ]


# ============================================================
# LOG RESULT
# ============================================================

def log_result(
    index: int,
    total: int,
    path: Path,
    success: bool,
    message: str,
) -> None:

    status = "OK" if success else "FAIL"

    logger.info(
        "[%d/%d] %-4s %s (%s)",
        index,
        total,
        status,
        path.name,
        message,
    )


# ============================================================
# ENHANCE DATASET
# ============================================================

def enhance_dataset(
    input_folder: Path,
    output_folder: Path,
    overwrite: bool = False,
    workers: int = 1,
    clip_limit: float = DEFAULT_CLIP_LIMIT,
    tile_grid_size: int = DEFAULT_TILE_GRID_SIZE,
) -> None:
    """Apply CLAHE enhancement to all supported images."""

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if not input_folder.exists():

        logger.error(
            "Input folder does not exist: %s",
            input_folder,
        )

        return

    if workers < 1:
        raise ValueError(
            "workers must be >= 1"
        )

    validate_clahe_parameters(
        clip_limit,
        tile_grid_size,
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # COLLECT IMAGES
    # --------------------------------------------------------

    image_paths = collect_image_paths(
        input_folder
    )

    total = len(image_paths)

    # --------------------------------------------------------
    # LOG CONFIGURATION
    # --------------------------------------------------------

    logger.info("=" * 60)
    logger.info(
        "DERMAVISION AI - CLAHE IMAGE ENHANCEMENT"
    )
    logger.info("=" * 60)

    logger.info(
        "Input       : %s",
        input_folder,
    )

    logger.info(
        "Output      : %s",
        output_folder,
    )

    logger.info(
        "Clip limit  : %.2f",
        clip_limit,
    )

    logger.info(
        "Tile grid   : %d x %d",
        tile_grid_size,
        tile_grid_size,
    )

    logger.info(
        "Images      : %d found",
        total,
    )

    logger.info(
        "Workers     : %d",
        workers,
    )

    logger.info("")

    if total == 0:

        logger.warning(
            "No supported images found."
        )

        return

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # BUILD JOBS
    # --------------------------------------------------------

    jobs = []

    for image_path in image_paths:

        relative_path = (
            image_path
            .relative_to(input_folder)
            .with_suffix(OUTPUT_FORMAT)
        )

        output_path = (
            output_folder /
            relative_path
        )

        jobs.append(
            (
                image_path,
                output_path,
            )
        )

    # ========================================================
    # SINGLE PROCESS
    # ========================================================

    if workers == 1:

        for index, (
            input_path,
            output_path,
        ) in enumerate(
            jobs,
            start=1,
        ):

            (
                _,
                success,
                message,
            ) = enhance_image(
                input_path=input_path,
                output_path=output_path,
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size,
                overwrite=overwrite,
            )

            log_result(
                index,
                total,
                input_path,
                success,
                message,
            )

            if success:
                successful += 1
            else:
                failed += 1

    # ========================================================
    # MULTI-PROCESS
    # ========================================================

    else:

        with ProcessPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = {
                executor.submit(
                    enhance_image,
                    input_path,
                    output_path,
                    clip_limit,
                    tile_grid_size,
                    overwrite,
                ): input_path

                for input_path, output_path in jobs
            }

            for index, future in enumerate(
                as_completed(futures),
                start=1,
            ):

                (
                    input_path,
                    success,
                    message,
                ) = future.result()

                log_result(
                    index,
                    total,
                    input_path,
                    success,
                    message,
                )

                if success:
                    successful += 1
                else:
                    failed += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("CLAHE ENHANCEMENT COMPLETED")
    logger.info("=" * 60)

    logger.info(
        "Total images : %d",
        total,
    )

    logger.info(
        "Successful   : %d",
        successful,
    )

    logger.info(
        "Failed       : %d",
        failed,
    )

    logger.info("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Apply CLAHE enhancement to "
            "HAM10000 images"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "ai_model/dataset/HAM10000/"
            "preprocessed/denoised"
        ),
        help="Folder containing denoised images",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "ai_model/dataset/HAM10000/"
            "preprocessed/enhanced"
        ),
        help="Folder for enhanced images",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess existing images",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers",
    )

    parser.add_argument(
        "--clip-limit",
        type=float,
        default=DEFAULT_CLIP_LIMIT,
        help="CLAHE clip limit",
    )

    parser.add_argument(
        "--tile-grid-size",
        type=int,
        default=DEFAULT_TILE_GRID_SIZE,
        help="CLAHE tile grid size",
    )

    args = parser.parse_args()

    enhance_dataset(
        input_folder=args.input,
        output_folder=args.output,
        overwrite=args.overwrite,
        workers=args.workers,
        clip_limit=args.clip_limit,
        tile_grid_size=args.tile_grid_size,
    )