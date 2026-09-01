"""
DermaVision AI - Dataset Image Resizing

Resizes HAM10000 dataset images to a standardized
224 x 224 RGB JPEG format.

Input structure:

ai_model/
└── database/
    └── HAM1000/
        ├── HAM10000_metadata.csv
        ├── images_part_1/
        └── images_part_2/

Output:

ai_model/
└── dataset/
    └── HAM10000/
        └── preprocessed/
            └── resized/
"""

import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image, ImageOps


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

SUPPORTED_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

OUTPUT_FORMAT = "JPEG"
OUTPUT_EXT = ".jpg"
JPEG_QUALITY = 95


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
# RESIZE SINGLE IMAGE
# ============================================================

def resize_image(
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> tuple[Path, bool, str]:

    """
    Resize one image to 224 x 224 RGB JPEG.

    Returns:
        (input_path, success, message)
    """

    # Skip already processed images
    if output_path.exists() and not overwrite:
        return input_path, True, "skipped - already exists"

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        with Image.open(input_path) as image:

            # Correct EXIF orientation
            image = ImageOps.exif_transpose(image)

            # Convert image to RGB
            image = image.convert("RGB")

            # ------------------------------------------------
            # RESIZE
            # ------------------------------------------------

            resized_image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS,
            )

            # ------------------------------------------------
            # CREATE OUTPUT DIRECTORY
            # ------------------------------------------------

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ------------------------------------------------
            # SAVE JPEG
            # ------------------------------------------------

            resized_image.save(
                output_path,
                format=OUTPUT_FORMAT,
                quality=JPEG_QUALITY,
                optimize=True,
            )

        return input_path, True, "OK"

    except Exception as error:

        return input_path, False, str(error)


# ============================================================
# COLLECT IMAGE PATHS
# ============================================================

def collect_image_paths(
    input_folder: Path,
) -> list[Path]:

    """
    Recursively collect all supported image files.

    This automatically finds images inside:

    images_part_1
    images_part_2
    and any nested folders.
    """

    image_paths = [
        path
        for path in input_folder.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_FORMATS
        )
    ]

    return sorted(image_paths)


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
# RESIZE DATASET
# ============================================================

def resize_dataset(
    input_folder: Path,
    output_folder: Path,
    overwrite: bool = False,
    workers: int = 1,
) -> None:

    """
    Resize all supported images.

    The relative folder structure is preserved.
    """

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if not input_folder.exists():

        logger.error(
            "Input folder does not exist: %s",
            input_folder,
        )

        return

    if not input_folder.is_dir():

        logger.error(
            "Input path is not a directory: %s",
            input_folder,
        )

        return

    # ========================================================
    # VALIDATE WORKERS
    # ========================================================

    if workers < 1:

        raise ValueError(
            "workers must be >= 1"
        )

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # COLLECT IMAGES
    # ========================================================

    image_paths = collect_image_paths(
        input_folder
    )

    total = len(image_paths)

    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    logger.info("=" * 65)
    logger.info("DERMAVISION AI - IMAGE RESIZING")
    logger.info("=" * 65)

    logger.info(
        "Input       : %s",
        input_folder.resolve(),
    )

    logger.info(
        "Output      : %s",
        output_folder.resolve(),
    )

    logger.info(
        "Image size  : %d x %d",
        *IMAGE_SIZE,
    )

    logger.info(
        "JPEG quality: %d",
        JPEG_QUALITY,
    )

    logger.info(
        "Images found: %d",
        total,
    )

    logger.info(
        "Workers     : %d",
        workers,
    )

    logger.info("")

    # ========================================================
    # NO IMAGES
    # ========================================================

    if total == 0:

        logger.warning(
            "No supported images found."
        )

        logger.warning(
            "Check the input folder."
        )

        return

    # ========================================================
    # COUNTERS
    # ========================================================

    successful = 0
    failed = 0
    skipped = 0

    # ========================================================
    # BUILD JOBS
    # ========================================================

    jobs = []

    for image_path in image_paths:

        relative_path = (
            image_path
            .relative_to(input_folder)
            .with_suffix(OUTPUT_EXT)
        )

        output_path = (
            output_folder / relative_path
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
            ) = resize_image(
                input_path,
                output_path,
                overwrite,
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

                if "skipped" in message:
                    skipped += 1

            else:

                failed += 1

    # ========================================================
    # MULTI PROCESS
    # ========================================================

    else:

        with ProcessPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = {
                executor.submit(
                    resize_image,
                    input_path,
                    output_path,
                    overwrite,
                ): input_path

                for (
                    input_path,
                    output_path,
                ) in jobs
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

                    if "skipped" in message:
                        skipped += 1

                else:

                    failed += 1

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    logger.info("")
    logger.info("=" * 65)
    logger.info("RESIZING COMPLETED")
    logger.info("=" * 65)

    logger.info(
        "Total images : %d",
        total,
    )

    logger.info(
        "Successful   : %d",
        successful,
    )

    logger.info(
        "Skipped      : %d",
        skipped,
    )

    logger.info(
        "Failed       : %d",
        failed,
    )

    logger.info(
        "Output size  : %d x %d",
        *IMAGE_SIZE,
    )

    logger.info(
        "Output folder: %s",
        output_folder.resolve(),
    )

    logger.info("=" * 65)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Resize HAM10000 images "
            "for DermaVision AI"
        )
    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "ai_model/database/HAM1000"
        ),
        help=(
            "HAM1000 database folder containing "
            "images_part_1 and images_part_2"
        ),
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "ai_model/dataset/HAM10000/"
            "preprocessed/resized"
        ),
        help=(
            "Folder for resized images"
        ),
    )

    # --------------------------------------------------------
    # OVERWRITE
    # --------------------------------------------------------

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Reprocess images even if "
            "output already exists"
        ),
    )

    # --------------------------------------------------------
    # WORKERS
    # --------------------------------------------------------

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel workers. "
            "Use 1 for safe CPU processing."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # RUN
    # ========================================================

    resize_dataset(
        input_folder=args.input,
        output_folder=args.output,
        overwrite=args.overwrite,
        workers=args.workers,
    )