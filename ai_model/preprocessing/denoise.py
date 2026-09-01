"""
DermaVision AI - Image Denoising

Applies:
1. Median filtering
2. Gaussian filtering

Input:
    ai_model/dataset/HAM10000/preprocessed/resized

Output:
    ai_model/dataset/HAM10000/preprocessed/denoised
"""

import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2


# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTPUT_EXT = ".jpg"
JPEG_QUALITY = 95

DEFAULT_MEDIAN_KERNEL = 3
DEFAULT_GAUSSIAN_KERNEL = 3
DEFAULT_GAUSSIAN_SIGMA = 0

# Status constants — used for counting, not just display, so log
# message wording can change freely without breaking the summary counts.
STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


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
# VALIDATE KERNEL
# ============================================================

def validate_kernel(value: int, name: str) -> None:
    if value < 1 or value % 2 == 0:
        raise ValueError(f"{name} must be a positive odd integer.")


# ============================================================
# DENOISE SINGLE IMAGE
# ============================================================

def denoise_image(
    input_path: Path,
    output_path: Path,
    median_kernel: int,
    gaussian_kernel: int,
    gaussian_sigma: float,
    overwrite: bool = False,
) -> tuple[Path, str, str]:
    """
    Returns (input_path, status, message) where status is one of
    STATUS_OK / STATUS_SKIPPED / STATUS_FAILED.
    """
    if output_path.exists() and not overwrite:
        return input_path, STATUS_SKIPPED, "already exists"

    try:
        image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if image is None:
            return input_path, STATUS_FAILED, "unable to read image"

        median_image = cv2.medianBlur(image, median_kernel)
        denoised_image = cv2.GaussianBlur(
            median_image, (gaussian_kernel, gaussian_kernel), gaussian_sigma
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        saved = cv2.imwrite(
            str(output_path), denoised_image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )
        if not saved:
            return input_path, STATUS_FAILED, "failed to save image"

        return input_path, STATUS_OK, "OK"

    except Exception as error:
        return input_path, STATUS_FAILED, str(error)


# ============================================================
# COLLECT IMAGES
# ============================================================

def collect_image_paths(input_folder: Path) -> list[Path]:
    return sorted(
        path
        for path in input_folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS
    )


# ============================================================
# LOG RESULT
# ============================================================

def log_result(index: int, total: int, path: Path, status: str, message: str) -> None:
    label = {STATUS_OK: "OK", STATUS_SKIPPED: "SKIP", STATUS_FAILED: "FAIL"}[status]
    logger.info("[%d/%d] %-4s %s (%s)", index, total, label, path.name, message)


# ============================================================
# DENOISE DATASET
# ============================================================

def denoise_dataset(
    input_folder: Path,
    output_folder: Path,
    overwrite: bool = False,
    workers: int = 1,
    median_kernel: int = DEFAULT_MEDIAN_KERNEL,
    gaussian_kernel: int = DEFAULT_GAUSSIAN_KERNEL,
    gaussian_sigma: float = DEFAULT_GAUSSIAN_SIGMA,
) -> None:

    if not input_folder.exists():
        logger.error("Input folder does not exist: %s", input_folder)
        return

    if workers < 1:
        raise ValueError("workers must be >= 1")

    validate_kernel(median_kernel, "median_kernel")
    validate_kernel(gaussian_kernel, "gaussian_kernel")

    output_folder.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(input_folder)
    total = len(image_paths)

    logger.info("=" * 65)
    logger.info("DERMAVISION AI - IMAGE DENOISING")
    logger.info("=" * 65)
    logger.info("Input          : %s", input_folder.resolve())
    logger.info("Output         : %s", output_folder.resolve())
    logger.info("Median kernel  : %d x %d", median_kernel, median_kernel)
    logger.info("Gaussian kernel: %d x %d", gaussian_kernel, gaussian_kernel)
    logger.info("Gaussian sigma : %s", gaussian_sigma)
    logger.info("Images found   : %d", total)
    logger.info("Workers        : %d", workers)
    logger.info("")

    if total == 0:
        logger.warning("No supported images found.")
        return

    jobs = []
    for image_path in image_paths:
        relative_path = image_path.relative_to(input_folder).with_suffix(OUTPUT_EXT)
        output_path = output_folder / relative_path
        jobs.append((image_path, output_path))

    successful = 0
    failed = 0
    skipped = 0

    def _record(status: str) -> None:
        nonlocal successful, failed, skipped
        if status == STATUS_OK:
            successful += 1
        elif status == STATUS_SKIPPED:
            skipped += 1
        else:
            failed += 1

    if workers == 1:
        for index, (input_path, output_path) in enumerate(jobs, start=1):
            _, status, message = denoise_image(
                input_path, output_path,
                median_kernel, gaussian_kernel, gaussian_sigma,
                overwrite,
            )
            log_result(index, total, input_path, status, message)
            _record(status)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    denoise_image,
                    input_path, output_path,
                    median_kernel, gaussian_kernel, gaussian_sigma,
                    overwrite,
                ): input_path
                for input_path, output_path in jobs
            }
            for index, future in enumerate(as_completed(futures), start=1):
                input_path, status, message = future.result()
                log_result(index, total, input_path, status, message)
                _record(status)

    logger.info("")
    logger.info("=" * 65)
    logger.info("DENOISING COMPLETED")
    logger.info("=" * 65)
    logger.info("Total images : %d", total)
    logger.info("Successful   : %d", successful)
    logger.info("Skipped      : %d", skipped)
    logger.info("Failed       : %d", failed)
    logger.info("Output folder: %s", output_folder.resolve())
    logger.info("=" * 65)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise HAM10000 images for DermaVision AI")

    parser.add_argument("--input", type=Path,
                         default=Path("ai_model/dataset/HAM10000/preprocessed/resized"))
    parser.add_argument("--output", type=Path,
                         default=Path("ai_model/dataset/HAM10000/preprocessed/denoised"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--median-kernel", type=int, default=DEFAULT_MEDIAN_KERNEL)
    parser.add_argument("--gaussian-kernel", type=int, default=DEFAULT_GAUSSIAN_KERNEL)
    parser.add_argument("--gaussian-sigma", type=float, default=DEFAULT_GAUSSIAN_SIGMA)

    args = parser.parse_args()

    denoise_dataset(
        input_folder=args.input,
        output_folder=args.output,
        overwrite=args.overwrite,
        workers=args.workers,
        median_kernel=args.median_kernel,
        gaussian_kernel=args.gaussian_kernel,
        gaussian_sigma=args.gaussian_sigma,
    )