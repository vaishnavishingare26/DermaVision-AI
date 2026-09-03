import base64

import cv2
import numpy as np


def _to_data_url(image_bgr, quality=92):
    success, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not success:
        raise ValueError("Unable to encode segmentation image.")

    data = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def _largest_component(mask):
    binary = (mask > 0).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if count <= 1:
        return binary * 255

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(
        labels == largest_label,
        255,
        0,
    ).astype(np.uint8)


def _fallback_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu gives a deterministic fallback for images where GrabCut
    # cannot produce a useful foreground.
    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Try the opposite polarity when the first result is too large.
    ratio = float(np.count_nonzero(mask)) / float(mask.size)

    if ratio > 0.85:
        _, mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    return mask


def segment_skin_lesion(image_bytes: bytes):
    if not image_bytes:
        raise ValueError("No image data received.")

    image_array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    original = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if original is None:
        raise ValueError("Invalid or unsupported image.")

    image = cv2.resize(
        original,
        (224, 224),
        interpolation=cv2.INTER_AREA,
    )

    # Mild smoothing reduces isolated noise before segmentation.
    work = cv2.GaussianBlur(
        image,
        (5, 5),
        0,
    )

    height, width = work.shape[:2]

    # GrabCut initialization:
    # a border rectangle is background and the central area is
    # treated as probable foreground.
    grab_mask = np.full(
        (height, width),
        cv2.GC_BGD,
        dtype=np.uint8,
    )

    border = max(4, int(min(height, width) * 0.04))
    inner = max(border + 2, int(min(height, width) * 0.08))

    grab_mask[
        border:height - border,
        border:width - border
    ] = cv2.GC_PR_BGD

    grab_mask[
        inner:height - inner,
        inner:width - inner
    ] = cv2.GC_PR_FGD

    bgd_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )
    fgd_model = np.zeros(
        (1, 65),
        dtype=np.float64,
    )

    try:
        cv2.grabCut(
            work,
            grab_mask,
            None,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_MASK,
        )

        mask = np.where(
            (grab_mask == cv2.GC_FGD) |
            (grab_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)

        method = "OpenCV GrabCut + morphology"

    except cv2.error:
        mask = _fallback_mask(work)
        method = "OpenCV threshold fallback + morphology"

    # Morphological cleanup.
    kernel = np.ones(
        (5, 5),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    mask = _largest_component(mask)

    area_ratio = float(
        np.count_nonzero(mask)
    ) / float(mask.size)

    # Guard against obviously unusable masks.
    if area_ratio < 0.01 or area_ratio > 0.85:
        mask = _fallback_mask(work)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2,
        )

        mask = _largest_component(mask)

        area_ratio = float(
            np.count_nonzero(mask)
        ) / float(mask.size)

        method = "OpenCV fallback threshold + morphology"

    # Smooth mask edges for a cleaner visual boundary.
    mask = cv2.GaussianBlur(
        mask,
        (5, 5),
        0,
    )

    _, mask_binary = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    # Lesion-only crop.
    lesion = cv2.bitwise_and(
        image,
        image,
        mask=mask_binary,
    )

    # Build a visible boundary overlay.
    overlay = image.copy()

    outside = mask_binary == 0
    overlay[outside] = (
        overlay[outside] * 0.35
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 220, 120),
        2,
    )

    area_percentage = round(
        area_ratio * 100.0,
        2,
    )

    return {
        "available": True,
        "method": method,
        "width": int(width),
        "height": int(height),
        "area_percentage": area_percentage,
        "mask": _to_data_url(
            cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2BGR)
        ),
        "overlay": _to_data_url(overlay),
        "lesion": _to_data_url(lesion),
        "processing": [
            "Resize 224x224",
            "Gaussian smoothing",
            "GrabCut foreground extraction",
            "Morphological cleanup",
            "Largest connected component",
            "Boundary overlay",
        ],
        "message": (
            "The highlighted boundary is a classical computer-vision "
            "segmentation result. It is not a trained U-Net mask."
        ),
    }
