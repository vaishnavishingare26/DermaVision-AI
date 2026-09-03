from pathlib import Path
import base64
import cv2
import numpy as np


def preprocess_image(image_bytes: bytes):
    """
    Preprocess uploaded skin image.

    Steps:
    1. Decode image
    2. Resize to 224x224
    3. Median + Gaussian denoising
    4. CLAHE enhancement
    5. Return enhanced image as base64
    """

    # Decode uploaded image
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid or unsupported image.")

    # Resize
    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    # Denoising
    image = cv2.medianBlur(image, 3)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # CLAHE enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    # Encode enhanced image
    success, encoded = cv2.imencode(
        ".jpg",
        enhanced,
        [cv2.IMWRITE_JPEG_QUALITY, 92]
    )

    if not success:
        raise ValueError("Unable to encode processed image.")

    base64_image = base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")

    return {
        "image": f"data:image/jpeg;base64,{base64_image}",
        "width": 224,
        "height": 224,
        "processing": [
            "Resize 224x224",
            "Median denoising",
            "Gaussian denoising",
            "CLAHE enhancement"
        ]
    }