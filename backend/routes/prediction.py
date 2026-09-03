from flask import Blueprint, jsonify, request

from backend.services.prediction_service import predict_image
from backend.services.preprocessing_service import preprocess_image
from backend.services.segmentation_service import segment_skin_lesion


prediction_bp = Blueprint(
    "prediction",
    __name__,
)


def _get_upload():
    return (
        request.files.get("image")
        or request.files.get("file")
    )


@prediction_bp.post("/predict")
def predict():
    upload = _get_upload()

    if upload is None:
        return jsonify({
            "error": "No image uploaded. Use form field 'image'."
        }), 400

    try:
        result = predict_image(
            upload.read()
        )

        return jsonify(result), 200

    except FileNotFoundError as exc:
        return jsonify({
            "error": str(exc)
        }), 500

    except ValueError as exc:
        return jsonify({
            "error": str(exc)
        }), 400

    except Exception as exc:
        return jsonify({
            "error": f"Prediction failed: {exc}"
        }), 500


@prediction_bp.post("/preprocess")
def preprocess():
    upload = _get_upload()

    if upload is None:
        return jsonify({
            "error": "No image uploaded. Use form field 'image'."
        }), 400

    try:
        result = preprocess_image(
            upload.read()
        )

        return jsonify(result), 200

    except ValueError as exc:
        return jsonify({
            "error": str(exc)
        }), 400

    except Exception as exc:
        return jsonify({
            "error": f"Preprocessing failed: {exc}"
        }), 500


@prediction_bp.post("/segment")
def segment():
    upload = _get_upload()

    if upload is None:
        return jsonify({
            "error": "No image uploaded. Use form field 'image'."
        }), 400

    try:
        result = segment_skin_lesion(
            upload.read()
        )

        return jsonify(result), 200

    except ValueError as exc:
        return jsonify({
            "error": str(exc)
        }), 400

    except Exception as exc:
        return jsonify({
            "error": f"Segmentation failed: {exc}"
        }), 500
