from flask import Blueprint, jsonify, request

from backend.services.prediction_service import predict_image

prediction_bp = Blueprint("prediction", __name__)


@prediction_bp.post("/predict")
def predict():
    upload = request.files.get("image") or request.files.get("file")
    if upload is None:
        return jsonify({"error": "No image uploaded. Use form field 'image'."}), 400

    try:
        result = predict_image(upload.read())
        return jsonify(result), 200
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 500
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500
