from flask import Flask, jsonify
from flask_cors import CORS

from backend.routes.prediction import prediction_bp

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
CORS(app)

app.register_blueprint(prediction_bp, url_prefix="/api")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "DermaVision AI backend"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
