"""
Flask-based inference server for anomaly detection
"""

import os
import logging
from flask import Flask, request, jsonify
from model.autoencoder import create_detector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Initialize model
MODEL_PATH = os.getenv("MODEL_PATH", None)
detector = create_detector(MODEL_PATH)
logger.info("Anomaly detector initialized")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "service": "anomaly-detection",
            "model": "pytorch-autoencoder",
        }
    )


@app.route("/anomaly/score", methods=["POST"])
def score():
    """
    Calculate anomaly score for a transaction.

    Request body:
    {
        "amount": 1500.00,
        "currency": "BRL",
        "country": "BR",
        "device": "mobile",
        "hour": 14,
        "day_of_week": 3,
        "user_id": "user-123"
    }

    Response:
    {
        "anomaly_score": 0.87
    }
    """
    try:
        features = request.get_json()

        if not features:
            return jsonify({"error": "No features provided"}), 400

        logger.info(f"Scoring transaction: {features.get('user_id', 'unknown')}")

        # Calculate anomaly score
        score = detector.predict(features)

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))

        logger.info(f"Anomaly score: {score:.4f}")

        return jsonify({"anomaly_score": round(score, 4)})

    except Exception as e:
        logger.error(f"Error scoring transaction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model information"""
    return jsonify(
        {
            "model_type": "Autoencoder",
            "framework": "PyTorch",
            "input_dim": 10,
            "encoding_dim": 4,
            "output": "anomaly_score [0, 1]",
        }
    )


def main():
    """Run the inference server"""
    port = int(os.getenv("PORT", 5001))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"Starting anomaly detection server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()
