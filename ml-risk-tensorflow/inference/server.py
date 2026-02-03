"""
Flask-based inference server for risk classification
"""

import os
import logging
from flask import Flask, request, jsonify
from model.classifier import create_classifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Initialize model
MODEL_PATH = os.getenv("MODEL_PATH", None)
classifier = create_classifier(MODEL_PATH)
logger.info("Risk classifier initialized")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "healthy",
            "service": "risk-classification",
            "model": "tensorflow-classifier",
        }
    )


@app.route("/risk/classify", methods=["POST"])
def classify():
    """
    Classify transaction risk.

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
        "risk_probability": 0.76
    }
    """
    try:
        features = request.get_json()

        if not features:
            return jsonify({"error": "No features provided"}), 400

        logger.info(f"Classifying transaction: {features.get('user_id', 'unknown')}")

        # Calculate risk probability
        probability = classifier.predict(features)

        # Clamp to [0, 1]
        probability = max(0.0, min(1.0, probability))

        logger.info(f"Risk probability: {probability:.4f}")

        return jsonify({"risk_probability": round(probability, 4)})

    except Exception as e:
        logger.error(f"Error classifying transaction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/risk/batch", methods=["POST"])
def batch_classify():
    """Classify multiple transactions"""
    try:
        data = request.get_json()

        if not data or "transactions" not in data:
            return jsonify({"error": "No transactions provided"}), 400

        transactions = data["transactions"]
        results = []

        for tx in transactions:
            probability = classifier.predict(tx)
            results.append(
                {
                    "transaction_id": tx.get("transaction_id", "unknown"),
                    "risk_probability": round(probability, 4),
                }
            )

        return jsonify({"results": results})

    except Exception as e:
        logger.error(f"Error in batch classification: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model information"""
    return jsonify(
        {
            "model_type": "Binary Classifier",
            "framework": "TensorFlow",
            "input_dim": 12,
            "hidden_layers": [64, 32, 16],
            "output": "risk_probability [0, 1]",
        }
    )


def main():
    """Run the inference server"""
    port = int(os.getenv("PORT", 5002))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"Starting risk classification server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()
