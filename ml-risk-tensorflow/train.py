"""
Training script for the risk classification model
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import os
import logging
from model.classifier import create_risk_classifier
from model.preprocessing import RiskFeaturePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 10000) -> tuple:
    """
    Generate synthetic labeled transaction data for training.
    In production, this would load from a labeled database.

    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(42)
    preprocessor = RiskFeaturePreprocessor()

    features_list = []
    labels = []

    for _ in range(n_samples):
        # Generate random transaction features
        amount = np.random.exponential(500) + 10
        hour = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)

        # Assign risk levels based on features (synthetic labels)
        country = np.random.choice(
            ["BR", "US", "UK", "DE", "XX", "NG", "CN"],
            p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05],
        )
        device = np.random.choice(
            ["mobile", "desktop", "tablet", "unknown"], p=[0.5, 0.3, 0.15, 0.05]
        )
        currency = np.random.choice(
            ["BRL", "USD", "EUR", "GBP", "CNY"], p=[0.4, 0.25, 0.15, 0.1, 0.1]
        )

        features = {
            "amount": amount,
            "currency": currency,
            "country": country,
            "device": device,
            "hour": hour,
            "day_of_week": day_of_week,
        }

        # Create synthetic label based on risk factors
        risk_score = 0.0

        # High risk country
        if country in ["XX", "NG", "CN"]:
            risk_score += 0.3

        # Unknown device
        if device == "unknown":
            risk_score += 0.25

        # High amount
        if amount > 5000:
            risk_score += 0.2
        elif amount > 2000:
            risk_score += 0.1

        # Night time
        if hour in [0, 1, 2, 3, 4, 5, 23]:
            risk_score += 0.15

        # Weekend
        if day_of_week >= 5:
            risk_score += 0.05

        # Add noise
        risk_score += np.random.normal(0, 0.1)
        risk_score = max(0, min(1, risk_score))

        # Binary label
        label = 1 if risk_score > 0.4 else 0

        processed = preprocessor.preprocess(features)
        features_list.append(processed)
        labels.append(label)

    return np.array(features_list), np.array(labels)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
) -> keras.Model:
    """Train the risk classification model"""

    # Create model
    model = create_risk_classifier(
        input_dim=X_train.shape[1],
        hidden_units=(64, 32, 16),
        dropout_rate=0.3,
        num_classes=1,
    )

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
    ]

    # Calculate class weights for imbalanced data
    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    weight_for_0 = (1 / n_neg) * (len(y_train) / 2.0) if n_neg > 0 else 1.0
    weight_for_1 = (1 / n_pos) * (len(y_train) / 2.0) if n_pos > 0 else 1.0
    class_weight = {0: weight_for_0, 1: weight_for_1}

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Validation Accuracy: {val_acc:.4f}")
    logger.info(f"Validation AUC: {val_auc:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train risk classification model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--samples", type=int, default=10000, help="Number of training samples"
    )
    parser.add_argument(
        "--output", type=str, default="model/trained_model", help="Output model path"
    )
    args = parser.parse_args()

    logger.info("Generating synthetic training data...")
    X, y = generate_synthetic_data(n_samples=args.samples)

    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Positive class: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    logger.info(
        f"Negative class: {len(y) - sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)"
    )

    # Split into train/val
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # Train model
    logger.info("Starting training...")
    model = train_model(
        X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size
    )

    # Save model
    os.makedirs(
        os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
        exist_ok=True,
    )
    model.save_weights(args.output)
    logger.info(f"Model saved to {args.output}")

    # Also save in SavedModel format
    model.save(args.output + "_saved_model")
    logger.info(f"SavedModel saved to {args.output}_saved_model")


if __name__ == "__main__":
    main()
