"""
TensorFlow Risk Classification Model
Binary/Multi-class classifier for transaction risk assessment
"""

from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Optional


def create_risk_classifier(
    input_dim: int = 12,
    hidden_units: Tuple[int, ...] = (64, 32, 16),
    dropout_rate: float = 0.3,
    num_classes: int = 1,  # 1 for binary, >1 for multi-class
) -> Model:
    """
    Create a neural network classifier for risk assessment.

    Args:
        input_dim: Number of input features
        hidden_units: Tuple of hidden layer sizes
        dropout_rate: Dropout rate for regularization
        num_classes: Number of output classes (1 for binary)

    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=(input_dim,), name="features")

    x = layers.BatchNormalization()(inputs)

    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, name=f"dense_{i}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(dropout_rate)(x)

    # Output layer
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="risk_probability")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="risk_classes")(
            x
        )

    model = Model(inputs=inputs, outputs=outputs, name="risk_classifier")

    # Compile model
    if num_classes == 1:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    return model


class RiskClassifier:
    """
    High-level interface for risk classification.
    Handles feature preprocessing and model inference.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.input_dim = 12
        self.model = create_risk_classifier(input_dim=self.input_dim)

        if model_path:
            self.load_model(model_path)
        else:
            # Initialize with random weights for demo
            self._initialize_demo_weights()

        # Feature normalization parameters
        self.feature_stats = {
            "amount_mean": 500.0,
            "amount_std": 1000.0,
            "hour_mean": 12.0,
            "hour_std": 6.0,
        }

        # Risk factors for feature engineering
        self.high_risk_countries = {"XX", "NG", "RU", "CN"}
        self.high_risk_devices = {"unknown", "emulator"}
        self.risky_hours = {0, 1, 2, 3, 4, 5, 23}  # Late night/early morning

    def _initialize_demo_weights(self):
        """Initialize with weights that produce reasonable demo outputs"""
        # Run a dummy prediction to build the model
        dummy_input = np.random.randn(1, self.input_dim).astype(np.float32)
        _ = self.model.predict(dummy_input, verbose=0)

    def load_model(self, path: str):
        """Load trained model weights"""
        self.model.load_weights(path)

    def save_model(self, path: str):
        """Save model weights"""
        self.model.save_weights(path)

    def preprocess_features(self, features: dict) -> np.ndarray:
        """
        Convert transaction features to model input array.

        Expected features:
        - amount: float
        - currency: str
        - country: str
        - device: str
        - hour: int
        - day_of_week: int
        - user_id: str
        """
        # Normalize numerical features
        amount = features.get("amount", 0)
        amount_norm = (amount - self.feature_stats["amount_mean"]) / self.feature_stats[
            "amount_std"
        ]

        # Log transform for amount (handles large values)
        amount_log = np.log1p(amount) / 10.0

        hour = features.get("hour", 12)
        hour_norm = (hour - self.feature_stats["hour_mean"]) / self.feature_stats[
            "hour_std"
        ]

        day_of_week = features.get("day_of_week", 3)
        dow_norm = day_of_week / 6.0  # Normalize to [0, 1]

        # Binary risk indicators
        is_high_risk_country = (
            1.0 if features.get("country", "BR") in self.high_risk_countries else 0.0
        )
        is_high_risk_device = (
            1.0 if features.get("device", "mobile") in self.high_risk_devices else 0.0
        )
        is_risky_hour = 1.0 if hour in self.risky_hours else 0.0
        is_weekend = 1.0 if day_of_week >= 5 else 0.0
        is_high_amount = 1.0 if amount > 5000 else 0.0

        # Currency risk (simplified)
        currency = features.get("currency", "BRL")
        currency_risk = 0.0
        if currency in {"USD", "EUR", "GBP"}:
            currency_risk = 0.2
        elif currency not in {"BRL", "USD", "EUR", "GBP", "JPY"}:
            currency_risk = 0.5

        # Device type encoding
        device = features.get("device", "mobile")
        device_map = {"mobile": 0.2, "desktop": 0.3, "tablet": 0.25, "unknown": 0.8}
        device_score = device_map.get(device, 0.5)

        # Combined velocity indicator (simulated)
        velocity_score = min(amount / 10000.0, 1.0)

        # Build feature vector (12 dimensions)
        feature_vector = np.array(
            [
                amount_norm,
                amount_log,
                hour_norm,
                dow_norm,
                is_high_risk_country,
                is_high_risk_device,
                is_risky_hour,
                is_weekend,
                is_high_amount,
                currency_risk,
                device_score,
                velocity_score,
            ],
            dtype=np.float32,
        )

        return feature_vector.reshape(1, -1)

    def predict(self, features: dict) -> float:
        """
        Predict risk probability for a transaction.

        Returns:
            float: Risk probability between 0 and 1
        """
        input_array = self.preprocess_features(features)

        # Add small noise for demo variety
        noise = np.random.randn(*input_array.shape) * 0.05
        input_array = input_array + noise

        prediction = self.model.predict(input_array, verbose=0)
        probability = float(prediction[0][0])

        # Clamp to valid range
        return max(0.0, min(1.0, probability))

    def predict_batch(self, features_list: list) -> np.ndarray:
        """Predict risk for multiple transactions"""
        inputs = np.vstack([self.preprocess_features(f) for f in features_list])
        predictions = self.model.predict(inputs, verbose=0)
        return predictions.flatten()


def create_classifier(model_path: Optional[str] = None) -> RiskClassifier:
    """Factory function to create risk classifier"""
    return RiskClassifier(model_path)
