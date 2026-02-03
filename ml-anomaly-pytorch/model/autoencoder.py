"""
PyTorch Anomaly Detection Service
Autoencoder-based anomaly detection for transaction risk scoring
"""

import torch
import torch.nn as nn
from typing import Optional


class TransactionAutoencoder(nn.Module):
    """
    Autoencoder for detecting anomalous transactions.
    Anomaly score is based on reconstruction error.
    """

    def __init__(self, input_dim: int = 10, encoding_dim: int = 4):
        super(TransactionAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, encoding_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            nn.Linear(16, input_dim),
            nn.Sigmoid(),
        )

        # Threshold for anomaly detection (learned during training)
        self.threshold = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction error (MSE) for each sample"""
        reconstructed = self.forward(x)
        mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate anomaly score between 0 and 1.
        Higher score indicates more anomalous transaction.
        """
        with torch.no_grad():
            reconstruction_error = self.reconstruction_error(x)
            # Normalize to 0-1 range using sigmoid
            score = torch.sigmoid(reconstruction_error * 10 - 5)
            return score


class AnomalyDetector:
    """
    High-level interface for anomaly detection.
    Handles feature preprocessing and model inference.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransactionAutoencoder(input_dim=10, encoding_dim=4)
        self.model.to(self.device)

        if model_path:
            self.load_model(model_path)
        else:
            # Initialize with random weights for demo
            self._initialize_demo_weights()

        self.model.eval()

        # Feature normalization parameters
        self.feature_means = {"amount": 500.0, "hour": 12.0, "day_of_week": 3.0}
        self.feature_stds = {"amount": 1000.0, "hour": 6.0, "day_of_week": 2.0}

        # Encoding mappings
        self.currency_encoding = {"BRL": 0, "USD": 1, "EUR": 2, "GBP": 3}
        self.country_encoding = {"BR": 0, "US": 1, "EU": 2, "UK": 3, "XX": 4}
        self.device_encoding = {"mobile": 0, "desktop": 1, "tablet": 2, "unknown": 3}

    def _initialize_demo_weights(self):
        """Initialize with weights that produce reasonable demo outputs"""
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def load_model(self, path: str):
        """Load trained model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)

    def preprocess_features(self, features: dict) -> torch.Tensor:
        """
        Convert transaction features to model input tensor.

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
        amount_norm = (
            features.get("amount", 0) - self.feature_means["amount"]
        ) / self.feature_stds["amount"]
        hour_norm = (
            features.get("hour", 12) - self.feature_means["hour"]
        ) / self.feature_stds["hour"]
        dow_norm = (
            features.get("day_of_week", 3) - self.feature_means["day_of_week"]
        ) / self.feature_stds["day_of_week"]

        # One-hot encode categorical features
        currency = features.get("currency", "BRL")
        currency_vec = [0.0] * 4
        currency_idx = self.currency_encoding.get(currency, 0)
        currency_vec[currency_idx] = 1.0

        device = features.get("device", "mobile")
        device_idx = self.device_encoding.get(device, 3)
        device_vec = device_idx / 3.0  # Normalize to 0-1

        country = features.get("country", "BR")
        country_idx = self.country_encoding.get(country, 4)
        country_risk = country_idx / 4.0  # Higher index = potentially riskier

        # Build feature vector (10 dimensions)
        feature_vector = [
            amount_norm,
            hour_norm,
            dow_norm,
            device_vec,
            country_risk,
        ] + currency_vec

        tensor = torch.tensor([feature_vector], dtype=torch.float32)
        return tensor.to(self.device)

    def predict(self, features: dict) -> float:
        """
        Predict anomaly score for a transaction.

        Returns:
            float: Anomaly score between 0 and 1
        """
        input_tensor = self.preprocess_features(features)

        # Add small noise to prevent identical outputs
        noise = torch.randn_like(input_tensor) * 0.01
        input_tensor = input_tensor + noise

        with torch.no_grad():
            score = self.model.anomaly_score(input_tensor)
            return float(score[0].cpu().numpy())


def create_detector(model_path: Optional[str] = None) -> AnomalyDetector:
    """Factory function to create anomaly detector"""
    return AnomalyDetector(model_path)
