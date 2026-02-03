"""
Feature preprocessing utilities for anomaly detection
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature preprocessing"""

    amount_mean: float = 500.0
    amount_std: float = 1000.0
    hour_mean: float = 12.0
    hour_std: float = 6.0
    day_of_week_mean: float = 3.0
    day_of_week_std: float = 2.0


class FeaturePreprocessor:
    """
    Preprocesses raw transaction features for ML models.
    Handles normalization, encoding, and feature engineering.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

        # Categorical mappings
        self.currency_map = {
            "BRL": 0,
            "USD": 1,
            "EUR": 2,
            "GBP": 3,
            "JPY": 4,
            "CNY": 5,
            "INR": 6,
            "AUD": 7,
        }

        self.country_map = {
            "BR": 0,
            "US": 1,
            "UK": 2,
            "DE": 3,
            "FR": 4,
            "JP": 5,
            "CN": 6,
            "IN": 7,
            "AU": 8,
            "CA": 9,
        }

        self.device_map = {
            "mobile": 0,
            "desktop": 1,
            "tablet": 2,
            "smart_tv": 3,
            "unknown": 4,
        }

        # Risk profiles for countries (for feature engineering)
        self.country_risk_scores = {
            "BR": 0.3,
            "US": 0.2,
            "UK": 0.2,
            "DE": 0.15,
            "FR": 0.2,
            "JP": 0.1,
            "CN": 0.4,
            "IN": 0.35,
            "AU": 0.15,
            "CA": 0.2,
        }

    def normalize_amount(self, amount: float) -> float:
        """Normalize transaction amount using z-score"""
        return (amount - self.config.amount_mean) / self.config.amount_std

    def normalize_hour(self, hour: int) -> float:
        """Normalize hour of day"""
        return (hour - self.config.hour_mean) / self.config.hour_std

    def normalize_day_of_week(self, day: int) -> float:
        """Normalize day of week (0=Monday, 6=Sunday)"""
        return (day - self.config.day_of_week_mean) / self.config.day_of_week_std

    def encode_currency(self, currency: str) -> List[float]:
        """One-hot encode currency"""
        encoding = [0.0] * len(self.currency_map)
        idx = self.currency_map.get(currency.upper(), 0)
        encoding[idx] = 1.0
        return encoding

    def encode_device(self, device: str) -> float:
        """Encode device type as normalized value"""
        idx = self.device_map.get(device.lower(), 4)
        return idx / (len(self.device_map) - 1)

    def get_country_risk(self, country: str) -> float:
        """Get risk score for country"""
        return self.country_risk_scores.get(country.upper(), 0.5)

    def engineer_time_features(self, hour: int, day_of_week: int) -> Dict[str, float]:
        """Create engineered time-based features"""
        # Is weekend
        is_weekend = 1.0 if day_of_week >= 5 else 0.0

        # Is night time (22:00 - 06:00)
        is_night = 1.0 if hour >= 22 or hour <= 6 else 0.0

        # Is business hours (09:00 - 18:00)
        is_business_hours = 1.0 if 9 <= hour <= 18 else 0.0

        # Cyclical encoding for hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Cyclical encoding for day of week
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        return {
            "is_weekend": is_weekend,
            "is_night": is_night,
            "is_business_hours": is_business_hours,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
        }

    def preprocess(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess raw features into model input.

        Args:
            features: Dictionary with transaction features

        Returns:
            numpy array of preprocessed features
        """
        # Extract and normalize numerical features
        amount = self.normalize_amount(features.get("amount", 0))
        hour = features.get("hour", 12)
        day_of_week = features.get("day_of_week", 3)

        # Get time features
        time_features = self.engineer_time_features(hour, day_of_week)

        # Encode categorical features
        device = self.encode_device(features.get("device", "unknown"))
        country_risk = self.get_country_risk(features.get("country", "BR"))

        # Build feature vector
        feature_vector = [
            amount,
            self.normalize_hour(hour),
            self.normalize_day_of_week(day_of_week),
            device,
            country_risk,
            time_features["is_weekend"],
            time_features["is_night"],
            time_features["is_business_hours"],
            time_features["hour_sin"],
            time_features["hour_cos"],
        ]

        return np.array(feature_vector, dtype=np.float32)

    def validate_features(self, features: Dict[str, Any]) -> bool:
        """Validate that required features are present"""
        required = ["amount", "currency", "country", "device"]
        return all(key in features for key in required)
