"""
Feature preprocessing utilities for risk classification
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class RiskFeatureConfig:
    """Configuration for risk feature preprocessing"""

    amount_mean: float = 500.0
    amount_std: float = 1000.0
    amount_max: float = 50000.0

    high_risk_countries: List[str] = field(
        default_factory=lambda: ["XX", "NG", "RU", "CN", "IR", "KP"]
    )
    medium_risk_countries: List[str] = field(
        default_factory=lambda: ["IN", "PK", "BD", "VN", "PH"]
    )

    high_risk_devices: List[str] = field(
        default_factory=lambda: ["unknown", "emulator", "rooted"]
    )

    risky_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 23])


class RiskFeaturePreprocessor:
    """
    Preprocesses raw transaction features for risk classification.
    """

    def __init__(self, config: Optional[RiskFeatureConfig] = None):
        self.config = config or RiskFeatureConfig()

        # Currency mappings with risk weights
        self.currency_risk = {
            "BRL": 0.1,
            "USD": 0.15,
            "EUR": 0.15,
            "GBP": 0.15,
            "JPY": 0.1,
            "CNY": 0.3,
            "INR": 0.25,
            "RUB": 0.4,
            "BTC": 0.6,
            "ETH": 0.6,
        }

        # Device risk scores
        self.device_risk = {
            "mobile": 0.2,
            "desktop": 0.25,
            "tablet": 0.2,
            "smart_tv": 0.3,
            "unknown": 0.7,
            "emulator": 0.9,
        }

    def normalize_amount(self, amount: float) -> Dict[str, float]:
        """Create multiple normalized amount features"""
        return {
            "amount_zscore": (amount - self.config.amount_mean)
            / self.config.amount_std,
            "amount_log": np.log1p(amount) / 12.0,  # Normalize log
            "amount_ratio": min(amount / self.config.amount_max, 1.0),
            "is_micro": 1.0 if amount < 10 else 0.0,
            "is_small": 1.0 if 10 <= amount < 100 else 0.0,
            "is_medium": 1.0 if 100 <= amount < 1000 else 0.0,
            "is_large": 1.0 if 1000 <= amount < 10000 else 0.0,
            "is_very_large": 1.0 if amount >= 10000 else 0.0,
        }

    def get_time_features(self, hour: int, day_of_week: int) -> Dict[str, float]:
        """Extract time-based risk features"""
        return {
            "hour_norm": hour / 23.0,
            "is_night": 1.0 if hour in self.config.risky_hours else 0.0,
            "is_business_hours": 1.0 if 9 <= hour <= 18 else 0.0,
            "is_weekend": 1.0 if day_of_week >= 5 else 0.0,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
        }

    def get_country_risk(self, country: str) -> Dict[str, float]:
        """Calculate country-based risk features"""
        country = country.upper()

        is_high_risk = country in self.config.high_risk_countries
        is_medium_risk = country in self.config.medium_risk_countries

        if is_high_risk:
            risk_score = 0.8
        elif is_medium_risk:
            risk_score = 0.5
        else:
            risk_score = 0.2

        return {
            "country_risk": risk_score,
            "is_high_risk_country": 1.0 if is_high_risk else 0.0,
            "is_medium_risk_country": 1.0 if is_medium_risk else 0.0,
        }

    def get_device_risk(self, device: str) -> Dict[str, float]:
        """Calculate device-based risk features"""
        device = device.lower()
        risk = self.device_risk.get(device, 0.5)

        return {
            "device_risk": risk,
            "is_mobile": 1.0 if device == "mobile" else 0.0,
            "is_unknown_device": (
                1.0 if device in self.config.high_risk_devices else 0.0
            ),
        }

    def get_currency_risk(self, currency: str) -> Dict[str, float]:
        """Calculate currency-based risk features"""
        currency = currency.upper()
        risk = self.currency_risk.get(currency, 0.3)

        return {
            "currency_risk": risk,
            "is_crypto": 1.0 if currency in {"BTC", "ETH", "USDT"} else 0.0,
            "is_major_currency": (
                1.0 if currency in {"USD", "EUR", "GBP", "JPY"} else 0.0
            ),
        }

    def preprocess(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess raw features into model input.

        Returns:
            numpy array of shape (12,) for model input
        """
        # Get individual feature groups
        amount_features = self.normalize_amount(features.get("amount", 0))
        time_features = self.get_time_features(
            features.get("hour", 12), features.get("day_of_week", 3)
        )
        country_features = self.get_country_risk(features.get("country", "BR"))
        device_features = self.get_device_risk(features.get("device", "mobile"))
        currency_features = self.get_currency_risk(features.get("currency", "BRL"))

        # Build final feature vector (12 dimensions to match model)
        feature_vector = [
            amount_features["amount_zscore"],
            amount_features["amount_log"],
            time_features["hour_norm"],
            time_features["is_weekend"],
            country_features["country_risk"],
            country_features["is_high_risk_country"],
            device_features["device_risk"],
            device_features["is_unknown_device"],
            time_features["is_night"],
            amount_features["is_very_large"],
            currency_features["currency_risk"],
            amount_features["amount_ratio"],
        ]

        return np.array(feature_vector, dtype=np.float32)

    def validate(self, features: Dict[str, Any]) -> List[str]:
        """Validate features and return list of warnings"""
        warnings = []

        if "amount" not in features:
            warnings.append("Missing 'amount' field")
        elif features["amount"] < 0:
            warnings.append("Negative amount")

        if "country" not in features:
            warnings.append("Missing 'country' field")

        if "currency" not in features:
            warnings.append("Missing 'currency' field")

        return warnings
