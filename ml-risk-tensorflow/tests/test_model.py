"""
Tests for the risk classification model
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.classifier import create_risk_classifier, RiskClassifier, create_classifier
from model.preprocessing import RiskFeaturePreprocessor, RiskFeatureConfig


class TestRiskClassifierModel:
    """Tests for the TensorFlow classifier model"""

    def setup_method(self):
        self.model = create_risk_classifier(input_dim=12)

    def test_model_output_shape(self):
        """Test model output shape"""
        x = np.random.randn(16, 12).astype(np.float32)
        output = self.model.predict(x, verbose=0)
        assert output.shape == (16, 1)

    def test_model_output_range(self):
        """Test output is in [0, 1]"""
        x = np.random.randn(100, 12).astype(np.float32)
        output = self.model.predict(x, verbose=0)
        assert np.all(output >= 0)
        assert np.all(output <= 1)

    def test_model_with_extreme_inputs(self):
        """Test model handles extreme inputs"""
        x_extreme = np.array(
            [[1e6, -1e6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32
        )
        output = self.model.predict(x_extreme, verbose=0)
        assert np.isfinite(output).all()


class TestRiskClassifier:
    """Tests for the high-level classifier interface"""

    def setup_method(self):
        self.classifier = create_classifier()

    def test_predict_valid_features(self):
        """Test prediction with valid features"""
        features = {
            "amount": 1500.00,
            "currency": "BRL",
            "country": "BR",
            "device": "mobile",
            "hour": 14,
            "day_of_week": 3,
            "user_id": "user-123",
        }
        probability = self.classifier.predict(features)
        assert 0 <= probability <= 1

    def test_predict_high_risk_features(self):
        """Test prediction with high risk features"""
        features = {
            "amount": 50000.00,
            "currency": "BTC",
            "country": "XX",
            "device": "unknown",
            "hour": 3,
            "day_of_week": 6,
        }
        probability = self.classifier.predict(features)
        assert 0 <= probability <= 1

    def test_predict_low_risk_features(self):
        """Test prediction with low risk features"""
        features = {
            "amount": 50.00,
            "currency": "BRL",
            "country": "BR",
            "device": "mobile",
            "hour": 10,
            "day_of_week": 2,
        }
        probability = self.classifier.predict(features)
        assert 0 <= probability <= 1

    def test_predict_minimal_features(self):
        """Test prediction with minimal features"""
        features = {
            "amount": 100.00,
            "currency": "USD",
            "country": "US",
            "device": "desktop",
        }
        probability = self.classifier.predict(features)
        assert 0 <= probability <= 1

    def test_preprocess_features(self):
        """Test feature preprocessing"""
        features = {
            "amount": 1000.00,
            "currency": "EUR",
            "country": "DE",
            "device": "tablet",
            "hour": 15,
            "day_of_week": 4,
        }
        processed = self.classifier.preprocess_features(features)
        assert processed.shape == (1, 12)

    def test_predict_batch(self):
        """Test batch prediction"""
        features_list = [
            {"amount": 100, "currency": "BRL", "country": "BR", "device": "mobile"},
            {"amount": 500, "currency": "USD", "country": "US", "device": "desktop"},
            {"amount": 10000, "currency": "EUR", "country": "XX", "device": "unknown"},
        ]
        predictions = self.classifier.predict_batch(features_list)
        assert predictions.shape == (3,)
        assert all(0 <= p <= 1 for p in predictions)


class TestRiskFeaturePreprocessor:
    """Tests for feature preprocessing"""

    def setup_method(self):
        self.preprocessor = RiskFeaturePreprocessor()

    def test_normalize_amount(self):
        """Test amount normalization"""
        result = self.preprocessor.normalize_amount(500.0)
        assert "amount_zscore" in result
        assert "amount_log" in result
        assert "is_very_large" in result
        assert result["amount_zscore"] == 0.0  # Mean should give 0

    def test_get_time_features(self):
        """Test time feature extraction"""
        features = self.preprocessor.get_time_features(14, 2)  # 2pm Wednesday

        assert "hour_norm" in features
        assert "is_night" in features
        assert "is_weekend" in features

        assert features["is_night"] == 0.0
        assert features["is_weekend"] == 0.0

    def test_get_country_risk(self):
        """Test country risk calculation"""
        br_risk = self.preprocessor.get_country_risk("BR")
        xx_risk = self.preprocessor.get_country_risk("XX")

        assert br_risk["country_risk"] < xx_risk["country_risk"]
        assert xx_risk["is_high_risk_country"] == 1.0

    def test_get_device_risk(self):
        """Test device risk calculation"""
        mobile_risk = self.preprocessor.get_device_risk("mobile")
        unknown_risk = self.preprocessor.get_device_risk("unknown")

        assert mobile_risk["device_risk"] < unknown_risk["device_risk"]
        assert unknown_risk["is_unknown_device"] == 1.0

    def test_preprocess_output_shape(self):
        """Test preprocessed output shape"""
        features = {
            "amount": 1000.00,
            "currency": "BRL",
            "country": "BR",
            "device": "mobile",
            "hour": 12,
            "day_of_week": 3,
        }
        result = self.preprocessor.preprocess(features)
        assert result.shape == (12,)

    def test_validate_features(self):
        """Test feature validation"""
        valid = {"amount": 100, "currency": "BRL", "country": "BR", "device": "mobile"}
        warnings = self.preprocessor.validate(valid)
        assert len(warnings) == 0

        invalid = {"device": "mobile"}
        warnings = self.preprocessor.validate(invalid)
        assert len(warnings) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
