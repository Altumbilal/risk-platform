"""
Tests for the anomaly detection model
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.autoencoder import TransactionAutoencoder, create_detector
from model.preprocessing import FeaturePreprocessor


class TestTransactionAutoencoder:
    """Tests for the autoencoder model"""

    def setup_method(self):
        self.model = TransactionAutoencoder(input_dim=10, encoding_dim=4)

    def test_model_forward(self):
        """Test forward pass"""
        x = torch.randn(16, 10)
        output = self.model(x)
        assert output.shape == (16, 10)

    def test_encoder_output_dim(self):
        """Test encoder produces correct output dimension"""
        x = torch.randn(16, 10)
        encoded = self.model.encode(x)
        assert encoded.shape == (16, 4)

    def test_reconstruction_error(self):
        """Test reconstruction error calculation"""
        x = torch.randn(16, 10)
        error = self.model.reconstruction_error(x)
        assert error.shape == (16,)
        assert (error >= 0).all()

    def test_anomaly_score_range(self):
        """Test that anomaly scores are in [0, 1]"""
        x = torch.randn(100, 10)
        scores = self.model.anomaly_score(x)
        assert scores.shape == (100,)
        assert (scores >= 0).all()
        assert (scores <= 1).all()


class TestAnomalyDetector:
    """Tests for the high-level detector interface"""

    def setup_method(self):
        self.detector = create_detector()

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
        score = self.detector.predict(features)
        assert 0 <= score <= 1

    def test_predict_high_amount(self):
        """Test prediction with high amount"""
        features = {
            "amount": 50000.00,
            "currency": "USD",
            "country": "XX",
            "device": "unknown",
            "hour": 3,
            "day_of_week": 6,
            "user_id": "user-456",
        }
        score = self.detector.predict(features)
        assert 0 <= score <= 1

    def test_predict_minimal_features(self):
        """Test prediction with minimal features"""
        features = {
            "amount": 100.00,
            "currency": "BRL",
            "country": "BR",
            "device": "mobile",
        }
        score = self.detector.predict(features)
        assert 0 <= score <= 1

    def test_predict_consistency(self):
        """Test that same input produces similar outputs"""
        features = {
            "amount": 500.00,
            "currency": "EUR",
            "country": "DE",
            "device": "desktop",
            "hour": 10,
            "day_of_week": 1,
        }
        scores = [self.detector.predict(features) for _ in range(10)]
        # Allow some variance due to noise
        assert max(scores) - min(scores) < 0.2

    def test_preprocess_features(self):
        """Test feature preprocessing"""
        features = {
            "amount": 1000.00,
            "currency": "BRL",
            "country": "BR",
            "device": "mobile",
            "hour": 12,
            "day_of_week": 3,
        }
        tensor = self.detector.preprocess_features(features)
        assert tensor.shape == (1, 10)


class TestFeaturePreprocessor:
    """Tests for feature preprocessing"""

    def setup_method(self):
        self.preprocessor = FeaturePreprocessor()

    def test_normalize_amount(self):
        """Test amount normalization"""
        result = self.preprocessor.normalize_amount(500.0)
        assert result == 0.0  # Mean should give 0

    def test_encode_currency(self):
        """Test currency encoding"""
        encoding = self.preprocessor.encode_currency("USD")
        assert len(encoding) == 8
        assert sum(encoding) == 1.0
        assert encoding[1] == 1.0  # USD is index 1

    def test_encode_device(self):
        """Test device encoding"""
        mobile = self.preprocessor.encode_device("mobile")
        desktop = self.preprocessor.encode_device("desktop")
        assert mobile == 0.0
        assert desktop == 0.25

    def test_get_country_risk(self):
        """Test country risk scores"""
        br_risk = self.preprocessor.get_country_risk("BR")
        de_risk = self.preprocessor.get_country_risk("DE")
        unknown_risk = self.preprocessor.get_country_risk("XX")

        assert br_risk == 0.3
        assert de_risk == 0.15
        assert unknown_risk == 0.5  # Default for unknown

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
        assert result.shape == (10,)

    def test_validate_features(self):
        """Test feature validation"""
        valid = {"amount": 100, "currency": "BRL", "country": "BR", "device": "mobile"}
        invalid = {"amount": 100}

        assert self.preprocessor.validate_features(valid)
        assert not self.preprocessor.validate_features(invalid)

    def test_engineer_time_features(self):
        """Test time feature engineering"""
        features = self.preprocessor.engineer_time_features(14, 2)  # 2pm Wednesday

        assert "is_weekend" in features
        assert "is_night" in features
        assert "is_business_hours" in features

        assert features["is_weekend"] == 0.0
        assert features["is_night"] == 0.0
        assert features["is_business_hours"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
