"""
Tests for the inference server
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.server import app


@pytest.fixture
def client():
    """Create test client"""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Tests for health endpoint"""

    def test_health_returns_200(self, client):
        """Test health endpoint returns 200"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Test health endpoint returns status"""
        response = client.get("/health")
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["service"] == "anomaly-detection"


class TestScoreEndpoint:
    """Tests for anomaly score endpoint"""

    def test_score_valid_request(self, client):
        """Test scoring with valid request"""
        payload = {
            "amount": 1500.00,
            "currency": "BRL",
            "country": "BR",
            "device": "mobile",
            "hour": 14,
            "day_of_week": 3,
            "user_id": "user-123",
        }
        response = client.post(
            "/anomaly/score", data=json.dumps(payload), content_type="application/json"
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "anomaly_score" in data
        assert 0 <= data["anomaly_score"] <= 1

    def test_score_empty_request(self, client):
        """Test scoring with empty request"""
        response = client.post(
            "/anomaly/score", data=json.dumps({}), content_type="application/json"
        )
        # Should still work with defaults
        assert response.status_code == 200

    def test_score_no_body(self, client):
        """Test scoring with no body"""
        response = client.post("/anomaly/score")
        assert response.status_code == 400

    def test_score_high_amount_transaction(self, client):
        """Test scoring high amount transaction"""
        payload = {
            "amount": 100000.00,
            "currency": "USD",
            "country": "XX",
            "device": "unknown",
            "hour": 3,
            "day_of_week": 6,
        }
        response = client.post(
            "/anomaly/score", data=json.dumps(payload), content_type="application/json"
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 0 <= data["anomaly_score"] <= 1


class TestModelInfoEndpoint:
    """Tests for model info endpoint"""

    def test_model_info_returns_200(self, client):
        """Test model info returns 200"""
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_content(self, client):
        """Test model info content"""
        response = client.get("/model/info")
        data = json.loads(response.data)

        assert data["model_type"] == "Autoencoder"
        assert data["framework"] == "PyTorch"
        assert "input_dim" in data
        assert "encoding_dim" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
