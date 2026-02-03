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
        assert data["service"] == "risk-classification"


class TestClassifyEndpoint:
    """Tests for risk classification endpoint"""

    def test_classify_valid_request(self, client):
        """Test classification with valid request"""
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
            "/risk/classify", data=json.dumps(payload), content_type="application/json"
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "risk_probability" in data
        assert 0 <= data["risk_probability"] <= 1

    def test_classify_empty_request(self, client):
        """Test classification with empty request"""
        response = client.post(
            "/risk/classify", data=json.dumps({}), content_type="application/json"
        )
        # Should still work with defaults
        assert response.status_code == 200

    def test_classify_no_body(self, client):
        """Test classification with no body"""
        response = client.post("/risk/classify")
        assert response.status_code == 400

    def test_classify_high_risk_transaction(self, client):
        """Test classification of high risk transaction"""
        payload = {
            "amount": 100000.00,
            "currency": "BTC",
            "country": "XX",
            "device": "unknown",
            "hour": 3,
            "day_of_week": 6,
        }
        response = client.post(
            "/risk/classify", data=json.dumps(payload), content_type="application/json"
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 0 <= data["risk_probability"] <= 1


class TestBatchEndpoint:
    """Tests for batch classification endpoint"""

    def test_batch_classify(self, client):
        """Test batch classification"""
        payload = {
            "transactions": [
                {
                    "transaction_id": "tx1",
                    "amount": 100,
                    "currency": "BRL",
                    "country": "BR",
                    "device": "mobile",
                },
                {
                    "transaction_id": "tx2",
                    "amount": 5000,
                    "currency": "USD",
                    "country": "US",
                    "device": "desktop",
                },
            ]
        }
        response = client.post(
            "/risk/batch", data=json.dumps(payload), content_type="application/json"
        )
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "results" in data
        assert len(data["results"]) == 2

    def test_batch_empty_transactions(self, client):
        """Test batch with empty transactions"""
        response = client.post(
            "/risk/batch", data=json.dumps({}), content_type="application/json"
        )
        assert response.status_code == 400


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

        assert data["model_type"] == "Binary Classifier"
        assert data["framework"] == "TensorFlow"
        assert "input_dim" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
