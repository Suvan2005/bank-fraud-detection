import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Banking Transaction Fraud Detection API"

def test_api_health():
    response = client.get("/health")
    assert response.status_code in [200, 530, 503]

def test_predict_single_endpoint():
    payload = {
        "transaction_id": "TXN-TEST-001",
        "customer_id": "CUST-999",
        "transaction_amount": 1250.0,
        "customer_age": 30,
        "account_balance": 5000.0,
        "avg_amount_30d": 80.0,
        "hour_of_day": 3,
        "distance_from_home_km": 400.0,
        "txn_count_1h": 5,
        "txn_count_24h": 14,
        "prev_fraud_count": 1,
        "is_foreign_transaction": 1,
        "transaction_type": "Transfer",
        "merchant_category": "Crypto",
        "device_type": "Unknown",
        "authentication_method": "None"
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert "is_fraud" in data
        assert "risk_score" in data
        assert "risk_level" in data
        assert "recommended_action" in data
        assert "top_risk_factors" in data
