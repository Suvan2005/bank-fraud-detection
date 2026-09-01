from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TransactionInput(BaseModel):
    model_config = {"json_schema_extra": {"example": {
        "transaction_id": "TXN-SAMPLE001",
        "customer_id": "CUST-10023",
        "transaction_amount": 850.0,
        "customer_age": 34,
        "account_balance": 4500.0,
        "avg_amount_30d": 65.0,
        "hour_of_day": 3,
        "distance_from_home_km": 350.0,
        "txn_count_1h": 4,
        "txn_count_24h": 12,
        "prev_fraud_count": 1,
        "is_foreign_transaction": 1,
        "transaction_type": "Transfer",
        "merchant_category": "Crypto",
        "device_type": "Unknown",
        "authentication_method": "None"
    }}}

    transaction_id: Optional[str] = Field(default="TXN-SAMPLE001", description="Unique transaction ID")
    customer_id: Optional[str] = Field(default="CUST-10023", description="Customer ID")
    transaction_amount: float = Field(..., gt=0, description="Amount of the transaction in USD")
    customer_age: int = Field(..., ge=18, le=100, description="Customer age in years")
    account_balance: float = Field(..., ge=0, description="Current account balance in USD")
    avg_amount_30d: float = Field(..., ge=0, description="30-day average transaction amount")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of the transaction (0-23, 24h format)")
    distance_from_home_km: float = Field(..., ge=0, description="Distance from home address in km")
    txn_count_1h: int = Field(..., ge=0, description="Number of transactions by this customer in last 1 hour")
    txn_count_24h: int = Field(..., ge=0, description="Number of transactions by this customer in last 24 hours")
    prev_fraud_count: int = Field(..., ge=0, description="Count of past confirmed frauds on this account")
    is_foreign_transaction: int = Field(..., ge=0, le=1, description="1 if transaction is foreign/international, else 0")
    transaction_type: str = Field(..., description="One of: Transfer, Withdrawal, Payment, Online Purchase, ATM")
    merchant_category: str = Field(..., description="One of: Grocery, Electronics, Luxury, Travel, Crypto, Gambling, Utility")
    device_type: str = Field(..., description="One of: Mobile, Web, POS, Unknown")
    authentication_method: str = Field(..., description="One of: PIN, Biometric, Password, OTP, None")


class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    feature_value: Any
    impact: str


class PredictionResponse(BaseModel):
    transaction_id: str
    is_fraud: int
    fraud_probability: float
    risk_score: float
    risk_level: str
    recommended_action: str
    top_risk_factors: List[RiskFactor]


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]


class BatchPredictionResponse(BaseModel):
    total_processed: int
    flagged_fraud_count: int
    blocked_count: int
    predictions: List[PredictionResponse]


class ModelInfoResponse(BaseModel):
    project_name: str
    version: str
    best_model: str
    metrics: Dict[str, Any]
    feature_names: List[str]
