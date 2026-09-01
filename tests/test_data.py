import pytest
import pandas as pd
from src.data.generator import generate_synthetic_transactions
from src.features.build_features import engineer_features
from src.data.preprocessor import FraudDataPreprocessor

def test_generate_synthetic_transactions():
    df = generate_synthetic_transactions(n_samples=200, fraud_ratio=0.1, random_state=42)
    assert len(df) == 200
    assert "is_fraud" in df.columns
    # Fraud count may vary ±10% due to label noise; expect 14–26 for fraud_ratio=0.1
    fraud_n = int(df["is_fraud"].sum())
    assert 14 <= fraud_n <= 26, f"Expected ~20 fraud rows, got {fraud_n}"
    assert "transaction_amount" in df.columns
    assert df["transaction_amount"].min() > 0
    # Fraud amounts should average higher than legit (some noise expected)
    fraud_mean = df.loc[df["is_fraud"] == 1, "transaction_amount"].mean()
    legit_mean = df.loc[df["is_fraud"] == 0, "transaction_amount"].mean()
    assert fraud_mean > legit_mean * 1.5, (
        f"Expected fraud amounts >> legit; got fraud={fraud_mean:.0f}, legit={legit_mean:.0f}"
    )


def test_engineer_features():
    df_raw = generate_synthetic_transactions(n_samples=50, random_state=42)
    df_feat = engineer_features(df_raw)
    assert 'amount_to_avg_ratio' in df_feat.columns
    assert 'amount_to_balance_ratio' in df_feat.columns
    assert 'is_high_risk_time' in df_feat.columns
    assert 'velocity_ratio' in df_feat.columns

def test_preprocessor_pipeline():
    df_raw = generate_synthetic_transactions(n_samples=100, random_state=42)
    df_feat = engineer_features(df_raw)

    num_cols = ["transaction_amount", "customer_age", "account_balance", "txn_count_1h", "txn_count_24h", "avg_amount_30d"]
    cat_cols = ["transaction_type", "merchant_category", "device_type", "authentication_method"]

    preprocessor = FraudDataPreprocessor(numerical_cols=num_cols, categorical_cols=cat_cols)
    X_trans = preprocessor.fit_transform(df_feat[num_cols + cat_cols])

    assert isinstance(X_trans, pd.DataFrame)
    assert len(X_trans) == 100
    assert X_trans.isnull().sum().sum() == 0
