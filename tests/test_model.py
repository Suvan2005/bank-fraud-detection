import pytest
import numpy as np
from src.models.evaluate import evaluate_model_performance
from src.inference.predictor import FraudPredictor

def test_evaluate_model_performance():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.85, 0.9, 0.3, 0.7])
    metrics = evaluate_model_performance(y_true, y_proba)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert metrics["roc_auc"] > 0.8
    assert metrics["f1_score"] > 0.8

def test_fraud_predictor_risk_rules():
    predictor = FraudPredictor()
    
    score_low, level_low, act_low = predictor.evaluate_risk(0.15)
    assert score_low == 15.0
    assert level_low == "Low"
    assert act_low == "Approve"

    score_med, level_med, act_med = predictor.evaluate_risk(0.55)
    assert score_med == 55.0
    assert level_med == "Medium"
    assert act_med == "Review"

    score_high, level_high, act_high = predictor.evaluate_risk(0.88)
    assert score_high == 88.0
    assert level_high == "High"
    assert act_high == "Block"
