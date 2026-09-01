import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    brier_score_loss
)
from typing import Dict, Any
from src.logger import get_logger

logger = get_logger("src.models.evaluate")

def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Calculates comprehensive classification metrics for fraud detection."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    pr_auc = float(average_precision_score(y_true, y_pred_proba))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_pred_proba))

    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "accuracy": round(accuracy, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": cm,
        "classification_threshold": threshold
    }

    logger.info(f"Evaluated Metrics -> ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")
    return metrics
