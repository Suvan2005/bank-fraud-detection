import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from src.logger import get_logger

logger = get_logger("src.models.explain")

class ModelExplainer:
    """Computes SHAP explanations for individual predictions and global feature attribution."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("Initialized SHAP TreeExplainer.")
        except Exception as e:
            logger.warning(f"TreeExplainer failed ({e}). Falling back to KernelExplainer.")
            dummy_bg = np.zeros((10, len(self.feature_names)))
            self.explainer = shap.KernelExplainer(self.model.predict_proba, dummy_bg)

    def _get_fraud_shap_values(self, shap_values) -> np.ndarray:
        """Extract 1D SHAP values for the fraud class from any output format."""
        if isinstance(shap_values, list):
            # Old sklearn-style: list of arrays per class
            return np.array(shap_values[1]).flatten()
        sv = np.array(shap_values)
        if sv.ndim == 3:
            # Shape (n_samples, n_features, n_classes)
            return sv[0, :, 1]
        elif sv.ndim == 2:
            # Shape (n_samples, n_features) — single output
            return sv[0]
        else:
            return sv

    def _get_base_value(self) -> float:
        """Safely extract scalar base value for fraud class."""
        ev = self.explainer.expected_value
        if isinstance(ev, (list, np.ndarray)):
            arr = np.array(ev).flatten()
            return float(arr[1]) if len(arr) > 1 else float(arr[0])
        return float(ev)

    def explain_instance(self, X_row: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
        """Calculates SHAP values for a single transaction row and extracts top risk contributors."""
        if isinstance(X_row, pd.Series):
            X_row = pd.DataFrame([X_row])

        shap_values = self.explainer.shap_values(X_row)
        vals = self._get_fraud_shap_values(shap_values)

        raw_vals = X_row.iloc[0].values

        feature_contribs = []
        for feature, shap_val, raw_val in zip(self.feature_names, vals, raw_vals):
            feature_contribs.append({
                "feature": feature,
                "shap_value": float(round(float(shap_val), 4)),
                "feature_value": float(round(float(raw_val), 4)) if isinstance(raw_val, (int, float, np.integer, np.floating)) else str(raw_val),
                "impact": "Increases Fraud Risk" if shap_val > 0 else "Decreases Fraud Risk"
            })

        sorted_contribs = sorted(feature_contribs, key=lambda x: abs(x["shap_value"]), reverse=True)
        base_value = self._get_base_value()

        return {
            "base_value": round(base_value, 4),
            "top_risk_factors": sorted_contribs[:top_n],
            "all_contributions": sorted_contribs
        }

    def get_global_importance(self, X_sample: pd.DataFrame) -> Dict[str, float]:
        """Calculates mean absolute SHAP values across a sample set for global feature ranking."""
        shap_values = self.explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            vals = np.abs(np.array(shap_values[1])).mean(axis=0)
        else:
            sv = np.array(shap_values)
            if sv.ndim == 3:
                vals = np.abs(sv[:, :, 1]).mean(axis=0)
            else:
                vals = np.abs(sv).mean(axis=0)

        importance_series = pd.Series(vals, index=self.feature_names).sort_values(ascending=False)
        return {feat: float(round(val, 4)) for feat, val in importance_series.items()}
