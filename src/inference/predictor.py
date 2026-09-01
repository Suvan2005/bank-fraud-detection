import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Tuple
from src.config import Config
from src.logger import get_logger
from src.utils import load_joblib
from src.features.build_features import engineer_features
from src.models.explain import ModelExplainer

logger = get_logger("src.inference.predictor")

class FraudPredictor:
    """End-to-end inference engine for real-time and batch fraud prediction."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.feature_names = []
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            self.model = load_joblib(self.config.best_model_path)
            self.preprocessor = load_joblib(self.config.preprocessor_path)
            self.feature_names = load_joblib(self.config.feature_names_path)
            self.explainer = ModelExplainer(self.model, self.feature_names)
            logger.info("Successfully loaded model, preprocessor, and explainer artifacts.")
        except Exception as e:
            logger.warning(f"Could not load artifacts during initialization: {e}. Model will need to be trained first.")

    def evaluate_risk(self, probability: float) -> Tuple[float, str, str]:
        """Converts fraud probability to risk score (0-100), risk level, and recommended action."""
        risk_score = round(probability * 100.0, 1)
        low_thresh = self.config.get("risk_rules.low_threshold", 30.0)
        high_thresh = self.config.get("risk_rules.high_threshold", 70.0)

        if risk_score < low_thresh:
            level = "Low"
            action = self.config.get("risk_rules.actions.low", "Approve")
        elif risk_score < high_thresh:
            level = "Medium"
            action = self.config.get("risk_rules.actions.medium", "Review")
        else:
            level = "High"
            action = self.config.get("risk_rules.actions.high", "Block")

        return risk_score, level, action

    def _preprocess(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineering then preprocessor transform."""
        num_cols = self.config.get("features.numerical")
        cat_cols = self.config.get("features.categorical")
        df_feat = engineer_features(df_raw)
        # Select only expected columns; fill any missing with 0
        for col in num_cols + cat_cols:
            if col not in df_feat.columns:
                df_feat[col] = 0
        return self.preprocessor.transform(df_feat[num_cols + cat_cols])

    def predict_single(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates a single raw transaction dictionary."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model artifacts are not loaded. Train the pipeline first.")

        df_raw = pd.DataFrame([transaction_data])
        X_trans = self._preprocess(df_raw)

        prob = float(self.model.predict_proba(X_trans)[:, 1][0])
        is_fraud = int(prob >= 0.5)
        risk_score, risk_level, action = self.evaluate_risk(prob)

        explanation = self.explainer.explain_instance(X_trans, top_n=5)

        return {
            "transaction_id": transaction_data.get("transaction_id", "N/A"),
            "is_fraud": is_fraud,
            "fraud_probability": round(prob, 4),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "recommended_action": action,
            "top_risk_factors": explanation["top_risk_factors"]
        }

    def predict_batch(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Evaluates a batch DataFrame of raw transactions."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model artifacts are not loaded. Train the pipeline first.")

        X_trans = self._preprocess(transactions_df)
        probs = self.model.predict_proba(X_trans)[:, 1]

        results = []
        for p in probs:
            is_f = int(p >= 0.5)
            score, level, act = self.evaluate_risk(float(p))
            results.append({
                "is_fraud": is_f,
                "fraud_probability": round(float(p), 4),
                "risk_score": score,
                "risk_level": level,
                "recommended_action": act
            })

        res_df = pd.DataFrame(results)
        return pd.concat([transactions_df.reset_index(drop=True), res_df], axis=1)
