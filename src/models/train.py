import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Tuple, Any

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

from src.config import Config
from src.logger import get_logger
from src.utils import save_joblib, save_json
from src.models.evaluate import evaluate_model_performance

logger = get_logger("src.models.train")

class ModelTrainer:
    """Trains, benchmarks, and logs multiple Machine Learning algorithms for fraud detection."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        if HAS_MLFLOW:
            # Allow filesystem-based MLflow tracking store
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            self.mlflow_uri = self.config.get("paths.mlflow_tracking_uri", "file:./mlruns")
            mlflow.set_tracking_uri(self.mlflow_uri)
            try:
                mlflow.set_experiment("Banking_Fraud_Detection")
            except Exception as e:
                logger.warning(f"Could not set MLflow experiment: {e}")
        else:
            logger.info("MLflow not installed. Proceeding with standard file metrics logging.")

    def get_candidate_models(self, scale_pos_weight: float = 10.0) -> Dict[str, Any]:
        """Returns candidate model instances configured for imbalanced classification."""
        return {
            "LogisticRegression": LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=150, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.08,
                scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss"
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.08,
                scale_pos_weight=scale_pos_weight, random_state=42, verbose=-1
            )
        }

    def train_and_evaluate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[Any, Dict[str, Dict[str, Any]]]:
        """Splits data, trains all models, selects best performer, and saves artifacts."""
        test_size = self.config.get("training.test_size", 0.2)
        random_state = self.config.get("training.random_state", 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = float(n_neg / max(n_pos, 1))

        models = self.get_candidate_models(scale_pos_weight=scale_pos_weight)
        all_metrics = {}

        best_model_name = None
        best_roc_auc = -1.0
        best_model = None

        logger.info(f"Starting model comparison on {len(X_train)} train samples and {len(X_test)} test samples...")

        for name, model in models.items():
            logger.info(f"Training model: {name}...")
            model.fit(X_train, y_train)

            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_model_performance(y_test, y_pred_proba)

            if HAS_MLFLOW:
                try:
                    with mlflow.start_run(run_name=name):
                        mlflow.log_param("model_name", name)
                        for metric_name, val in metrics.items():
                            if isinstance(val, (int, float)):
                                mlflow.log_metric(metric_name, val)
                        mlflow.sklearn.log_model(model, artifact_path="model")
                except Exception as ex:
                    logger.warning(f"MLflow logging error for {name}: {ex}")

            all_metrics[name] = metrics

            if metrics["roc_auc"] > best_roc_auc:
                best_roc_auc = metrics["roc_auc"]
                best_model_name = name
                best_model = model

        logger.info(f"Winner Model: {best_model_name} with ROC-AUC = {best_roc_auc:.4f}")

        # Save winner model and metrics summary
        save_joblib(best_model, self.config.best_model_path)
        save_joblib(list(X.columns), self.config.feature_names_path)
        
        summary = {
            "best_model_name": best_model_name,
            "best_metrics": all_metrics[best_model_name],
            "all_models_metrics": all_metrics,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_count": len(X.columns)
        }
        save_json(summary, self.config.get("paths.metrics_path"))

        return best_model, summary
