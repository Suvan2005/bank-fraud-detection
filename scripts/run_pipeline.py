import os
import sys
import pandas as pd

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.logger import get_logger
from src.utils import ensure_dir, save_joblib
from src.data.generator import generate_synthetic_transactions
from src.features.build_features import engineer_features
from src.data.preprocessor import FraudDataPreprocessor
from src.models.train import ModelTrainer

logger = get_logger("scripts.run_pipeline")

def run_pipeline():
    """Executes full ML pipeline: Data Generation -> Feature Engineering -> Preprocessing -> Model Training."""
    logger.info("=" * 60)
    logger.info("  Banking Fraud Detection -- End-to-End ML Pipeline")
    logger.info("=" * 60)

    config = Config()

    # Step 1: Generate Synthetic Raw Data
    n_samples = config.get("data_generation.n_samples", 12000)
    fraud_ratio = config.get("data_generation.fraud_ratio", 0.045)
    random_state = config.get("data_generation.random_state", 42)

    logger.info(f"[1/4] Generating {n_samples} synthetic transactions (fraud_ratio={fraud_ratio})...")
    raw_df = generate_synthetic_transactions(
        n_samples=n_samples, fraud_ratio=fraud_ratio, random_state=random_state
    )
    ensure_dir(os.path.dirname(config.raw_data_path))
    raw_df.to_csv(config.raw_data_path, index=False)
    logger.info(f"     Raw data saved -> {config.raw_data_path}")

    # Step 2: Feature Engineering
    logger.info("[2/4] Engineering features (ratios, velocity flags, time features)...")
    feat_df = engineer_features(raw_df)
    ensure_dir(os.path.dirname(config.processed_data_path))
    feat_df.to_csv(config.processed_data_path, index=False)
    logger.info(f"     Feature data saved  -> {config.processed_data_path}")

    # Step 3: Preprocessing
    logger.info("[3/4] Fitting preprocessing pipeline (scaling + encoding)...")
    num_cols = config.get("features.numerical")
    cat_cols = config.get("features.categorical")
    target_col = config.get("features.target")

    X_raw = feat_df[num_cols + cat_cols]
    y = feat_df[target_col]

    preprocessor = FraudDataPreprocessor(numerical_cols=num_cols, categorical_cols=cat_cols)
    X_processed = preprocessor.fit_transform(X_raw)

    # Save preprocessor
    ensure_dir(os.path.dirname(config.preprocessor_path))
    save_joblib(preprocessor, config.preprocessor_path)
    logger.info(f"     Preprocessor saved  -> {config.preprocessor_path}")
    logger.info(f"     Processed feature matrix: {X_processed.shape}")

    # Step 4: Model Training & Benchmarking
    logger.info("[4/4] Training & benchmarking candidate models with MLflow logging...")
    trainer = ModelTrainer(config=config)
    best_model, summary = trainer.train_and_evaluate(X_processed, y)

    logger.info("=" * 60)
    logger.info("  Pipeline Completed Successfully!")
    logger.info(f"  Best Model : {summary['best_model_name']}")
    logger.info(f"  ROC-AUC   : {summary['best_metrics']['roc_auc']:.4f}")
    logger.info(f"  PR-AUC    : {summary['best_metrics']['pr_auc']:.4f}")
    logger.info(f"  F1-Score  : {summary['best_metrics']['f1_score']:.4f}")
    logger.info(f"  Recall    : {summary['best_metrics']['recall']:.4f}")
    logger.info(f"  Precision : {summary['best_metrics']['precision']:.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_pipeline()
