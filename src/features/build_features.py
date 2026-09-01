import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger("src.features.build_features")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes derived statistical, temporal, and risk-based features from raw transactions."""
    df_feat = df.copy()

    # 1. Ratios
    df_feat['amount_to_avg_ratio'] = np.round(
        df_feat['transaction_amount'] / (df_feat['avg_amount_30d'] + 1e-5), 4
    )
    df_feat['amount_to_balance_ratio'] = np.round(
        df_feat['transaction_amount'] / (df_feat['account_balance'] + 1e-5), 4
    )
    df_feat['velocity_ratio'] = np.round(
        df_feat['txn_count_1h'] / (df_feat['txn_count_24h'] + 1.0), 4
    )

    # 2. Time-based binary flag
    if 'hour_of_day' not in df_feat.columns and 'timestamp' in df_feat.columns:
        df_feat['hour_of_day'] = pd.to_datetime(df_feat['timestamp']).dt.hour

    df_feat['is_high_risk_time'] = df_feat['hour_of_day'].apply(
        lambda h: 1 if h in [1, 2, 3, 4, 5] else 0
    )

    logger.info(f"Engineered {df_feat.shape[1]} features for {len(df_feat)} records.")
    return df_feat
