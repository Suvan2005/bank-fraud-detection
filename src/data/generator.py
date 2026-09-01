import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timedelta
from src.logger import get_logger

logger = get_logger("src.data.generator")


class SyntheticTransactionGenerator:
    """Generates realistic banking transaction dataset with embedded fraud patterns."""

    def __init__(self, n_samples: int = 12000, fraud_ratio: float = 0.045, random_state: int = 42):
        self.n_samples = n_samples
        self.fraud_ratio = fraud_ratio
        self.random_state = random_state

    def generate(self) -> pd.DataFrame:
        return generate_synthetic_transactions(
            n_samples=self.n_samples,
            fraud_ratio=self.fraud_ratio,
            random_state=self.random_state,
        )


def generate_synthetic_transactions(
    n_samples: int = 12000,
    fraud_ratio: float = 0.045,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a realistic, overlapping synthetic banking dataset.

    Fraud signals are probabilistic — not deterministic — so models
    achieve realistic ROC-AUC scores (0.92–0.97) rather than trivial 1.0.
    """
    logger.info(f"Generating {n_samples} synthetic banking transactions (fraud_ratio={fraud_ratio})…")
    rng = np.random.default_rng(random_state)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # ── Categorical lookup tables ────────────────────────────────────────────
    transaction_types = ["Transfer", "Withdrawal", "Payment", "Online Purchase", "ATM"]
    merchant_categories = ["Grocery", "Electronics", "Luxury", "Travel", "Crypto", "Gambling", "Utility"]
    device_types = ["Mobile", "Web", "POS", "Unknown"]
    auth_methods = ["PIN", "Biometric", "Password", "OTP", "None"]

    # Probability tables (legit vs fraud)
    txn_type_p = {
        "legit": [0.25, 0.20, 0.35, 0.15, 0.05],
        "fraud": [0.40, 0.15, 0.08, 0.32, 0.05],
    }
    merch_p = {
        "legit": [0.35, 0.15, 0.05, 0.10, 0.02, 0.01, 0.32],
        "fraud": [0.06, 0.22, 0.22, 0.14, 0.16, 0.14, 0.06],
    }
    device_p = {
        "legit": [0.55, 0.30, 0.14, 0.01],
        "fraud": [0.22, 0.38, 0.10, 0.30],
    }
    auth_p = {
        "legit": [0.30, 0.35, 0.20, 0.14, 0.01],
        "fraud": [0.07, 0.06, 0.14, 0.23, 0.50],
    }

    def _norm(p):
        a = np.array(p, dtype=float)
        return a / a.sum()

    start_date = datetime(2026, 1, 1)

    # ────────────────────────────────────────────────────────────────────────
    # 1. LEGITIMATE TRANSACTIONS
    # ────────────────────────────────────────────────────────────────────────
    legit_ages = rng.integers(18, 76, size=n_legit)
    legit_balances = rng.exponential(scale=7000, size=n_legit) + 500
    legit_avg_30d = rng.gamma(shape=3.0, scale=40.0, size=n_legit) + 20
    # Amounts cluster near avg (0.4–2.0×), with occasional outliers
    legit_amount_mult = rng.lognormal(mean=0.0, sigma=0.50, size=n_legit)
    legit_amounts = legit_avg_30d * np.clip(legit_amount_mult, 0.15, 2.5)  # cap legit at 2.5x

    _legit_hour_p = _norm([
        0.010, 0.010, 0.006, 0.005, 0.010, 0.020,
        0.040, 0.060, 0.070, 0.080, 0.070, 0.060,
        0.060, 0.060, 0.060, 0.060, 0.060, 0.060,
        0.060, 0.050, 0.040, 0.030, 0.020, 0.019,
    ])
    legit_hours = rng.choice(np.arange(24), size=n_legit, p=_legit_hour_p)
    legit_dist = rng.exponential(scale=12.0, size=n_legit)
    legit_txn_1h = rng.poisson(lam=0.5, size=n_legit)
    legit_txn_24h = legit_txn_1h + rng.poisson(lam=2.5, size=n_legit)
    legit_prev_fraud = rng.choice([0, 1], size=n_legit, p=[0.97, 0.03])
    legit_foreign = rng.choice([0, 1], size=n_legit, p=[0.94, 0.06])

    # ────────────────────────────────────────────────────────────────────────
    # 2. FRAUDULENT TRANSACTIONS  (overlapping, not perfectly separated)
    # ────────────────────────────────────────────────────────────────────────
    fraud_ages = rng.integers(20, 66, size=n_fraud)
    fraud_balances = rng.exponential(scale=11000, size=n_fraud) + 1500
    fraud_avg_30d = rng.gamma(shape=3.0, scale=50.0, size=n_fraud) + 30
    # Fraud: bimodal — minimum 3x spike, most 3-8x, some 8-18x — target ~0.90-0.95 AUC
    spike_mask = rng.random(n_fraud) < 0.60
    fraud_mult_low  = rng.uniform(3.0, 8.0,  size=n_fraud)   # 60%: clear spike
    fraud_mult_high = rng.uniform(8.0, 18.0, size=n_fraud)   # 40%: extreme spike
    fraud_mult = np.where(spike_mask, fraud_mult_low, fraud_mult_high)
    fraud_amounts = fraud_avg_30d * fraud_mult

    _fraud_hour_p = _norm([
        0.06, 0.17, 0.21, 0.19, 0.11, 0.05,
        0.03, 0.02, 0.02, 0.02, 0.02, 0.01,
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    ])
    fraud_hours = rng.choice(np.arange(24), size=n_fraud, p=_fraud_hour_p)
    # Distance: some fraud is local (card-present), most is far
    fraud_dist_local = rng.exponential(scale=10.0, size=n_fraud)
    fraud_dist_far = rng.exponential(scale=300.0, size=n_fraud) + 80.0
    local_mask = rng.random(n_fraud) < 0.25
    fraud_dist = np.where(local_mask, fraud_dist_local, fraud_dist_far)
    # Velocity: elevated but not always extreme
    fraud_txn_1h = rng.poisson(lam=3.0, size=n_fraud)   # clearer velocity signal
    fraud_txn_24h = fraud_txn_1h + rng.poisson(lam=7.0, size=n_fraud)
    fraud_prev_fraud = rng.choice([0, 1, 2, 3], size=n_fraud, p=[0.52, 0.30, 0.13, 0.05])
    fraud_foreign = rng.choice([0, 1], size=n_fraud, p=[0.42, 0.58])

    # ────────────────────────────────────────────────────────────────────────
    # 3. COMBINE
    # ────────────────────────────────────────────────────────────────────────
    def _cat(choices, p_dict, key, size):
        return rng.choice(choices, size=size, p=_norm(p_dict[key]))

    df_legit = pd.DataFrame({
        "customer_age":         legit_ages,
        "account_balance":      np.round(legit_balances, 2),
        "avg_amount_30d":       np.round(legit_avg_30d, 2),
        "transaction_amount":   np.round(legit_amounts, 2),
        "hour_of_day":          legit_hours,
        "distance_from_home_km": np.round(legit_dist, 1),
        "txn_count_1h":         legit_txn_1h,
        "txn_count_24h":        legit_txn_24h,
        "prev_fraud_count":     legit_prev_fraud,
        "is_foreign_transaction": legit_foreign,
        "transaction_type":     _cat(transaction_types, txn_type_p, "legit", n_legit),
        "merchant_category":    _cat(merchant_categories, merch_p, "legit", n_legit),
        "device_type":          _cat(device_types, device_p, "legit", n_legit),
        "authentication_method": _cat(auth_methods, auth_p, "legit", n_legit),
        "is_fraud": 0,
    })

    df_fraud = pd.DataFrame({
        "customer_age":         fraud_ages,
        "account_balance":      np.round(fraud_balances, 2),
        "avg_amount_30d":       np.round(fraud_avg_30d, 2),
        "transaction_amount":   np.round(fraud_amounts, 2),
        "hour_of_day":          fraud_hours,
        "distance_from_home_km": np.round(fraud_dist, 1),
        "txn_count_1h":         fraud_txn_1h,
        "txn_count_24h":        fraud_txn_24h,
        "prev_fraud_count":     fraud_prev_fraud,
        "is_foreign_transaction": fraud_foreign,
        "transaction_type":     _cat(transaction_types, txn_type_p, "fraud", n_fraud),
        "merchant_category":    _cat(merchant_categories, merch_p, "fraud", n_fraud),
        "device_type":          _cat(device_types, device_p, "fraud", n_fraud),
        "authentication_method": _cat(auth_methods, auth_p, "fraud", n_fraud),
        "is_fraud": 1,
    })

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # ── Label Noise (2%) — simulates real-world labelling uncertainty ─────────
    # Flips a small fraction of labels so models score realistic ~0.90-0.95 AUC
    noise_rate = 0.02
    noise_mask = rng.random(len(df)) < noise_rate
    df.loc[noise_mask, "is_fraud"] = 1 - df.loc[noise_mask, "is_fraud"]


    # ── IDs & Timestamps ─────────────────────────────────────────────────────
    df["transaction_id"] = [f"TXN-{uuid.uuid4().hex[:10].upper()}" for _ in range(len(df))]
    # Pool of ~5000 customers (some are repeat offenders)
    customer_pool = [f"CUST-{rng.integers(10000, 60000)}" for _ in range(5000)]
    df["customer_id"] = [customer_pool[rng.integers(0, 5000)] for _ in range(len(df))]

    timestamps = []
    for h in df["hour_of_day"]:
        d = int(rng.integers(0, 30))
        m = int(rng.integers(0, 60))
        s = int(rng.integers(0, 60))
        ts = start_date + timedelta(days=d, hours=int(h), minutes=m, seconds=s)
        timestamps.append(ts.strftime("%Y-%m-%d %H:%M:%S"))
    df["timestamp"] = timestamps

    fraud_count = int(df["is_fraud"].sum())
    logger.info(
        f"Generated {len(df)} transactions - "
        f"{fraud_count} fraud ({fraud_count/len(df)*100:.2f}%) / "
        f"{n_legit} legitimate."
    )
    return df
