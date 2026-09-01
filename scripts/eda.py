"""
Exploratory Data Analysis (EDA) Script
---------------------------------------
Generates visual reports of the synthetic transaction dataset and saves
them as PNG files under reports/figures/ for documentation purposes.

Usage:
    python scripts/eda.py
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from src.config import Config
from src.logger import get_logger
from src.data.generator import generate_synthetic_transactions
from src.features.build_features import engineer_features
from src.utils import ensure_dir

logger = get_logger("scripts.eda")

FIGURES_DIR = "reports/figures"
ensure_dir(FIGURES_DIR)
sns.set_theme(style="whitegrid", palette="muted")
PALETTE = {"Legitimate": "#3B82F6", "Fraud": "#EF4444"}


def load_data() -> pd.DataFrame:
    cfg = Config()
    path = cfg.raw_data_path
    if os.path.exists(path):
        logger.info(f"Loading raw data from {path}")
        df = pd.read_csv(path)
    else:
        logger.info("Raw data not found — generating 5000 sample rows for EDA")
        df = generate_synthetic_transactions(n_samples=5000)
    df = engineer_features(df)
    df["label"] = df["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
    return df


def plot_class_balance(df: pd.DataFrame):
    """Bar chart showing class imbalance."""
    counts = df["is_fraud"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        ["Legitimate", "Fraud"],
        [counts[0], counts.get(1, 0)],
        color=["#3B82F6", "#EF4444"],
        width=0.5,
        edgecolor="white",
    )
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{int(bar.get_height()):,}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Class Distribution (Fraud vs Legitimate)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "class_balance.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved: {out}")


def plot_amount_distribution(df: pd.DataFrame):
    """Log-scale violin plot of transaction amounts by label."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for lbl, color in PALETTE.items():
        sub = df[df["label"] == lbl]["transaction_amount"]
        ax.hist(np.log1p(sub), bins=50, alpha=0.65, label=lbl, color=color)
    ax.set_xlabel("log(1 + Amount $)")
    ax.set_ylabel("Count")
    ax.set_title("Transaction Amount Distribution (log scale)")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    data_violin = [
        df.loc[df["label"] == "Legitimate", "transaction_amount"].values,
        df.loc[df["label"] == "Fraud", "transaction_amount"].values,
    ]
    parts = ax2.violinplot(data_violin, positions=[1, 2], showmedians=True)
    parts["cmedians"].set_color("white")
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(list(PALETTE.values())[i])
        pc.set_alpha(0.75)
    ax2.set_yscale("log")
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Legitimate", "Fraud"])
    ax2.set_title("Amount Distribution — Violin (log scale)")
    ax2.set_ylabel("Amount ($)")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "amount_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved: {out}")


def plot_fraud_by_hour(df: pd.DataFrame):
    """Fraud count and rate by hour of the day."""
    hour_stats = df.groupby("hour_of_day").agg(
        fraud_count=("is_fraud", "sum"),
        total=("is_fraud", "count"),
    ).reset_index()
    hour_stats["fraud_rate"] = hour_stats["fraud_count"] / hour_stats["total"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.bar(hour_stats["hour_of_day"], hour_stats["fraud_count"], color="#EF4444", alpha=0.85)
    ax1.set_ylabel("Fraud Count")
    ax1.set_title("Fraud Volume and Rate by Hour of Day", fontsize=13, fontweight="bold")
    ax1.axvspan(1, 5, alpha=0.08, color="#EF4444", label="High-risk window")
    ax1.legend(loc="upper right")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(hour_stats["hour_of_day"], hour_stats["fraud_rate"] * 100, marker="o",
             color="#F59E0B", linewidth=2.5, markersize=6)
    ax2.fill_between(hour_stats["hour_of_day"], hour_stats["fraud_rate"] * 100,
                     alpha=0.12, color="#F59E0B")
    ax2.set_xlabel("Hour of Day (0–23)")
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.axvspan(1, 5, alpha=0.08, color="#EF4444")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "fraud_by_hour.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved: {out}")


def plot_categorical_fraud_rates(df: pd.DataFrame):
    """Fraud rate for key categorical variables."""
    cats = ["merchant_category", "device_type", "authentication_method", "transaction_type"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for ax, col in zip(axes.flat, cats):
        stats = df.groupby(col)["is_fraud"].mean().sort_values(ascending=False) * 100
        bars = ax.barh(stats.index, stats.values, color="#6366F1", alpha=0.85, edgecolor="white")
        ax.set_xlabel("Fraud Rate (%)")
        ax.set_title(f"Fraud Rate by {col.replace('_', ' ').title()}", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        for bar in bars:
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "categorical_fraud_rates.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved: {out}")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Pearson correlation heatmap of numeric features."""
    num_cols = [
        "transaction_amount", "customer_age", "account_balance",
        "avg_amount_30d", "txn_count_1h", "txn_count_24h",
        "distance_from_home_km", "prev_fraud_count",
        "amount_to_avg_ratio", "amount_to_balance_ratio",
        "velocity_ratio", "is_fraud",
    ]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr, mask=mask, cmap="RdBu_r", center=0,
        annot=True, fmt=".2f", linewidths=0.5,
        square=True, ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "correlation_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved: {out}")


def plot_velocity_analysis(df: pd.DataFrame):
    """Fraud rate as a function of transaction velocity."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    v1 = df.groupby("txn_count_1h")["is_fraud"].mean().reset_index()
    v1 = v1[v1["txn_count_1h"] <= 10]
    ax1.bar(v1["txn_count_1h"], v1["is_fraud"] * 100, color="#F59E0B", alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Transactions in Last 1 Hour")
    ax1.set_ylabel("Fraud Rate (%)")
    ax1.set_title("1-Hour Velocity vs Fraud Rate", fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)

    v24 = df.groupby("txn_count_24h")["is_fraud"].mean().reset_index()
    v24 = v24[v24["txn_count_24h"] <= 25]
    ax2.plot(v24["txn_count_24h"], v24["is_fraud"] * 100, color="#EF4444", marker="o",
             linewidth=2.5, markersize=5)
    ax2.set_xlabel("Transactions in Last 24 Hours")
    ax2.set_ylabel("Fraud Rate (%)")
    ax2.set_title("24-Hour Velocity vs Fraud Rate", fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "velocity_analysis.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info(f"Saved: {out}")


def print_summary(df: pd.DataFrame):
    n = len(df)
    n_fraud = df["is_fraud"].sum()
    logger.info("=" * 55)
    logger.info("  EDA Summary")
    logger.info("=" * 55)
    logger.info(f"  Total Transactions : {n:,}")
    logger.info(f"  Fraud Transactions : {n_fraud} ({n_fraud/n*100:.2f}%)")
    logger.info(f"  Legit Transactions : {n - n_fraud}")
    logger.info(f"  Features           : {df.shape[1]}")
    logger.info(f"  Amount (fraud) avg : ${df.loc[df['is_fraud']==1,'transaction_amount'].mean():,.0f}")
    logger.info(f"  Amount (legit) avg : ${df.loc[df['is_fraud']==0,'transaction_amount'].mean():,.0f}")
    logger.info(f"  Figures saved to   : {FIGURES_DIR}/")
    logger.info("=" * 55)


if __name__ == "__main__":
    df = load_data()
    print_summary(df)

    logger.info("Generating EDA plots…")
    plot_class_balance(df)
    plot_amount_distribution(df)
    plot_fraud_by_hour(df)
    plot_categorical_fraud_rates(df)
    plot_correlation_heatmap(df)
    plot_velocity_analysis(df)

    logger.info("All EDA figures generated successfully.")
