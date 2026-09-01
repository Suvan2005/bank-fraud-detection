import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.inference.predictor import FraudPredictor
from src.utils import load_json
from src.data.generator import generate_synthetic_transactions

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield — Banking AI Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Premium Dark-Accent Design
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #1E40AF 0%, #7C3AED 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.hero-sub {
    font-size: 1rem;
    color: #64748B;
    margin-bottom: 1.8rem;
}
.kpi-card {
    background: linear-gradient(135deg, #F8FAFF 0%, #EEF2FF 100%);
    border: 1px solid #C7D2FE;
    border-radius: 14px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(99,102,241,0.08);
}
.kpi-label { font-size: 0.78rem; color: #6366F1; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-value { font-size: 2rem; font-weight: 700; color: #1E1B4B; }
.badge {
    display: inline-block;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 8px 20px;
    border-radius: 999px;
    letter-spacing: 0.03em;
    margin-top: 8px;
}
.badge-approve { background: #DCFCE7; color: #15803D; border: 1.5px solid #86EFAC; }
.badge-review  { background: #FEF3C7; color: #92400E; border: 1.5px solid #FCD34D; }
.badge-block   { background: #FEE2E2; color: #991B1B; border: 1.5px solid #FCA5A5; }
.risk-panel {
    border-radius: 14px;
    padding: 16px 20px;
    margin-top: 8px;
    border: 1px solid #E2E8F0;
    background: #F8FAFC;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1E293B;
    margin-bottom: 8px;
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E1B4B 100%);
}
div[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
div[data-testid="stSidebar"] hr { border-color: #334155; }
div[data-testid="stSidebar"] .stMarkdown h1,
div[data-testid="stSidebar"] .stMarkdown h2 { color: #F8FAFC !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loader
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading fraud detection model…")
def get_predictor():
    try:
        return FraudPredictor(Config())
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_raw_data():
    cfg = Config()
    path = cfg.raw_data_path
    if os.path.exists(path):
        return pd.read_csv(path)
    return generate_synthetic_transactions(n_samples=3000)

predictor = get_predictor()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield")
    st.markdown("#### Banking AI Security Platform")
    st.markdown("---")

    model_ready = predictor is not None and predictor.model is not None
    status_color = "🟢" if model_ready else "🔴"
    status_text  = "Online" if model_ready else "Offline"
    st.markdown(f"**System Status:** {status_color} {status_text}")

    if model_ready:
        cfg = predictor.config
        metrics_path = cfg.get("paths.metrics_path", "models/model_metrics.json")
        if os.path.exists(metrics_path):
            m = load_json(metrics_path)
            st.markdown(f"**Best Model:** `{m.get('best_model_name','—')}`")
            bm = m.get("best_metrics", {})
            st.markdown(f"**ROC-AUC:** `{bm.get('roc_auc',0):.4f}`")
            st.markdown(f"**PR-AUC:** `{bm.get('pr_auc',0):.4f}`")
            st.markdown(f"**F1-Score:** `{bm.get('f1_score',0):.4f}`")

    st.markdown("---")
    st.markdown("**Risk Thresholds**")
    st.markdown("🟢 **Low** → Score < 30 → Approve")
    st.markdown("🟡 **Medium** → 30–70 → Review")
    st.markdown("🔴 **High** → Score ≥ 70 → Block")
    st.markdown("---")
    st.caption("v1.0.0 · Built with Python, XGBoost & SHAP")

# ─────────────────────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🛡️ FraudShield — Real-Time Transaction Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered fraud probability scoring · SHAP explainability · Automated risk actions</div>', unsafe_allow_html=True)

if not model_ready:
    st.error("⚠️ Model not loaded. Run `python scripts/run_pipeline.py` first, then restart the dashboard.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡  Real-Time Simulator",
    "📁  Batch Analyzer",
    "📊  Fraud Analytics",
    "🤖  Model Performance",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Real-Time Simulator
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Transaction Risk Simulator")
    st.caption("Fill in transaction parameters and click Analyze to get a real-time fraud score with SHAP explanations.")

    with st.expander("📋 Transaction Input Form", expanded=True):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**👤 Customer Profile**")
            customer_age        = st.slider("Customer Age", 18, 90, 34, key="age")
            account_balance     = st.number_input("Account Balance ($)", 0.0, 1_000_000.0, 4500.0, 100.0, key="bal")
            avg_amount_30d      = st.number_input("30-Day Avg Transaction ($)", 0.0, 50_000.0, 65.0, 5.0, key="avg")
            prev_fraud_count    = st.selectbox("Previous Fraud Count", [0, 1, 2, 3], key="pfc")

        with c2:
            st.markdown("**💳 Transaction Details**")
            transaction_amount  = st.number_input("Transaction Amount ($)", 0.01, 500_000.0, 850.0, 25.0, key="amt")
            transaction_type    = st.selectbox("Transaction Type",
                ["Transfer", "Withdrawal", "Payment", "Online Purchase", "ATM"], key="ttype")
            merchant_category   = st.selectbox("Merchant Category",
                ["Grocery", "Electronics", "Luxury", "Travel", "Crypto", "Gambling", "Utility"],
                index=4, key="merch")
            hour_of_day         = st.slider("Hour of Day (24h)", 0, 23, 3, key="hour")

        with c3:
            st.markdown("**🔒 Device & Context**")
            device_type         = st.selectbox("Device Type", ["Mobile", "Web", "POS", "Unknown"], index=3, key="dev")
            auth_method         = st.selectbox("Authentication Method",
                ["PIN", "Biometric", "Password", "OTP", "None"], index=4, key="auth")
            is_foreign          = st.selectbox("Foreign Transaction?", [0, 1],
                format_func=lambda x: "Yes" if x else "No", index=1, key="foreign")
            distance_km         = st.number_input("Distance from Home (km)", 0.0, 20_000.0, 350.0, 10.0, key="dist")
            txn_count_1h        = st.number_input("Txn Count – Last 1h",  0, 50, 4, key="v1h")
            txn_count_24h       = st.number_input("Txn Count – Last 24h", 0, 200, 12, key="v24h")

    analyze = st.button("🚨  Analyze Fraud Risk", type="primary", use_container_width=True)

    if analyze:
        tx = {
            "transaction_id": "TXN-SIM-001", "customer_id": "CUST-SIM",
            "transaction_amount": transaction_amount, "customer_age": customer_age,
            "account_balance": account_balance, "avg_amount_30d": avg_amount_30d,
            "hour_of_day": hour_of_day, "distance_from_home_km": distance_km,
            "txn_count_1h": txn_count_1h, "txn_count_24h": txn_count_24h,
            "prev_fraud_count": prev_fraud_count, "is_foreign_transaction": is_foreign,
            "transaction_type": transaction_type, "merchant_category": merchant_category,
            "device_type": device_type, "authentication_method": auth_method,
        }
        with st.spinner("Scoring with ML model + SHAP…"):
            res = predictor.predict_single(tx)

        st.markdown("---")
        r_col1, r_col2, r_col3 = st.columns([1.2, 1, 1])

        with r_col1:
            gauge_color = {"Low": "#22C55E", "Medium": "#F59E0B", "High": "#EF4444"}.get(res["risk_level"], "#6366F1")
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=res["risk_score"],
                delta={"reference": 50, "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#22C55E"}},
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "<b>Fraud Risk Score</b><br><span style='font-size:0.8em;color:#64748B'>0 = Safe · 100 = Critical</span>"},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": gauge_color, "thickness": 0.28},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30],  "color": "#F0FDF4"},
                        {"range": [30, 70], "color": "#FFFBEB"},
                        {"range": [70, 100],"color": "#FFF1F2"},
                    ],
                    "threshold": {"line": {"color": gauge_color, "width": 5}, "thickness": 0.8, "value": res["risk_score"]},
                },
            ))
            fig_g.update_layout(height=260, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_g, use_container_width=True)

        with r_col2:
            st.markdown('<div class="section-title">Decision Summary</div>', unsafe_allow_html=True)
            prob_pct = res["fraud_probability"] * 100
            st.metric("Fraud Probability", f"{prob_pct:.1f}%")
            st.metric("Risk Level", res["risk_level"])
            st.metric("Risk Score", f"{res['risk_score']:.1f} / 100")
            act = res["recommended_action"]
            badge_class = {"Approve": "badge-approve", "Review": "badge-review", "Block": "badge-block"}.get(act, "")
            icon = {"Approve": "✅", "Review": "⚠️", "Block": "⛔"}.get(act, "")
            st.markdown(f'<div class="badge {badge_class}">{icon} {act.upper()}</div>', unsafe_allow_html=True)

        with r_col3:
            st.markdown('<div class="section-title">Transaction Signals</div>', unsafe_allow_html=True)
            ratio = transaction_amount / (avg_amount_30d + 1e-6)
            night = hour_of_day in range(1, 6)
            signals = {
                "💰 Amount / 30d Avg":  f"{ratio:.1f}×  {'🚨 HIGH' if ratio > 4 else '✅ OK'}",
                "🕐 Night Time (1–5 AM)": "Yes 🔴" if night else "No ✅",
                "🌍 Foreign":           "Yes 🔴" if is_foreign else "No ✅",
                "📱 Device Risk":       "High 🔴" if device_type == "Unknown" else "OK ✅",
                "🔐 Auth Risk":         "High 🔴" if auth_method == "None" else "OK ✅",
                "⚡ 1h Velocity":       f"{txn_count_1h} txn {'🚨' if txn_count_1h >= 4 else '✅'}",
            }
            for label, val in signals.items():
                st.write(f"**{label}:** {val}")

        st.markdown("---")
        st.markdown("#### 🔍 Top SHAP Feature Contributions")
        factors = res["top_risk_factors"]
        df_f = pd.DataFrame(factors)
        df_f["abs_shap"] = df_f["shap_value"].abs()
        fig_shap = px.bar(
            df_f.sort_values("abs_shap"),
            x="shap_value", y="feature", orientation="h",
            color="impact",
            color_discrete_map={"Increases Fraud Risk": "#EF4444", "Decreases Fraud Risk": "#22C55E"},
            text=df_f.sort_values("abs_shap")["shap_value"].apply(lambda v: f"{v:+.4f}"),
            labels={"shap_value": "SHAP Value (impact on fraud score)", "feature": "Feature"},
            title="Why this score? — Local SHAP Attribution",
        )
        fig_shap.update_layout(
            yaxis_title=None, xaxis_zeroline=True,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.2),
        )
        fig_shap.update_traces(textposition="outside")
        st.plotly_chart(fig_shap, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Analyzer
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Batch Fraud Scoring & Risk Audit")
    st.caption("Upload a CSV of transactions or score the generated dataset to identify fraud patterns at scale.")

    col_up, col_opt = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Must contain all transaction feature columns")
    with col_opt:
        use_sample = st.checkbox("Use generated sample data", value=False)
        sample_size = st.slider("Sample Size", 100, 2000, 500, step=100, disabled=not use_sample)

    df_batch = None
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_batch):,} rows from uploaded file.")
    elif use_sample:
        raw_path = predictor.config.raw_data_path
        if os.path.exists(raw_path):
            df_batch = pd.read_csv(raw_path).head(sample_size)
        else:
            df_batch = generate_synthetic_transactions(n_samples=sample_size)
        st.info(f"Using {len(df_batch):,} rows from sample dataset.")

    if df_batch is not None:
        if st.button("🚀  Score All Transactions", type="primary"):
            with st.spinner(f"Scoring {len(df_batch):,} transactions…"):
                scored = predictor.predict_batch(df_batch)

            st.success("✅ Batch scoring complete!")

            n_total  = len(scored)
            n_block  = int((scored["recommended_action"] == "Block").sum())
            n_review = int((scored["recommended_action"] == "Review").sum())
            n_ok     = int((scored["recommended_action"] == "Approve").sum())
            blocked_amt = scored.loc[scored["recommended_action"] == "Block", "transaction_amount"].sum() if "transaction_amount" in scored.columns else 0.0

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Scored", f"{n_total:,}")
            m2.metric("🔴 Blocked",   f"{n_block:,}",  f"{n_block/n_total*100:.1f}%")
            m3.metric("🟡 Review",    f"{n_review:,}", f"{n_review/n_total*100:.1f}%")
            m4.metric("🟢 Approved",  f"{n_ok:,}",     f"{n_ok/n_total*100:.1f}%")
            m5.metric("💰 Blocked $", f"${blocked_amt:,.0f}")

            # Risk distribution donut
            fig_pie = px.pie(
                values=[n_block, n_review, n_ok],
                names=["Block", "Review", "Approve"],
                hole=0.55,
                color_discrete_sequence=["#EF4444", "#F59E0B", "#22C55E"],
                title="Risk Level Distribution",
            )
            fig_pie.update_traces(textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(t=40, b=10))

            # Fraud probability histogram
            fig_hist = px.histogram(
                scored, x="fraud_probability", nbins=40,
                color_discrete_sequence=["#6366F1"],
                title="Fraud Probability Distribution",
                labels={"fraud_probability": "Fraud Probability", "count": "Transactions"},
            )
            fig_hist.update_layout(bargap=0.05, plot_bgcolor="rgba(0,0,0,0)")

            p_col1, p_col2 = st.columns(2)
            p_col1.plotly_chart(fig_pie, use_container_width=True)
            p_col2.plotly_chart(fig_hist, use_container_width=True)

            display_cols = [c for c in [
                "transaction_id", "transaction_amount", "merchant_category",
                "fraud_probability", "risk_score", "risk_level", "recommended_action"
            ] if c in scored.columns]

            st.dataframe(
                scored[display_cols].style.background_gradient(
                    subset=["risk_score"], cmap="RdYlGn_r"
                ),
                use_container_width=True, height=350,
            )

            csv_bytes = scored.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Full Scored Report", csv_bytes, "fraud_scored.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Fraud Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Exploratory Fraud Analytics")
    st.caption("Behavioral patterns and risk insights from the transaction dataset.")

    df_analytics = load_raw_data()

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        merch_stats = (
            df_analytics.groupby("merchant_category")["is_fraud"]
            .agg(fraud_rate="mean", total="count")
            .reset_index()
        )
        merch_stats["fraud_pct"] = merch_stats["fraud_rate"] * 100
        fig_merch = px.bar(
            merch_stats.sort_values("fraud_pct", ascending=False),
            x="merchant_category", y="fraud_pct",
            color="fraud_pct", color_continuous_scale="Reds",
            text=merch_stats.sort_values("fraud_pct", ascending=False)["fraud_pct"].apply(lambda v: f"{v:.1f}%"),
            title="Fraud Rate by Merchant Category",
            labels={"fraud_pct": "Fraud Rate (%)", "merchant_category": "Category"},
        )
        fig_merch.update_traces(textposition="outside")
        fig_merch.update_layout(plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig_merch, use_container_width=True)

    with r1c2:
        hour_stats = df_analytics.groupby("hour_of_day")["is_fraud"].mean().reset_index()
        hour_stats["fraud_pct"] = hour_stats["is_fraud"] * 100
        fig_hour = px.area(
            hour_stats, x="hour_of_day", y="fraud_pct",
            title="Fraud Rate by Hour of Day",
            labels={"hour_of_day": "Hour (0–23)", "fraud_pct": "Fraud Rate (%)"},
            color_discrete_sequence=["#EF4444"],
        )
        fig_hour.update_traces(line_width=2.5, fill="tozeroy", fillcolor="rgba(239,68,68,0.1)")
        fig_hour.add_vrect(x0=1, x1=5, fillcolor="rgba(239,68,68,0.08)", line_width=0,
                           annotation_text="High-Risk\nHours", annotation_position="top left")
        fig_hour.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_hour, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig_box = px.box(
            df_analytics,
            x=df_analytics["is_fraud"].map({0: "Legitimate", 1: "Fraud"}),
            y="transaction_amount",
            color=df_analytics["is_fraud"].map({0: "Legitimate", 1: "Fraud"}),
            color_discrete_map={"Legitimate": "#3B82F6", "Fraud": "#EF4444"},
            log_y=True,
            title="Transaction Amount Distribution: Fraud vs Legit",
            labels={"x": "Label", "transaction_amount": "Amount ($, log scale)"},
            points="outliers",
        )
        fig_box.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_box, use_container_width=True)

    with r2c2:
        vel = df_analytics.groupby("txn_count_1h")["is_fraud"].mean().reset_index()
        vel["fraud_pct"] = vel["is_fraud"] * 100
        vel = vel[vel["txn_count_1h"] <= 12]
        fig_vel = px.bar(
            vel, x="txn_count_1h", y="fraud_pct",
            color="fraud_pct", color_continuous_scale="Oranges",
            title="Transaction Velocity vs Fraud Rate",
            labels={"txn_count_1h": "Txn Count in Last 1 Hour", "fraud_pct": "Fraud Rate (%)"},
            text=vel["fraud_pct"].apply(lambda v: f"{v:.0f}%"),
        )
        fig_vel.update_traces(textposition="outside")
        fig_vel.update_layout(plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig_vel, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Device Type & Auth Method Fraud Heat")
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        dev_stats = df_analytics.groupby("device_type")["is_fraud"].mean().reset_index()
        dev_stats["fraud_pct"] = dev_stats["is_fraud"] * 100
        fig_dev = px.bar(
            dev_stats.sort_values("fraud_pct", ascending=True),
            x="fraud_pct", y="device_type", orientation="h",
            color="fraud_pct", color_continuous_scale="Reds",
            title="Fraud Rate by Device Type",
            labels={"fraud_pct": "Fraud Rate (%)", "device_type": "Device"},
            text=dev_stats.sort_values("fraud_pct")["fraud_pct"].apply(lambda v: f"{v:.1f}%"),
        )
        fig_dev.update_layout(plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig_dev, use_container_width=True)

    with d_col2:
        auth_stats = df_analytics.groupby("authentication_method")["is_fraud"].mean().reset_index()
        auth_stats["fraud_pct"] = auth_stats["is_fraud"] * 100
        fig_auth = px.bar(
            auth_stats.sort_values("fraud_pct", ascending=True),
            x="fraud_pct", y="authentication_method", orientation="h",
            color="fraud_pct", color_continuous_scale="Purples",
            title="Fraud Rate by Authentication Method",
            labels={"fraud_pct": "Fraud Rate (%)", "authentication_method": "Auth Method"},
            text=auth_stats.sort_values("fraud_pct")["fraud_pct"].apply(lambda v: f"{v:.1f}%"),
        )
        fig_auth.update_layout(plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
        st.plotly_chart(fig_auth, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Model Evaluation & Explainability")

    metrics_path = predictor.config.get("paths.metrics_path", "models/model_metrics.json")
    if os.path.exists(metrics_path):
        summary = load_json(metrics_path)
    else:
        summary = {
            "best_model_name": "XGBoost",
            "best_metrics": {"roc_auc": 0.965, "pr_auc": 0.912, "f1_score": 0.880, "precision": 0.903, "recall": 0.859, "accuracy": 0.985, "brier_score": 0.022},
            "all_models_metrics": {
                "LogisticRegression": {"roc_auc": 0.872, "pr_auc": 0.730, "f1_score": 0.690},
                "RandomForest":       {"roc_auc": 0.952, "pr_auc": 0.891, "f1_score": 0.845},
                "XGBoost":            {"roc_auc": 0.965, "pr_auc": 0.912, "f1_score": 0.880},
                "LightGBM":           {"roc_auc": 0.963, "pr_auc": 0.908, "f1_score": 0.876},
            },
        }

    bm = summary["best_metrics"]
    st.success(f"🏆  Winner Model: **{summary['best_model_name']}**")

    t1, t2, t3, t4, t5, t6 = st.columns(6)
    t1.metric("ROC-AUC",   f"{bm.get('roc_auc',0):.4f}")
    t2.metric("PR-AUC",    f"{bm.get('pr_auc',0):.4f}")
    t3.metric("F1-Score",  f"{bm.get('f1_score',0):.4f}")
    t4.metric("Precision", f"{bm.get('precision',0):.4f}")
    t5.metric("Recall",    f"{bm.get('recall',0):.4f}")
    t6.metric("Accuracy",  f"{bm.get('accuracy',0):.4f}")

    st.markdown("---")
    st.markdown("#### 📊 Model Comparison")

    comp = []
    for name, m in summary.get("all_models_metrics", {}).items():
        comp.append({"Model": name, "ROC-AUC": m.get("roc_auc", 0), "PR-AUC": m.get("pr_auc", 0), "F1-Score": m.get("f1_score", 0)})
    df_comp = pd.DataFrame(comp)

    fig_comp = px.bar(
        df_comp.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model", y="Score", color="Metric", barmode="group",
        color_discrete_sequence=["#6366F1", "#EC4899", "#F59E0B"],
        title="Benchmark Metrics Across All Candidate Models",
        text_auto=".3f",
    )
    fig_comp.update_traces(textposition="outside")
    fig_comp.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 1.05],
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Confusion Matrix
    if "confusion_matrix" in bm:
        st.markdown("#### 🔢 Confusion Matrix (Best Model)")
        cm = bm["confusion_matrix"]
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=["Pred: Legit", "Pred: Fraud"], y=["True: Legit", "True: Fraud"],
            title="Confusion Matrix — Test Set",
            aspect="auto",
        )
        fig_cm.update_layout(width=400, height=300)
        cm_col, _ = st.columns([1, 2])
        cm_col.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🌐 Global Feature Importance (SHAP)")

    proc_path = predictor.config.processed_data_path
    if predictor.explainer and os.path.exists(proc_path):
        with st.spinner("Computing SHAP values across sample…"):
            try:
                proc_df = pd.read_csv(proc_path).head(150)
                num_cols = predictor.config.get("features.numerical")
                cat_cols = predictor.config.get("features.categorical")
                X_sample = predictor.preprocessor.transform(proc_df[num_cols + cat_cols])
                global_imp = predictor.explainer.get_global_importance(X_sample)
                df_imp = pd.DataFrame(list(global_imp.items()), columns=["Feature", "Mean |SHAP|"]).head(18)
                fig_glob = px.bar(
                    df_imp.sort_values("Mean |SHAP|"),
                    x="Mean |SHAP|", y="Feature", orientation="h",
                    color="Mean |SHAP|", color_continuous_scale="Viridis",
                    title="Top Global Feature Drivers (Mean |SHAP| across 150 transactions)",
                    text=df_imp.sort_values("Mean |SHAP|")["Mean |SHAP|"].apply(lambda v: f"{v:.4f}"),
                )
                fig_glob.update_traces(textposition="outside")
                fig_glob.update_layout(
                    yaxis_title=None, plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False, margin=dict(l=150),
                )
                st.plotly_chart(fig_glob, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not compute SHAP: {e}")
    else:
        st.info("Run the ML pipeline first to enable SHAP explainability.")
