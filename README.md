# 🛡️ FraudShield — Real-Time Banking Transaction Fraud Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" />
  <img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/XGBoost-2.0%2B-EB5424?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/License-MIT-green.style=for-the-badge" alt="License" />
</p>

An enterprise-grade, real-time Machine Learning system designed to inspect banking transactions, predict fraud probabilities ($P(\text{fraud})$), compute calibrated **Risk Scores ($0 - 100$)**, assign automated triage actions (**`Approve`**, **`Review`**, **`Block`**), and generate explainable **SHAP (SHapley Additive exPlanations)** feature attributions.

---

## 📸 Website & Dashboard Preview

> *Place your application screenshots in the `docs/screenshots/` folder to display them below.*

### 1. Real-Time Transaction Risk Simulator
> Evaluate single transactions on the fly, visualize the calibrated risk gauge, and inspect instant SHAP local feature attribution explanations.
<p align="center">
  <img src="docs/screenshots/realtime_simulator.png" alt="Real-Time Transaction Simulator" width="90%" />
</p>

---

### 2. Batch Transaction CSV Analyzer
> Upload transaction batches in CSV format, perform automated high-throughput bulk inference, analyze risk score distributions, and export tagged reports.
<p align="center">
  <img src="docs/screenshots/batch_analyzer.png" alt="Batch CSV Analyzer" width="90%" />
</p>

---

### 3. Fraud Analytics & Risk Intelligence Hub
> High-level executive KPI metrics, fraud patterns across merchant categories, transaction velocity spikes, and temporal fraud heatmaps.
<p align="center">
  <img src="docs/screenshots/fraud_analytics.png" alt="Fraud Analytics Dashboard" width="90%" />
</p>

---

### 4. Model Performance Benchmark & Global Explainability
> Compare ROC-AUC, PR-AUC, and F1-Scores across Logistic Regression, Random Forest, XGBoost, and LightGBM with global SHAP summary feature importance.
<p align="center">
  <img src="docs/screenshots/model_performance.png" alt="Model Performance and SHAP Explainability" width="90%" />
</p>

---

### 5. Interactive FastAPI REST Documentation (`/docs`)
> OpenAPI Swagger interface supporting sub-millisecond single and bulk JSON transaction payload scoring.
<p align="center">
  <img src="docs/screenshots/fastapi_docs.png" alt="FastAPI Swagger Documentation" width="90%" />
</p>

---

## 🌟 Key Capabilities & Highlights

- **Synthetic Realistic Banking Generator**: Simulates 12,000+ realistic transaction streams complete with velocity spikes, midnight transaction anomalies (3 AM spikes), high amount-to-balance ratios, foreign locations, and high-risk merchant categories.
- **Multi-Model Benchmark Suite**: Rigorously evaluates Logistic Regression, Random Forest, XGBoost, and LightGBM using stratifying folds and `scale_pos_weight` imbalance handling.
- **Explainable AI (XAI)**: Native SHAP integration delivering local transaction force/waterfall plots and global feature rankings to build trust with risk officers.
- **Production FastAPI Backend**: Async REST API serving `/predict` (single), `/predict/batch`, `/health`, and `/model/info` with strict Pydantic validation.
- **Interactive Streamlit Portal**: 4-in-1 UI featuring Risk Simulator, Batch Scoring & Export, Analytics Hub, and Model Diagnostic Suite.
- **MLflow Tracking**: Complete lifecycle tracking for parameters, ROC-AUC, PR-AUC, F1-Scores, confusion matrices, and model serialization.
- **DevOps Ready**: Automated `pytest` unit test suites, Docker Compose container orchestration, and GitHub Actions CI pipelines.

---

## 📐 System Architecture

```
                               ┌───────────────────────────────────┐
                               │   Synthetic Transaction Stream    │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │ Feature Engineering & Risk Ratios │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │  Scikit-Learn Imputer & Scaler    │
                               └─────────────────┬─────────────────┘
                                                 │
                                                 ▼
                               ┌───────────────────────────────────┐
                               │    XGBoost / LightGBM Engine      │
                               └─────────────────┬─────────────────┘
                                                 │
                         ┌───────────────────────┴───────────────────────┐
                         │                                               │
                         ▼                                               ▼
        ┌───────────────────────────────────┐           ┌───────────────────────────────────┐
        │    FastAPI Backend (Port 8000)    │           │  Streamlit Dashboard (Port 8501)  │
        │  • Single Prediction: /predict    │           │  • Live Risk Simulator & Gauge    │
        │  • Bulk Inference: /predict/batch │           │  • CSV Batch Analyzer & Export    │
        │  • Swagger Docs: /docs            │           │  • Fraud Analytics & SHAP Suite   │
        └───────────────────────────────────┘           └───────────────────────────────────┘
```

---

## 🎯 Risk Classification Matrix & Decision Rules

Transactions are mapped to calibrated Risk Scores ($0 - 100$) and automated mitigation workflows:

| Fraud Probability ($P$) | Risk Score (0-100) | Risk Tier | Recommended Action | Operational Action |
|:---:|:---:|:---:|:---:|:---|
| **$P < 0.30$** | `0.0 - 29.9` | 🟢 **`Low`** | **`Approve`** | Cleared instantly without customer friction |
| **$0.30 \le P < 0.70$** | `30.0 - 69.9` | 🟡 **`Medium`** | **`Review`** | Step-up 2FA prompt / routed to Fraud Ops queue |
| **$P \ge 0.70$** | `70.0 - 100.0` | 🔴 **`High`** | **`Block`** | Transaction halted; alert sent to account holder |

---

## 📂 Repository Structure

```
bank-fraud-detection/
├── .github/workflows/ci.yml       # CI/CD pipeline (Lint, Test, Docker Build)
├── config/
│   ├── config.yaml                # Model parameters, thresholds, feature mappings
│   └── logging.yaml               # Structured logging configuration
├── data/
│   ├── raw/                       # Generated raw transaction logs (transactions.csv)
│   └── processed/                 # Engineered feature matrix (features.csv)
├── docs/
│   └── screenshots/               # Application UI screenshots & preview assets
├── models/                        # Serialized models (.joblib) and metrics (JSON)
├── src/
│   ├── data/                      # Synthetic data generator and preprocessor
│   ├── features/                  # Velocity, ratio, and time-based feature engineering
│   ├── models/                    # Model training, evaluation, and SHAP explainers
│   └── inference/                 # Real-time predictor engine
├── api/
│   ├── schemas.py                 # Pydantic request/response validation schemas
│   └── main.py                    # FastAPI application & REST endpoints
├── dashboard/
│   └── app.py                     # Streamlit frontend dashboard
├── scripts/
│   └── run_pipeline.py            # End-to-end ML data & training pipeline
├── tests/                         # Unit tests for data, model, and API endpoints
├── Dockerfile                     # Multi-stage production Dockerfile
├── docker-compose.yml             # Orchestration for FastAPI, Streamlit, and MLflow
├── requirements.txt               # Project dependencies
├── setup.py                       # Local package installer
└── README.md                      # Project documentation
```

---

## ⚡ Quick Start Guide (VS Code / Terminal)

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/Suvan2005/bank-fraud-detection.git
cd bank-fraud-detection

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# Install dependencies and local package
pip install -r requirements.txt
pip install -e .
```

---

### 2. Train Models & Build Artifacts

Run the end-to-end automated pipeline to generate synthetic data, compute feature matrices, benchmark models, and save `.joblib` model artifacts:

```bash
python scripts/run_pipeline.py
```

---

### 3. Launch the Streamlit Dashboard (Frontend UI)

Start the interactive dashboard:

```bash
streamlit run dashboard/app.py
```
👉 Access the web interface at: **`http://localhost:8501`**

---

### 4. Launch the FastAPI Backend (REST API)

Start the production REST API server:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
👉 Interactive Swagger documentation available at: **`http://localhost:8000/docs`**

---

## 🧪 Running Unit Tests

Run test suites covering data generation, feature pipelines, model predictions, and API endpoints:

```bash
pytest tests/ -v
```

---

## 🐳 Docker Deployment

Run the complete multi-container stack (FastAPI Backend + Streamlit Dashboard + MLflow Server) using Docker Compose:

```bash
docker-compose up --build
```

| Service | Local URL |
|---|---|
| **Streamlit Dashboard** | `http://localhost:8501` |
| **FastAPI REST Service** | `http://localhost:8000/docs` |
| **MLflow Tracking Hub** | `http://localhost:5000` |

---

## 🛠️ API Reference

### Key Endpoints

| HTTP Method | Route | Description |
|:---:|:---|:---|
| `GET` | `/` | API version & health overview |
| `GET` | `/health` | Model status and readiness probe |
| `POST` | `/predict` | Single transaction fraud evaluation with SHAP explanation |
| `POST` | `/predict/batch` | Bulk transaction evaluation with aggregate analytics |
| `GET` | `/model/info` | Active model metadata, metrics, and feature list |

### Example Single Prediction (`POST /predict`)

**Request Payload:**
```json
{
  "transaction_id": "TXN_99812",
  "customer_id": "CUST_00412",
  "amount": 2850.00,
  "account_balance": 1200.00,
  "transaction_type": "ONLINE",
  "merchant_category": "electronics",
  "location": "FOREIGN",
  "device_type": "UNRECOGNIZED_DEVICE",
  "hour_of_day": 3,
  "day_of_week": 6,
  "tx_count_last_1h": 6,
  "tx_count_last_24h": 14,
  "avg_amount_last_30d": 120.00,
  "failed_attempts_last_24h": 3
}
```

**Response Payload:**
```json
{
  "transaction_id": "TXN_99812",
  "fraud_probability": 0.8924,
  "risk_score": 89.24,
  "risk_level": "High",
  "recommended_action": "Block",
  "top_risk_factors": [
    { "feature": "amount_to_balance_ratio", "shap_value": 0.3412 },
    { "feature": "tx_count_last_1h", "shap_value": 0.2845 },
    { "feature": "location_FOREIGN", "shap_value": 0.2103 }
  ],
  "latency_ms": 12.4
}
```

---

## 💻 Tech Stack & Tools

- **Core**: Python 3.10+
- **Machine Learning**: Scikit-Learn, XGBoost, LightGBM
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **APIs & Validation**: FastAPI, Uvicorn, Pydantic
- **Dashboard & Visuals**: Streamlit, Plotly, Seaborn, Matplotlib
- **MLOps & Tracking**: MLflow, Joblib
- **Testing & Containerization**: Pytest, Docker, Docker Compose

---

## 📄 License
Distributed under the **MIT License**. See `LICENSE` for details.
