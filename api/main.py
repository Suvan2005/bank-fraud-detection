import os
import sys
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchTransactionInput,
    BatchPredictionResponse,
    ModelInfoResponse
)
from src.config import Config
from src.logger import get_logger
from src.inference.predictor import FraudPredictor
from src.utils import load_json

logger = get_logger("api.main")
config = Config()
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    try:
        predictor = FraudPredictor(config)
        logger.info("FastAPI startup: FraudPredictor loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load predictor on startup: {e}")
    yield
    logger.info("FastAPI shutdown.")

app = FastAPI(
    title="Real-Time Banking Fraud Detection API",
    description="Production-ready REST API for evaluating real-time transaction fraud probability, risk scores, and recommended security actions.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "Banking Transaction Fraud Detection API",
        "status": "Online",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health"])
def health_check():
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts not initialized. Please run training pipeline."
        )
    return {"status": "healthy", "model_ready": True}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_transaction(input_data: TransactionInput):
    """Evaluate fraud risk for a single real-time banking transaction."""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor engine is not available."
        )
    
    try:
        raw_dict = input_data.model_dump()
        result = predictor.predict_single(raw_dict)
        return result
    except Exception as e:
        logger.error(f"Error during single transaction prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch_transactions(batch_data: BatchTransactionInput):
    """Evaluate fraud risk for an array of banking transactions."""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictor engine is not available."
        )

    try:
        dict_list = [tx.model_dump() for tx in batch_data.transactions]
        df = pd.DataFrame(dict_list)
        
        predictions_res = []
        flagged_count = 0
        blocked_count = 0

        for tx in dict_list:
            res = predictor.predict_single(tx)
            predictions_res.append(res)
            if res["is_fraud"] == 1:
                flagged_count += 1
            if res["recommended_action"] == "Block":
                blocked_count += 1

        return {
            "total_processed": len(dict_list),
            "flagged_fraud_count": flagged_count,
            "blocked_count": blocked_count,
            "predictions": predictions_res
        }
    except Exception as e:
        logger.error(f"Error during batch transaction prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model Info"])
def get_model_info():
    """Returns model metadata, performance metrics, and feature configuration."""
    metrics_path = config.get("paths.metrics_path")
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Model metrics metadata not found.")

    metrics_data = load_json(metrics_path)
    return {
        "project_name": config.get("project.name", "Banking Fraud Detection"),
        "version": config.get("project.version", "1.0.0"),
        "best_model": metrics_data.get("best_model_name", "Unknown"),
        "metrics": metrics_data.get("best_metrics", {}),
        "feature_names": predictor.feature_names if predictor else []
    }
