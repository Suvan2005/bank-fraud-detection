import os
import json
import joblib
from typing import Any
from src.logger import get_logger

logger = get_logger("src.utils")

def ensure_dir(dir_path: str):
    """Ensures a directory exists."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def save_joblib(obj: Any, path: str):
    """Saves a python object using joblib."""
    ensure_dir(os.path.dirname(path))
    joblib.dump(obj, path)
    logger.info(f"Successfully saved artifact to: {path}")

def load_joblib(path: str) -> Any:
    """Loads a python object using joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact path does not exist: {path}")
    obj = joblib.load(path)
    logger.info(f"Successfully loaded artifact from: {path}")
    return obj

def save_json(data: dict, path: str):
    """Saves dictionary to JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Successfully saved JSON to: {path}")

def load_json(path: str) -> dict:
    """Loads dictionary from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON path does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
