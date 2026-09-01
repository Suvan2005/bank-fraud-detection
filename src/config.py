import os
import yaml
from typing import Any, Dict

class Config:
    """Singleton Config class to load config.yaml across modules."""
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: str = None):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: str = None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "config.yaml")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """Fetch nested configuration values using dot notation (e.g. 'paths.model_dir')"""
        keys = key_path.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def raw_data_path(self) -> str:
        return self.get("paths.raw_data")

    @property
    def processed_data_path(self) -> str:
        return self.get("paths.processed_data")

    @property
    def model_dir(self) -> str:
        return self.get("paths.model_dir")

    @property
    def best_model_path(self) -> str:
        return self.get("paths.best_model_path")

    @property
    def preprocessor_path(self) -> str:
        return self.get("paths.preprocessor_path")

    @property
    def explainer_path(self) -> str:
        return self.get("paths.explainer_path")

    @property
    def feature_names_path(self) -> str:
        return self.get("paths.feature_names_path")
