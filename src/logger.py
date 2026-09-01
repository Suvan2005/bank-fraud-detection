import logging
import logging.config
import os
import yaml

def get_logger(name: str = "fraud_detection") -> logging.Logger:
    """Returns a configured logger instance."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "logging.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                log_config = yaml.safe_load(f)
            logging.config.dictConfig(log_config)
        except Exception:
            logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)
    
    return logging.getLogger(name)
