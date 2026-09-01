import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import Tuple, List
from src.logger import get_logger

logger = get_logger("src.data.preprocessor")

class FraudDataPreprocessor:
    """Preprocessor pipeline for scaling numeric features and encoding categorical features."""

    def __init__(self, numerical_cols: List[str], categorical_cols: List[str]):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.pipeline: ColumnTransformer = None
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame) -> "FraudDataPreprocessor":
        """Fits numerical scaling and categorical one-hot encoder on training data."""
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, self.numerical_cols),
                ('cat', cat_pipeline, self.categorical_cols)
            ]
        )

        self.pipeline.fit(X)

        # Get feature names after one-hot encoding
        cat_encoder = self.pipeline.named_transformers_['cat'].named_steps['encoder']
        encoded_cat_cols = list(cat_encoder.get_feature_names_out(self.categorical_cols))
        self.feature_names = self.numerical_cols + encoded_cat_cols

        logger.info(f"Preprocessor fitted. Total engineered features output count: {len(self.feature_names)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw input DataFrame into scaled/encoded feature matrix."""
        if self.pipeline is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform.")
        
        X_trans = self.pipeline.transform(X)
        return pd.DataFrame(X_trans, columns=self.feature_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform input DataFrame."""
        return self.fit(X).transform(X)
