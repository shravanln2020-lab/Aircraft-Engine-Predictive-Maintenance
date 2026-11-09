"""
RUL Predictor module for Aircraft Engine Predictive Maintenance (AEPM).

Provides functions to load trained scikit-learn models and perform RUL predictions
on engineered telemetry feature matrices with feature ordering enforcement and sanity checks.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src import config
from src.utils import setup_logger

logger = setup_logger(__name__)

# Default model path pointing to production model (random_forest_v2.joblib)
DEFAULT_MODEL_PATH: Path = config.MODELS_DIR / "random_forest_v2.joblib"


class RULPredictor:
    """Class wrapper for loading trained RUL regression model and generating predictions."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize RULPredictor and load model object from joblib file.

        Args:
            model_path: Path to trained .joblib model. Defaults to DEFAULT_MODEL_PATH.
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model = self._load_model(self.model_path)

    def _load_model(self, path: Path) -> RandomForestRegressor:
        """Load joblib model file safely.

        Args:
            path: Target model file path.

        Returns:
            RandomForestRegressor: Loaded model instance.

        Raises:
            FileNotFoundError: If model file does not exist.
        """
        if not path.is_file():
            # Fallback to random_forest_final.joblib if v2 not present
            fallback_path = config.MODELS_DIR / "random_forest_final.joblib"
            if fallback_path.is_file():
                logger.warning("Primary model '%s' not found. Loading fallback model '%s'.", path, fallback_path)
                path = fallback_path
            else:
                logger.error("Model file not found at '%s'.", path)
                raise FileNotFoundError(f"Trained model file not found at '{path}'.")

        logger.info("Loading trained RUL model from: %s", path)
        model = joblib.load(path)
        logger.info("RUL model loaded successfully (%s).", type(model).__name__)
        return model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate RUL predictions for feature matrix X with feature ordering enforcement.

        Args:
            X: Input feature DataFrame.

        Returns:
            np.ndarray: Predicted RUL values (finite, clipped to non-negative floats).

        Raises:
            ValueError: If X is empty or missing required model features.
        """
        if X is None or X.empty:
            raise ValueError("Feature matrix X is empty or None.")

        # Enforce exact feature selection and ordering if model stores feature_names_in_
        if hasattr(self.model, "feature_names_in_"):
            expected_feats = list(self.model.feature_names_in_)
            missing_feats = [f for f in expected_feats if f not in X.columns]
            if missing_feats:
                raise ValueError(
                    f"Feature matrix is missing {len(missing_feats)} required model features: {missing_feats[:5]}"
                )
            X_eval = X[expected_feats]
        else:
            # Fallback filtering of index and target columns
            feature_cols = [c for c in X.columns if c not in set(config.INDEX_COLS + [config.RUL_COL])]
            X_eval = X[feature_cols]

        raw_preds = self.model.predict(X_eval)

        # Sanity check: Ensure predictions are finite numbers
        if not np.isfinite(raw_preds).all():
            raise ValueError("Model prediction output contains infinite or NaN values.")

        # RUL cannot be negative in physical operational domain
        clipped_preds = np.clip(raw_preds, 0.0, None)

        logger.info(
            "Generated RUL predictions for %d samples (Range: %.2f to %.2f cycles).",
            len(clipped_preds),
            float(clipped_preds.min()),
            float(clipped_preds.max()),
        )
        return clipped_preds


def load_model(
    model_path: Optional[Union[str, Path]] = None
) -> RandomForestRegressor:
    """Standalone function to load a joblib model file.

    Args:
        model_path: Target model file path. Defaults to DEFAULT_MODEL_PATH.

    Returns:
        RandomForestRegressor: Loaded model object.
    """
    predictor = RULPredictor(model_path=model_path)
    return predictor.model


def predict_rul(
    model: RandomForestRegressor, X: pd.DataFrame
) -> np.ndarray:
    """Predict RUL using a loaded scikit-learn model object with feature ordering enforcement.

    Args:
        model: Loaded RandomForestRegressor instance.
        X: Feature matrix DataFrame.

    Returns:
        np.ndarray: Non-negative RUL predictions array.
    """
    if hasattr(model, "feature_names_in_"):
        expected_feats = list(model.feature_names_in_)
        missing_feats = [f for f in expected_feats if f not in X.columns]
        if missing_feats:
            raise ValueError(
                f"Feature matrix is missing {len(missing_feats)} required model features: {missing_feats[:5]}"
            )
        X_eval = X[expected_feats]
    else:
        feature_cols = [c for c in X.columns if c not in set(config.INDEX_COLS + [config.RUL_COL])]
        X_eval = X[feature_cols]

    preds = model.predict(X_eval)
    if not np.isfinite(preds).all():
        raise ValueError("Model prediction output contains infinite or NaN values.")
    return np.clip(preds, 0.0, None)

