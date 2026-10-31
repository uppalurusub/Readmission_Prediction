"""
Implementation layer for Patient Readmission Risk Prediction.

Responsibilities:
- Load the trained classifier and fitted preprocessing transformer.
- Validate raw API features.
- Reproduce feature engineering performed before training.
- Transform input with the saved preprocessor and predict class-1 probability.
- Return prediction metadata and log operational events without patient data.

Expected artifacts:
    artifacts/best_model.pkl
    artifacts/preprocessor.pkl

The current training script saves a classifier trained on pre-transformed features,
so the preprocessor must be applied before calling the classifier.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config.settings import settings
logger = settings.get_logger()

ARTIFACT_DIR = settings.artifact_dir

MODEL_PATH = ARTIFACT_DIR / "best_model.pkl"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.pkl"
MODEL_NAME = "readmission-classifier"



DEFAULT_THRESHOLD = float(os.getenv("READMISSION_DECISION_THRESHOLD", "0.50"))

EXPECTED_FEATURES = [
    "age",
    "length_of_stay",
    "previous_admissions",
    "diagnosis_count",
    "comorbidity_count",
    "discharge_destination",
]
NUMERIC_FEATURES = [
    "age",
    "length_of_stay",
    "previous_admissions",
    "diagnosis_count",
    "comorbidity_count",
]


class ReadmissionImplementation:
    """Loads the model artifacts and generates readmission predictions."""

    def __init__(self) -> None:
        self.model_path = MODEL_PATH
        self.preprocessor_path = PREPROCESSOR_PATH
        self.model: Any | None = None
        self.preprocessor: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.model_name = MODEL_NAME
        self.model_version: str | None = os.getenv("READMISSION_MODEL_VERSION")
        self.decision_threshold = DEFAULT_THRESHOLD

        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError(
                "READMISSION_DECISION_THRESHOLD must be between 0 and 1."
            )

        self.load_model()

    def load_model(self) -> None:
        """Load the classifier and its matching fitted preprocessor."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Readmission model artifact not found: {self.model_path}"
            )
        if not self.preprocessor_path.is_file():
            raise FileNotFoundError(
                f"Readmission preprocessor artifact not found: "
                f"{self.preprocessor_path}"
            )

        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)

            if not callable(getattr(self.model, "predict_proba", None)):
                raise TypeError(
                    "Loaded model does not support predict_proba()."
                )
            if not callable(getattr(self.preprocessor, "transform", None)):
                raise TypeError(
                    "Loaded preprocessor does not support transform()."
                )

            logger.success(
                "Readmission artifacts loaded | model_path={} | "
                "preprocessor_path={} | model_name={} | threshold={}",
                self.model_path,
                self.preprocessor_path,
                self.model_name,
                self.decision_threshold,
            )
        except Exception:
            logger.exception("Unable to load readmission model artifacts")
            raise

    @staticmethod
    def _validate_features(features: dict[str, Any]) -> pd.DataFrame:
        """Validate the original six API fields and preserve their order."""
        missing = [name for name in EXPECTED_FEATURES if name not in features]
        unexpected = [name for name in features if name not in EXPECTED_FEATURES]

        if missing:
            raise ValueError(f"Missing required features: {missing}")
        if unexpected:
            raise ValueError(f"Unexpected features supplied: {unexpected}")

        frame = pd.DataFrame(
            [{name: features[name] for name in EXPECTED_FEATURES}],
            columns=EXPECTED_FEATURES,
        )

        for column in NUMERIC_FEATURES:
            value = frame.at[0, column]
            if pd.isna(value):
                raise ValueError(f"Feature '{column}' cannot be missing.")
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise ValueError(f"Feature '{column}' must be numeric.")
            if not np.isfinite(float(value)):
                raise ValueError(f"Feature '{column}' must be finite.")

        age = float(frame.at[0, "age"])
        if not 0 <= age <= 120:
            raise ValueError("Feature 'age' must be between 0 and 120.")

        for column in (
            "length_of_stay",
            "previous_admissions",
            "diagnosis_count",
            "comorbidity_count",
        ):
            if float(frame.at[0, column]) < 0:
                raise ValueError(f"Feature '{column}' cannot be negative.")

        destination = frame.at[0, "discharge_destination"]
        if not isinstance(destination, str) or destination not in {
            "home",
            "facility",
            "other",
        }:
            raise ValueError(
                "discharge_destination must be 'home', 'facility', or 'other'."
            )

        return frame

    @staticmethod
    def _apply_feature_engineering(frame: pd.DataFrame) -> pd.DataFrame:
        """Mirror DataPreprocessor.feature_engineering() from training."""
        engineered = frame.copy()

        engineered["age_group"] = pd.cut(
            engineered["age"],
            bins=[0, 18, 40, 60, 120],
            labels=["Young", "Adult", "Middle", "Senior"],
        )
        engineered["high_risk"] = (
            (engineered["age"] >= 60)
            & (engineered["comorbidity_count"] >= 3)
        ).astype(int)

        return engineered

    def _transform_input(self, raw_frame: pd.DataFrame):
        """Apply the saved preprocessor using its fitted input column order."""
        if self.preprocessor is None:
            raise RuntimeError("Readmission preprocessor has not been loaded.")

        engineered = self._apply_feature_engineering(raw_frame)
        fitted_columns = getattr(self.preprocessor, "feature_names_in_", None)

        if fitted_columns is None:
            raise RuntimeError(
                "The saved preprocessor has no feature_names_in_. "
                "Verify that it was fitted on a pandas DataFrame and retrain/save "
                "the preprocessor if necessary."
            )

        fitted_columns = list(fitted_columns)
        missing = [column for column in fitted_columns if column not in engineered.columns]
        if missing:
            raise ValueError(
                "Saved preprocessor expects columns that are not present in the "
                f"API input or feature engineering: {missing}. Align the API "
                "features and training feature set, then retrain the artifacts."
            )

        # The preprocessor may have been fitted on additional columns. We select
        # only its recorded columns and preserve the exact fitted order.
        model_frame = engineered.loc[:, fitted_columns]
        return self.preprocessor.transform(model_frame)

    def predict(
        self,
        features: dict[str, Any],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """Return class-1 probability and the threshold-based prediction."""
        started_at = time.perf_counter()

        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Readmission model artifacts have not been loaded.")

        try:
            raw_frame = self._validate_features(features)
            transformed_input = self._transform_input(raw_frame)

            probabilities = np.asarray(
                self.model.predict_proba(transformed_input),
                dtype=float,
            )

            classes = getattr(self.model, "classes_", None)
            if classes is None:
                classifier = getattr(self.model, "named_steps", {}).get("classifier")
                classes = getattr(classifier, "classes_", None)

            if classes is None or 1 not in list(classes):
                raise ValueError(
                    "Loaded model does not expose the expected positive class 1."
                )
            if probabilities.ndim != 2 or probabilities.shape[0] != 1:
                raise ValueError(
                    "Model returned an unexpected predict_proba output shape."
                )

            positive_index = list(classes).index(1)
            probability = float(probabilities[0, positive_index])

            if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("Model returned an invalid readmission probability.")

            prediction = int(probability >= self.decision_threshold)
            elapsed_seconds = round(time.perf_counter() - started_at, 4)

            logger.bind(
                request_id=request_id,
                elapsed_seconds=elapsed_seconds,
                predicted_readmission=prediction,
                model_name=self.model_name,
            ).info("Readmission inference completed")

            return {
                "readmission_probability": probability,
                "predicted_readmission": prediction,
                "decision_threshold": self.decision_threshold,
                "model_name": self.model_name,
                #"model_version": self.model_version,
            }

        except Exception:
            logger.bind(request_id=request_id).exception(
                "Readmission inference failed"
            )
            raise