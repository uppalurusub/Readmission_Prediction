"""
Service layer for Patient Readmission Risk Prediction.

Delegates preprocessing and model inference to ReadmissionImplementation,
then applies the configured operational risk categories.
Patient feature values are never written to logs.
"""

from __future__ import annotations

import os
import time
from typing import Any
import mlflow
from datetime import datetime
from app.config.settings import settings
logger = settings.get_logger()

from app.implementations.readmission_impl import ReadmissionImplementation
MODEL_NAME = "readmission-classifier"




# Operational policy thresholds only; these are not clinically validated.
MODERATE_RISK_THRESHOLD = float(
    os.getenv("READMISSION_MODERATE_RISK_THRESHOLD", "0.30")
)
HIGH_RISK_THRESHOLD = float(
    os.getenv("READMISSION_HIGH_RISK_THRESHOLD", "0.60")
)


class ReadmissionService:
    """Orchestrates prediction and maps probability to a risk category."""

    def __init__(self, implementation: ReadmissionImplementation | None = None) -> None:
        if MODERATE_RISK_THRESHOLD >= HIGH_RISK_THRESHOLD:
            raise ValueError(
                "READMISSION_MODERATE_RISK_THRESHOLD must be lower than "
                "READMISSION_HIGH_RISK_THRESHOLD."
            )

        self.implementation = implementation or ReadmissionImplementation()

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability >= HIGH_RISK_THRESHOLD:
            return "HIGH"
        if probability >= MODERATE_RISK_THRESHOLD:
            return "MODERATE"
        return "LOW"

    def predict(
        self,
        features: dict[str, Any],
        request_id: str,
    ) -> dict[str, Any]:
        """Return a prediction result using the implementation layer."""

        mlflow.set_tracking_uri(
            "http://localhost:5000"
        )

        mlflow.set_experiment(
            "Readmission_Prediction"
        )

        started = time.perf_counter()

        with mlflow.start_run(
            run_name=
            f"predict_readmission_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):

            result = self.implementation.predict(
                features=features,
                request_id=request_id,
            )

            probability = float(result["readmission_probability"])
            result["risk_level"] = self._risk_level(probability)

            elapsed_seconds = round(time.perf_counter() - started, 4)
            logger.bind(
                request_id=request_id,
                model_name=result.get("model_name", MODEL_NAME),
                model_version=result.get("model_version"),
                elapsed_seconds=elapsed_seconds,
                predicted_readmission=result.get("predicted_readmission"),
                risk_level=result["risk_level"],
            ).info("Readmission service completed")

            mlflow.log_param(
                "input_text",
                features
            )

            mlflow.log_param(
                "prediction_result",
                result
            )

            

            mlflow.set_tag(
                "source",
                "FastAPI"
            )

            mlflow.set_tag(
                "endpoint",
                "/predict"
            )

        return result