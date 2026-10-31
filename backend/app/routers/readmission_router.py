
"""
FastAPI router for Patient Readmission Risk Prediction.

Responsibilities:
- Validate incoming requests using Pydantic schemas.
- Delegate prediction to the service layer.
- Log requests, outcomes, and errors using Loguru.
- Return structured prediction responses.
"""

import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from app.config.settings import settings
logger = settings.get_logger()

from pydantic import ValidationError

from app.schemas.readmission_schema import (
    ReadmissionPredictionRequest,
    ReadmissionPredictionResponse,
)
from app.services.readmission_service import ReadmissionService


router = APIRouter(
    prefix="/api/v1/readmission",
    tags=["Readmission Prediction"],
)

# The service should load or initialize the model through the
# implementation/model-loader layer, not on every request.
readmission_service = ReadmissionService()


@router.post(
    "/predict",
    response_model=ReadmissionPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict 30-day readmission risk",
    description=(
        "Estimates the probability of hospital readmission within "
        "30 days using the configured trained model."
    ),
)
async def predict_readmission(
    payload: ReadmissionPredictionRequest,
    request: Request,
) -> ReadmissionPredictionResponse:
    """Predict readmission risk for one patient."""

    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    start_time = time.perf_counter()

    logger.bind(
        request_id=request_id,
        endpoint="/api/v1/readmission/predict",
    ).info("Readmission prediction request received")

    try:
        # Convert the validated Pydantic request into model features.
        features = payload.model_dump()

        # Service contract:
        # predict(features: dict, request_id: str) -> dict
        result = readmission_service.predict(
            features=features,
            request_id=request_id,
        )

        response = ReadmissionPredictionResponse(
            **result,
            request_id=request_id,
        )

        elapsed_seconds = round(
            time.perf_counter() - start_time,
            4,
        )

        logger.bind(
            request_id=request_id,
            model_name=response.model_name,
            predicted_readmission=response.predicted_readmission,
            risk_level=response.risk_level,
            elapsed_seconds=elapsed_seconds,
        ).success("Readmission prediction completed")

        return response

    except HTTPException:
        raise

    except (ValueError, ValidationError) as exc:
        logger.bind(
            request_id=request_id,
            error=str(exc),
        ).warning("Invalid readmission prediction input")

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error_code": "INVALID_PREDICTION_INPUT",
                "message": str(exc),
                "request_id": request_id,
            },
        ) from exc

    except Exception as exc:
        elapsed_seconds = round(
            time.perf_counter() - start_time,
            4,
        )

        logger.bind(
            request_id=request_id,
            elapsed_seconds=elapsed_seconds,
            error_type=type(exc).__name__,
        ).exception("Readmission prediction failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "READMISSION_PREDICTION_FAILED",
                "message": "Unable to complete the prediction.",
                "request_id": request_id,
            },
        ) from exc