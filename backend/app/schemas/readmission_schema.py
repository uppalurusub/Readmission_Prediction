
"""
Pydantic schemas for Patient Readmission Risk Prediction.

Request features must match the model's expected input columns:
    - age
    - length_of_stay
    - previous_admissions
    - diagnosis_count
    - comorbidity_count
    - discharge_destination

Target:
    readmitted_30d (0 = no readmission, 1 = readmission)
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReadmissionFeatures(BaseModel):
    """Input features for one patient's readmission risk prediction."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    age: int = Field(
        ...,
        ge=18,
        le=120,
        description="Patient age in years.",
        examples=[55],
    )

    length_of_stay: int = Field(
        ...,
        ge=0,
        le=365,
        description="Length of the hospital stay in days.",
        examples=[5],
    )

    previous_admissions: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of previous hospital admissions.",
        examples=[2],
    )

    diagnosis_count: int = Field(
        ...,
        ge=1,
        le=100,
        description="Number of diagnoses recorded for the patient.",
        examples=[4],
    )

    comorbidity_count: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of documented comorbidities.",
        examples=[2],
    )

    discharge_destination: Literal["home", "facility", "other"] = Field(
        ...,
        description="Patient's discharge destination.",
        examples=["home"],
    )


class ReadmissionPredictionRequest(ReadmissionFeatures):
    """Request schema for the prediction API."""

    pass


class ReadmissionPredictionResponse(BaseModel):
    """Response schema returned by the prediction API."""

    model_config = ConfigDict(extra="forbid")

    readmission_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability of readmission within 30 days.",
    )

    predicted_readmission: Literal[0, 1] = Field(
        ...,
        description="Predicted binary outcome: 1 = positive, 0 = negative.",
    )

    risk_level: Literal["LOW", "MODERATE", "HIGH"] = Field(
        ...,
        description="Risk category determined by the application's policy.",
    )

    decision_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Decision threshold applied to the predicted probability.",
    )

    model_name: str = Field(
        ...,
        min_length=1,
        description="Name of the model used for prediction.",
    )

    """ model_version: str | None = Field(
        default=None,
        description="Deployed model version, if available.",
    ) """

    request_id: str | None = Field(
        default=None,
        description="Optional request correlation identifier.",
    )

    @model_validator(mode="after")
    def validate_prediction_consistency(self):
        """Ensure the binary prediction agrees with the threshold."""
        expected_prediction = int(
            self.readmission_probability >= self.decision_threshold
        )

        if self.predicted_readmission != expected_prediction:
            raise ValueError(
                "predicted_readmission must match the probability "
                "and decision_threshold."
            )

        return self


class ReadmissionHealthResponse(BaseModel):
    """Basic API health-check response."""

    status: Literal["healthy", "unhealthy"]
    service: str = "patient-readmission-risk-prediction"


class ReadmissionErrorResponse(BaseModel):
    """Standardized API error response."""

    error_code: str
    message: str
    request_id: str | None = None
