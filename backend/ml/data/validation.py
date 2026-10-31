
"""
Dataset validation for Patient Readmission Risk Prediction.

Expected target:
    readmitted_30d
        0 = No readmission within 30 days
        1 = Readmission within 30 days

Usage:
    from ml.data.validation import validate_readmission_data

    df = pd.read_csv("data/readmission.csv")
    validate_readmission_data(df)
"""

from typing import Optional

import pandas as pd
from app.config.logger import get_logger
logger = get_logger()

# ---------------------------------------------------------
# Schema configuration
# ---------------------------------------------------------

REQUIRED_COLUMNS = [
    "age",
    "length_of_stay",
    "previous_admissions",
    "diagnosis_count",
    "comorbidity_count",
    "discharge_destination",
    "readmitted_30d",
]

NUMERIC_COLUMNS = [
    "age",
    "length_of_stay",
    "previous_admissions",
    "diagnosis_count",
    "comorbidity_count",
    "readmitted_30d",
]

FEATURE_COLUMNS = [
    "age",
    "length_of_stay",
    "previous_admissions",
    "diagnosis_count",
    "comorbidity_count",
    "discharge_destination",
]

ALLOWED_DISCHARGE_DESTINATIONS = {
    "home",
    "facility",
    "other",
}

TARGET_COLUMN = "readmitted_30d"


# ---------------------------------------------------------
# Validation exception
# ---------------------------------------------------------

class DataValidationError(ValueError):
    """Raised when dataset validation fails."""


# ---------------------------------------------------------
# Main validation function
# ---------------------------------------------------------

def validate_readmission_data(
    df: pd.DataFrame,
    *,
    min_records: int = 1,
    reject_duplicates: bool = False,
    allow_extra_columns: bool = True,
    require_both_classes: bool = False,
    logger_context: Optional[str] = None,
) -> bool:
    """
    Validate the readmission prediction dataset.

    Args:
        df:
            Input patient encounter DataFrame.
        min_records:
            Minimum required number of records.
        reject_duplicates:
            If True, reject duplicate complete rows.
        allow_extra_columns:
            If False, reject columns outside the expected schema.
        require_both_classes:
            If True, require target classes 0 and 1.
            Useful before training or ROC-AUC evaluation.
        logger_context:
            Optional identifier for the calling pipeline.

    Returns:
        True if all validation checks pass.

    Raises:
        DataValidationError: If validation fails.
    """

    context = logger_context or "readmission_data"

    logger.info(
        "Starting dataset validation | context={} | rows={} | columns={}",
        context,
        len(df),
        len(df.columns),
    )

    # 1. Input validation
    if not isinstance(df, pd.DataFrame):
        logger.error(
            "Validation failed | context={} | reason=Input is not a DataFrame",
            context,
        )
        raise DataValidationError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        logger.error(
            "Validation failed | context={} | reason=Empty dataset",
            context,
        )
        raise DataValidationError(
            "Dataset must contain at least one record."
        )

    if min_records < 1:
        raise ValueError("min_records must be at least 1.")

    if len(df) < min_records:
        logger.error(
            "Validation failed | context={} | rows={} | minimum={}",
            context,
            len(df),
            min_records,
        )
        raise DataValidationError(
            f"Dataset contains {len(df)} records; "
            f"at least {min_records} are required."
        )

    # 2. Column validation
    if not df.columns.is_unique:
        duplicates = df.columns[
            df.columns.duplicated()
        ].tolist()

        logger.error(
            "Duplicate column names found | context={} | columns={}",
            context,
            duplicates,
        )
        raise DataValidationError(
            f"Duplicate column names: {duplicates}"
        )

    missing_columns = sorted(
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        logger.error(
            "Missing required columns | context={} | columns={}",
            context,
            missing_columns,
        )
        raise DataValidationError(
            f"Missing required columns: {missing_columns}"
        )

    if not allow_extra_columns:
        extra_columns = sorted(
            set(df.columns) - set(REQUIRED_COLUMNS)
        )

        if extra_columns:
            logger.error(
                "Unexpected columns | context={} | columns={}",
                context,
                extra_columns,
            )
            raise DataValidationError(
                f"Unexpected columns: {extra_columns}"
            )

    # 3. Missing-value validation
    missing_counts = df[REQUIRED_COLUMNS].isna().sum()
    missing_counts = missing_counts[
        missing_counts > 0
    ].to_dict()

    if missing_counts:
        logger.error(
            "Missing values detected | context={} | columns={}",
            context,
            missing_counts,
        )
        raise DataValidationError(
            f"Missing values detected: {missing_counts}"
        )

    # 4. Numeric conversion and type validation
    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_mask = converted.isna()

        if invalid_mask.any():
            invalid_count = int(invalid_mask.sum())

            logger.error(
                "Non-numeric values detected | context={} | "
                "column={} | count={}",
                context,
                column,
                invalid_count,
            )

            raise DataValidationError(
                f"Column '{column}' contains "
                f"{invalid_count} non-numeric or invalid values."
            )

        if not converted.map(
            lambda value: float(value).is_integer()
        ).all():
            logger.error(
                "Non-integer values detected | context={} | column={}",
                context,
                column,
            )
            raise DataValidationError(
                f"Column '{column}' must contain integer values."
            )

    # 5. Numeric range validation
    numeric_ranges = {
        "age": (18, 95),
        "length_of_stay": (1, 60),
        "previous_admissions": (0, 15),
        "diagnosis_count": (1, 25),
        "comorbidity_count": (0, 12),
    }

    for column, (minimum, maximum) in numeric_ranges.items():
        values = pd.to_numeric(df[column], errors="coerce")

        invalid_mask = ~values.between(minimum, maximum)

        if invalid_mask.any():
            invalid_count = int(invalid_mask.sum())

            logger.error(
                "Out-of-range values | context={} | column={} | "
                "minimum={} | maximum={} | invalid_count={}",
                context,
                column,
                minimum,
                maximum,
                invalid_count,
            )

            raise DataValidationError(
                f"Column '{column}' must be between "
                f"{minimum} and {maximum}; "
                f"{invalid_count} invalid records found."
            )

    # 6. Categorical validation
    destinations = (
        df["discharge_destination"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    invalid_categories = sorted(
        set(destinations.unique())
        - ALLOWED_DISCHARGE_DESTINATIONS
    )

    if invalid_categories:
        logger.error(
            "Invalid categorical values | context={} | "
            "column=discharge_destination | values={}",
            context,
            invalid_categories,
        )
        raise DataValidationError(
            "Invalid discharge_destination values: "
            f"{invalid_categories}"
        )

    # 7. Binary target validation
    target = pd.to_numeric(
        df[TARGET_COLUMN],
        errors="coerce",
    )

    invalid_target = (
        target.isna()
        | ~target.isin([0, 1])
    )

    if invalid_target.any():
        invalid_count = int(invalid_target.sum())

        logger.error(
            "Invalid binary target | context={} | "
            "column={} | invalid_count={}",
            context,
            TARGET_COLUMN,
            invalid_count,
        )

        raise DataValidationError(
            f"'{TARGET_COLUMN}' must contain only 0 or 1; "
            f"{invalid_count} invalid records found."
        )

    # 8. Duplicate record validation
    duplicate_count = int(
        df[REQUIRED_COLUMNS].duplicated().sum()
    )

    if duplicate_count:
        if reject_duplicates:
            logger.error(
                "Duplicate records rejected | context={} | count={}",
                context,
                duplicate_count,
            )
            raise DataValidationError(
                f"Found {duplicate_count} duplicate records."
            )

        logger.warning(
            "Duplicate records detected | context={} | count={}",
            context,
            duplicate_count,
        )

    # 9. Class-distribution validation
    class_counts = (
        target.astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    class_percentages = {
        int(label): round(
            count / len(df) * 100, 2
        )
        for label, count in class_counts.items()
    }

    if require_both_classes and set(class_counts) != {0, 1}:
        logger.error(
            "Both target classes are required | context={} | "
            "class_counts={}",
            context,
            class_counts,
        )
        raise DataValidationError(
            "Both target classes (0 and 1) are required."
        )

    # 10. Successful validation
    logger.success(
        "Dataset validation passed | context={} | "
        "rows={} | features={} | class_counts={} | "
        "class_percentages={} | duplicate_rows={}",
        context,
        len(df),
        len(FEATURE_COLUMNS),
        class_counts,
        class_percentages,
        duplicate_count,
    )

    return True