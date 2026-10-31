
"""
Application entry point for Patient Readmission Risk Prediction API.

Responsibilities:
- Configure application logging with Loguru.
- Create the FastAPI application.
- Register API routers.
- Handle application startup and shutdown.
- Provide health and readiness endpoints.
- Log HTTP request duration and request IDs.
"""

import sys
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.config.settings import settings
logger = settings.get_logger()


from app.routers.readmission_router import router as readmission_router


# ---------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------

def configure_logging() -> None:
    """Configure structured, rotating application logs."""

    logger.remove()

    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level:<8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    logger.add(
        "logs/readmission_api_{time:YYYY-MM-DD}.log",
        level="INFO",
        rotation="00:00",
        retention="14 days",
        compression="zip",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )


configure_logging()


# ---------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""

    logger.info("Starting Patient Readmission Risk Prediction API")

    try:
        # The router's service/implementation initializes the model.
        # Importing the router may already instantiate that service.
        logger.success("Application startup completed")
        yield
    except Exception:
        logger.exception("Application lifecycle failed")
        raise
    finally:
        logger.info("Shutting down Patient Readmission Risk Prediction API")


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Readmission Prediction API",
    description=(
        "API for estimating 30-day hospital readmission probability "
        "using a configured machine-learning model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Middleware
# ---------------------------------------------------------

@app.middleware("http")
async def log_http_requests(request: Request, call_next):
    """Log request metadata and response duration without patient data."""

    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id

    started_at = time.perf_counter()

    try:
        response = await call_next(request)

        elapsed_seconds = round(
            time.perf_counter() - started_at,
            4,
        )

        response.headers["X-Request-ID"] = request_id

        logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            elapsed_seconds=elapsed_seconds,
        ).info("HTTP request completed")

        return response

    except Exception:
        elapsed_seconds = round(
            time.perf_counter() - started_at,
            4,
        )

        logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            elapsed_seconds=elapsed_seconds,
        ).exception("Unhandled HTTP request failure")

        raise


# ---------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    """Return a consistent response for invalid requests."""

    request_id = getattr(request.state, "request_id", str(uuid4()))

    # Do not log the request body or submitted patient values.
    logger.bind(
        request_id=request_id,
        path=request.url.path,
        error_count=len(exc.errors()),
    ).warning("Request validation failed")

    return JSONResponse(
        status_code=422,
        content={
            "error_code": "REQUEST_VALIDATION_FAILED",
            "message": "The request contains invalid or missing fields.",
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):
    """Return a safe generic response for unexpected errors."""

    request_id = getattr(request.state, "request_id", str(uuid4()))

    logger.bind(
        request_id=request_id,
        path=request.url.path,
        error_type=type(exc).__name__,
    ).exception("Unhandled application exception")

    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected server error occurred.",
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


# ---------------------------------------------------------
# Routers
# ---------------------------------------------------------

app.include_router(readmission_router)


# ---------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------

@app.get(
    "/",
    tags=["System"],
    summary="API information",
)
async def root():
    return {
        "service": "Patient Readmission Risk Prediction API",
        "version": app.version,
        "status": "running",
        "docs": "/docs",
    }


@app.get(
    "/health",
    tags=["System"],
    summary="Application health check",
)
async def health():
    """Check whether the API process is responsive."""

    return {
        "status": "healthy",
        "service": "patient-readmission-risk-prediction",
    }


@app.get(
    "/ready",
    tags=["System"],
    summary="Application readiness check",
)
async def readiness():
    """
    Basic readiness endpoint.

    This confirms the application is responding. For production,
    also verify that the model is loaded and usable.
    """

    return {
        "status": "ready",
        "service": "patient-readmission-risk-prediction",
    }
