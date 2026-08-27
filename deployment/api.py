from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from .job_manager import (
    IdempotencyConflictError,
    JobManager,
    JobNotFoundError,
    JobQueueFullError,
    JobSettings,
)
from .validation import InputValidationError


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_or_none(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


def _env_delays(name: str, default: str = "0,60,300") -> tuple[float, ...]:
    values = os.environ.get(name, default)
    return tuple(float(item.strip()) for item in values.split(",") if item.strip())


def _env_csv(name: str) -> tuple[str, ...]:
    return tuple(
        item.strip() for item in os.environ.get(name, "").split(",") if item.strip()
    )


def manager_from_environment() -> JobManager:
    settings = JobSettings(
        work_root=Path(
            os.environ.get("SOURCE_INVERSION_WORK_ROOT", "deployment_runs/api")
        ),
        queue_size=int(os.environ.get("SOURCE_INVERSION_QUEUE_SIZE", "8")),
        epochs=_env_int_or_none("SOURCE_INVERSION_TEST_EPOCHS"),
        random_seed=int(os.environ.get("SOURCE_INVERSION_RANDOM_SEED", "0")),
        make_plots=_env_flag("SOURCE_INVERSION_MAKE_PLOTS", False),
        callback_token=os.environ.get("SOURCE_INVERSION_CALLBACK_TOKEN", ""),
        callback_timeout_seconds=float(
            os.environ.get("SOURCE_INVERSION_CALLBACK_TIMEOUT_SECONDS", "30")
        ),
        callback_retry_delays_seconds=_env_delays(
            "SOURCE_INVERSION_CALLBACK_RETRY_DELAYS_SECONDS"
        ),
        callback_url_override=os.environ.get(
            "SOURCE_INVERSION_CALLBACK_URL_OVERRIDE", ""
        ).strip(),
        callback_allowed_hosts=_env_csv("SOURCE_INVERSION_CALLBACK_ALLOWED_HOSTS"),
    )
    return JobManager(settings)


def create_app(manager: JobManager | None = None) -> FastAPI:
    job_manager = manager or manager_from_environment()
    api_token = os.environ.get("SOURCE_INVERSION_API_TOKEN", "")
    max_body_bytes = int(os.environ.get("SOURCE_INVERSION_MAX_BODY_BYTES", "10485760"))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        job_manager.start()
        try:
            yield
        finally:
            job_manager.stop()

    app = FastAPI(
        title="Source Inversion API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.job_manager = job_manager

    def error_response(
        request: Request,
        *,
        status_code: int,
        code: str,
        message: str,
        field: str | None = None,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={
                "request_id": request.headers.get("X-Request-ID"),
                "job_id": None,
                "error": {
                    "code": code,
                    "message": message,
                    "field": field,
                    "retryable": status_code in {429, 500, 503},
                },
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        codes = {
            400: "BAD_REQUEST",
            401: "UNAUTHORIZED",
            404: "JOB_NOT_FOUND",
            409: "CONFLICT",
            413: "BODY_TOO_LARGE",
            422: "INVALID_INPUT",
            429: "QUEUE_FULL",
        }
        return error_response(
            request,
            status_code=exc.status_code,
            code=codes.get(exc.status_code, "HTTP_ERROR"),
            message=str(exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(request: Request, exc: RequestValidationError):
        return error_response(
            request,
            status_code=400,
            code="INVALID_JSON",
            message=str(exc),
        )

    @app.middleware("http")
    async def body_size_limit(request: Request, call_next):
        length = request.headers.get("content-length")
        if length:
            try:
                too_large = int(length) > max_body_bytes
            except ValueError:
                too_large = False
            if too_large:
                return error_response(
                    request,
                    status_code=413,
                    code="BODY_TOO_LARGE",
                    message="request body exceeds size limit",
                )
        return await call_next(request)

    def authorize(authorization: str | None) -> None:
        if not api_token:
            return
        if authorization != f"Bearer {api_token}":
            raise HTTPException(status_code=401, detail="invalid bearer token")

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
    def submit_job(
        payload: dict = Body(...),
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ):
        authorize(authorization)
        if idempotency_key and idempotency_key != str(payload.get("request_id", "")):
            raise HTTPException(
                status_code=400,
                detail="Idempotency-Key must match request_id",
            )
        try:
            state, _ = job_manager.submit(payload)
        except InputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except IdempotencyConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except JobQueueFullError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return {
            "request_id": state["request_id"],
            "job_id": state["job_id"],
            "status": state["status"],
        }

    @app.get("/api/v1/jobs/{job_id}")
    def job_status(job_id: str, authorization: str | None = Header(default=None)):
        authorize(authorization)
        try:
            return job_manager.get_status(job_id)
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/api/v1/jobs/{job_id}/result")
    def download_result(job_id: str, authorization: str | None = Header(default=None)):
        authorize(authorization)
        try:
            path = job_manager.get_artifact(job_id, "result.zip")
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail="result is not ready") from exc
        return FileResponse(path, media_type="application/zip", filename="result.zip")

    @app.get("/api/v1/jobs/{job_id}/output")
    def get_output(job_id: str, authorization: str | None = Header(default=None)):
        authorize(authorization)
        try:
            path = job_manager.get_artifact(job_id, "output.json")
        except JobNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail="result is not ready") from exc
        return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))

    return app


app = create_app()
