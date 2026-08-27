from __future__ import annotations

import hashlib
import json
import queue
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from .callback_sender import send_result_callback
from .local_runner import _safe_name, run_local_job
from .packager import build_zip
from .validation import TIME_FORMAT, validate_input


class IdempotencyConflictError(RuntimeError):
    pass


class JobQueueFullError(RuntimeError):
    pass


class JobNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class JobSettings:
    work_root: Path
    queue_size: int = 8
    epochs: int | None = None
    random_seed: int = 0
    make_plots: bool = False
    callback_token: str = ""
    callback_timeout_seconds: float = 30.0
    callback_retry_delays_seconds: tuple[float, ...] = (0.0, 60.0, 300.0)
    callback_url_override: str = ""
    callback_allowed_hosts: tuple[str, ...] = ()


class JobManager:
    def __init__(
        self,
        settings: JobSettings,
        *,
        runner: Callable = run_local_job,
        callback_sender: Callable = send_result_callback,
    ):
        self.settings = settings
        self.settings.work_root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.settings.work_root / "registry.json"
        self._runner = runner
        self._callback_sender = callback_sender
        self._queue: queue.Queue[str | None] = queue.Queue(
            maxsize=max(int(settings.queue_size), 1)
        )
        self._lock = threading.RLock()
        self._registry = self._load_registry()
        self._worker: threading.Thread | None = None

    def _load_registry(self) -> dict:
        if not self.registry_path.exists():
            return {"requests": {}, "jobs": {}}
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"requests": {}, "jobs": {}}
        data.setdefault("requests", {})
        data.setdefault("jobs", {})
        return data

    def _save_registry(self) -> None:
        temporary = self.registry_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self._registry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.registry_path)

    @staticmethod
    def _digest(payload: dict) -> str:
        body = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def start(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._recover_jobs()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="source-inversion-worker",
                daemon=True,
            )
            self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        worker = self._worker
        if not worker or not worker.is_alive():
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            return
        worker.join(timeout=timeout)

    def _recover_jobs(self) -> None:
        for job_id, record in self._registry["jobs"].items():
            job_path = Path(record["job_dir"]) / "job.json"
            if not job_path.exists():
                continue
            state = self._read_json(job_path)
            if state.get("status") == "queued":
                try:
                    self._queue.put_nowait(job_id)
                except queue.Full:
                    break
            elif state.get("status") in {
                "validating",
                "running",
                "packaging",
                "callback_pending",
            }:
                state.update(
                    {
                        "status": "failed",
                        "error": {
                            "type": "ServiceRestarted",
                            "message": "job was interrupted by a service restart",
                        },
                    }
                )
                self._write_json(job_path, state)

    def submit(self, payload: dict) -> tuple[dict, bool]:
        validate_input(payload)
        request_id = str(payload["request_id"]).strip()
        digest = self._digest(payload)
        with self._lock:
            existing = self._registry["requests"].get(request_id)
            if existing:
                if existing["digest"] != digest:
                    raise IdempotencyConflictError(
                        f"request_id {request_id} was already used with different data"
                    )
                return self.get_status(existing["job_id"]), False
            if self._queue.full():
                raise JobQueueFullError("source inversion queue is full")

            job_id = str(uuid.uuid4())
            job_dir = (
                self.settings.work_root / f"{_safe_name(request_id)}_{job_id[:8]}"
            ).resolve()
            job_dir.mkdir(parents=True, exist_ok=False)
            request_path = job_dir / "request.json"
            self._write_json(request_path, payload)
            state = {
                "request_id": request_id,
                "job_id": job_id,
                "status": "queued",
                "started_at": None,
                "completed_at": None,
                "error": None,
                "callback": None,
            }
            self._write_json(job_dir / "job.json", state)
            self._registry["requests"][request_id] = {
                "job_id": job_id,
                "digest": digest,
            }
            self._registry["jobs"][job_id] = {
                "request_id": request_id,
                "job_dir": str(job_dir),
                "request_path": str(request_path),
            }
            self._save_registry()
            self._queue.put_nowait(job_id)
            return state, True

    def get_status(self, job_id: str) -> dict:
        with self._lock:
            record = self._registry["jobs"].get(job_id)
            if not record:
                raise JobNotFoundError(job_id)
            job_path = Path(record["job_dir"]) / "job.json"
            if not job_path.exists():
                raise JobNotFoundError(job_id)
            return self._read_json(job_path)

    def get_artifact(self, job_id: str, name: str) -> Path:
        with self._lock:
            record = self._registry["jobs"].get(job_id)
            if not record:
                raise JobNotFoundError(job_id)
            path = Path(record["job_dir"]) / "delivery" / name
            if not path.exists():
                raise FileNotFoundError(path)
            return path

    def _update_state(self, job_dir: Path, **changes) -> dict:
        state_path = job_dir / "job.json"
        state = self._read_json(state_path)
        state.update(changes)
        self._write_json(state_path, state)
        return state

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                self._queue.task_done()
                return
            try:
                self._execute_job(job_id)
            finally:
                self._queue.task_done()

    def _execute_job(self, job_id: str) -> None:
        record = self._registry["jobs"][job_id]
        job_dir = Path(record["job_dir"])
        request_path = Path(record["request_path"])
        payload = self._read_json(request_path)
        try:
            result = self._runner(
                input_path=request_path,
                work_root=self.settings.work_root,
                epochs=self.settings.epochs,
                random_seed=self.settings.random_seed,
                make_plots=self.settings.make_plots,
                job_id=job_id,
                job_dir=job_dir,
                completion_status="callback_pending",
            )
            self._update_state(job_dir, status="callback_pending", callback=None)
            callback_url = self.settings.callback_url_override or payload["callback_url"]
            allowed_hosts = {
                host.strip().lower()
                for host in self.settings.callback_allowed_hosts
                if host.strip()
            }
            callback_host = (urlparse(callback_url).hostname or "").lower()
            if allowed_hosts and callback_host not in allowed_hosts:
                raise ValueError(
                    f"callback host is not allowed by server configuration: {callback_host}"
                )
            callback = self._callback_sender(
                callback_url=callback_url,
                output_path=result["output_json"],
                request_id=payload["request_id"],
                job_id=job_id,
                token=self.settings.callback_token,
                timeout_seconds=self.settings.callback_timeout_seconds,
                retry_delays_seconds=self.settings.callback_retry_delays_seconds,
            )
            self._update_state(job_dir, status="completed", callback=callback, error=None)
            build_zip(job_dir)
        except Exception as exc:
            current = self._read_json(job_dir / "job.json")
            self._update_state(
                job_dir,
                status="failed",
                completed_at=current.get("completed_at"),
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            if (job_dir / "delivery" / "output.json").exists():
                build_zip(job_dir)
