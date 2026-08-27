from __future__ import annotations

import gzip
import json
import os
import tempfile
import threading
import time
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from deployment.api import create_app
from deployment.callback_sender import send_result_callback
from deployment.job_manager import JobManager, JobSettings
from deployment.packager import build_zip, write_manifest
from deployment.tests.test_adapter import sample_payload
from deployment.validation import TIME_FORMAT


def fake_runner(**kwargs) -> dict:
    job_dir = Path(kwargs["job_dir"])
    delivery = job_dir / "delivery"
    delivery.mkdir(parents=True, exist_ok=True)
    output = {
        "request_id": "test-001",
        "job_id": kwargs["job_id"],
        "status": "completed",
        "fields": [],
    }
    output_path = delivery / "output.json"
    output_path.write_text(json.dumps(output), encoding="utf-8")
    state_path = job_dir / "job.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "status": kwargs.get("completion_status", "completed"),
            "started_at": "2026-08-11 00:00:00",
            "completed_at": "2026-08-11 00:00:01",
        }
    )
    state_path.write_text(json.dumps(state), encoding="utf-8")
    manifest_path = write_manifest(
        job_dir=job_dir,
        request_id=state["request_id"],
        job_id=state["job_id"],
        started_at=state["started_at"],
        completed_at=state["completed_at"],
    )
    zip_path = build_zip(job_dir)
    return {
        "job_id": state["job_id"],
        "job_dir": str(job_dir),
        "output_json": str(output_path),
        "manifest_json": str(manifest_path),
        "zip_path": str(zip_path),
    }


def fake_callback(**kwargs) -> dict:
    return {"delivered": True, "attempts": 1, "http_status": 200, "response": "ok"}


class ApiTests(unittest.TestCase):
    def test_submit_idempotency_status_and_download(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"SOURCE_INVERSION_API_TOKEN": "secret"}
        ):
            manager = JobManager(
                JobSettings(work_root=Path(tmp), callback_retry_delays_seconds=(0.0,)),
                runner=fake_runner,
                callback_sender=fake_callback,
            )
            app = create_app(manager)
            headers = {
                "Authorization": "Bearer secret",
                "Idempotency-Key": "test-001",
            }
            with TestClient(app) as client:
                self.assertEqual(client.get("/api/v1/health").status_code, 200)
                self.assertEqual(client.post("/api/v1/jobs", json=sample_payload()).status_code, 401)
                response = client.post("/api/v1/jobs", json=sample_payload(), headers=headers)
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["job_id"]
                duplicate = client.post("/api/v1/jobs", json=sample_payload(), headers=headers)
                self.assertEqual(duplicate.status_code, 202)
                self.assertEqual(duplicate.json()["job_id"], job_id)

                deadline = time.time() + 3.0
                state = None
                while time.time() < deadline:
                    state = client.get(
                        f"/api/v1/jobs/{job_id}", headers={"Authorization": "Bearer secret"}
                    ).json()
                    if state["status"] in {"completed", "failed"}:
                        break
                    time.sleep(0.02)
                self.assertEqual(state["status"], "completed")
                self.assertTrue(state["callback"]["delivered"])

                output = client.get(
                    f"/api/v1/jobs/{job_id}/output",
                    headers={"Authorization": "Bearer secret"},
                )
                self.assertEqual(output.status_code, 200)
                archive = client.get(
                    f"/api/v1/jobs/{job_id}/result",
                    headers={"Authorization": "Bearer secret"},
                )
                self.assertEqual(archive.status_code, 200)
                zip_path = Path(tmp) / "download.zip"
                zip_path.write_bytes(archive.content)
                with zipfile.ZipFile(zip_path) as bundle:
                    packaged_state = json.loads(bundle.read("job.json"))
                self.assertEqual(packaged_state["status"], "completed")

                conflicting = sample_payload()
                conflicting["pollutant"] = "different"
                conflict = client.post("/api/v1/jobs", json=conflicting, headers=headers)
                self.assertEqual(conflict.status_code, 409)

    def test_invalid_input_returns_422(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                JobSettings(work_root=Path(tmp)),
                runner=fake_runner,
                callback_sender=fake_callback,
            )
            app = create_app(manager)
            payload = sample_payload()
            payload["wind"][0]["dir"] = 360
            with TestClient(app) as client:
                response = client.post("/api/v1/jobs", json=payload)
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["error"]["code"], "INVALID_INPUT")
            self.assertFalse(response.json()["error"]["retryable"])


class CallbackTests(unittest.TestCase):
    def test_gzip_callback_retries_and_preserves_headers(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                received.append(
                    {
                        "payload": json.loads(gzip.decompress(body)),
                        "encoding": self.headers.get("Content-Encoding"),
                        "authorization": self.headers.get("Authorization"),
                        "attempt": self.headers.get("X-Callback-Attempt"),
                    }
                )
                self.send_response(500 if len(received) == 1 else 200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *_):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "output.json"
                output.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
                result = send_result_callback(
                    callback_url=f"http://127.0.0.1:{server.server_port}/callback",
                    output_path=output,
                    request_id="request-1",
                    job_id="job-1",
                    token="callback-secret",
                    timeout_seconds=2,
                    retry_delays_seconds=(0.0, 0.0),
                )
        finally:
            server.shutdown()
            server.server_close()
        self.assertTrue(result["delivered"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(len(received), 2)
        self.assertEqual(received[0]["encoding"], "gzip")
        self.assertEqual(received[0]["authorization"], "Bearer callback-secret")
        self.assertEqual(received[1]["attempt"], "2")


if __name__ == "__main__":
    unittest.main()
