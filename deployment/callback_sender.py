from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


class CallbackDeliveryError(RuntimeError):
    pass


def send_result_callback(
    *,
    callback_url: str,
    output_path: str | Path,
    request_id: str,
    job_id: str,
    token: str = "",
    timeout_seconds: float = 30.0,
    retry_delays_seconds: tuple[float, ...] = (0.0, 60.0, 300.0),
) -> dict:
    raw_body = Path(output_path).read_bytes()
    body = gzip.compress(raw_body)
    last_error = None

    for attempt, delay in enumerate(retry_delays_seconds, start=1):
        if delay > 0:
            time.sleep(delay)
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Encoding": "gzip",
            "Accept": "application/json",
            "X-Request-ID": request_id,
            "X-Job-ID": job_id,
            "X-Callback-Attempt": str(attempt),
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            callback_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = int(response.status)
                response_body = response.read(4096).decode("utf-8", errors="replace")
            if 200 <= status < 300:
                return {
                    "delivered": True,
                    "attempts": attempt,
                    "http_status": status,
                    "response": response_body,
                }
            last_error = f"callback returned HTTP {status}"
        except urllib.error.HTTPError as exc:
            last_error = f"callback returned HTTP {exc.code}"
            exc.close()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)

    raise CallbackDeliveryError(
        f"Callback delivery failed after {len(retry_delays_seconds)} attempts: {last_error}"
    )
