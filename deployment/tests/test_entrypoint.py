from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from deployment.algorithm_runner import build_worker_command
from deployment.entrypoint import main


class EntrypointTests(unittest.TestCase):
    def test_python_worker_command_is_the_development_fallback(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SOURCE_INVERSION_EXECUTABLE", None)
            command = build_worker_command(["--pollutant", "test"])
        self.assertEqual(
            command[:4],
            [sys.executable, "-u", "-m", "deployment.worker_entry"],
        )
        self.assertEqual(command[4:], ["--pollutant", "test"])

    def test_compiled_worker_command_uses_configured_launcher(self):
        with patch.dict(
            os.environ,
            {"SOURCE_INVERSION_EXECUTABLE": "/opt/source-inversion/bin/app"},
        ):
            command = build_worker_command(["--pollutant", "test"])
        self.assertEqual(
            command,
            [
                "/opt/source-inversion/bin/app",
                "worker",
                "--pollutant",
                "test",
            ],
        )

    def test_worker_subcommand_delegates_remaining_arguments(self):
        with patch("deployment.worker_entry.main") as worker_main:
            main(["worker", "--pollutant", "test"])
        worker_main.assert_called_once_with(["--pollutant", "test"])

    def test_serve_subcommand_delegates_remaining_arguments(self):
        with patch("deployment.entrypoint._run_server") as run_server:
            main(["serve", "--port", "9000"])
        run_server.assert_called_once_with(["--port", "9000"])


if __name__ == "__main__":
    unittest.main()
