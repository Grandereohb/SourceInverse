from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from deployment.input_adapter import load_and_validate_input, write_algorithm_inputs
from deployment.output_adapter import build_output_payload
from deployment.packager import build_zip, write_manifest
from deployment.validation import InputValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]


def sample_payload() -> dict:
    return {
        "request_id": "test-001",
        "pollutant": "乙烯",
        "concentration_unit": "ug/m3",
        "coordinate_system": "WGS84",
        "stations": [
            {"station_id": "S01", "longitude": 121.30, "latitude": 30.71},
            {"station_id": "S02", "longitude": 121.31, "latitude": 30.72},
        ],
        "wind": [
            {"time": "2026-08-11 00:00:00", "sp": 2.6, "dir": 45.0},
            {"time": "2026-08-11 01:00:00", "sp": 2.1, "dir": 60.0},
        ],
        "concentrations": [
            {"time": "2026-08-11 00:00:00", "station_id": "S01", "value": 10.0},
            {"time": "2026-08-11 00:00:00", "station_id": "S02", "value": 2.0},
            {"time": "2026-08-11 01:00:00", "station_id": "S01", "value": 12.0},
        ],
        "callback_url": "https://client.example.com/result",
    }


class AdapterTests(unittest.TestCase):
    def test_validation_rejects_unknown_station(self):
        payload = sample_payload()
        payload["concentrations"][0]["station_id"] = "missing"
        with self.assertRaises(InputValidationError):
            from deployment.validation import validate_input

            validate_input(payload)

    def test_json_to_excel_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_json = root / "input.json"
            input_json.write_text(
                json.dumps(sample_payload(), ensure_ascii=False), encoding="utf-8"
            )
            payload = load_and_validate_input(input_json)
            paths = write_algorithm_inputs(payload, root / "input")

            sites = pd.read_excel(paths["sites"])
            concentration = pd.read_excel(paths["concentration"])
            wind = pd.read_excel(paths["wind"])
            self.assertEqual(list(sites.columns), ["station", "lon", "lat"])
            self.assertEqual(
                list(concentration.columns),
                ["time", "S01", "S02", "TARGET_POLLUTANT"],
            )
            self.assertEqual(list(wind.columns), ["time", "dir", "sp"])
            self.assertTrue(pd.isna(concentration.loc[1, "S02"]))
            self.assertEqual(concentration.loc[0, "TARGET_POLLUTANT"], "乙烯")

    def test_output_assembly_and_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            job_dir = Path(tmp) / "job"
            case_dir = job_dir / "algorithm_output" / "case"
            export_dir = case_dir / "溯源输出"
            field_dir = export_dir / "浓度场"
            delivery = job_dir / "delivery"
            logs = job_dir / "logs"
            input_dir = job_dir / "input"
            for directory in (field_dir, delivery, logs, input_dir):
                directory.mkdir(parents=True, exist_ok=True)

            strength = export_dir / "源强.txt"
            strength.write_text(
                "2026-08-11 00:00:00\t2.50000000\n", encoding="utf-8"
            )
            field = field_dir / "浓度场_20260811_h00.txt"
            field.write_text(
                "121.30000000\t30.71000000\t3.50000000\n", encoding="utf-8"
            )
            report = {
                "inversion_text_outputs": {
                    "source_strength_paths": [str(strength)],
                    "field_paths": [[str(field)]],
                },
                "is_reasonable": True,
                "warnings": [],
            }
            (case_dir / "result_quality_report.json").write_text(
                json.dumps(report, ensure_ascii=False), encoding="utf-8"
            )
            result = {
                "output_dir": str(case_dir),
                "pred_lon": 121.31,
                "pred_lat": 30.72,
            }
            output = build_output_payload(
                payload=sample_payload(),
                job_id="job-1",
                algorithm_result=result,
                repo_root=REPO_ROOT,
            )
            self.assertEqual(output["release_strength"][0]["value"], 2.5)
            self.assertEqual(output["fields"][0]["points"][0]["concentration"], 3.5)
            output_path = delivery / "output.json"
            output_path.write_text(json.dumps(output), encoding="utf-8")
            (job_dir / "job.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8"
            )
            manifest = write_manifest(
                job_dir=job_dir,
                request_id="test-001",
                job_id="job-1",
                started_at="2026-08-11 00:00:00",
                completed_at="2026-08-11 00:01:00",
            )
            archive = build_zip(job_dir)
            self.assertTrue(manifest.exists())
            with zipfile.ZipFile(archive) as bundle:
                names = set(bundle.namelist())
            self.assertIn("delivery/output.json", names)
            self.assertIn("delivery/manifest.json", names)
            self.assertIn("job.json", names)


if __name__ == "__main__":
    unittest.main()
