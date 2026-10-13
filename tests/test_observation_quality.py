import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "pinn_source"))
import extract_monitor_data_jjj as extractor
from observation_quality import (QUALITY_FILENAME, make_quality_record,
    build_quality_report, write_quality_report, archive_quality_report)
from pipeline import _copy_training_inputs


class ObservationQualityTests(unittest.TestCase):
    def record(self, raw="0.00(D)", number=0., **extra):
        return make_quality_record(raw, number, station="H2", time="2026-09-13 15:00:00", **extra)

    def test_unknown_status_never_certified(self):
        for code in ("D", "N", "C"):
            row = self.record(f"0.00({code})")
            self.assertEqual(row["status_code"], code)
            self.assertIsNone(row["status_meaning"])
            self.assertEqual(row["status_state"], "unknown_code")
            self.assertEqual(row["numeric_value"], 0.)

    def test_missing_not_zero(self):
        for raw in (None, "", "--"):
            row = self.record(raw, None)
            self.assertIsNone(row["numeric_value"])
            self.assertEqual(row["value_state"], "missing")

    def test_raw_bound_and_scientific_notation_preserved(self):
        row = self.record("<0.01(D)", .01)
        self.assertEqual(row["reported_relation"], "<")
        self.assertEqual(row["raw_text"], "<0.01(D)")
        self.assertTrue(row["warnings"])
        self.assertTrue(self.record("1e-3(N)", 1.)["warnings"])

    def test_unflagged_nonfinite_and_unparsed(self):
        self.assertEqual(self.record("1", 1.)["status_state"], "unflagged_unknown")
        self.assertEqual(self.record("bad", None)["value_state"], "unparsed")
        row = self.record("inf", float("inf"))
        self.assertEqual(row["value_state"], "nonfinite_numeric")
        json.dumps(row, allow_nan=False)

    def test_mixed_duplicate_flags_not_collapsed(self):
        report = build_quality_report([self.record(), self.record("10(N)", 10.)], {})
        summary = report["station_summary"][0]
        self.assertEqual(summary["duplicate_time_count"], 1)
        self.assertEqual(summary["mixed_status_duplicate_time_count"], 1)
        self.assertEqual(summary["status_code_counts"], {"D": 1, "N": 1})
        self.assertEqual(len(report["records"]), 2)

    def workbook(self, folder):
        path = folder / "monitor.xlsx"
        frame = pd.DataFrame({"时间": ["标准单位", "2026-09-13 15:00:00", "2026-09-13 15:00:00",
                                     "2026-09-13 16:00:00", "2026-09-13 17:00:00"],
            "丙烷": ["ug/m3", "0.00(D)", "10.00(N)", "--", "<0.20(N)"],
            "风速": ["m/s", 1., 3., 2., 2.], "风向": ["deg", 0., 0., 0., 0.]})
        frame.to_excel(path, sheet_name="H2(带标识)", index=False)
        return path

    def test_quality_collection_preserves_numeric_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.workbook(Path(temp))
            old = extractor.build_concentration_and_wind_tables(path, None, None, "丙烷")
            records = []
            new = extractor.build_concentration_and_wind_tables(path, None, None, "丙烷", quality_records=records)
            pd.testing.assert_frame_equal(old[0], new[0])
            pd.testing.assert_frame_equal(old[1], new[1])
            self.assertEqual(old[2], new[2])
            self.assertEqual(new[0]["H2"].iloc[0], 5.)
            self.assertTrue(pd.isna(new[0]["H2"].iloc[1]))
            self.assertEqual(records[0]["source_excel_row"], 3)
            self.assertEqual(records[0]["unit"], "ug/m3")
            self.assertEqual(records[2]["value_state"], "missing")

    def test_extraction_sidecar_and_pipeline_copy(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            monitor = self.workbook(folder)
            sites = folder / "coords.xlsx"
            pd.DataFrame([[1, "H2", "point", "120,39"]]).to_excel(sites, index=False, header=False)
            paths = extractor.extract_hourly_monitor_data(monitor, None, None, "丙烷", sites,
                                                         output_folder=str(folder / "out"))
            concentration, wind, site = paths
            sidecar = concentration.parent / QUALITY_FILENAME
            report = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(len(report["records"]), 4)
            self.assertEqual(report["policy"], "diagnostics_only_no_filtering")
            result = folder / "result"
            result.mkdir()
            copied = _copy_training_inputs(result, site, concentration, wind)
            self.assertEqual(set(copied), {"sites.xlsx", "concentration.xlsx", "wind.xlsx"})
            self.assertEqual(sidecar.read_bytes(), (result / QUALITY_FILENAME).read_bytes())

    def test_stale_or_corrupt_sidecar_is_not_archived(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source, concentration = folder / "source", folder / "concentration.xlsx"
            source.write_bytes(b"source")
            concentration.write_bytes(b"workbook")
            write_quality_report([self.record()], concentration, source)
            target = folder / "result"
            target.mkdir()
            concentration.write_bytes(b"different workbook")
            self.assertEqual(archive_quality_report(concentration, target), "stale_hash")
            self.assertFalse((target / QUALITY_FILENAME).exists())
            (folder / QUALITY_FILENAME).write_text("not json", encoding="utf-8")
            self.assertEqual(archive_quality_report(concentration, target), "unreadable")

    def test_optional_sidecar_and_failed_archive_do_not_block_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            concentration = folder / "concentration.xlsx"
            concentration.write_bytes(b"workbook")
            self.assertEqual(archive_quality_report(concentration, folder), "absent")
            write_quality_report([self.record()], concentration, concentration)
            with patch("observation_quality.shutil.copy2", side_effect=PermissionError):
                self.assertEqual(archive_quality_report(concentration, folder / "out"), "archive_failed")

    def test_quality_lock_uses_same_fallback_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(extractor, "output_file_is_locked", side_effect=lambda p: p.name == QUALITY_FILENAME):
                paths = extractor.prepare_output_paths(Path(temp))
            self.assertEqual(len({p.parent for p in paths}), 1)
            self.assertTrue(paths[0].parent.name.startswith("extracted_"))
