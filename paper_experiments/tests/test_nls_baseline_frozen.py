"""Protect the user-requested pre-optimization NLS implementation."""

import hashlib
import json
from pathlib import Path
import unittest

import metec_point_adapter


class FrozenNlsBaselineTests(unittest.TestCase):
    def test_implementation_matches_pre_optimization_snapshot(self):
        root=Path(__file__).resolve().parents[2]
        manifest=json.loads((root/'paper_experiments/nls_baseline_frozen.json').read_text(encoding='utf-8'))
        for name,expected in manifest['sha256_lf'].items():
            with self.subTest(path=name):
                source=(root/name).read_bytes().replace(b'\r\n',b'\n')
                self.assertEqual(hashlib.sha256(source).hexdigest(),expected)

    def test_nls_does_not_import_ia_transport_extensions(self):
        self.assertEqual(metec_point_adapter.PhysicalBeamPlan.__module__,'paper_physical_beam_adapter')
        self.assertEqual(metec_point_adapter.PhysicalBeamResponse.__module__,'paper_physical_beam_adapter')


if __name__=='__main__':
    unittest.main()
