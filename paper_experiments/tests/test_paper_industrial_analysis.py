import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import analyze_paper_industrial_cases as analysis


class IndustrialAnalysisTests(unittest.TestCase):
    def test_pending_cases_are_not_aggregated_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            prepared=root/'prepared'
            prepared.mkdir()
            manifest=prepared/'PREPARATION_FINISHED.json'
            manifest.write_text(json.dumps(dict(prepared_sha256={})),encoding='utf-8')
            study=root/'study'
            study.mkdir()
            (study/'protocol.json').write_text(json.dumps(dict(preparation=str(prepared),
                prepared_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                case_ids=['unit_a','unit_b'],mode='full')),encoding='utf-8')
            frame,status=analysis.analyze(study)
            self.assertTrue(frame.empty)
            self.assertFalse(status['final'])
            self.assertFalse(status['source_truth_available'])
            self.assertEqual(status['planned_cases'],2)
            self.assertFalse((study/'analysis/summary.csv').exists())

    def test_cluster_bootstrap_preserves_blocks_and_requires_replication(self):
        pairs=pd.DataFrame(dict(temporal_cluster=['a','a','b','b'],
            normalized_rmse_IA_minus_joint=[1.,2.,3.,4.]))
        report=analysis.summarize_pairs(pairs,repeats=200)
        self.assertEqual(report['temporal_clusters'],2)
        self.assertEqual(report['normalized_rmse_difference_median'],2.5)
        self.assertEqual(report['interval_low'],1.5)
        self.assertEqual(report['interval_high'],3.5)
        self.assertIsNone(analysis.summarize_pairs(pairs.iloc[:2],repeats=20)['interval_low'])


if __name__=='__main__':
    unittest.main()
