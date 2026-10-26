import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_paper_comparison_report import METHODS, check_inputs, interval
from plot_paper_comparison_v4 import complete_analysis


class ComparisonDeliverableTests(unittest.TestCase):
    def test_formal_figure_rejects_partial_analysis(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            (path/'analysis').mkdir()
            (path/'analysis/analysis_status.json').write_text('{"final": false}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Complete batch'):
                complete_analysis(path, False)
            self.assertFalse(complete_analysis(path, True)['final'])

    def test_report_check_does_not_treat_complete_metrics_as_complete_delivery(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            synthetic, public = root/'synthetic', root/'public'
            for folder in (synthetic, public):
                (folder/'analysis').mkdir(parents=True)
                (folder/'analysis/analysis_status.json').write_text('{"final": true}', encoding='utf-8')
            (synthetic/'protocol.json').write_text(json.dumps(dict(
                cases=[dict(case=dict(case_id='unit'))])), encoding='utf-8')
            (public/'protocol.json').write_text(json.dumps(dict(events=[dict(source='source_1', release='rel03')],
                heights_m=[.3])), encoding='utf-8')
            pd.DataFrame([dict(case_id='unit', method=method) for method in METHODS]).to_csv(
                synthetic/'analysis/case_metrics.csv', index=False)
            pd.DataFrame([dict(event='source1_rel03', height_m=.3, method=method) for method in METHODS]).to_csv(
                public/'analysis/case_metrics.csv', index=False)
            report = check_inputs(synthetic, public, root/'newman', root/'figures')
            self.assertFalse(report['ready'])
            self.assertIn('Newman result/audit missing: source1_rel03_h03', report['missing'])
            self.assertNotIn('synthetic case identities incomplete', report['missing'])
            self.assertFalse((root/'report.docx').exists())
            (public/'analysis/quadrature_replay_status.json').write_text(
                json.dumps(dict(final=True, failed=4)), encoding='utf-8')
            strict = check_inputs(synthetic, public, root/'newman', root/'figures')
            explicit = check_inputs(synthetic, public, root/'newman', root/'figures',
                                    allow_numerical_warnings=True)
            self.assertIn('public higher-quadrature replay incomplete or failed', strict['missing'])
            self.assertNotIn('public higher-quadrature replay incomplete or failed', explicit['missing'])
            self.assertFalse(explicit['ready'])

    def test_missing_measured_reference_not_numeric_zero(self):
        self.assertEqual(interval(pd.Series([None], dtype=float)), '无实测参考')
        self.assertEqual(interval(pd.Series([1., 2., 3.])), '2.0 [1.5, 2.5]')


if __name__ == '__main__':
    unittest.main()
