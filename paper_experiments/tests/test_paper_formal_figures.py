import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import plot_paper_controlled_release as controlled

SPEC = importlib.util.spec_from_file_location('formal_figures',
    Path(__file__).resolve().parents[1]/'plot_paper_formal.py')
figures = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(figures)


class FormalFigureTests(unittest.TestCase):
    def test_partial_real_release_study_cannot_be_rendered(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'analysis_status.json').write_text(json.dumps(dict(final=False,
                completed_height_jobs=2,completed_cases=1)))
            with self.assertRaises(ValueError):
                controlled.plot(root)
            self.assertFalse((root/'figures').exists())

    def test_partial_evaluation_cannot_be_rendered_as_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'analysis_status.json').write_text(json.dumps(dict(
                final=False, completed_cases=1, planned_cases=120)))
            with self.assertRaisesRegex(ValueError, 'partial evaluation'):
                figures.plot(root)
            self.assertFalse((root/'figures').exists())

    def test_wilson_interval_has_correct_edge_behavior(self):
        self.assertIsNone(figures.wilson(0, 0))
        p, low, high = figures.wilson(0, 10)
        self.assertEqual(p, 0.)
        self.assertAlmostEqual(low, 0.)
        self.assertGreater(high, 0.)
        p, low, high = figures.wilson(10, 10)
        self.assertEqual(p, 1.)
        self.assertLess(low, 1.)
        self.assertAlmostEqual(high, 1.)


if __name__ == '__main__':
    unittest.main()
