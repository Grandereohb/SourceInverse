from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_paper_alternatives_v4 import summarize_candidates


class AlternativeDiagnosticsTests(unittest.TestCase):
    def test_better_finite_candidate_does_not_silently_replace_selected_solution(self):
        surface = np.array([[.5, 0., .1, .05, .1, 1.], [.25, 0., .1, .05, .105, 1.]])
        selected = np.array([0., 0.])
        rows = summarize_candidates(surface, selected, .2, 1000.)
        self.assertTrue(all(not row['selected_within_tolerance'] for row in rows))
        self.assertEqual(rows[0]['selected_relative_gap'], 1.)
        self.assertEqual(rows[0]['farthest_alternative_m'], 500.)
        np.testing.assert_array_equal(selected, [0., 0.])
        self.assertTrue(all(a['sampled_locations'] <= b['sampled_locations']
                            for a, b in zip(rows, rows[1:])))

    def test_selected_solution_counted_without_duplicate_location(self):
        surface = np.array([[0., 0., .1, .05, .1, 1.], [.75, 0., .1, .05, .105, 1.]])
        rows = summarize_candidates(surface, [0., 0.], .1, 1000.)
        self.assertEqual(rows[0]['sampled_locations'], 1)
        self.assertEqual(rows[-1]['sampled_locations'], 2)
        self.assertTrue(rows[-1]['sampled_boundary_reached'])


if __name__ == '__main__':
    unittest.main()
