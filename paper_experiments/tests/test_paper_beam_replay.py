from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verify_paper_beam_comparison import relative_prediction_change


class BeamReplayTests(unittest.TestCase):
    def test_relative_change_uses_fine_reference(self):
        self.assertAlmostEqual(relative_prediction_change([1.01, 2.02], [1., 2.]), .01)
        self.assertEqual(relative_prediction_change([0., 0.], [0., 0.]), 0.)

    def test_nonfinite_replay_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Finite'):
            relative_prediction_change([np.nan], [1.])


if __name__ == '__main__':
    unittest.main()
