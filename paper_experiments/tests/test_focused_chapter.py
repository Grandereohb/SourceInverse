import unittest

import pandas as pd

from run_focused_chapter import classify, summarize_flags, representative_ids


class FocusedChapterTests(unittest.TestCase):
    def test_thresholds_keep_denominators_and_boundary(self):
        data = pd.DataFrame(dict(method=['a']*3, observed_mse=[.0004, .0001, .0009],
            noise_std=[.01]*3, localization_error_m=[100., 101., 300.], q_relative_l2=[0.]*3))
        flags = classify(data)
        self.assertEqual(flags.good_fit.tolist(), [True, True, False])
        result = summarize_flags(flags, ['method']).iloc[0]
        self.assertEqual(result.n, 3)
        self.assertEqual(result.good_fit_wrong, 1)
        self.assertEqual(result.conditional_rate, .5)

    def test_representative_does_not_select_on_ia_win(self):
        frame = pd.DataFrame(dict(case_id=['eval_020', 'eval_010'], cohort=['nominal']*2,
            method=['joint_nls_trf']*2, good_fit_wrong_location=[True]*2))
        self.assertEqual(representative_ids(frame), ['eval_007', 'eval_010'])


if __name__ == '__main__':
    unittest.main()
