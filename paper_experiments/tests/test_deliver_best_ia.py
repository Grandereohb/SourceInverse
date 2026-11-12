import unittest
import numpy as np
import pandas as pd
from deliver_best_ia import METHODS, validate_cohort


class DeliveryCohortTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(case_id=case, method=method, localization_error_m=float(index + 1))
            for case in ['a', 'b'] for index, method in enumerate(METHODS)])

    def test_complete_cohort(self):
        self.assertEqual(validate_cohort(self.frame(), 2).shape, (2, 4))

    def test_missing_or_duplicate_rows_rejected(self):
        frame = self.frame()
        for wrong in [frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]]), frame[frame.case_id == 'a']]:
            with self.assertRaises(ValueError):
                validate_cohort(wrong, 2)

    def test_nonfinite_or_negative_errors_rejected(self):
        for value in [np.nan, np.inf, -1.]:
            frame = self.frame()
            frame.loc[0, 'localization_error_m'] = value
            with self.assertRaises(ValueError):
                validate_cohort(frame, 2)


if __name__ == '__main__':
    unittest.main()
