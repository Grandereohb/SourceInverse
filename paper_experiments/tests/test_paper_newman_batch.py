from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_paper_newman_batch import planned_cases


class NewmanBatchTests(unittest.TestCase):
    def test_same_events_both_heights_no_best_height_selection(self):
        protocol = dict(events=[dict(source='source_1', release='rel03'),
                                dict(source='source_2', release='rel06')], heights_m=[.3, 1.6])
        cases = planned_cases(protocol)
        self.assertEqual(len(cases), 4)
        self.assertEqual(cases[0], ('source1_rel03', .3, 'source1_rel03_h03'))
        self.assertEqual(cases[-1], ('source2_rel06', 1.6, 'source2_rel06_h16'))

    def test_duplicate_event_height_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            planned_cases(dict(events=[dict(source='source_1', release='rel03')], heights_m=[.3, .3]))


if __name__ == '__main__':
    unittest.main()
