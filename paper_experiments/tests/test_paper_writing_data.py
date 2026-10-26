"""Writing-table provenance checks; no additional inverse solves."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import build_paper_writing_data as writing


class WritingTablesTests(unittest.TestCase):
    def test_same_release_table_keeps_both_heights_and_reference_scope(self):
        source=writing.ROOT/'controlled_release_fit_v2_precision/analysis/case_metrics.csv'
        if not source.exists():
            self.skipTest('Historical experiment data are not distributed with tests')
        before=source.read_bytes()
        with tempfile.TemporaryDirectory() as folder, patch.object(writing,'OUTPUT',Path(folder)):
            writing.literature_tables()
            table=pd.read_csv(Path(folder)/'controlled_release_same_event_comparison.csv')
            self.assertEqual(set(table.release),{'rel06','rel07'})
            self.assertEqual(len(table),8)
            own=table[table.method=='IA_transient_constant_Q']
            self.assertEqual(set(own.height_m),{.3,1.6})
            self.assertTrue(own.source.str.contains('NOT independent weighing').all())
            self.assertTrue((own[own.release=='rel07'].estimated_q_kg_h>10).all())
            self.assertEqual(source.read_bytes(),before)
            original=pd.read_csv(source)
            selected=original[original.event.isin(['source2_rel06','source1_rel07'])]
            self.assertTrue(selected.reference_q_kg_h.isna().all())


if __name__=='__main__':
    unittest.main()
