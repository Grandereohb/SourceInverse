"""The expanded design must keep all original cases and fixed sample counts."""

from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
import json

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import extend_paper_manuscript_supplement as extension
import build_paper_writing_data as writing


class SupplementExtensionTests(unittest.TestCase):
    def fixture(self):
        return dict(networks={
            'ring':dict(wind='steady',release='pulse'),
            'corridor':dict(wind='veering',release='double_peak')},
            case_design=[dict(case_id=f'supp_{i*6+j+1:03d}',seed=931001+i*6+j,
                cohort='nominal',network=name,wind=wind,release=release,repeat=j)
                for i,(name,wind,release) in enumerate([
                    ('ring','steady','pulse'),('corridor','veering','double_peak')]) for j in range(6)])

    def test_design_retains_original_twelve_and_balances_48_each(self):
        previous=self.fixture()
        copy=deepcopy(previous)
        specs=extension.expanded_design(previous)
        self.assertEqual(previous,copy)
        self.assertEqual(specs[:12],previous['case_design'])
        self.assertEqual(len(specs),96)
        self.assertEqual(Counter(s['network'] for s in specs),{'ring':48,'corridor':48})
        self.assertEqual(len({s['seed'] for s in specs}),96)
        self.assertEqual(len({s['case_id'] for s in specs}),96)
        for network in previous['networks']:
            self.assertEqual(sorted(s['repeat'] for s in specs if s['network']==network),list(range(48)))
        self.assertEqual(extension.expanded_design(previous),specs)

    def test_partial_expansion_cannot_replace_the_final_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            study=root/'study'
            study.mkdir()
            (study/'protocol.json').write_text(json.dumps(dict(case_count=96)),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Complete all 96'):
                writing.build(study,root/'packet')
            self.assertFalse((root/'packet').exists())


if __name__=='__main__':
    unittest.main()
