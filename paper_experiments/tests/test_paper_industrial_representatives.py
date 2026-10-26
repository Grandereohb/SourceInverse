"""The bounded application must retain failure and never retry it."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'paper_experiments'))
import run_paper_industrial_representatives as application


class RepresentativeBudgetTests(unittest.TestCase):
    def test_failure_retained_and_remaining_cases_attempted_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            preparation=root/'industrial_preparation_v2_flags'
            preparation.mkdir()
            (preparation/'PREPARATION_FINISHED.json').write_text(json.dumps(
                dict(representative_case_ids=['ind_003','ind_017','ind_039'])),encoding='utf-8')
            failed=root/'industrial_fit_v2_scaled_nnls/ind_003'
            failed.mkdir(parents=True)
            (failed/'STOPPED.json').write_text(json.dumps(dict(message='nonfinite',type='RuntimeError')),
                                             encoding='utf-8')
            def freeze(preparation,preflight,output,mode,device):
                output.mkdir(exist_ok=True)
                return dict(case_ids=[f'ind_{i:03d}' for i in range(1,60)])
            def execute(case,preparation,output,settings):
                if case=='ind_017':
                    raise FloatingPointError('Retained, not retried')
                target=output/case
                target.mkdir()
                (target/'result.json').write_text(json.dumps(dict(methods={
                    'ia_vpdi':dict(numerical_gate_passed=True),
                    'matched_joint_adam':dict(numerical_gate_passed=True)})),encoding='utf-8')
            with patch.object(application.runner,'STUDY',root), \
                 patch.object(application.runner,'ROOT',root), \
                 patch.object(application.runner,'freeze',side_effect=freeze), \
                 patch.object(application.runner,'execute_case',side_effect=execute) as worker:
                application.run()
                application.run()
            self.assertEqual([call.args[0] for call in worker.call_args_list],['ind_017','ind_039'])
            output=root/'industrial_representative_application_v1'
            state=json.loads((output/'APPLICATION_ATTEMPTS_FINISHED.json').read_text())
            self.assertEqual(state['states'],dict(ind_003='retained_failure',
                ind_017='failed_no_retry',ind_039='completed'))
            self.assertFalse(state['all_eligible_cohort_complete'])
            self.assertFalse((output/'RUN_FINISHED.json').exists())


if __name__=='__main__':
    unittest.main()
