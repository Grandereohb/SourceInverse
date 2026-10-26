"""Bounded application demonstration, not a replacement for cohort statistics.

The three cases were selected by input characteristics before any fitting.
Retain the stopped single-station case; run each remaining case once only.
"""

import hashlib
import json
from pathlib import Path

import torch

import run_paper_industrial_cases as runner


def run():
    torch.set_num_threads(1)
    preparation=runner.STUDY/'industrial_preparation_v2_flags'
    preflight=runner.STUDY/'industrial_transport_preflight_v1'
    output=runner.STUDY/'industrial_representative_application_v1'
    prior=runner.STUDY/'industrial_fit_v2_scaled_nnls'
    prepared=json.loads((preparation/'PREPARATION_FINISHED.json').read_text(encoding='utf-8'))
    selected=prepared['representative_case_ids']
    if selected!=['ind_003','ind_017','ind_039']:
        raise ValueError('Preselected representative identities changed')
    failure=prior/'ind_003/STOPPED.json'
    if not failure.exists():
        raise ValueError('The retained stopped result must exist before this bounded continuation')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    protocol=runner.freeze(preparation,preflight,output,'full',device)
    decision=dict(selected_case_ids=selected,original_eligible_cohort=len(protocol['case_ids']),
        scope='Three input-preselected application examples, NOT all-eligible aggregate evaluation',
        decision_timing='Budget stop after ind_003 failure, before fitting ind_017 or ind_039',
        no_retry=True,no_budget_extension=True,no_algorithm_change=True,
        retained_failure=str(failure.relative_to(runner.ROOT)),
        retained_failure_sha256=hashlib.sha256(failure.read_bytes()).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    decision_path=output/'representative_plan.json'
    if decision_path.exists():
        if json.loads(decision_path.read_text(encoding='utf-8'))!=decision:
            raise ValueError('Representative decision drift')
    else:
        runner.base.save_json(decision_path,decision)
    retained=output/'ind_003'
    retained.mkdir(exist_ok=True)
    runner.base.save_json(retained/'STOPPED.json',dict(
        **json.loads(failure.read_text(encoding='utf-8')),
        provenance=str(failure.relative_to(runner.ROOT)),not_rerun=True))
    states={selected[0]:'retained_failure'}
    for case_id in selected[1:]:
        target=output/case_id
        if (target/'STOPPED.json').exists():
            states[case_id]='failed_no_retry'
            continue
        if (target/'result.json').exists():
            record=json.loads((target/'result.json').read_text(encoding='utf-8'))
            if not all(v['numerical_gate_passed'] for v in record['methods'].values()):
                raise ValueError('Existing result has no passed numerical gate')
            states[case_id]='completed'
            continue
        print('START_REPRESENTATIVE',case_id,flush=True)
        try:
            runner.execute_case(case_id,preparation,output,protocol)
            states[case_id]='completed'
        except Exception as error:
            target.mkdir(exist_ok=True)
            runner.base.save_json(target/'STOPPED.json',dict(type=type(error).__name__,message=str(error)))
            states[case_id]='failed_no_retry'
        print('END_REPRESENTATIVE',case_id,states[case_id],flush=True)
        if device=='cuda':
            torch.cuda.empty_cache()
    runner.base.save_json(output/'APPLICATION_ATTEMPTS_FINISHED.json',dict(
        states=states,all_eligible_cohort_complete=False,
        scope='Bounded representative attempts ended; failures retained, not all successful fits'))


if __name__=='__main__':
    run()
