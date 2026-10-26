"""Two input-defined supplementary scenarios, six fresh seeds per scenario.

All cases are retained regardless of which method wins. They supplement, not
replace, the complete 120-case comparison. No production configuration changes.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import run_paper_shared_forward as shared
import analyze_paper_formal as analysis

STUDY = shared.formal.ROOT/'supplementary_scenarios_v1'


def scenarios():
    angles = np.linspace(0., 2*np.pi, 5, endpoint=False)
    ring = np.column_stack((-.2+.85*np.cos(angles), .85*np.sin(angles)))
    corridor = np.column_stack((np.linspace(0., 2., 6), [.15,-.15,.15,-.15,.15,-.15]))
    return {
        'surround5_pulse': dict(stations=ring.tolist(), length_m=1000., sigma=.15,
                               wind='steady', release='pulse'),
        'corridor6_double_peak': dict(stations=corridor.tolist(), length_m=1000., sigma=.15,
                                     wind='veering', release='double_peak')}


def prepare():
    definition = scenarios()
    specs = [dict(case_id=f'supp_{i*6+j+1:03d}', seed=931001+i*6+j,
        cohort='nominal', network=name, wind=network['wind'], release=network['release'],
        repeat=j) for i,(name, network) in enumerate(definition.items()) for j in range(6)]
    previous = shared.formal.ROOT/'evaluation_v3_shared_forward/protocol.json'
    protocol = deepcopy(json.loads(previous.read_text(encoding='utf-8')))
    protocol.update(case_design=specs, case_count=12, nominal_count=12,
        wind_mismatch_count=0, q_mismatch_count=0, networks=definition,
        role='Exploratory fresh-seed supplementary scenarios; not a replacement for the full cohort',
        scenario_question='Response to surrounding versus corridor monitoring, with prescribed release/wind combinations; not a one-factor causal comparison',
        selection='Input-defined before fitting; retain all 12 cases, no truth-based winner selection',
        stopping='One fixed-budget run per method and case; no ranking-based extensions')
    for path in [Path(__file__), Path(shared.__file__)]:
        protocol['frozen_files'][path.relative_to(shared.formal.base.ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    STUDY.mkdir(exist_ok=True)
    destination = STUDY/'protocol.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Supplement protocol changed; preserve previous version')
        return protocol
    shared.formal.base.save_json(destination, protocol)
    for spec in specs:
        case = STUDY/spec['case_id']
        case.mkdir()
        metadata, data = shared.formal.make_input(spec, definition[spec['network']])
        shared.formal.base.save_json(case/'input.json', metadata)
        np.savez_compressed(case/'input.npz', **data)
    return protocol


def run(device):
    protocol = prepare()
    for name, expected in protocol['frozen_files'].items():
        if hashlib.sha256((shared.formal.base.ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen dependency changed: {name}')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for spec in protocol['case_design']:
        case = STUDY/spec['case_id']
        if (case/'result.json').exists():
            continue
        print('START_SUPPLEMENT', spec['case_id'], spec['network'], flush=True)
        try:
            result = shared.fit_case(case, protocol, device)
        except Exception as exc:
            shared.formal.base.save_json(case/'STOPPED.json', dict(type=type(exc).__name__,message=str(exc)))
            raise
        print('DONE_SUPPLEMENT', spec['case_id'],
              {m: round(v['localization_error_m'],2) for m,v in result['methods'].items()},flush=True)
    analysis.analyze(STUDY)
    print('SUPPLEMENT_COMPLETE',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare_only else run(args.device)
