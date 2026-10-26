"""Versioned continuation after an independently reproduced roundoff failure.

Preserves evaluation_v1. Same data and settings, unchanged IA/joint code.
Existing IA/joint results are reused transparently; ALL puff demos are rerun.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np

import run_paper_formal_experiments as formal
from paper_bounded_baseline import fit_demo

V1 = formal.STUDY
V2 = formal.ROOT/'evaluation_v2_boundary_fix'


def prepare():
    original = json.loads((V1/'protocol.json').read_text(encoding='utf-8'))
    protocol = deepcopy(original)
    protocol.update(previous_version='evaluation_v1',
        correction='Powell isotropy boundary, <=16 float64 ULPs only; no data/priors/objective/budget changes',
        reuse='IA/joint+diagnostics+traces from completed v1 cases, unchanged code; rerun ALL independent puff fits')
    for name in ('resume_paper_formal_boundary_fix.py', 'paper_bounded_baseline.py'):
        path = Path(__file__).with_name(name)
        protocol['frozen_files'][path.relative_to(formal.base.ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    V2.mkdir(exist_ok=True)
    destination = V2/'protocol.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != protocol:
            raise ValueError('V2 source/protocol changed; preserve version integrity')
    else:
        formal.base.save_json(destination, protocol)
    for spec in protocol['case_design']:
        old, new = V1/spec['case_id'], V2/spec['case_id']
        new.mkdir(exist_ok=True)
        for name in ('input.json', 'input.npz'):
            if not (new/name).exists():
                shutil.copy2(old/name, new/name)
            if (old/name).read_bytes() != (new/name).read_bytes():
                raise ValueError(f'Input changed across versions: {spec["case_id"]}/{name}')
    return protocol


def run(device, limit):
    protocol = prepare()
    for name, expected in protocol['frozen_files'].items():
        if hashlib.sha256((formal.base.ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source changed: {name}')
    # Reuse only unaffected optimizers, never an old baseline under a new label.
    for spec in protocol['case_design'][:limit]:
        old, new = V1/spec['case_id'], V2/spec['case_id']
        if (new/'result.json').exists() or not (old/'result.json').exists():
            continue
        result = json.loads((old/'result.json').read_text(encoding='utf-8'))
        metadata = result['metadata']
        data = np.load(new/'input.npz')
        baseline = fit_demo(data['stations'], data['observed'], metadata['wind'],
            protocol['puff_powell_max_evaluations_per_start'], sigma=metadata['sigma'],
            search_step=protocol['search_step_norm'])
        result['methods']['independent_puff_demo'] = dict(**baseline, **formal.metrics(
            metadata, data, baseline['source_norm'], baseline['coefficients'], np.asarray(baseline['prediction'])))
        result['reuse_provenance'] = dict(ia_joint_and_information_from=str(old.relative_to(formal.base.ROOT)),
            original_result_sha256=hashlib.sha256((old/'result.json').read_bytes()).hexdigest(),
            independent_puff='rerun with roundoff boundary correction')
        shutil.copy2(old/'optimization_traces.npz', new/'optimization_traces.npz')
        formal.base.save_json(new/'result.json', result)
        print('BASELINE RERUN + VERIFIED IA/JOINT REUSE', spec['case_id'], flush=True)
    with patch.object(formal, 'STUDY', V2), patch.object(formal, 'fit_demo', fit_demo):
        formal.run(device, limit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--limit', type=int, default=120)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
        print('V2 inputs copied exactly; protocol prepared; no inference run')
    else:
        run(args.device, args.limit)
