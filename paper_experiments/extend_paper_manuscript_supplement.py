"""Expand each supplementary scenario to 48 cases without outcome selection."""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import torch

import run_paper_manuscript_supplement as original

ROOT = original.shared.formal.ROOT
STUDY = ROOT/'supplementary_scenarios_v2_48each'
PACKET = ROOT/'manuscript_results_v2'
FILES = ('input.json', 'input.npz', 'result.json', 'optimization_traces.npz')


def expanded_design(previous):
    specs = deepcopy(previous['case_design'])
    counts = Counter(s['network'] for s in specs)
    if set(counts.values()) != {6} or len(counts) != 2:
        raise ValueError('Expected the complete initial six-case design for both scenarios')
    for name, network in previous['networks'].items():
        for repeat in range(6, 48):
            index = len(specs)
            specs.append(dict(case_id=f'supp_{index+1:03d}', seed=932001+index-12,
                cohort='nominal', network=name, wind=network['wind'],
                release=network['release'], repeat=repeat))
    return specs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete(case):
    path=case/'result.json'
    return path.exists() and json.loads(path.read_text(encoding='utf-8')).get('comparison_version')=='shared_forward_v3'


def prepare(study=STUDY):
    prior = original.STUDY
    previous = json.loads((prior/'protocol.json').read_text(encoding='utf-8'))
    for name, expected in previous['frozen_files'].items():
        if digest(original.shared.formal.base.ROOT/name) != expected:
            raise ValueError(f'Original dependency changed: {name}')
    protocol = deepcopy(previous)
    protocol.update(case_design=expanded_design(previous), case_count=96, nominal_count=96,
        previous_version=prior.name, reused_cases=12, additional_cases=84,
        role='Exploratory extension to48 cases per input-defined scenario; initial12 retained',
        selection='All48 seeds per scenario retained; no case exclusion by errors or rankings',
        stopping='Same fixed budgets as original study; no result-dependent reruns',
        expansion_decision='2026-10-20; increased sample size after inspecting the initial12 results; not a preregistered independent replication')
    for name in ('extend_paper_manuscript_supplement.py','build_paper_writing_data.py',
                 'build_paper_results_summary.py'):
        path = Path(__file__).with_name(name)
        protocol['frozen_files'][path.relative_to(original.shared.formal.base.ROOT).as_posix()] = digest(path)
    study.mkdir(exist_ok=True)
    destination = study/'protocol.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Expansion protocol changed; preserve the frozen version')
        return protocol
    original.shared.formal.base.save_json(destination, protocol)
    provenance = []
    old_ids = {s['case_id'] for s in previous['case_design']}
    for spec in protocol['case_design']:
        case = study/spec['case_id']
        case.mkdir()
        if spec['case_id'] in old_ids:
            source = prior/spec['case_id']
            for name in FILES:
                shutil.copy2(source/name, case/name)
                if digest(source/name) != digest(case/name):
                    raise ValueError('Reused case copy differs from original')
                provenance.append(dict(path=(case/name).relative_to(study).as_posix(),
                    source=(source/name).relative_to(ROOT).as_posix(), sha256=digest(case/name)))
        else:
            metadata, data = original.shared.formal.make_input(spec, protocol['networks'][spec['network']])
            original.shared.formal.base.save_json(case/'input.json', metadata)
            np.savez_compressed(case/'input.npz', **data)
    hashes = {f'{s["case_id"]}/{name}':digest(study/s['case_id']/name)
              for s in protocol['case_design'] for name in ('input.json','input.npz')}
    original.shared.formal.base.save_json(study/'input_hash_manifest.json', dict(sha256=hashes))
    original.shared.formal.base.save_json(study/'reuse_provenance.json', provenance)
    with zipfile.ZipFile(study/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in protocol['frozen_files']:
            archive.write(original.shared.formal.base.ROOT/name, name)
    return protocol


def status(study, protocol, state, error=None):
    completed = [s['case_id'] for s in protocol['case_design'] if complete(study/s['case_id'])]
    original.shared.formal.base.save_json(study/'run_status.json', dict(
        state=state, planned=96, completed=len(completed), reused=12,
        updated_at=datetime.now().isoformat(timespec='seconds'), error=error))


def run(device, bundle_python=None, bundle_node=None, artifact_marker=None):
    protocol = prepare()
    hashes = json.loads((STUDY/'input_hash_manifest.json').read_text(encoding='utf-8'))['sha256']
    if any(digest(STUDY/name)!=value for name,value in hashes.items()):
        raise ValueError('Frozen input changed')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    status(STUDY, protocol, 'running')
    try:
        for spec in protocol['case_design']:
            case = STUDY/spec['case_id']
            if complete(case):
                continue
            print('START_EXPANSION', spec['case_id'], spec['network'], flush=True)
            original.shared.fit_case(case, protocol, device)
            status(STUDY, protocol, 'running')
            print('DONE_EXPANSION', spec['case_id'], flush=True)
        original.analysis.analyze(STUDY)
        import build_paper_writing_data as writing
        writing.build(supplement=STUDY, output=PACKET)
        if bundle_python:
            if artifact_marker:
                subprocess.run([str(bundle_node),str(artifact_marker),'--operation-kind','create',
                    '--expected-output-count','1','--output-format','docx'],check=True)
            subprocess.run([str(bundle_python), str(Path(__file__).with_name('build_paper_results_summary.py')),
                '--packet', str(PACKET), '--date', '2026-10-20'], check=True)
        status(STUDY, protocol, 'completed_report_needs_visual_review' if bundle_python else 'completed_data')
        print('EXPANSION_COMPLETE', flush=True)
    except Exception as exc:
        status(STUDY, protocol, 'failed', dict(type=type(exc).__name__, message=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--bundle-python', type=Path)
    parser.add_argument('--bundle-node', type=Path)
    parser.add_argument('--artifact-marker', type=Path)
    args = parser.parse_args()
    if args.artifact_marker and not args.bundle_node:
        parser.error('--artifact-marker requires --bundle-node')
    prepare() if args.prepare_only else run(args.device, args.bundle_python,args.bundle_node,args.artifact_marker)
