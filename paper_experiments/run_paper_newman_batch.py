"""Sequential author-method evaluation on the existing public-event cohort.

Run only after inspecting the pilot's completed chain. This driver does not
alter the author sampler, tune settings or add experimental conditions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'result/paper_experiments_20261017'


def planned_cases(protocol):
    cases = [(event['source'].replace('_', '')+'_'+event['release'], height)
             for event in protocol['events'] for height in protocol['heights_m']]
    if len(cases) != len(set(cases)):
        raise ValueError('Duplicate event-height in public protocol')
    return [(event, height, f'{event}_h{int(round(height*10)):02d}') for event, height in cases]


def run(output, public, python, iterations=10000):
    if iterations < 10000:
        raise ValueError('This formal driver requires the declared full sampling budget')
    output = Path(output).resolve()
    protocol_path = Path(public)/'protocol.json'
    public_protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    cases = planned_cases(public_protocol)
    if public_protocol['technical_smoke']:
        raise ValueError('Do not use a technical-smoke cohort as formal public validation')
    paths = [Path(__file__), *[Path(__file__).with_name(name) for name in (
        'run_paper_newman_event.py', 'paper_newman_adapter.py',
        'paper_newman_initialization.py', 'audit_paper_newman_result.py')]]
    protocol = dict(method='author sourceinversion0.1.5 Gaussian plume + M-MALA-within-Gibbs',
        iterations=iterations, burn_in=iterations//2, initial_step_size=.5,
        initialization='author likelihood selecting1000 SciPy LHS points, seed1729',
        events=[dict(event=e, height_m=h, directory=d) for e, h, d in cases],
        public_protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in paths},
        scope='same existing windows and both source heights, no new optimization conditions',
        posterior_warning='single fixed-budget chains, ESS/state-change audit required; no automatic convergence claim')
    output.mkdir(parents=True, exist_ok=True)
    destination = output/'batch_protocol.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8')) != protocol:
            raise ValueError('Preserve frozen author batch; changed protocol needs a new version')
    else:
        destination.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    log_dir = output/'logs'
    log_dir.mkdir(exist_ok=True)
    for event, height, directory in cases:
        target = output/directory
        result = target/'result.json'
        if result.exists():
            saved = json.loads(result.read_text(encoding='utf-8'))
            if (saved['event'] != event or saved['height_m'] != height or
                    saved['iterations'] != iterations or not saved['finite_required_chains'] or
                    saved.get('initialization', {}).get('method') !=
                    'SciPy LHS with unchanged author latin_hypercube_likelihood'):
                raise ValueError('Existing result is not a compatible full-window author run: '+directory)
        else:
            if target.exists():
                raise RuntimeError('Incomplete directory exists: verify its process before any manual recovery: '+directory)
            print('START_NEWMAN', directory, flush=True)
            with (log_dir/(directory+'.stdout.log')).open('w', encoding='utf-8') as stdout, \
                 (log_dir/(directory+'.stderr.log')).open('w', encoding='utf-8') as stderr:
                subprocess.run([str(python), str(Path(__file__).with_name('run_paper_newman_event.py')),
                    '--event', event, '--height', str(height), '--iterations', str(iterations),
                    '--initialization', 'author-lhs', '--output', str(target)],
                    cwd=ROOT, stdout=stdout, stderr=stderr, check=True)
        if not (target/'chain_audit.json').exists():
            with (log_dir/(directory+'.audit.log')).open('w', encoding='utf-8') as audit_log:
                subprocess.run([str(python), str(Path(__file__).with_name('audit_paper_newman_result.py')),
                    '--result', str(target)], cwd=ROOT, stdout=audit_log, stderr=subprocess.STDOUT, check=True)
        print('DONE_NEWMAN', directory, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=STUDY/'newman_lhs_full_v1')
    parser.add_argument('--public', type=Path, default=STUDY/'beam_comparison_v1')
    parser.add_argument('--python', type=Path, default=ROOT/'deployment_runs/newman_env/Scripts/python.exe')
    args = parser.parse_args()
    run(args.output, args.public, args.python)
