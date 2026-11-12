"""Audited same-quadrature acceleration of the frozen complete-cohort driver."""

import argparse
import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from run_metec_comparison import ROOT, OUT, read, save
from run_ia_3d_affine_full import DEST, WIND, run
from ia_3d_affine_transport import general_step
from ia_3d_affine_fast import shared_gram_step


def audit_wind():
    protocol = read(DEST / 'protocol.json')
    rows = []
    for case in protocol['cases']:
        wind = read(WIND / case / 'wind_fit.json')
        with np.load(OUT / 'inputs' / case / 'input.npz') as data:
            vectors = data['wind_xy'].copy()
        maximum = 0.
        for a, b, vector in zip(wind['gradient'], wind['intercept'], vectors):
            direction = vector / np.linalg.norm(vector) if np.linalg.norm(vector) else np.array([1., 0.])
            arguments = (np.asarray(a), np.asarray(b), np.linspace(0., 60., 33), direction)
            reference = general_step(*arguments, order=32)
            current = shared_gram_step(*arguments, order=32)
            for index in [0, 1, 2, 3, 4, 5]:
                x, y = reference[index], current[index]
                if index in [2, 3]:
                    x, y = x @ x.swapaxes(-1, -2), y @ y.swapaxes(-1, -2)
                maximum = max(maximum, float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30)))
        rows.append(dict(event_id=case, minutes=len(vectors), relative_error=maximum))
    if any(row['relative_error'] > 1e-10 for row in rows):
        raise ValueError('Shared Gram backend does not match the frozen local quadrature')
    save(DEST / 'shared_gram_wind_audit.json', dict(rows=rows, passed=True,
        source_truth_accessed=False, concentration_fits=False,
        scope='Local steps at all event wind records; long-history response is separately replayed'))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    rows = audit_wind()
    print('FULL_WIND_KERNEL_AUDIT', len(rows), max(row['relative_error'] for row in rows), flush=True)
    if not args.audit_only:
        protocol = read(DEST / 'protocol.json')
        previous = [DEST / case / 'ia_fit.json' for case in protocol['cases']]
        sources = [Path(__file__), ROOT / 'paper_experiments/ia_3d_affine_fast.py']
        manifest = dict(mode='Equivalent shared scalar Gram QR; same quadrature, physics, fitting protocol and cases',
            reused_fit_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in previous if p.exists()},
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            audit_sha256=hashlib.sha256((DEST / 'shared_gram_wind_audit.json').read_bytes()).hexdigest(),
            complete=False, nls_called=False)
        save(DEST / 'acceleration_manifest.json', manifest)
        with patch('ia_3d_affine_transport.general_step', shared_gram_step):
            run(fit_cases=True)
        manifest['complete'] = True
        save(DEST / 'acceleration_manifest.json', manifest)
