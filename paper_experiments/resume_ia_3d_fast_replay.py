"""Resume the existing fixed-parameter replay using an audited QR factorization.

This changes neither the physical model nor quadrature nodes/weights. The
original kernel remains intact; the temporary substitution is process-scoped.
"""

import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from run_metec_comparison import ROOT, read, save
from refine_ia_3d_affine_numerics import DEST, run
from ia_3d_affine_fast import shared_gram_step


if __name__ == '__main__':
    audit = read(DEST / 'shared_gram_audit.json')
    if (not audit['passed'] or audit['local_covariance_translation_relative_error'] > 1e-10 or
            max(audit['sampled_response_relative_l2'], audit['sampled_jacobian_relative_l2']) > 1e-8):
        raise ValueError('Shared Gram factorization has not passed the fixed response audit')
    protocol = read(DEST / 'protocol.json')
    rows = []
    for case in protocol['refine_events']:
        folder = DEST / case
        if not (folder / 'ia_fit.json').exists():
            raise ValueError('Only resume replay after original-kernel fitting is saved')
        checkpoint = folder / 'replay_checkpoint.npz'
        first = 0
        if checkpoint.exists():
            with np.load(checkpoint) as old:
                first = int(old['minutes_done'])
        rows.append(dict(event_id=case, reference_prefix_minutes=first,
            fit_sha256=hashlib.sha256((folder / 'ia_fit.json').read_bytes()).hexdigest()))
    sources = [Path(__file__), ROOT / 'paper_experiments/ia_3d_affine_fast.py']
    manifest = dict(mode='Same-quadrature shared scalar Gram factorization for remaining replay only',
        audit_sha256=hashlib.sha256((DEST / 'shared_gram_audit.json').read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        reused_reference_prefix=rows, original_fit_not_retrained=True,
        nls_called=False, complete=False)
    save(DEST / 'replay_acceleration_manifest.json', manifest)
    with patch('ia_3d_affine_transport.general_step', shared_gram_step):
        run()
    manifest['complete'] = True
    save(DEST / 'replay_acceleration_manifest.json', manifest)
