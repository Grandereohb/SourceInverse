"""One-ULP-safe isotropy boundary for the frozen independent puff demo.

SciPy Powell can evaluate a tiny negative diffusion contrast at a closed lower
bound. Only machine-roundoff violations are projected; real violations raise.
The original frozen demo and its ordinary feasible responses remain unchanged.
"""

from unittest.mock import patch

import numpy as np
import paper_baseline_demos as original

ORIGINAL_MEAN = original.mean_response


def roundoff_safe_mean_response(stations, source, d_parallel, d_perp, wind,
                               *args, audit=None, **kwargs):
    if np.isfinite(d_parallel) and np.isfinite(d_perp) and 0 < d_perp > d_parallel:
        gap = float(d_perp-d_parallel)
        tolerance = 16 * np.spacing(max(abs(float(d_parallel)), abs(float(d_perp))))
        if gap <= tolerance:
            d_parallel = d_perp
            if audit is not None:
                audit['corrected_evaluations'] += 1
                audit['maximum_diffusion_correction'] = max(audit['maximum_diffusion_correction'], gap)
    return ORIGINAL_MEAN(stations, source, d_parallel, d_perp, wind, *args, **kwargs)


def fit_demo(*args, **kwargs):
    audit = dict(corrected_evaluations=0, maximum_diffusion_correction=0.,
                 rule='only D_parallel<D_perp by <=16 float64 ULPs at the isotropy boundary')
    def response(*values, **options):
        return roundoff_safe_mean_response(*values, audit=audit, **options)
    with patch.object(original, 'mean_response', response):
        result = original.fit_demo(*args, **kwargs)
    result['diffusion_norm'][0] = max(result['diffusion_norm'])
    return dict(result, boundary_roundoff_audit=audit)
