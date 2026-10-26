"""Audit a saved author chain and replay its physical parameter-mean estimate.

No sampler, priors, observations or saved chain is changed. Requires the isolated
author environment for the optional unchanged-posterior forward evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


MH_KEYS = ('source_x', 'source_y', 's', 'a_H', 'a_V', 'b_H', 'b_V')


def chain_audit(arrays, burn):
    count = len(arrays['source_x'])
    if not 0 <= burn < count-1:
        raise ValueError('Burn-in must leave at least two draws')
    report = {}
    for key in MH_KEYS:
        values = np.asarray(arrays[key], dtype=float).reshape(count, -1)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite author chain: {key}')
        retained = values[burn:]
        changes = np.any(retained[1:] != retained[:-1], axis=1)
        split = len(retained)//2
        report[key] = dict(range=np.ptp(retained, axis=0).tolist(),
            distinct_retained_states=len(np.unique(retained, axis=0)),
            retained_change_fraction=float(np.mean(changes)),
            first_half_mean=retained[:split].mean(axis=0).tolist(),
            second_half_mean=retained[split:].mean(axis=0).tolist())
    # Every M-MALA proposal updates one shared MH block; Gibbs updates do not
    # count as MH acceptance. Count changed MH states, excluding the first draw.
    block = np.column_stack([np.asarray(arrays[key]).reshape(count, -1) for key in MH_KEYS])
    report['mh_block'] = dict(
        all_draw_transition_fraction=float(np.mean(np.any(np.diff(block, axis=0) != 0, axis=1))),
        retained_transition_fraction=float(np.mean(np.any(np.diff(block[burn:], axis=0) != 0, axis=1))),
        interpretation='accepted state changes inferred from stored MH parameters; first transition unavailable')
    return report


def audit(directory, replay=True):
    from run_paper_newman_event import STUDY, build_event
    directory = Path(directory)
    result_path = directory/'result.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    arrays = np.load(directory/'chains.npz', allow_pickle=False)
    burn = result.get('burn_in', result['iterations']//2)
    report = dict(event=result['event'], height_m=result['height_m'], burn_in=burn,
        result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        chains_sha256=hashlib.sha256((directory/'chains.npz').read_bytes()).hexdigest(),
        chains=chain_audit(arrays, burn),
        effective_sample_size=result.get('effective_sample_size'),
        convergence_status='not certified; inspect ESS, state changes and trajectories',
        eligible_for_converged_posterior_ranking=False,
        reason='one fixed-budget chain has no independent-chain convergence assessment')
    if replay:
        inputs, weather = build_event(STUDY/'controlled_release_preparation_v2'/result['event'],
                                    result['height_m'], result['minutes'])
        import jax.numpy as jnp
        params = {}
        for key in MH_KEYS:
            mean = np.asarray(arrays[key][burn:]).mean(axis=0)
            params[key if key.startswith('source_') else 'log_'+key] = jnp.asarray(
                mean if key.startswith('source_') else np.log(mean))
        background = np.asarray(arrays['background'][burn:]).mean(axis=0)
        betas = jnp.asarray(np.repeat(background, len(weather)).reshape(-1, 1))
        priors = inputs['priors']
        _, response = inputs['log_posterior'](params,
            jnp.asarray(np.mean(arrays['sigma_squared'][burn:])), betas,
            jnp.asarray(priors.log_slab_var).reshape(1, 1),
            jnp.asarray(priors.log_slab_mean).reshape(1, 1),
            jnp.asarray(inputs['data']), priors, True, len(weather))
        prediction = np.asarray(response)*float(np.mean(arrays['s'][burn:]))+np.asarray(betas)
        observed = inputs['data']
        if prediction.shape != observed.shape or not np.isfinite(prediction).all():
            raise ValueError('Author posterior forward replay is invalid')
        report['parameter_mean_forward_rmse_ppm'] = float(np.sqrt(np.mean((prediction-observed)**2)))
        report['prediction_definition'] = 'forward at physical posterior parameter means; NOT posterior predictive mean'
        frame = pd.DataFrame(dict(time=np.tile(weather.index, 7),
            beam=np.repeat(np.arange(1, 8), len(weather)),
            observed_residual_ppm=observed.ravel()-1.92,
            predicted_residual_ppm=prediction.ravel()-1.92))
        frame.to_csv(directory/'parameter_mean_prediction.csv', index=False)
    (directory/'chain_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--no-replay', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.result, not args.no_replay), indent=2))
