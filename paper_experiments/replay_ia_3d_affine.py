"""Checkpointed fixed-parameter high-order replay of the 3D affine IA pilot."""

import gc
import hashlib
from pathlib import Path
import numpy as np
import torch
from run_metec_comparison import ROOT, OUT, read, save
from run_ia_3d_affine_pilot import DEST, WIND
from ia_3d_affine_adapter import configure_3d_affine


def run():
    torch.set_num_threads(1)
    pilot = read(DEST / 'protocol.json')
    for name, digest in pilot['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Pilot implementation changed: ' + name)
    for name, digest in read(ROOT / 'paper_experiments/nls_baseline_frozen.json')['sha256_lf'].items():
        if hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != digest:
            raise ValueError('Frozen NLS changed: ' + name)
    protocol = dict(cases=pilot['cases'], time_order=32, release_order=32,
        history_minutes=40, flow_order=32, batch_minutes=8, relative_threshold=.01,
        optimization=False, nls_called=False,
        pilot_protocol_sha256=hashlib.sha256((DEST / 'protocol.json').read_bytes()).hexdigest(),
        fit_sha256={case: hashlib.sha256((DEST / case / 'ia_fit.json').read_bytes()).hexdigest() for case in pilot['cases']},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path = DEST / 'replay_protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Preserve previous replay with different parameters')
    save(path, protocol)
    rows = []
    for identifier in protocol['cases']:
        folder = DEST / identifier
        completed = folder / 'replay.json'
        if completed.exists():
            rows.append({key: value for key, value in read(completed).items() if key != 'prediction'})
            continue
        data = dict(np.load(OUT / 'inputs' / identifier / 'input.npz'))
        meta = read(OUT / 'inputs' / identifier / 'metadata.json')
        record = read(WIND / identifier / 'wind_fit.json')
        fit = read(folder / 'ia_fit.json')
        velocity, gradient = np.asarray(record['intercept']), np.asarray(record['gradient'])
        checkpoint = folder / 'replay_checkpoint.npz'
        prediction = np.empty(0)
        first = 0
        if checkpoint.exists():
            with np.load(checkpoint) as old:
                if str(old['fit_sha256']) != protocol['fit_sha256'][identifier]:
                    raise ValueError('Replay checkpoint belongs to another fit')
                first = int(old['minutes_done'])
                prediction = old['prediction'].copy()
            expected = np.isfinite(data['concentration'][:first]).sum()
            if len(prediction) != expected:
                raise ValueError('Incomplete replay prefix')
        print('THREE_D_REPLAY_START', identifier, 'resume_minute', first, flush=True)
        for start in range(first, len(data['concentration']), protocol['batch_minutes']):
            end = min(start + protocol['batch_minutes'], len(data['concentration']))
            query = np.arange(start, end)
            model, xyt, basis, response, observed = configure_3d_affine(data, meta['source_height_m'],
                velocity, gradient, device='cuda', time_order=32, release_order=32,
                history_minutes=40, flow_order=32, observation_minutes=query)
            theta = model.xs.new_tensor([*fit['source_norm'], fit['diffusion_norm'][1],
                fit['diffusion_norm'][0] - fit['diffusion_norm'][1]])
            with torch.no_grad():
                value = (response(model, 0., xyt, basis, physical_parameters=theta) @
                         model.xs.new_tensor(fit['coefficients'])).cpu().numpy()
            if len(value) != len(observed) or not np.isfinite(value).all():
                raise FloatingPointError('Nonfinite or mismatched high-order response')
            prediction = np.r_[prediction, value]
            temporary = checkpoint.with_name('replay_checkpoint.tmp.npz')
            np.savez_compressed(temporary, minutes_done=end, prediction=prediction,
                                fit_sha256=protocol['fit_sha256'][identifier])
            temporary.replace(checkpoint)
            save(folder / 'replay_progress.json', dict(minutes_done=end,
                 total_minutes=len(data['concentration']), complete=False))
            print('THREE_D_REPLAY_BATCH', identifier, end, '/', len(data['concentration']), flush=True)
            del response, model
            gc.collect()
            torch.cuda.empty_cache()
        relative = float(np.linalg.norm(prediction - np.asarray(fit['prediction'])) /
                         max(np.linalg.norm(prediction), 1e-30))
        row = dict(event_id=identifier, relative_l2=relative,
                   passed=relative <= protocol['relative_threshold'], parameters_not_changed=True)
        save(completed, dict(**row, prediction=prediction))
        save(folder / 'replay_progress.json', dict(minutes_done=len(data['concentration']),
             total_minutes=len(data['concentration']), complete=True))
        checkpoint.unlink(missing_ok=True)
        rows.append(row)
        print('THREE_D_REPLAY_DONE', row, flush=True)
    save(DEST / 'replay_summary.json', dict(rows=rows, all_passed=all(row['passed'] for row in rows),
        interpretation='Numerical response consistency, not proof of physical validity or source accuracy'))
    status = read(DEST / 'status.json')
    status.update(numerical_replay_pending=False, numerically_passed=all(row['passed'] for row in rows))
    save(DEST / 'status.json', status)


if __name__ == '__main__':
    run()
