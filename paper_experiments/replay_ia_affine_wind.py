"""Higher-order replay of frozen affine-wind pilot parameters, without refitting."""

import gc
import hashlib
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_ia_affine_wind_pilot import DEST,calibration,covariance_preflight
from run_metec_comparison import ROOT,OUT,AUDIT,read,save
from ia_metec_point_adapter import configure
from paper_sqrt_affine_wind import SquareRootAffineWindHistory
from run_ia_affine_wind_full import DEST as FULL,factor_preflight


def run(full=False):
    destination=FULL if full else DEST
    torch.set_num_threads(1)
    parent=read(destination/'protocol.json')
    if full:
        for name,digest in parent['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Frozen fitting implementation changed before replay: '+name)
    if full and any(not (destination/identifier/'complete.json').exists() for identifier in parent['cases']):
        raise ValueError('Finish the complete frozen cohort before replay')
    raw=pd.read_csv(AUDIT/'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp']=pd.to_datetime(raw.time,utc=True)-pd.Timedelta(minutes=1)
    sensors=pd.read_csv(AUDIT/'raw/input_data/sensor_locations.csv').set_index('name')
    protocol=dict(orders=[32,32],history_minutes=40,batch_minutes=8,square_root=full,
        comparison='Frozen16/16,20min-history fits; same full wind and Q clocks',
        threshold=.01,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parent_protocol_sha256=hashlib.sha256((destination/'protocol.json').read_bytes()).hexdigest(),
        wind_input_helper_sha256=hashlib.sha256((ROOT/'paper_experiments/run_ia_wind_ensemble_pilot.py').read_bytes()).hexdigest())
    if (destination/'replay_protocol.json').exists() and read(destination/'replay_protocol.json')!=protocol:
        raise ValueError('Preserve earlier replay; use a new output version')
    save(destination/'replay_protocol.json',protocol)
    snapshot=destination/'replay_source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for source in [Path(__file__),ROOT/'paper_experiments/run_ia_wind_ensemble_pilot.py']:
        if not (snapshot/source.name).exists():
            shutil.copy2(source,snapshot/source.name)
    rows=[]
    for identifier in parent['cases']:
        data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        history,audit=calibration(data,meta,raw,sensors)
        if full:
            history=SquareRootAffineWindHistory(history.edges,history.velocity,history.gradient,data['wind_xy'])
        fits={method:read(destination/identifier/(method+'.json')) for method in ['ia_affine_wind','joint_nls_affine_wind']}
        path=destination/identifier/'replay32.json'
        if path.exists():
            rows.extend(read(path))
            continue
        print('AFFINE_REPLAY_START',identifier,flush=True)
        predictions={key:[] for key in fits}
        try:
            for start in range(0,len(data['concentration']),8):
                model,xyt,basis,response,observed=configure(data,meta['source_height_m'],device='cuda',
                    time_order=32,release_order=32,history_minutes=40,transport_history=history,
                    observation_minutes=np.arange(start,min(start+8,len(data['concentration']))))
                (factor_preflight if full else covariance_preflight)(response.plan)
                for key,fit in fits.items():
                    dp,dq=fit['diffusion_norm']
                    theta=model.xs.new_tensor([*fit['source_norm'],dq,dp-dq])
                    value=(response(model,0.,xyt,basis,physical_parameters=theta)@
                        model.xs.new_tensor(fit['coefficients'])).cpu().numpy()
                    if not np.isfinite(value).all():
                        raise FloatingPointError('Nonfinite affine replay response')
                    predictions[key].append(value)
                del model,response
        except FloatingPointError as exc:
            save(destination/identifier/'replay_failure.json',dict(error=str(exc),passed=False,
                no_clipping_applied=True,numerical_or_affine_extrapolation_issue=True))
            rows.extend(dict(event_id=identifier,method=key,passed=False,error=str(exc)) for key in fits)
            gc.collect()
            torch.cuda.empty_cache()
            continue
        event=[]
        observed=data['concentration'].ravel()
        observed=observed[np.isfinite(observed)]
        for key,fit in fits.items():
            prediction=np.concatenate(predictions[key])
            delta=float(np.linalg.norm(prediction-np.array(fit['prediction']))/max(np.linalg.norm(prediction),1e-30))
            event.append(dict(event_id=identifier,method=key,relative_l2=delta,passed=delta<=.01,
                high_order_rmse=float(np.sqrt(np.mean((prediction-observed)**2)))))
        np.savez_compressed(destination/identifier/'replay32_predictions.npz',
            **{key:np.concatenate(parts) for key,parts in predictions.items()})
        save(path,event)
        rows.extend(event)
        print('AFFINE_REPLAY_DONE',identifier,event,flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    save(destination/'replay_summary.json',dict(complete=len(rows)==2*len(parent['cases']),passed=all(row['passed'] for row in rows),
        rows=rows,full_goal_achieved=False))
    if full:
        status=read(destination/'status.json')
        status.update(replay_pending=False,ready=bool(len(rows)==2*len(parent['cases']) and all(row['passed'] for row in rows)),
            numerical_status='passed' if all(row['passed'] for row in rows) else 'warnings_retained')
        save(destination/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true')
    run(parser.parse_args().full)
