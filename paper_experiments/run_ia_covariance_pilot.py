"""Development pilot: independent pre-event covariance, not a new held-out objective."""

import argparse
import gc
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from run_metec_comparison import ROOT,OUT,AUDIT,read,save,load_input,JointBounds
from paper_observation_covariance import estimate_background_noise,StationAR1Whitening,CovarianceResponse
from paper_beam_joint_baselines import BeamJointResidual
from paper_joint_baselines import fit_joint_nls
from paper_q_model_selection import fit_profiled_q_models
from metec_point_adapter import xy_from_lon_lat,window_mean_release

DEST=OUT.parent/'ia_covariance_pilot_20261027'


def noise_table(identifier,raw):
    meta=read(OUT/'inputs'/identifier/'metadata.json')
    start=pd.Timestamp(meta['start_utc'])
    stations=meta['station_names']
    clock=pd.date_range(start-pd.Timedelta(minutes=30),start,freq='min',inclusive='left')
    table=raw.pivot_table(index='timestamp',columns='name',values='methane',aggfunc='mean')
    background=table.reindex(index=clock,columns=stations).to_numpy(copy=True)
    for index,name in enumerate(stations):
        if np.isfinite(background[:,index]).sum()<3:
            prior=table.loc[(table.index<start)&(table.index>=start-pd.Timedelta(hours=3)),name].dropna().tail(30)
            background[:,index]=np.nan
            background[-len(prior):,index]=prior.to_numpy()
    return estimate_background_noise(background)


def run(device):
    torch.set_num_threads(1)
    parent=read(OUT/'protocol.json')
    selected=[]
    heights=set()
    for identifier in parent['event_ids']:
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        if meta['source_height_m'] not in heights:
            selected.append(identifier)
            heights.add(meta['source_height_m'])
    sources=[Path(__file__),Path(__file__).with_name('paper_observation_covariance.py')]
    protocol=dict(version='covariance_pilot_v1',cases=selected,
        selection='First event per supplied source height in frozen chronological order; development, not independent validation',
        baseline='Original NLS and IA retained read-only; add matched GLS NLS to attribute the effect',
        calibration='Pre-event30min raw data only; same prior3h fallback if station absent',
        event_observations='All finite rows retained, no station/time withholding',
        likelihood='Stationary Gaussian AR1, independently calibrated station sigma/rho; no additive empirical penalty',
        covariance_comparison='IA GLS vs original NLS is a full-method comparison, not isolated Q-elimination evidence',
        max_nfev=250,rho_bound=.98,orders=parent['numerical_orders'],
        parent_protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Pilot changed; preserve the earlier version')
    save(DEST/'protocol.json',protocol)
    raw=pd.read_csv(AUDIT/'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp']=pd.to_datetime(raw.time,utc=True)-pd.Timedelta(minutes=1)
    rows=[]
    for identifier in selected:
        path=DEST/identifier
        data,meta,model,xyt,basis,response,observed=load_input(identifier,device,protocol['orders'])
        noise=noise_table(identifier,raw)
        indexes=np.flatnonzero(np.isfinite(data['concentration'].ravel()))
        count=data['concentration'].shape[1]
        whitening=StationAR1Whitening(indexes//count,indexes%count,noise['station_sigma'],noise['station_rho'])
        wrapped=CovarianceResponse(response,whitening)
        initialization=read(OUT/'fits'/identifier/'initialization.json')
        bounds=JointBounds(**initialization['bounds'])
        starts=[np.asarray(s) for s in initialization['starts']]
        save(path/'calibration.json',noise)
        for method in ['ia_gls','joint_nls_gls']:
            output=path/(method+'.json')
            if not output.exists():
                print('COVARIANCE_START',identifier,method,flush=True)
                objective=BeamJointResidual(wrapped,model,0.,xyt,basis,whitening.numpy(observed),bounds)
                fit=(fit_profiled_q_models(objective,starts,max_nfev=protocol['max_nfev'])
                     if method=='ia_gls' else fit_joint_nls(objective,starts,protocol['max_nfev']))
                fit['generalized_sse']=fit['sse']
                fit['generalized_mse']=fit['mse']
                fit['observation_model']='Independently calibrated station AR1 covariance; all event rows retained'
                fit['generalized_prediction']=fit.pop('prediction')
                theta=model.xs.new_tensor([*fit['source_norm'],fit['diffusion_norm'][1],
                    fit['diffusion_norm'][0]-fit['diffusion_norm'][1]])
                prediction=(response(model,0.,xyt,basis,physical_parameters=theta)@model.xs.new_tensor(fit['coefficients'])).cpu().numpy()
                fit.update(prediction=prediction,sse=float(np.sum((prediction-observed)**2)),
                    mse=float(np.mean((prediction-observed)**2)))
                save(output,fit)
                print('COVARIANCE_DONE',identifier,method,round(fit['seconds'],2),flush=True)
        # Evaluation-only truth access after both new solvers have completed.
        truth=read(OUT/'inputs'/identifier/'truth.json')
        true_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        duration=len(data['concentration'])*60.
        for method in ['ia_gls','joint_nls_gls','ia_vpdi','joint_nls_trf']:
            fit=read(path/(method+'.json')) if method.endswith('gls') else read(OUT/'fits'/identifier/(method+'.json'))
            q=3600*window_mean_release(fit['coefficients'],duration,data['release_window_s'])
            rows.append(dict(event_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.array(fit['source_norm'])*100-true_xy)),
                q_kg_h=q,true_q_kg_h=truth['mean_q_kg_h'],
                rmse_ppm=float(np.sqrt(fit['mse'])),converged=fit.get('converged'),
                q_basis_count=fit.get('q_basis_count',5)))
        del model,response,wrapped,objective
        gc.collect()
        if device=='cuda':
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    save(DEST/'status.json',dict(pilot_complete=True,full_goal_achieved=False,
        note='Two development events are not evidence of full4.2/4.4 superiority',cases=selected))


if __name__=='__main__':
    raise SystemExit('Archived matched-NLS control. NLS is frozen; use IA-only experiments.')
