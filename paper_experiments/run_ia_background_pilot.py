"""Frozen small test of jointly eliminated source and shared ambient background."""

import argparse
import gc
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_metec_comparison import ROOT,OUT,AUDIT,read,save,load_input,JointBounds
from paper_beam_joint_baselines import BeamJointResidual
from paper_linear_background import background_design,fit_source_and_background
from metec_point_adapter import xy_from_lon_lat,window_mean_release

DEST=OUT.parent/'ia_linear_background_pilot_20261027'


def pre_event(identifier,table):
    meta=read(OUT/'inputs'/identifier/'metadata.json')
    start=pd.Timestamp(meta['start_utc'])
    clock=pd.date_range(start-pd.Timedelta(minutes=30),start,freq='min',inclusive='left')
    values=table.reindex(index=clock,columns=meta['station_names']).to_numpy(copy=True)
    for index,name in enumerate(meta['station_names']):
        if not np.isfinite(values[:,index]).any():
            prior=table.loc[(table.index<start)&(table.index>=start-pd.Timedelta(hours=3)),name].dropna().tail(30)
            values[-len(prior):,index]=prior.to_numpy()
    return values


def run(device):
    torch.set_num_threads(1)
    parent=read(OUT/'protocol.json')
    cases=[]
    heights=set()
    for identifier in parent['event_ids']:
        height=read(OUT/'inputs'/identifier/'metadata.json')['source_height_m']
        if height not in heights:
            cases.append(identifier)
            heights.add(height)
    files=[Path(__file__),Path(__file__).with_name('paper_linear_background.py'),ROOT/'pinn_source/reduced_trust_region.py']
    protocol=dict(version='linear_background_pilot_v1',cases=cases,
        selection='Same first event per known source height as covariance pilot; development only',
        source='Same physical operator, bounds and geometry starts; Q1/3/5 selected by working BIC',
        background='Three linear hats common to all stations; fixed station offsets from independent pre-event medians',
        calibration='All finite pre-event30min measurements used as source-zero linear rows, no event withholding',
        objective='One Gaussian data likelihood, no extra empirical penalty or Q smoothing',
        max_nfev=250,orders=parent['numerical_orders'],
        parent_protocol_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Changed background pilot; preserve prior results')
    save(DEST/'protocol.json',protocol)
    raw=pd.read_csv(AUDIT/'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp']=pd.to_datetime(raw.time,utc=True)-pd.Timedelta(minutes=1)
    table=raw.pivot_table(index='timestamp',columns='name',values='methane',aggfunc='mean')
    rows=[]
    for identifier in cases:
        path=DEST/identifier/'ia_background.json'
        data,meta,model,xyt,basis,response,_=load_input(identifier,device,protocol['orders'])
        mask=np.isfinite(data['raw_concentration'].ravel())
        observed=data['raw_concentration'].ravel()[mask]
        design=background_design(data['raw_concentration'].shape,np.flatnonzero(mask),pre_event(identifier,table))
        if not path.exists():
            init=read(OUT/'fits'/identifier/'initialization.json')
            objective=BeamJointResidual(response,model,0.,xyt,basis,observed,JointBounds(**init['bounds']))
            print('BACKGROUND_START',identifier,flush=True)
            fit=fit_source_and_background(objective,[np.asarray(s) for s in init['starts']],design,
                max_nfev=protocol['max_nfev'])
            save(path,fit)
            print('BACKGROUND_DONE',identifier,fit['q_basis_count'],flush=True)
        truth=read(OUT/'inputs'/identifier/'truth.json')
        true_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        for method in ['ia_background','ia_vpdi','joint_nls_trf']:
            fit=read(path) if method=='ia_background' else read(OUT/'fits'/identifier/(method+'.json'))
            q=window_mean_release(fit['coefficients'],len(data['concentration'])*60.,data['release_window_s'])*3600
            rows.append(dict(event_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.array(fit['source_norm'])*100-true_xy)),
                q_kg_h=q,true_q_kg_h=truth['mean_q_kg_h'],rmse_ppm=float(np.sqrt(fit['mse'])),
                q_basis_count=fit.get('q_basis_count',5),converged=fit.get('converged')))
        del model,response
        gc.collect()
        if device=='cuda':
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    save(DEST/'status.json',dict(pilot_complete=True,full_goal_achieved=False,
        note='Two development cases; no full-cohort or significant-advantage claim'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    run(parser.parse_args().device)
