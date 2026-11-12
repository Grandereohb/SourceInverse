"""Two metadata-selected public events; measured wind ensemble vs original mean wind."""

import argparse
import gc
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_metec_comparison import ROOT,OUT,AUDIT,read,save,load_input,JointBounds
from ia_metec_point_adapter import configure,xy_from_lon_lat,window_mean_release
from paper_wind_ensemble import LocalWindEnsembleResponse
from paper_beam_joint_baselines import BeamJointResidual
from paper_joint_baselines import fit_joint_nls
from paper_q_model_selection import fit_profiled_q_models

DEST=OUT.parent/'ia_wind_ensemble_pilot_20261027'


def load_wind_members(raw,clock,names,global_wind):
    direction=np.deg2rad(raw['wind.direction'])
    vectors=raw.assign(u=-raw['wind.speed']*np.sin(direction),
        v=-raw['wind.speed']*np.cos(direction))
    grouped=vectors.groupby(['timestamp','name'])[['u','v']].mean()
    members=[]
    audit=[]
    for name in names:
        series=grouped.xs(name,level='name').sort_index()
        full_clock=pd.date_range(series.index.min(),series.index.max(),freq='min')
        original=series.reindex(full_clock)
        interpolated=original.interpolate(method='time',limit=5,limit_area='inside')
        values=interpolated.reindex(clock).to_numpy(copy=True)
        missing=~np.isfinite(values).all(axis=1)
        values[missing]=global_wind[missing]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite observed wind after declared fallback')
        audit.append(dict(station=name,global_fallback_minutes=int(missing.sum()),
            interpolated_minutes=int((original.reindex(clock).isna().any(axis=1)&~missing).sum())))
        members.append(values)
    return members,audit


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
    sources=[Path(__file__),Path(__file__).with_name('paper_wind_ensemble.py')]
    protocol=dict(version='observed_wind_ensemble_pilot_v1',cases=cases,
        case_selection='First event per supplied height in frozen chronological order; development pilot',
        wind_stations=['E','NW','SW'],weights='Normalized1/(source-to-wind-station distance squared +1m squared)',
        interpretation='Mixture of measured wind histories, not a spatially resolved wind PDE',
        missing_wind='Time-interpolate up to5 missing samples, then use original common global wind with explicit flags',
        likelihood='Unchanged unweighted squared residual; no empirical penalties or withholding',
        fit='IA BIC1/3/5 and matched fixed5 NLS; old starts, bounds, observations and quadratures',
        max_nfev=250,orders=parent['numerical_orders'],device=device,
        parent_sha256=hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Preserve previous pilot; changed protocol requires a new output directory')
    save(DEST/'protocol.json',protocol)
    raw=pd.read_csv(AUDIT/'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp']=pd.to_datetime(raw.time,utc=True)-pd.Timedelta(minutes=1)
    sensors=pd.read_csv(AUDIT/'raw/input_data/sensor_locations.csv').set_index('name')
    rows=[]
    for identifier in cases:
        folder=DEST/identifier
        data,meta,model,xyt,basis,original,observed=load_input(identifier,device,protocol['orders'])
        clock=pd.to_datetime(data['time_utc'],utc=True)
        winds,audit=load_wind_members(raw,clock,protocol['wind_stations'],data['wind_xy'])
        locations=sensors.loc[protocol['wind_stations']]
        positions=xy_from_lon_lat(locations.lon,locations.lat,meta['origin'])
        members=[configure(dict(data,wind_xy=wind),meta['source_height_m'],device=device,
            time_order=protocol['orders']['time'],release_order=protocol['orders']['release'],
            history_minutes=protocol['orders']['history_minutes'])[3] for wind in winds]
        response=LocalWindEnsembleResponse(members,positions)
        initialization=read(OUT/'fits'/identifier/'initialization.json')
        bounds=JointBounds(**initialization['bounds'])
        starts=[np.asarray(start) for start in initialization['starts']]
        save(folder/'wind_audit.json',dict(missing=audit,
            member_rms_vector_deviation_m_s=[float(np.sqrt(np.mean((wind-data['wind_xy'])**2))) for wind in winds]))
        for method in ['ia_wind_ensemble','joint_nls_wind_ensemble']:
            output=folder/(method+'.json')
            if not output.exists():
                print('WIND_START',identifier,method,flush=True)
                objective=BeamJointResidual(response,model,0.,xyt,basis,observed,bounds)
                fit=(fit_profiled_q_models(objective,starts,max_nfev=protocol['max_nfev'])
                    if method.startswith('ia_') else fit_joint_nls(objective,starts,protocol['max_nfev']))
                save(output,fit)
                print('WIND_DONE',identifier,method,round(fit['seconds'],2),flush=True)
        # Truth is accessed only after both fits; not used for wind weights or selection.
        truth=read(OUT/'inputs'/identifier/'truth.json')
        true_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        for method in ['ia_wind_ensemble','joint_nls_wind_ensemble','ia_vpdi','joint_nls_trf']:
            fit=read(folder/(method+'.json')) if 'ensemble' in method else read(OUT/'fits'/identifier/(method+'.json'))
            rows.append(dict(event_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.array(fit['source_norm'])*100-true_xy)),
                q_kg_h=3600*window_mean_release(fit['coefficients'],len(clock)*60.,data['release_window_s']),
                true_q_kg_h=truth['mean_q_kg_h'],rmse_ppm=float(np.sqrt(fit['mse'])),
                q_basis_count=fit.get('q_basis_count',5),converged=fit.get('converged')))
        del members,response,original,model,objective
        gc.collect()
        if device=='cuda':
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    save(DEST/'status.json',dict(pilot_complete=True,full_goal_achieved=False,
        note='Two development events cannot establish full cohort or solver-specific superiority'))


if __name__=='__main__':
    raise SystemExit('Archived matched-NLS control. NLS is frozen; use IA-only experiments.')
