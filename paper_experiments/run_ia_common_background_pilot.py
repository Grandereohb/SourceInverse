"""Fixed two-event IA pilot: eliminate Q and a common nonnegative background."""

import gc
import hashlib
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from run_metec_comparison import ROOT,OUT,STUDY,read,save,load_input,JointBounds
from paper_beam_joint_baselines import BeamJointResidual
from paper_location_capacity import capacity_response
from ia_common_background import fit_common_background
from q_capacity import gaussian_bic
from metec_point_adapter import xy_from_lon_lat,window_mean_release

DEST=STUDY/'ia_common_background_pilot_20261028'


def run():
    torch.set_num_threads(1)
    frozen=read(ROOT/'paper_experiments/nls_baseline_frozen.json')
    for name,digest in frozen['sha256_lf'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            raise ValueError('NLS baseline changed: '+name)
    parent=read(OUT/'protocol.json')
    cases=read(STUDY/'ia_location_capacity_pilot_20261028/protocol.json')['public'][:2]
    sources=[Path(__file__),ROOT/'paper_experiments/ia_common_background.py',
             ROOT/'paper_experiments/paper_location_capacity.py']
    protocol=dict(cases=cases,
        selection='Same fixed first two public development events, no source-truth selection',
        measurement='Raw concentration=source+nonnegative common background per minute+fixed sensor offset',
        offsets='Original independent pre-event q20 station offsets relative to their median; no quantile search',
        fit='All valid rows, original source response/starts/bounds/Qcap; scaled reduced TRF, nfev250',
        inner='Convex Q>=0 and background>=0; group projection with background active constraints',
        capacity='Q1/3/5 Gaussian working BIC, nominal parameter count4+K+number_of_observed_times',
        risks='Common background can absorb source information; unrestricted time variation is not a calibrated ambient model',
        nls='Frozen original outputs only, no runs or modifications',orders=parent['numerical_orders'],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Preserve previous common-background pilot')
    save(DEST/'protocol.json',protocol)
    for source in sources:
        target=DEST/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            shutil.copy2(source,target)
    rows=[]
    for identifier in cases:
        data,meta,model,xyt,basis,response,_=load_input(identifier,'cuda',protocol['orders'])
        indices=np.flatnonzero(np.isfinite(data['raw_concentration'].ravel()))
        stations=data['raw_concentration'].shape[1]
        offsets=data['baseline']-np.median(data['baseline'])
        raw=data['raw_concentration'].ravel()[indices]
        observed=raw-offsets[indices%stations]
        times=indices//stations
        initial=read(OUT/'fits'/identifier/'initialization.json')
        objective=BeamJointResidual(response,model,0.,xyt,basis,observed,JointBounds(**initial['bounds']))
        output=DEST/identifier/'ia_fit.json'
        if output.exists():
            fit=read(output)
        else:
            models=[]
            for count in [1,3,5]:
                print('COMMON_BG_START',identifier,count,flush=True)
                function,mapping=capacity_response(objective,count)
                fit=fit_common_background(function,observed,times,
                    [np.array(start)[:4] for start in initial['starts']],
                    objective.bounds.lower[:4],objective.bounds.upper[:4],float(objective.bounds.upper[4]),max_nfev=250)
                theta=fit.pop('theta')
                native=fit['coefficients'].copy()
                fit.update(source_norm=theta[:2],diffusion_norm=[theta[2]+theta[3],theta[2]],
                    native_coefficients=native,coefficients=mapping@native,q_basis_count=count,
                    background_time_indices=np.unique(times),fixed_sensor_offsets_ppm=offsets,
                    raw_prediction=fit['prediction']+offsets[indices%stations],
                    mse=fit['sse']/len(observed),bic=gaussian_bic(fit['sse'],len(observed),4+count+fit['background_groups']))
                models.append(fit)
                print('COMMON_BG_MODEL_DONE',identifier,count,fit['sse'],flush=True)
            fit=dict(min(models,key=lambda item:(item['bic'],item['q_basis_count'])))
            fit['release_models']=[{key:item[key] for key in
                ['q_basis_count','source_norm','diffusion_norm','coefficients','sse','bic','converged','seconds']} for item in models]
            save(output,fit)
        truth=read(OUT/'inputs'/identifier/'truth.json')
        xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        for method,item in [('ia_common_background',fit),('original_ia',read(OUT/'fits'/identifier/'ia_vpdi.json')),
                            ('frozen_nls',read(OUT/'fits'/identifier/'joint_nls_trf.json'))]:
            rows.append(dict(event_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.array(item['source_norm'])*100-xy)),
                q_kg_h=3600*window_mean_release(item['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                true_q_kg_h=truth['mean_q_kg_h'],raw_rmse_ppm=float(np.sqrt(item['mse'])),
                q_basis_count=item.get('q_basis_count',5),converged=item.get('converged')))
        print('COMMON_BG_DONE',identifier,rows[-3],flush=True)
        pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
        del objective,response,model
        gc.collect()
        torch.cuda.empty_cache()
    save(DEST/'status.json',dict(pilot_complete=True,nls_called=False,production_promoted=False,
        numerical_replay_pending=True,full_goal_achieved=False))


if __name__=='__main__':
    run()
