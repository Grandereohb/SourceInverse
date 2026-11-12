"""Matched high-order local refit of numerically sensitive IA/NLS solutions."""

import gc
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_metec_comparison import ROOT,OUT,read,save,load_input,JointBounds
from paper_beam_joint_baselines import BeamJointResidual
from paper_joint_baselines import fit_joint_nls
from paper_reduced_solver import fit_profiled_trf
from metec_point_adapter import xy_from_lon_lat,window_mean_release

DEST=OUT.parent/'metec_precision_refit_20261027'


def run():
    torch.set_num_threads(1)
    cases=[]
    parent=read(OUT/'protocol.json')
    for identifier in parent['event_ids']:
        replay=read(OUT/'replay'/(identifier+'.json'))
        if any(not replay[key]['passed'] for key in ['ia_vpdi_0','joint_nls_trf_0']):
            cases.append(identifier)
    protocol=dict(version='matched_precision_refit_v1',cases=cases,
        orders=dict(time=32,release=32,history_minutes=40),q_count=5,max_nfev=100,
        starts='Both saved IA and NLS physical parameters and amplitudes, shared by both solvers; no truth',
        scope='Local numerical refinement, not a fresh independent benchmark or exhaustive global solve',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Changed refit protocol; preserve the prior results')
    save(DEST/'protocol.json',protocol)
    rows=[]
    for identifier in cases:
        data,meta,model,xyt,basis,response,observed=load_input(identifier,'cuda',protocol['orders'])
        initial=read(OUT/'fits'/identifier/'initialization.json')
        bounds=JointBounds(**initial['bounds'])
        starts=[]
        for method in ['ia_vpdi','joint_nls_trf']:
            saved=read(OUT/'fits'/identifier/(method+'.json'))
            starts.append(np.r_[saved['source_norm'],saved['diffusion_norm'][1],
                saved['diffusion_norm'][0]-saved['diffusion_norm'][1],saved['coefficients']])
        for method in ['ia_vpdi','joint_nls_trf']:
            output=DEST/identifier/(method+'.json')
            if not output.exists():
                print('PRECISION_REFIT_START',identifier,method,flush=True)
                objective=BeamJointResidual(response,model,0.,xyt,basis,observed,bounds)
                fit=(fit_profiled_trf(objective,starts,protocol['max_nfev']) if method=='ia_vpdi'
                    else fit_joint_nls(objective,starts,protocol['max_nfev']))
                save(output,fit)
                print('PRECISION_REFIT_DONE',identifier,method,round(fit['seconds'],1),flush=True)
        truth=read(OUT/'inputs'/identifier/'truth.json')
        true_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        for method in ['ia_vpdi','joint_nls_trf']:
            for variant,path in [('original',OUT/'fits'/identifier/(method+'.json')),
                ('high_order_refit',DEST/identifier/(method+'.json'))]:
                fit=read(path)
                rows.append(dict(event_id=identifier,method=method,variant=variant,
                    localization_error_m=float(np.linalg.norm(np.asarray(fit['source_norm'])*100-true_xy)),
                    q_kg_h=3600*window_mean_release(fit['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                    true_q_kg_h=truth['mean_q_kg_h'],rmse_ppm=float(np.sqrt(fit['mse'])),
                    converged=fit.get('converged'),seconds=fit['seconds']))
        del model,response,objective
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    save(DEST/'status.json',dict(fits_complete=True,full_goal_achieved=False,
        replay_after_refit_pending=True,note='64-order saved-parameter replay remains required'))


if __name__=='__main__':
    raise SystemExit('Archived matched-NLS control. NLS is frozen; use IA-only experiments.')
