"""Frozen small development panel for a location-oriented capacity selector."""

import gc
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_paper_updated_evidence import STUDY,synthetic_setup,prior
from run_metec_comparison import ROOT,OUT,read,save,load_input,JointBounds
from paper_beam_joint_baselines import BeamJointResidual
from paper_location_capacity import fit_location_capacity
from metec_point_adapter import xy_from_lon_lat,window_mean_release
from q_capacity import gaussian_bic

DEST=STUDY/'ia_location_capacity_pilot_20261028'


def variants(result,objective):
    models=result['release_models']
    bic=min(models,key=lambda model:gaussian_bic(model['sse'],objective.observations.numel(),4+model['q_basis_count']))
    secondary=dict(bic)
    theta=objective.tensor([*bic['source_norm'],bic['diffusion_norm'][1],
        bic['diffusion_norm'][0]-bic['diffusion_norm'][1]])
    secondary['prediction']=(objective.response(objective.model,objective.sigma,objective.xyt,
        objective.basis,physical_parameters=theta)@objective.tensor(bic['coefficients'])).detach().cpu().numpy()
    return [('ia_location_capacity',result),('ia_continuation_bic',secondary)]


def run():
    torch.set_num_threads(1)
    synthetic=[]
    geometries=set()
    for entry in read(STUDY/'new_comparison_v4/protocol.json')['cases']:
        if entry['case']['cohort']=='nominal' and entry['case']['network'] not in geometries:
            synthetic.append(entry)
            geometries.add(entry['case']['network'])
    public=[]
    heights=set()
    for identifier in read(OUT/'protocol.json')['event_ids']:
        height=read(OUT/'inputs'/identifier/'metadata.json')['source_height_m']
        if height not in heights:
            public.append(identifier)
            heights.add(height)
    public.append('20220512005')
    files=[Path(__file__),Path(__file__).with_name('paper_location_capacity.py')]
    protocol=dict(version='location_capacity_pilot_v1',synthetic=synthetic,public=public,
        selection='First nominal case per geometry; first public event per supplied height; plus one known fit/location paradox for diagnosis',
        diagnostic_public='20220512005',
        diagnostic_public_note='Development failure diagnosis, not independent validation',
        fit='Original mean-wind physics and original observation-only starts/bounds; Q1/3/5, nested continuation',
        selector='Working Gaussian fit compatibility alpha0.05, then minimum profiled local source variance',
        all_observations_retained=True,new_training_penalty=False,max_nfev=250,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if (DEST/'protocol.json').exists() and read(DEST/'protocol.json')!=protocol:
        raise ValueError('Changed location-capacity pilot; preserve the earlier version')
    save(DEST/'protocol.json',protocol)
    rows=[]
    for entry in synthetic:
        identifier=entry['case']['case_id']
        old,data,objective,starts=synthetic_setup(entry,'cpu')
        path=DEST/'synthetic'/identifier/'result.json'
        if path.exists():
            result=read(path)
        else:
            print('LOCATION_PILOT_START',identifier,flush=True)
            result=fit_location_capacity(objective,starts,float(old['metadata']['length_m']),max_nfev=250)
            save(path,result)
            print('LOCATION_PILOT_DONE',identifier,flush=True)
        for method,fit in variants(result,objective):
            metric=prior.formal.metrics(old['metadata'],data,fit['source_norm'],
                np.array(fit['coefficients']),np.array(fit['prediction']))
            rows.append(dict(dataset='synthetic',case_id=identifier,method=method,
                localization_error_m=metric['localization_error_m'],q_relative_l2=metric['q_relative_l2'],
                q_basis_count=fit['q_basis_count']))
        old_ia=read(STUDY/'ia_release_capacity_full_v1'/identifier/'result.json')['methods']['ia_vpdi_release_bic']
        for method,fit in [('original_ia',old_ia),('joint_nls',old['methods']['joint_nls_trf'])]:
            rows.append(dict(dataset='synthetic',case_id=identifier,method=method,
                localization_error_m=fit['localization_error_m'],q_relative_l2=fit['q_relative_l2'],
                q_basis_count=fit.get('q_basis_count',5)))
        del objective
        gc.collect()
    for identifier in public:
        data,meta,model,xyt,basis,response,observed=load_input(identifier,'cuda',dict(time=16,release=16,history_minutes=20))
        initial=read(OUT/'fits'/identifier/'initialization.json')
        objective=BeamJointResidual(response,model,0.,xyt,basis,observed,JointBounds(**initial['bounds']))
        path=DEST/'public'/identifier/'result.json'
        if path.exists():
            result=read(path)
        else:
            print('LOCATION_PILOT_START',identifier,flush=True)
            result=fit_location_capacity(objective,[np.array(start) for start in initial['starts']],100.,max_nfev=250)
            save(path,result)
            print('LOCATION_PILOT_DONE',identifier,flush=True)
        # True coordinates/rates are evaluated only after fitting and model selection.
        truth=read(OUT/'inputs'/identifier/'truth.json')
        true_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        comparisons=variants(result,objective)
        comparisons.extend((method,read(OUT/'fits'/identifier/(name+'.json'))) for method,name in
            [('original_ia','ia_vpdi'),('joint_nls','joint_nls_trf')])
        for method,fit in comparisons:
            rows.append(dict(dataset='public',case_id=identifier,method=method,
                localization_error_m=float(np.linalg.norm(np.array(fit['source_norm'])*100-true_xy)),
                q_kg_h=3600*window_mean_release(fit['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
                q_basis_count=fit.get('q_basis_count',5),
                diagnostic_only=identifier==protocol['diagnostic_public']))
        del objective,response,model
        gc.collect()
        torch.cuda.empty_cache()
    frame=pd.DataFrame(rows)
    frame.to_csv(DEST/'comparison.csv',index=False,encoding='utf-8-sig')
    save(DEST/'status.json',dict(pilot_complete=True,full_goal_achieved=False,
        note='Development panel only; no production promotion or general accuracy claim'))


if __name__=='__main__':
    run()
