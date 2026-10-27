"""Post-fit truth evaluation and numerical replay of the complete frozen METEC cohort."""

import argparse
import gc
import hashlib
from importlib.metadata import version
import time

import numpy as np
import pandas as pd
import torch

from run_metec_comparison import ROOT,OUT,METHODS,read,save,load_input,xy_from_lon_lat
from metec_point_adapter import window_mean_release


def evaluate(device='cuda'):
    protocol=read(OUT/'protocol.json')
    for name,digest in protocol['input_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Public input changed: '+name)
    missing=[k for k in protocol['event_ids'] if not (OUT/'fits'/k/'complete.json').exists()]
    if missing:
        raise ValueError('Full cohort is not complete: '+','.join(missing))
    torch.set_num_threads(1)
    rows=[]
    for identifier in protocol['event_ids']:
        meta=read(OUT/'inputs'/identifier/'metadata.json')
        truth=read(OUT/'inputs'/identifier/'truth.json')
        with np.load(OUT/'inputs'/identifier/'input.npz') as input_data:
            release_window=input_data['release_window_s'].copy()
            duration=len(input_data['concentration'])*60.
        truth_xy=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]
        replay_path=OUT/'replay'/f'{identifier}.json'
        replay=read(replay_path) if replay_path.exists() else {}
        data=meta_new=model=xyt=basis=response=observed=None
        for method in METHODS:
            saved=read(OUT/'fits'/identifier/(method+'.json'))
            fits=saved.get('replicates',[saved])
            for repeat,fit in enumerate(fits):
                key=f'{method}_{repeat}'
                if key not in replay:
                    if model is None:
                        data,meta_new,model,xyt,basis,response,observed=load_input(identifier,device,protocol['replay_orders'])
                    parallel,perpendicular=fit['diffusion_norm']
                    theta=model.xs.new_tensor([*fit['source_norm'],perpendicular,parallel-perpendicular])
                    begun=time.perf_counter()
                    refined=(response(model,0.,xyt,basis,physical_parameters=theta) @ model.xs.new_tensor(fit['coefficients'])).cpu().numpy()
                    original=np.array(fit['prediction'])
                    delta=float(np.linalg.norm(refined-original)/max(np.linalg.norm(refined),1e-30))
                    replay[key]=dict(relative_l2=delta,passed=delta<=protocol['replay_threshold'],
                        rmse_ppm=float(np.sqrt(np.mean((refined-observed)**2))),seconds=time.perf_counter()-begun)
                    save(replay_path,replay)
                location=np.array(fit['source_norm'])*100
                q=window_mean_release(fit['coefficients'],duration,release_window)*3600
                initial=read(OUT/'fits'/identifier/'initialization.json')
                theta=np.r_[fit['source_norm'],fit['diffusion_norm'][1],
                    fit['diffusion_norm'][0]-fit['diffusion_norm'][1],fit['coefficients']]
                lower=np.array(initial['bounds']['lower'])
                upper=np.array(initial['bounds']['upper'])
                on_bound=(np.minimum(theta-lower,upper-theta)/(upper-lower)<1e-5)
                rows.append(dict(event_id=identifier,method=method,repeat=repeat,
                    seed=protocol['pso_seeds'][repeat] if len(fits)>1 else None,
                    equipment_group=truth['equipment_group'],source_height_m=meta['source_height_m'],
                    localization_error_m=float(np.linalg.norm(location-truth_xy)),
                    source_x_m=location[0],source_y_m=location[1],true_x_m=truth_xy[0],true_y_m=truth_xy[1],
                    source_lon=meta['origin'][0]+np.rad2deg(location[0]/(6371000*np.cos(np.deg2rad(meta['origin'][1])))),
                    source_lat=meta['origin'][1]+np.rad2deg(location[1]/6371000),
                    q_kg_h=q,true_q_kg_h=truth['mean_q_kg_h'],true_q_uncertainty_kg_h=truth['q_uncertainty_kg_h'],
                    q_relative_error=abs(q-truth['mean_q_kg_h'])/truth['mean_q_kg_h'],
                    rmse_ppm=float(np.sqrt(fit['mse'])),seconds=fit['seconds'],
                    common_initialization_seconds=initial['seconds'],
                    q_basis_count=fit.get('q_basis_count',5),
                    diffusion_parallel_m2_s=fit['diffusion_norm'][0]/.006,
                    diffusion_perp_m2_s=fit['diffusion_norm'][1]/.006,
                    converged=fit.get('converged'),bound_parameters=np.flatnonzero(on_bound).tolist(),
                    replay_relative_l2=replay[key]['relative_l2'],replay_passed=replay[key]['passed'],
                    n_observations=meta['observations'],missing_observations=meta['missing_observations'],
                    wind_filled_minutes=meta['wind_filled_minutes'],background_fallback=bool(meta['background_fallback'])))
        del model,response,data
        gc.collect()
        if device=='cuda':
            torch.cuda.empty_cache()
        print('METEC_REPLAY',identifier,flush=True)
    frame=pd.DataFrame(rows)
    folder=OUT/'analysis'
    folder.mkdir(exist_ok=True)
    save(folder/'environment.json',dict(device=device,
        gpu=torch.cuda.get_device_name() if device=='cuda' else None,
        versions={name:version(name) for name in ['torch','numpy','scipy','pandas','pyswarms']},
        fitting_dtype='float64; PSO population response float32, final predictions float64',
        derivative='Point analytic Jacobian tested against CPU/CUDA automatic differentiation; Adam reverse AD',
        early_checkpoints='First event and second IA/NLS used equivalent forward AD; source objective and budgets unchanged'))
    frame.to_csv(folder/'replicate_metrics.csv',index=False,encoding='utf-8-sig')
    # PSO technical repeats do not increase n: aggregate each metric within each event.
    metrics=['localization_error_m','q_relative_error','rmse_ppm','seconds','replay_relative_l2']
    per_case=frame.groupby(['event_id','method'],sort=False)[metrics].median().reset_index()
    per_case.to_csv(folder/'case_metrics.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for method,g in per_case.groupby('method',sort=False):
        item=dict(method=method,events=len(g))
        for metric in metrics:
            item[metric+'_median']=float(g[metric].median())
            item[metric+'_mean']=float(g[metric].mean())
        summary.append(item)
    pd.DataFrame(summary).to_csv(folder/'method_summary.csv',index=False,encoding='utf-8-sig')
    wide=per_case.pivot(index='event_id',columns='method',values='localization_error_m')
    difference=wide.ia_vpdi-wide.joint_nls_trf
    count=len(frame)
    save(folder/'summary.json',dict(final=True,ready=bool(frame.replay_passed.all()),
        numerical_status='passed' if frame.replay_passed.all() else 'warnings_retained',
        expected_events=17,complete_events=len(wide),
        method_runs=count,replay_passes=int(frame.replay_passed.sum()),replay_total=count,
        replay_max_relative_l2=float(frame.replay_relative_l2.max()),
        ia_minus_nls_location_m=dict(mean=float(difference.mean()),median=float(difference.median()),
            ia_better_by_more_than_1m=int((difference< -1).sum()),nls_better_by_more_than_1m=int((difference>1).sum()),
            within_1m=int((abs(difference)<=1).sum())),
        inference_scope='All17 releases; one site,one week,5 groups,11 coordinates. No independent-site generalization or guaranteed solver advantage.',
        caveats=['Known group-reference source height, not separately surveyed release height',
            'One-sided prior background estimate; no concentration imputation',
            'Fixed kg/m3-to-ppm conversion and wind-history approximation',
            'IA model-capacity selection vs fixed5 joint baselines; not pure elimination ablation',
            'Unequal evaluation budgets; PSO three technical repeats per event',
            'Numerical replay checks prediction stability, not physical truth or location uniqueness']))
    sources=[OUT/'protocol.json',*list((OUT/'fits').glob('*/*.json')),
        *list((OUT/'replay').glob('*.json')),*list((OUT/'inputs').glob('*/*'))]
    sources += [ROOT/'paper_experiments'/name for name in
        ['run_metec_comparison.py','metec_point_adapter.py','paper_physical_beam_adapter.py',
         'paper_joint_baselines.py','paper_reduced_solver.py','paper_beam_joint_baselines.py',
         'paper_q_model_selection.py','analyze_metec_comparison.py']]
    sources += [ROOT/'pinn_source/reduced_trust_region.py',ROOT/'pinn_source/q_capacity.py']
    save(folder/'sources.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    evaluate(parser.parse_args().device)
