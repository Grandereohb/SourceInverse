"""Paired development feasibility on independent analytic-puff inputs.

Uses existing production parcel responses without changing production defaults.
This is not the frozen independent evaluation or controlled-release experiment.
"""

import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

import run_paper_experiments as experiment
from gaussian_parcels import gaussian_parcel_response

ROOT = experiment.ROOT/'result/paper_experiments_20261017'
SAMPLES = 16


def response_mean(model, sigma, xyt, basis, track_gradients=False):
    with nullcontext() if track_gradients else torch.no_grad():
        response = gaussian_parcel_response(model, sigma, xyt, model.recurrent_integration_times,
            basis, subdivisions=8, initial_release_fraction=0.0)
    return experiment.hourly_mean_response(response, samples=SAMPLES)


def configure(record, stations, device='cpu',sigma=experiment.SIGMA,wind_interval_mean=False):
    clock = np.linspace(0.,6.,193)
    angle = np.full_like(clock,0.05) if record['wind']=='steady' else -0.35+0.15*clock
    u,v=1.2*np.cos(angle),1.2*np.sin(angle)
    if wind_interval_mean and record['wind']=='veering':
        duration=np.diff(clock)
        u_mean=1.2*(np.sin(angle[1:])-np.sin(angle[:-1]))/(.15*duration)
        v_mean=1.2*(np.cos(angle[:-1])-np.cos(angle[1:]))/(.15*duration)
        u,v=np.r_[u_mean,u_mean[-1]],np.r_[v_mean,v_mean[-1]]
    model = experiment.PINN(hidden=8,depth=1).to(device)
    experiment.field.configure_recurrent_context(model,-2.,4.,-3.,3.,clock,
        u,v,0.,0.05,anisotropic_diffusion=True,
        d_parallel_min_norm=0.001,d_perp_min_norm=0.001,decay_norm=0.1,
        initial_release_dt=0.,nx=36,ny=36,source_mode='gaussian',source_sigma_norm=sigma,
        diffusion_solver='gaussian',source_decay_integrator='exponential_linear',
        transport_solver='gaussian_parcels',parcel_subdivisions=8)
    experiment.ia.set_model_anisotropic_diffusion_norm(model,0.05,0.02)
    stations=torch.as_tensor(stations,dtype=torch.float32,device=device)
    times=torch.tensor((np.arange(6*SAMPLES)+0.5)/SAMPLES,dtype=torch.float32,device=device)
    xyt=torch.cat((stations.repeat(len(times),1),times.repeat_interleave(len(stations))[:,None]),1)
    basis=experiment.vp.build_q_basis(model.recurrent_integration_times,5)
    return model,xyt,basis


def run(steps,output,cached=False,device='cpu'):
    if output.exists():
        raise FileExistsError('Use a fresh study directory; do not overwrite earlier evidence')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    experiment.save_json(output/'manifest.json',{'case_indices':[0,2,10],'steps_per_start':steps,
        'cached_fixed_transport_factors':cached,'device':device,
        'dataset':'NEW analytic-reference truth + saved development noise; baseline_demo/*_input.npz',
        'split':'development, not independent evaluation','production_core_changed':False,
        'transport':'existing Gaussian parcels, dt=1/32h,8 release subdivisions, zero initial field',
        'observations':'16 midpoint samples/hour, all30 averaged observations fitted',
        'both_methods':'same5 nonnegative hats, same common coarse starts, same feasible source/D, unweighted SSE',
        'coarse_grid':'4x4 source points x6 diffusion seeds,2 spatially diverse starts',
        'truth_not_used_for_initialization_or_selection':True,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reference_driver_sha256':hashlib.sha256((Path(__file__).parent/'run_paper_experiments.py').read_bytes()).hexdigest(),
        'core_hashes':{str(p.relative_to(experiment.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (experiment.ROOT/'pinn_source').rglob('*.py')}})
    rows=[]
    try:
        from paper_fast_parcels import CachedMeanResponse
        callback=CachedMeanResponse(samples=SAMPLES) if cached else response_mean
        with ExitStack() as stack:
            stack.enter_context(experiment.experiment_context())
            stack.enter_context(patch.object(experiment.ia,'build_response_matrix',callback))
            stack.enter_context(patch.object(experiment,'averaged_response',callback))
            for index in (0,2,10):
                case=f'pilot_{index+1:02d}'
                saved=np.load(ROOT/'baseline_demo'/f'{case}_input.npz')
                record=json.loads((ROOT/'development_12'/case/'input.json').read_text(encoding='utf-8'))
                model,xyt,basis=configure(record,saved['stations'],device)
                observed,clean=saved['observed'],saved['clean']
                experiment.save_json(output/f'{case}_inputs.json',{
                    'input_sha256':hashlib.sha256((ROOT/'baseline_demo'/f'{case}_input.npz').read_bytes()).hexdigest(),
                    'wind':record['wind'],'source_truth':saved['source'],'q_coefficients_truth':saved['coefficients']})
                print(f'START {case} parcel pilot',flush=True)
                common_started=time.perf_counter()
                coarse=experiment.ia.run_ia_vpdi(model,experiment.SIGMA,xyt,observed,
                    experiment.BOUNDS,experiment.BOUNDS,0.5,q_basis_count=5,top_k=2,
                    diffusion_scale_seeds=(0.5,1.,2.),diffusion_ratio_seeds=(1.,3.),outer_steps=0,
                    start_strategy='spatial_diverse',start_min_distance=0.4)
                common_seconds=time.perf_counter()-common_started
                starts=[r.start for r in coarse.refinement_runs]
                started=time.perf_counter()
                profiles=[]
                for i,start in enumerate(starts):
                    print(f'profile start{i+1}/2,steps={steps}',flush=True)
                    profiles.append(experiment.ia.refine_profiled_candidate(model,experiment.SIGMA,xyt,
                        observed,basis,start,experiment.BOUNDS,experiment.BOUNDS,steps=steps,
                        lr=0.02,patience=steps+1,min_delta=0.0))
                profile_seconds=time.perf_counter()-started
                if not all(r.success for r in profiles):
                    raise RuntimeError(f'Profile refinement failed: {[r.message for r in profiles]}')
                best=min((r.best for r in profiles),key=lambda r:r.profile_score)
                joints=[experiment.joint_refine(model,xyt,basis,observed,start,steps) for start in starts]
                joint=min(joints,key=lambda r:r['score'])
                true_q=basis.cpu().numpy()@saved['coefficients']
                for method,source,coefficients,prediction,seconds in (
                    ('ia_vpdi',[best.source_x,best.source_y],best.coefficients,best.prediction,profile_seconds),
                    ('matched_joint_adam',joint['source'],joint['coefficients'],joint['prediction'],
                     sum(j['seconds'] for j in joints))):
                    metric=experiment.metrics(source,coefficients,prediction,saved['source'],true_q,basis,clean,observed)
                    rows.append({'case':case,'method':method,'steps':steps,
                        'seconds_including_common_search':common_seconds+seconds,**metric})
                experiment.save_json(output/f'{case}_result.json',{
                    'common_starts':[asdict(s) for s in starts],'common_seconds':common_seconds,
                    'profile_best':asdict(best),'joint_best':joint,
                    'profile_trajectories':[r.trajectory for r in profiles],
                    'joint_runs':joints,'metrics':rows[-2:]})
                experiment.save_json(output/'summary.json',rows)
                print(f'DONE {case}: IA={rows[-2]["localization_error_m"]:.2f}m, '
                      f'joint={rows[-1]["localization_error_m"]:.2f}m',flush=True)
    except Exception as exc:
        experiment.save_json(output/'STOPPED.json',{'type':type(exc).__name__,'message':str(exc)})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--output',type=Path,default=ROOT/'parcel_paired_development')
    parser.add_argument('--cached',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args()
    if args.steps<0 or not args.output.resolve().is_relative_to((experiment.ROOT/'result').resolve()):
        parser.error('Nonnegative steps and an output under result/ required')
    run(args.steps,args.output,args.cached,args.device)
