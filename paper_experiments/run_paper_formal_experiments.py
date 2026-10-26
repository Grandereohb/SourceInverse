"""Prespecified case-level independent evaluation for paper Results4.1/4.2.

No station/time withholding and no tuning from evaluation truth. The generic
puff baseline is an adapted demo, not a named-paper replication. Results4.3
and additional reliability/layout experiments remain separate requirements.
"""

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import run_paper_experiments as base
import run_paper_parcel_pilot as parcel
from paper_fast_parcels import CachedMeanResponse
from paper_reference_transport import puff_response
from paper_baseline_demos import fit_demo

ROOT=base.ROOT/'result/paper_experiments_20261017'
STUDY=ROOT/'evaluation_v1'
RELEASES=('constant','pulse','double_peak')
WIND=('steady','veering')
NETWORKS=('ideal5','jjj6')


def design():
    cases=[]
    for repeat in range(8):
        for network in NETWORKS:
            for wind in WIND:
                for release in RELEASES:
                    cases.append(dict(cohort='nominal',network=network,wind=wind,release=release,repeat=repeat))
    for network in NETWORKS:
        for wind in WIND:
            for release in RELEASES:
                cases.append(dict(cohort='wind_mismatch',network=network,wind=wind,release=release,repeat=0))
    q_combinations=[(n,w,r) for n in NETWORKS for w in WIND for r in RELEASES[1:]]
    q_combinations += [('ideal5','steady','pulse'),('ideal5','veering','double_peak'),
                       ('jjj6','steady','double_peak'),('jjj6','veering','pulse')]
    for index,(network,wind,release) in enumerate(q_combinations):
        cases.append(dict(cohort='q_mismatch',network=network,wind=wind,release=release,repeat=index//8))
    return [dict(item,case_id=f'eval_{i+1:03d}',seed=730001+i) for i,item in enumerate(cases)]


def networks():
    sites=base.ROOT/'data/jjj/sites.xlsx'
    frame=pd.read_excel(sites).set_index('station')
    lon,lat=frame.loc['lon'].to_numpy(float),frame.loc['lat'].to_numpy(float)
    center=np.array([lon.mean(),lat.mean()])
    xy=np.column_stack(((lon-center[0])*np.pi/180*6371000*np.cos(center[1]*np.pi/180),
                        (lat-center[1])*np.pi/180*6371000))
    if len(lon)!=6 or not np.isfinite(xy).all():
        raise ValueError('Expected all6 finite current JJJ station coordinates')
    return {'ideal5':dict(stations=[[.3,-.1],[.5,1.],[1.,-1.],[1.7,.9],[2.,-.9]],
                length_m=1000.,sigma=.15,labels=[f'ideal{i+1}' for i in range(5)]),
            'jjj6':dict(stations=(xy/4000.).tolist(),length_m=4000.,sigma=.025,
                labels=list(frame.columns),lon=lon.tolist(),lat=lat.tolist(),
                center_lon_lat=center.tolist(),sites_sha256=hashlib.sha256(sites.read_bytes()).hexdigest())}


def make_input(spec,network):
    rng=np.random.default_rng(spec['seed'])
    source=rng.uniform([-.65,-.5],[.15,.5])
    perpendicular=float(rng.uniform(.015,.045))
    parallel=perpendicular*float(rng.uniform(1.4,4.))
    count=6 if spec['cohort']=='q_mismatch' else 5
    knots=np.linspace(0.,6.,count)
    if count==6:
        knots[1:-1]+=rng.uniform(-.3,.3,count-2)
    if spec['release']=='constant':
        coefficients=np.ones(count)
    elif spec['release']=='pulse':
        coefficients=(np.array([0.,0.,1.,0.,0.]) if count==5 else np.array([0.,.05,1.,.15,0.,0.]))
        coefficients[coefficients>0]*=rng.uniform(.8,1.2,np.count_nonzero(coefficients))
    else:
        coefficients=(np.array([0.,1.,.02,.8,0.]) if count==5 else np.array([0.,1.,.02,.02,.8,0.]))
        coefficients[coefficients>0]*=rng.uniform(.8,1.2,np.count_nonzero(coefficients))
    stations=np.asarray(network['stations'])
    mismatch=spec['cohort']=='wind_mismatch'
    sign=1 if spec['seed']%2 else -1
    kwargs=dict(wind=spec['wind'],sigma=network['sigma'],
                speed=1.2*(1+sign*.1) if mismatch else 1.2,
                angle_offset=sign*np.pi/36 if mismatch else 0.)

    def truth(samples,order):
        t=(np.arange(6*samples)+.5)/samples
        response=puff_response(stations,t,knots,source,parallel,perpendicular,order=order,**kwargs)
        return (response@coefficients).reshape(6,samples,len(stations)).mean(1).ravel()

    clean=truth(64,128)
    release_check=np.linalg.norm(truth(64,64)-clean)/np.linalg.norm(clean)
    temporal_check=np.linalg.norm(truth(32,128)-clean)/np.linalg.norm(clean)
    if release_check>1e-4 or temporal_check>1e-3:
        raise RuntimeError(f'Independent generator quadrature gate failed: {release_check}, {temporal_check}')
    scale=float(clean.max())
    if not np.isfinite(scale) or scale<=0:
        raise RuntimeError('No finite positive generated signal; do not replace the case')
    coefficients=coefficients/scale
    clean=clean/scale
    observed=clean+rng.normal(0.,.01,len(clean))
    metadata=dict(spec,source_norm=source.tolist(),true_d_parallel_norm=parallel,
        true_d_perp_norm=perpendicular,true_knots=knots.tolist(),true_coefficients=coefficients.tolist(),
        length_m=network['length_m'],sigma=network['sigma'],noise_std=.01,
        true_wind_speed_norm=kwargs['speed'],true_angle_offset_rad=kwargs['angle_offset'],
        observed_wind_speed_norm=1.2,observed_angle_offset_rad=0.,
        release_quadrature_relative_change=float(release_check),temporal_quadrature_relative_change=float(temporal_check),
        station_peak_ratio_second_to_first=float(np.sort(clean.reshape(6,len(stations)).max(0))[-2]),
        q_unit='model units, NOT kg/h',coordinate_unit='normalized Cartesian; errors multiplied by length_m')
    return metadata,dict(stations=stations,clean=clean,observed=observed,source=source,
        true_knots=knots,true_coefficients=coefficients)


def freeze_inputs():
    if STUDY.exists():
        raise FileExistsError('Study already exists; preserve the frozen protocol and data')
    STUDY.mkdir()
    nets=networks()
    protocol={'case_count':120,'nominal_count':96,'wind_mismatch_count':12,'q_mismatch_count':12,
        'case_design':design(),'networks':nets,'primary_q_basis_count':5,
        'initial_source_norm':[0.,0.],'initial_diffusion_norm':[.05,.02],
        'source_bounds_norm':[[-.75,.75],[-.75,.75]],'search_step_norm':.25,
        'top_k':2,'start_strategy':'spatial_diverse','start_min_distance_norm':.4,
        'diffusion_scale_seeds':[.5,1.,2.],'diffusion_ratio_seeds':[1.,3.],
        'adam_steps':1000,'adam_lr':.02,'max_gradient_norm':10.,
        'stopping':'fixed1000 updates/start; convergence limitations reported, no truth-based extension',
        'puff_powell_max_evaluations_per_start':3000,'prior_or_extra_losses':'none',
        'inference':'existing zero-field Gaussian parcels; exact interval-MEAN prescribed wind at dt=1/32h;8 release subdivisions;16 midpoint means/hour',
        'generator':'independent NumPy analytic wind/tensor integrals;128 Gauss-Legendre;64 midpoint means/hour',
        'objective':'all valid observations, unweighted SSE, nonnegative5-hat Q; no station/time withholding',
        'source_distribution':'uniform x in[-.65,.15], y in[-.5,.5]; inference retains the FULL larger search box',
        'diffusion_distribution':'D_perp uniform[.015,.045]; D_parallel/D_perp uniform[1.4,4]',
        'noise':'independent Gaussian sigma=.01 after noiseless peak normalized to1; no clipping',
        'evaluation_status':'new seeds730001..730120; no previous development inputs reused',
        'baselines':'matched joint Adam and generic puff/NNLS/Powell; not strict named-paper reproductions',
        'timing':'accuracy runs GPU for IA/joint, CPU for independent demo; no same-hardware speed superiority claim',
        'outstanding_requirements':['layout and reliability full analysis','constant-Q probabilistic baseline track',
            'other external-demo feasibility','controlled releases4.3 physical adaptation and validation'],
        'frozen_files':{p.relative_to(base.ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),Path(__file__).with_name('paper_fast_parcels.py'),
                      Path(__file__).with_name('paper_reference_transport.py'),
                      Path(__file__).with_name('paper_baseline_demos.py'),
                      Path(__file__).with_name('run_paper_parcel_pilot.py'),
                      Path(__file__).with_name('run_paper_experiments.py')]+list((base.ROOT/'pinn_source').rglob('*.py'))}}
    base.save_json(STUDY/'protocol.json',protocol)
    for spec in design():
        metadata,data=make_input(spec,nets[spec['network']])
        case=STUDY/spec['case_id'];case.mkdir()
        base.save_json(case/'input.json',metadata)
        np.savez_compressed(case/'input.npz',**data)
        print(f'FROZEN {spec["case_id"]}',flush=True)


def metrics(metadata,data,source,coefficients,prediction):
    times=np.linspace(0.,6.,193)
    true=np.interp(times,data['true_knots'],data['true_coefficients'])
    predicted=np.interp(times,np.linspace(0.,6.,len(coefficients)),coefficients)
    return dict(localization_error_m=float(np.linalg.norm(np.asarray(source)-data['source'])*metadata['length_m']),
        q_relative_l2=float(np.linalg.norm(predicted-true)/np.linalg.norm(true)),
        concentration_rmse_clean=float(np.sqrt(np.mean((prediction-data['clean'])**2))),
        observed_mse=float(np.mean((prediction-data['observed'])**2)),
        source_at_boundary=bool(np.any(np.abs(source)>=.74)))


def fit_case(case,protocol,device):
    metadata=json.loads((case/'input.json').read_text(encoding='utf-8'))
    data=np.load(case/'input.npz')
    sigma=metadata['sigma']
    ns=len(data['stations'])
    callback=CachedMeanResponse(stations=ns)
    torch.manual_seed(metadata['seed'])
    model,xyt,basis=parcel.configure(metadata,data['stations'],device,sigma=sigma,wind_interval_mean=True)
    with ExitStack() as stack:
        stack.enter_context(base.experiment_context())
        stack.enter_context(patch.object(base,'SIGMA',sigma))
        stack.enter_context(patch.object(base,'averaged_response',callback))
        stack.enter_context(patch.object(base.ia,'build_response_matrix',callback))
        started=time.perf_counter()
        candidates=[]
        for label,dp,dq in base.ia.build_diffusion_seeds(model):
            for x,y in base.vp.generate_candidate_grid(base.BOUNDS,base.BOUNDS,protocol['search_step_norm']):
                candidates.append(base.ia.score_all_data_candidate(model,sigma,xyt,data['observed'],basis,x,y,dp,dq,seed_label=label))
        candidates.sort(key=lambda c:c.profile_score)
        starts=base.ia.select_refinement_starts(candidates,2,'spatial_diverse',.4)
        common_seconds=time.perf_counter()-started
        started=time.perf_counter()
        profiles=[base.ia.refine_profiled_candidate(model,sigma,xyt,data['observed'],basis,start,
            base.BOUNDS,base.BOUNDS,steps=protocol['adam_steps'],lr=.02,
            patience=protocol['adam_steps']+1,min_delta=0.) for start in starts]
        profile_seconds=time.perf_counter()-started
        if not all(r.success for r in profiles):
            raise RuntimeError(f'Profile run failed: {[r.message for r in profiles]}')
        best=min((r.best for r in profiles),key=lambda r:r.profile_score)
        joints=[base.joint_refine(model,xyt,basis,data['observed'],start,protocol['adam_steps']) for start in starts]
        joint=min(joints,key=lambda r:r['score'])
        result={'case_id':metadata['case_id'],'metadata':metadata,'device':device,
            'methods':{},'common_search_seconds':common_seconds,'common_candidate_count':len(candidates),
            'common_starts':[asdict(s) for s in starts]}
        for method,source,coefficients,prediction,diffusion,seconds in (
            ('ia_vpdi',[best.source_x,best.source_y],best.coefficients,best.prediction,
             [best.diffusion_parallel_norm,best.diffusion_perp_norm],profile_seconds),
            ('matched_joint_adam',joint['source'],joint['coefficients'],joint['prediction'],
             joint['diffusion'],sum(r['seconds'] for r in joints))):
            base.ia.set_model_anisotropic_diffusion_norm(model,*diffusion)
            with torch.no_grad():
                model.xs.fill_(source[0]);model.ys.fill_(source[1])
            replay=callback(model,sigma,xyt,basis).cpu().numpy()@coefficients
            np.testing.assert_allclose(replay,prediction,rtol=2e-5,atol=2e-6)
            result['methods'][method]=dict(source_norm=source,diffusion_norm=diffusion,
                coefficients=coefficients,prediction=prediction,
                seconds_including_common_search=common_seconds+seconds,
                **metrics(metadata,data,source,coefficients,prediction))
        # Restore the IA optimum for its conditional information diagnostic.
        base.ia.set_model_anisotropic_diffusion_norm(model,best.diffusion_parallel_norm,best.diffusion_perp_norm)
        with torch.no_grad():
            model.xs.fill_(best.source_x);model.ys.fill_(best.source_y)
        info=base.ia.compute_profile_information(model,sigma,xyt,data['observed'],basis,best.coefficients,
            length_m=metadata['length_m'],source_bounds=(base.BOUNDS,base.BOUNDS))
        result['conditional_information']=asdict(info)
        np.savez_compressed(case/'optimization_traces.npz',
            profile_scores=np.asarray([[t['profile_score'] for t in r.trajectory] for r in profiles]),
            joint_scores=np.asarray([r['trajectory'] for r in joints]),
            candidate_surface=np.asarray([[c.source_x,c.source_y,c.diffusion_parallel_norm,c.diffusion_perp_norm,
                c.profile_score,float(np.max(c.q_integration))] for c in candidates]))
        baseline=fit_demo(data['stations'],data['observed'],metadata['wind'],
            protocol['puff_powell_max_evaluations_per_start'],sigma=sigma,search_step=protocol['search_step_norm'])
        result['methods']['independent_puff_demo']=dict(**baseline,
            **metrics(metadata,data,baseline['source_norm'],baseline['coefficients'],np.asarray(baseline['prediction'])))
        base.save_json(case/'result.json',result)
    return result


def run(device,limit):
    protocol=json.loads((STUDY/'protocol.json').read_text(encoding='utf-8'))
    for name,expected in protocol['frozen_files'].items():
        if hashlib.sha256((base.ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Frozen code changed: {name}; do not mix evaluation versions')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    runtime=dict(device=device,torch=torch.__version__,platform=platform.platform(),
                 gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
                 jacobian_batch=os.environ.get('PINN_IA_VP_JACOBIAN_BATCH_SIZE'))
    if not (STUDY/'runtime.json').exists():
        base.save_json(STUDY/'runtime.json',runtime)
    for item in protocol['case_design'][:limit]:
        case=STUDY/item['case_id']
        if (case/'result.json').exists():
            continue
        print(f'START {item["case_id"]} {item["cohort"]} {item["network"]}',flush=True)
        try:
            result=fit_case(case,protocol,device)
        except Exception as exc:
            base.save_json(case/'STOPPED.json',dict(type=type(exc).__name__,message=str(exc)))
            raise
        print(f'DONE {item["case_id"]}: '+str({m:round(r['localization_error_m'],1)
              for m,r in result['methods'].items()}),flush=True)


def preflight(device):
    output=ROOT/'formal_preflight_interval_mean'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    for i,network in enumerate(NETWORKS):
        spec=dict(case_id=f'preflight_{i+1}',cohort='nominal',network=network,
                  wind='veering',release='pulse',seed=990001+i,repeat=0)
        metadata,data=make_input(spec,networks()[network])
        case=output/spec['case_id'];case.mkdir()
        base.save_json(case/'input.json',metadata)
        np.savez_compressed(case/'input.npz',**data)
        protocol=dict(search_step_norm=.5,adam_steps=2,puff_powell_max_evaluations_per_start=5)
        result=fit_case(case,protocol,device)
        if not all(np.isfinite(r['observed_mse']) for r in result['methods'].values()):
            raise RuntimeError('Preflight nonfinite objective')
    base.save_json(output/'PASS.json',dict(purpose='plumbing/finite values only; NOT evaluation',cases=2,steps=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--limit',type=int,default=120)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.preflight:
        preflight(args.device)
    elif args.freeze:
        freeze_inputs()
    else:
        run(args.device,args.limit)
