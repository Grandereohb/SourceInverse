"""Matched industrial application, distinct from truth-validated experiments.

Explicit --mode is required: smoke is a flow check, full retains the complete
eligible cohort. Unknown location/rate truth is never invented from fitted data.
"""

import argparse
from contextlib import ExitStack,redirect_stdout
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np
import torch

import paper_industrial_transport as transport
import run_paper_experiments as base
import paper_scaled_inner_q as scaled_inner

ROOT=base.ROOT
STUDY=ROOT/'result/paper_experiments_20261017'


def raw_prediction_jacobian(context,callback,coefficients):
    """Four exact forward-mode tangents, not observation-wise backward passes."""
    class Prediction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model=context.model
        def forward(self):
            response=callback(self.model,context.sigma_norm,context.xyt,context.basis)
            return response@torch.as_tensor(coefficients,dtype=response.dtype,device=response.device)
    module=Prediction()
    names=('xs','ys','rawD_perp','rawD_parallel_delta')
    columns=[]
    for index in range(4):
        with torch.autograd.forward_ad.dual_level():
            parameters={}
            for j,name in enumerate(names):
                primal=getattr(context.model,name).detach()
                tangent=torch.ones_like(primal) if index==j else torch.zeros_like(primal)
                parameters['model.'+name]=torch.autograd.forward_ad.make_dual(primal,tangent)
            value=torch.func.functional_call(module,parameters,())
            derivative=torch.autograd.forward_ad.unpack_dual(value).tangent
            columns.append(torch.zeros_like(value).detach() if derivative is None else derivative.detach())
    matrix=torch.stack(columns,dim=1)
    if not bool(torch.isfinite(matrix).all()):
        raise FloatingPointError('Nonfinite forward-mode information Jacobian')
    return matrix


def freeze(preparation,preflight,output,mode,device):
    prepared=json.loads((preparation/'PREPARATION_FINISHED.json').read_text(encoding='utf-8'))
    check=json.loads((preflight/'PREFLIGHT_FINISHED.json').read_text(encoding='utf-8'))
    if (preparation/'SUPERSEDED.json').exists() or not check['ready_for_smoke']:
        raise ValueError('A current preparation and successful numerical preflight are required')
    prior=json.loads((preflight/'protocol.json').read_text(encoding='utf-8'))
    for name,value in prior['source_sha256'].items():
        if hashlib.sha256((ROOT/name.replace('\\','/')).read_bytes()).hexdigest()!=value:
            raise ValueError('Numerically checked transport changed: '+name)
    for name,value in prepared['prepared_sha256'].items():
        if hashlib.sha256((preparation/name.replace('\\','/')).read_bytes()).hexdigest()!=value:
            raise ValueError('Prepared input changed: '+name)
    ids=prepared['representative_case_ids'] if mode=='smoke' else sorted({Path(v.replace('\\','/')).parent.name for v in prepared['prepared_sha256']})
    files=[Path(__file__).resolve(),Path(transport.__file__).resolve(),Path(base.__file__).resolve(),
        Path(scaled_inner.__file__).resolve(),
        Path(__file__).with_name('paper_fast_parcels.py'),*sorted((ROOT/'pinn_source').rglob('*.py'))]
    protocol=dict(scope='Flow-only industrial smoke' if mode=='smoke' else 'Industrial all-eligible fixed-budget application',
        mode=mode,case_ids=ids,device=device,steps_per_start=5 if mode=='smoke' else 1000,
        release_order=check['recommended_release_order'],hour_samples=check['recommended_hour_samples'],
        q_basis_count=5,source_bounds_norm=[-.75,.75],source_step_norm=.25,
        diffusion_perp_seeds_m2s=[75.,150.,300.],diffusion_ratio_seeds=[1.,3.],
        starts=2,initialization='Identical all-data coarse starts for both methods; score+spatial diversity .4norm',
        objective='All finite signed background residuals; same unweighted SSE, Q>=0; no station/time withholding',
        inner_q_solver='Column-scaled SciPy BVLS, same box-constrained objective, projected KKT audit; no weak-column threshold',
        outer_lr=.02,stopping='Fixed steps per start, best evaluated state retained; convergence unverified',
        information_jacobian='Exact four-direction forward-mode AD, tested against existing reverse-mode Jacobian',
        information_precision='Float64 kernel/Jacobian and fixed-position reprofiled Q; primary source positions unchanged',
        numerical_gate='Prediction change <=1% against64 midpoint means/hour and GL128; reference128->256 <=0.25%',
        source_truth='Unavailable: no location-error, source-rate-error or claimed calibration fields',
        coordinate_reference='Same supplied coordinates; geodetic datum not certified',
        source_rate='Effective2D normalized model units, NOT kg/h',
        preparation=str(preparation.relative_to(ROOT)),preflight=str(preflight.relative_to(ROOT)),
        prepared_manifest_sha256=hashlib.sha256((preparation/'PREPARATION_FINISHED.json').read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        torch=torch.__version__,numpy=np.__version__)
    if output.exists():
        if json.loads((output/'protocol.json').read_text(encoding='utf-8'))!=protocol:
            raise ValueError('Protocol drift; preserve this version and use a new output')
    else:
        output.mkdir(parents=True)
        base.save_json(output/'protocol.json',protocol)
        for source in files:
            target=output/'source_snapshot'/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    return protocol


def execute_case(case_id,preparation,output,protocol):
    destination=output/case_id
    destination.mkdir(exist_ok=True)
    metadata=json.loads((preparation/case_id/'input.json').read_text(encoding='utf-8'))
    with np.load(preparation/case_id/'input.npz') as loaded:
        arrays={name:loaded[name] for name in loaded.files}
    context=transport.configure(arrays,device=protocol['device'],samples=protocol['hour_samples'])
    model,xyt,basis=context.model,context.xyt,context.basis
    observed=context.observations
    callback=transport.HourlyResponse(context,protocol['release_order'])
    steps=protocol['steps_per_start']
    if protocol['device']=='cuda':
        torch.cuda.reset_peak_memory_stats()
    began=time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(base,'SIGMA',context.sigma_norm))
        stack.enter_context(patch.object(base,'averaged_response',callback))
        stack.enter_context(patch.object(base.ia,'build_response_matrix',callback))
        stack.enter_context(patch.object(base.ia,'solve_nonnegative_q',scaled_inner.solve_scaled_nonnegative_q))
        coarse_started=time.perf_counter()
        candidates=[]
        for perp in protocol['diffusion_perp_seeds_m2s']:
            for ratio in protocol['diffusion_ratio_seeds']:
                dp=transport.normalize_diffusivity_m2s(perp*ratio,1.,context.length_m)
                dq=transport.normalize_diffusivity_m2s(perp,1.,context.length_m)
                for x,y in base.vp.generate_candidate_grid(base.BOUNDS,base.BOUNDS,protocol['source_step_norm']):
                    candidates.append(base.ia.score_all_data_candidate(model,context.sigma_norm,xyt,observed,basis,
                        x,y,dp,dq,seed_label=f'perp{perp:g}_ratio{ratio:g}'))
        candidates.sort(key=lambda v:v.profile_score)
        starts=base.ia.select_refinement_starts(candidates,2,'spatial_diverse',.4)
        coarse_seconds=time.perf_counter()-coarse_started
        print('COARSE_DONE',case_id,'seconds',round(coarse_seconds,2),flush=True)
        t=time.perf_counter()
        profiles=[base.ia.refine_profiled_candidate(model,context.sigma_norm,xyt,observed,basis,start,
            base.BOUNDS,base.BOUNDS,steps=steps,lr=.02,patience=steps+1,min_delta=0.) for start in starts]
        profile_seconds=time.perf_counter()-t
        if not all(v.success for v in profiles):
            raise RuntimeError('IA refinement exception: '+str([v.message for v in profiles]))
        best=min((v.best for v in profiles),key=lambda v:v.profile_score)
        with (destination/'joint_solver.log').open('w',encoding='utf-8') as log,redirect_stdout(log):
            joints=[base.joint_refine(model,xyt,basis,observed,start,steps) for start in starts]
        joint=min(joints,key=lambda v:v['score'])
        reference=transport.configure(arrays,dtype=torch.float64,samples=64)
        result=dict(case_id=case_id,mode=protocol['mode'],metadata=metadata,
            common_coarse_seconds=coarse_seconds,common_candidate_count=len(candidates),
            common_starts=[asdict(v) for v in starts],methods={},source_truth_available=False,
            coordinate_system='Relative to supplied station coordinate datum; no WGS84 assertion',
            normalization=dict(length_m=context.length_m,time_unit='hour',lon0=context.lon0,lat0=context.lat0,
                               concentration_scale=metadata['concentration_scale']))
        for name,source,diffusion,coefficients,prediction,seconds in (
            ('ia_vpdi',[best.source_x,best.source_y],[best.diffusion_parallel_norm,best.diffusion_perp_norm],
             best.coefficients,best.prediction,profile_seconds),
            ('matched_joint_adam',joint['source'],joint['diffusion'],joint['coefficients'],
             joint['prediction'],sum(v['seconds'] for v in joints))):
            base.ia.set_model_anisotropic_diffusion_norm(model,*diffusion)
            with torch.no_grad():
                model.xs.fill_(source[0]);model.ys.fill_(source[1])
            matrix=callback(model,context.sigma_norm,xyt,basis).cpu().numpy().astype(float)
            replay=matrix@coefficients
            np.testing.assert_allclose(replay,prediction,rtol=3e-5,atol=3e-6)
            physical_d=np.asarray(diffusion)*context.length_m**2/3600.
            finer=transport.numpy_hourly_reference(reference,source,*physical_d,np.linspace(0.,context.hours,5),order=128)@coefficients
            finest=transport.numpy_hourly_reference(reference,source,*physical_d,np.linspace(0.,context.hours,5),order=256)@coefficients
            relative=float(np.linalg.norm(finer-prediction)/max(np.linalg.norm(finer),1e-12))
            reference_change=float(np.linalg.norm(finest-finer)/max(np.linalg.norm(finest),1e-12))
            peak=int(np.argmax(observed))
            lon,lat=transport.source_lonlat(context,source)
            result['methods'][name]=dict(source_norm=source,longitude=lon,latitude=lat,
                diffusion_norm=diffusion,diffusion_m2_s=physical_d,coefficients=coefficients,
                prediction_normalized=prediction,
                normalized_rmse=float(np.sqrt(np.mean((prediction-observed)**2))),
                concentration_rmse_ug_m3=float(np.sqrt(np.mean((prediction-observed)**2))*metadata['concentration_scale']),
                peak_prediction_to_observation=float(prediction[peak]/observed[peak]),
                seconds_including_common_search=coarse_seconds+seconds,
                convergence='fixed_budget_unverified',near_source_search_boundary=bool(np.any(np.abs(source)>=.74)),
                response_column_norms=np.linalg.norm(matrix,axis=0),
                effective_log10_coefficients=[float(np.log10(v)) if v>0 else None for v in coefficients],
                amplitude_caution='Unbounded effective Q may compensate a weak response; no physical release-rate truth',
                numerical_replay_relative_difference=relative,reference_release_relative_difference=reference_change,
                numerical_gate_passed=relative<=.01 and reference_change<=.0025,
                q_unit='Effective2D normalized model units, NOT kg/h')
        # Core information formulas stay unchanged; only their Jacobian engine differs.
        diagnostic=transport.configure(arrays,device=protocol['device'],dtype=torch.float64,
                                         samples=protocol['hour_samples'])
        diagnostic_callback=transport.HourlyResponse(diagnostic,protocol['release_order'])
        dm=diagnostic.model
        base.ia.set_model_anisotropic_diffusion_norm(dm,best.diffusion_parallel_norm,best.diffusion_perp_norm)
        with torch.no_grad():
            dm.xs.fill_(best.source_x);dm.ys.fill_(best.source_y)
        diagnostic_response=diagnostic_callback(dm,diagnostic.sigma_norm,diagnostic.xyt,diagnostic.basis)
        diagnostic_q=scaled_inner.solve_scaled_nonnegative_q(diagnostic_response,observed)
        if not diagnostic_q.success:
            raise RuntimeError('Float64 fixed-position diagnostic Q solve failed')
        with patch.object(base.ia,'_prediction_jacobian_raw',
                lambda model,prediction,batch_size=None:raw_prediction_jacobian(diagnostic,diagnostic_callback,diagnostic_q.coefficients)), \
             patch.object(base.ia,'build_response_matrix',
                lambda model,sigma,xyt,basis,track_gradients=False:diagnostic_callback(model,sigma,xyt,basis,False)):
            result['conditional_information']=asdict(base.ia.compute_profile_information(
                dm,diagnostic.sigma_norm,diagnostic.xyt,observed,diagnostic.basis,diagnostic_q.coefficients,
                length_m=context.length_m,source_bounds=(base.BOUNDS,base.BOUNDS)))
        result['information_numerics']=dict(dtype='float64',fixed_position_refit_only=True,
            diagnostic_coefficients=diagnostic_q.coefficients,solver_message=diagnostic_q.message,
            warning='Diagnostic reprofile does not replace either primary source position or rate curve')
        result['whole_case_seconds']=time.perf_counter()-began
        result['peak_cuda_allocated_bytes']=int(torch.cuda.max_memory_allocated()) if protocol['device']=='cuda' else None
        np.savez_compressed(destination/'optimization_traces.npz',
            profile_scores=np.asarray([[v['profile_score'] for v in run.trajectory] for run in profiles]),
            joint_scores=np.asarray([run['trajectory'] for run in joints]),
            coarse_surface=np.asarray([[v.source_x,v.source_y,v.diffusion_parallel_norm,v.diffusion_perp_norm,
                                        v.profile_score] for v in candidates]))
        base.save_json(destination/'result.json',result)
        if not all(v['numerical_gate_passed'] for v in result['methods'].values()):
            raise RuntimeError('Measured-wind numerical replay gate failed; result preserved, stop before batch expansion')
    return result


def run(mode,device,preparation,preflight,output,max_new_cases=None):
    if max_new_cases is not None and max_new_cases<1:
        raise ValueError('Execution batch size must be positive')
    torch.set_num_threads(1)
    protocol=freeze(preparation,preflight,output,mode,device)
    executed=0
    for case in protocol['case_ids']:
        destination=output/case
        if (destination/'STOPPED.json').exists():
            raise RuntimeError('Preserved stopped case requires investigation: '+case)
        if (destination/'result.json').exists():
            continue
        if max_new_cases is not None and executed>=max_new_cases:
            break
        print('START_INDUSTRIAL',case,mode,flush=True)
        try:
            result=execute_case(case,preparation,output,protocol)
        except Exception as error:
            destination.mkdir(exist_ok=True)
            base.save_json(destination/'STOPPED.json',dict(type=type(error).__name__,message=str(error)))
            raise
        print('DONE_INDUSTRIAL',case,'seconds',round(result['whole_case_seconds'],2),flush=True)
        executed+=1
        completed=[v for v in protocol['case_ids'] if (output/v/'result.json').exists()]
        base.save_json(output/'RUN_PROGRESS.json',dict(mode=mode,planned_cases=len(protocol['case_ids']),
            completed_cases=len(completed),pending=[v for v in protocol['case_ids'] if v not in completed],
            final=len(completed)==len(protocol['case_ids']),
            scope='Execution batching only; the complete frozen cohort and per-case budgets are unchanged'))
    if all((output/v/'result.json').exists() for v in protocol['case_ids']):
        base.save_json(output/'RUN_FINISHED.json',dict(mode=mode,cases=len(protocol['case_ids']),
            scope='Execution and replay gates passed; not source-location accuracy or optimizer convergence proof'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('smoke','full'),required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--preparation',type=Path,default=STUDY/'industrial_preparation_v2_flags')
    parser.add_argument('--preflight',type=Path,default=STUDY/'industrial_transport_preflight_v1')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--max-new-cases',type=int,
                        help='Resource batching only, never truncates the frozen cohort or writes false completion')
    args=parser.parse_args()
    destination=args.output or STUDY/('industrial_fit_smoke_v3_scaled_nnls' if args.mode=='smoke' else 'industrial_fit_v2_scaled_nnls')
    run(args.mode,args.device,args.preparation.resolve(),args.preflight.resolve(),destination.resolve(),args.max_new_cases)
