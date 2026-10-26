"""Matched constant-Q point estimates and an explicitly adapted Bayesian demo.

Not a Newman M-MALA reproduction. Eight balanced repeat0/4 nominal inputs,
selected by identifiers rather than earlier method errors; not new blind data.
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

import emcee
import numpy as np
from scipy.optimize import minimize
from scipy.stats import truncnorm
import torch

import run_paper_formal_experiments as formal
from paper_fast_parcels import CachedMeanResponse
from paper_probabilistic_demo import (ConstantPuffPlan,CollapsedPosterior,BatchedPosterior,
    positive_normal_mean,rank_split_rhat)

base=formal.base
ROOT=formal.ROOT
IDS=tuple(f'eval_{v:03d}' for v in (1,4,7,10,49,52,55,58))


def freeze(output,device,smoke):
    files=[Path(__file__),Path(__file__).with_name('paper_probabilistic_demo.py'),
        Path(formal.__file__),Path(base.__file__),Path(formal.parcel.__file__),
        Path(__file__).with_name('paper_fast_parcels.py'),
        Path(__file__).with_name('paper_baseline_demos.py'),
        Path(__file__).with_name('paper_reference_transport.py'),
        *sorted((base.ROOT/'pinn_source').rglob('*.py'))]
    inputs=[ROOT/'evaluation_v2_boundary_fix'/case/name for case in IDS
            for name in ('input.json','input.npz')]
    protocol=dict(case_ids=IDS,smoke=smoke,device=device,scope='flow only' if smoke else 'matched constant-Q supplemental track',
        selection='repeat0 and4, both networks and winds, nominal constant releases; not outcome selection or new blind cases',
        source_bounds=[-.75,.75],q_count=1,objective='All supplied observations, no withholding; unweighted SSE for point methods',
        point_steps=2 if smoke else 1000,starts=2,source_step=.25,
        mcmc=dict(independent_ensembles=4,walkers=16,warmup=4 if smoke else 1000,
            retained_stages=[20] if smoke else [3000,6000,12000],
            kernel='emcee DEMove .8 + DESnookerMove .2; no sampler implemented manually',
            rank_rhat_max=1.01,tau_length_multiplier=50,minimum_total_ess=400,
            rhat='Rank/folded split-Rhat on fixed walker0 from each independent ensemble, NOT16 dependent walkers',
            ess='Sum of independent-ensemble Nwalkers*Ndraws/tau; tau from all walkers in each ensemble',
            rule='Continue same ensembles only at fixed convergence checks, never from source-truth error'),
        priors=dict(source='Uniform square [-.75,.75]^2',
            log_D_perp_minus_floor='Normal(log(.02),1.5^2)',log_D_parallel_minus_perp='Normal(log(.03),1.5^2)',
            rate='HalfNormal(tau=5 model units), analytically marginalized',
            noise='Known synthetic Gaussian std .01; no fitted noise or extra background'),
        floor_D=.00100005,release_order=32,hour_samples=16,
        differences='Point estimators have no priors; Bayes has proper Q/D priors and marginalizes Q. Capacity matched, objectives NOT identical.',
        bayesian_scope='Independent puff physical response plus ensemble MCMC, NOT a named-paper implementation',
        runtime=dict(torch=torch.__version__,numpy=np.__version__,emcee=emcee.__version__,
            gpu=torch.cuda.get_device_name() if device=='cuda' else None),
        source_sha256={str(p.relative_to(base.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        input_sha256={str(p.relative_to(base.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    # Tuples round-trip as lists; compare the serialized representation on resume.
    protocol=json.loads(json.dumps(protocol))
    if output.exists():
        if json.loads((output/'protocol.json').read_text(encoding='utf-8'))!=protocol:
            raise ValueError('Protocol/source drift: preserve outputs and use a new version')
    else:
        output.mkdir(parents=True)
        for source in files:
            target=output/'source_snapshot'/source.relative_to(base.ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
        base.save_json(output/'protocol.json',protocol)
    return protocol


def point_methods(metadata,data,protocol,destination):
    model,xyt,_=formal.parcel.configure(metadata,data['stations'],protocol['device'],
        sigma=metadata['sigma'],wind_interval_mean=True)
    basis=torch.ones((len(model.recurrent_integration_times),1),dtype=model.xs.dtype,device=model.xs.device)
    callback=CachedMeanResponse(stations=len(data['stations']))
    with ExitStack() as stack:
        stack.enter_context(base.experiment_context())
        stack.enter_context(patch.object(base.ia,'build_response_matrix',callback))
        stack.enter_context(patch.object(base,'averaged_response',callback))
        stack.enter_context(patch.object(base,'SIGMA',metadata['sigma']))
        started=time.perf_counter()
        candidates=[base.ia.score_all_data_candidate(model,metadata['sigma'],xyt,data['observed'],basis,
            x,y,dp,dq,seed_label=label)
            for label,dp,dq in base.ia.build_diffusion_seeds(model)
            for x,y in base.vp.generate_candidate_grid(base.BOUNDS,base.BOUNDS,.25)]
        candidates.sort(key=lambda c:c.profile_score)
        starts=base.ia.select_refinement_starts(candidates,2,'spatial_diverse',.4)
        coarse_seconds=time.perf_counter()-started
        t=time.perf_counter()
        runs=[base.ia.refine_profiled_candidate(model,metadata['sigma'],xyt,data['observed'],basis,
            start,base.BOUNDS,base.BOUNDS,steps=protocol['point_steps'],lr=.02,
            patience=protocol['point_steps']+1,min_delta=0.) for start in starts]
        if not all(v.success for v in runs):
            raise RuntimeError('Constant-Q IA point fit failed')
        best=min((v.best for v in runs),key=lambda v:v.profile_score)
        ia_seconds=time.perf_counter()-t
        with (destination/'joint_solver.log').open('w',encoding='utf-8') as log,redirect_stdout(log):
            joints=[base.joint_refine(model,xyt,basis,data['observed'],v,protocol['point_steps']) for v in starts]
        joint=min(joints,key=lambda v:v['score'])
    return dict(ia_constant=dict(source_norm=[best.source_x,best.source_y],q=float(best.coefficients[0]),
            prediction=best.prediction,diffusion_norm=[best.diffusion_parallel_norm,best.diffusion_perp_norm],
            mse=best.profile_score,seconds=coarse_seconds+ia_seconds,convergence='fixed budget unverified'),
        joint_constant=dict(source_norm=joint['source'],q=float(joint['coefficients'][0]),
            prediction=joint['prediction'],diffusion_norm=joint['diffusion'],mse=joint['score'],
            seconds=coarse_seconds+sum(v['seconds'] for v in joints),convergence='fixed budget unverified'))


def coarse_thetas():
    return np.asarray([[x,y,np.log(dq-.00100005),np.log(max(dq*(ratio-1),1e-5))]
        for dq in (.0175,.035,.07) for ratio in (1.,3.)
        for x in np.linspace(-.75,.75,7) for y in np.linspace(-.75,.75,7)])


def fit_puff_mle(posterior,smoke):
    best=None
    calls=0
    def score(theta):
        nonlocal best,calls
        calls+=1
        try:
            g=posterior.plan.response(theta)
        except FloatingPointError:
            return np.inf
        norm=float(g@g)
        q=max(float(g@posterior.observed)/norm,0.) if norm>0 else 0.
        mse=float(np.mean((g*q-posterior.observed)**2))
        if best is None or mse<best['mse']:
            dq=.00100005+np.exp(theta[2])
            best=dict(source_norm=np.asarray(theta[:2]),q=q,mse=mse,prediction=g*q,
                      diffusion_norm=[dq+np.exp(theta[3]),dq])
        return mse
    started=time.perf_counter()
    ranked=sorted((score(v),tuple(v)) for v in coarse_thetas())
    first=np.asarray(ranked[0][1])
    second=next(np.asarray(v) for _,v in ranked if np.linalg.norm(np.asarray(v[:2])-first[:2])>=.4)
    runs=[minimize(score,start,method='Powell',bounds=[(-.75,.75),(-.75,.75),(None,None),(None,None)],
        options=dict(maxfev=10 if smoke else 3000,maxiter=30,xtol=1e-4,ftol=1e-8)) for start in (first,second)]
    return dict(**best,seconds=time.perf_counter()-started,calls=calls,
        convergence='reported_converged' if all(r.success for r in runs) else 'reported_nonconverged',
        terminations=[dict(success=bool(r.success),message=str(r.message),nfev=r.nfev) for r in runs])


def sample_diagnostics(chains,blobs):
    augmented=[]
    for chain,blob in zip(chains,blobs):
        dq=.00100005+np.exp(chain[:,:,2]);dp=dq+np.exp(chain[:,:,3])
        q=positive_normal_mean(blob['mean'],blob['sd'])
        augmented.append(np.stack((chain[:,:,0],chain[:,:,1],dp,dq,q),axis=-1))
    independent=np.stack([v[:,0,:] for v in augmented])
    rhat=rank_split_rhat(independent)
    taus=[];long_enough=[]
    for v in augmented:
        try:
            tau=emcee.autocorr.integrated_time(v,tol=50,quiet=False)
            long_enough.append(True)
        except emcee.autocorr.AutocorrError as error:
            tau=error.tau;long_enough.append(False)
        taus.append(tau)
    taus=np.asarray(taus)
    ess=np.sum(chains[0].shape[0]*chains[0].shape[1]/np.maximum(taus,1.),axis=0)
    passed=bool(np.isfinite(rhat).all() and np.isfinite(taus).all() and
        np.all(rhat<=1.01) and all(long_enough) and np.all(ess>=400))
    return dict(passed=passed,rhat=rhat,tau_by_ensemble=taus,ess_approx=ess,
        length_at_least_50tau_by_ensemble=long_enough,
        names=['source_x','source_y','D_parallel','D_perp','conditional_mean_Q'])


def bayesian(posterior,protocol,destination,seed):
    started=time.perf_counter()
    batch=BatchedPosterior(posterior,protocol['device'])
    candidates=coarse_thetas()
    scores=np.asarray(batch(candidates))[:,0]
    centers=[]
    for i in np.argsort(-scores):
        if all(np.linalg.norm(candidates[i,:2]-v[:2])>=.2 for v in centers):
            centers.append(candidates[i])
        if len(centers)==4:
            break
    if len(centers)!=4:
        raise RuntimeError('Four observation-selected spatially diverse starts required')
    settings=protocol['mcmc']
    samplers=[]
    for ensemble in range(4):
        rng=np.random.default_rng(seed+101*ensemble)
        positions=np.stack([centers[i%4] for i in range(16)])+rng.normal(size=(16,4))*[.025,.025,.1,.1]
        positions[:,:2]=np.clip(positions[:,:2],-.749999,.749999)
        sampler=emcee.EnsembleSampler(16,4,batch,vectorize=True,
            moves=[(emcee.moves.DEMove(),.8),(emcee.moves.DESnookerMove(),.2)],
            blobs_dtype=[('mean',float),('sd',float)])
        sampler.random_state=np.random.RandomState(seed+101*ensemble).get_state()
        sampler.run_mcmc(positions,settings['warmup'],progress=False)
        sampler.reset()
        samplers.append(sampler)
    previous=0
    history=[]
    for total in settings['retained_stages']:
        for sampler in samplers:
            sampler.run_mcmc(None,total-previous,progress=False)
        chains=[s.get_chain() for s in samplers]
        blobs=[s.get_blobs() for s in samplers]
        diagnostic=sample_diagnostics(chains,blobs)
        history.append(dict(draws_per_walker=total,**diagnostic))
        base.save_json(destination/'sampling_progress.json',dict(history=history,
            seconds=time.perf_counter()-started,complete=False))
        print('MCMC',destination.name,'draws',total,'Rhat',float(np.max(diagnostic['rhat'])),
              'pass',diagnostic['passed'],flush=True)
        previous=total
        if diagnostic['passed']:
            break
    theta=np.stack(chains)
    raw_mean=np.stack([v['mean'] for v in blobs])
    raw_sd=np.stack([v['sd'] for v in blobs])
    q_mean=positive_normal_mean(raw_mean,raw_sd)
    rng=np.random.default_rng(seed+9001)
    q_draw=truncnorm.ppf(rng.uniform(1e-12,1-1e-12,size=raw_mean.shape),
        -raw_mean/raw_sd,np.inf,loc=raw_mean,scale=raw_sd)
    means=theta.mean(axis=(0,1,2))
    dq=.00100005+np.exp(theta[:,:,:,2]);dp=dq+np.exp(theta[:,:,:,3])
    dp_mean,dq_mean=float(dp.mean()),float(dq.mean())
    theta_point=np.r_[means[:2],np.log(dq_mean-.00100005),np.log(dp_mean-dq_mean)]
    q=float(q_mean.mean())
    prediction=posterior.plan.response(theta_point)*q
    np.savez_compressed(destination/'posterior_samples.npz',theta=theta,q=q_draw,
        conditional_q_mean=q_mean,log_prob=np.stack([s.get_log_prob() for s in samplers]))
    return dict(source_norm=means[:2],q=q,prediction=prediction,
        diffusion_norm=[dp_mean,dq_mean],mse=float(np.mean((prediction-posterior.observed)**2)),
        prediction_definition='plug-in posterior mean source,D,Q; NOT posterior predictive average',
        seconds=time.perf_counter()-started,calls=batch.calls,diagnostic=diagnostic,
        diagnostic_history=history,convergence='diagnostic_passed' if diagnostic['passed'] else 'diagnostic_failed',
        coordinate_marginal_95_intervals=np.quantile(theta[:,:,:,:2].reshape(-1,2),[.025,.975],axis=0),
        q_marginal_95_interval=np.quantile(q_draw,[.025,.975]),
        acceptance_fraction=[s.acceptance_fraction for s in samplers],
        warning='Marginal coordinate intervals are not a joint95% region; failed diagnostics invalidate inference claims')


def run(device,smoke,output):
    torch.set_num_threads(1)
    protocol=freeze(output,device,smoke)
    for case_id in (IDS[:1] if smoke else IDS):
        destination=output/case_id
        if (destination/'result.json').exists():
            continue
        destination.mkdir(exist_ok=True)
        source=ROOT/'evaluation_v2_boundary_fix'/case_id
        metadata=json.loads((source/'input.json').read_text(encoding='utf-8'))
        data=np.load(source/'input.npz')
        if metadata['release']!='constant' or metadata['cohort']!='nominal':
            raise ValueError('Only predefined nominal constant-Q cases belong to this track')
        print('START_CONSTANT',case_id,flush=True)
        point=point_methods(metadata,data,protocol,destination)
        posterior=CollapsedPosterior(ConstantPuffPlan(data['stations'],metadata['wind'],metadata['sigma']),data['observed'])
        point['puff_mle_constant']=fit_puff_mle(posterior,smoke)
        point['bayesian_constant']=bayesian(posterior,protocol,destination,metadata['seed']+190000)
        # Truth is accessed for metrics only after all four fits have returned.
        if not np.allclose(data['true_coefficients'],data['true_coefficients'][0]):
            raise ValueError('Evaluation truth is not a constant release')
        q_true=float(data['true_coefficients'][0])
        for method in point.values():
            method['localization_error_m']=float(np.linalg.norm(np.asarray(method['source_norm'])-data['source'])*metadata['length_m'])
            method['q_relative_error']=abs(method['q']-q_true)/q_true
        base.save_json(destination/'result.json',dict(case_id=case_id,smoke=smoke,methods=point,
            network=metadata['network'],wind=metadata['wind'],q_units='model units, not kg/h',
            truth_access='post-fit evaluation only; same original data as main evaluation'))
        print('DONE_CONSTANT',case_id,flush=True)
    base.save_json(output/'RUN_FINISHED.json',dict(cases=1 if smoke else 8,smoke=smoke,
        scope='Execution complete; convergence failures retained and must be audited before inference claims'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    run(args.device,args.smoke,args.output or ROOT/('constant_q_preflight_v1' if args.smoke else 'constant_q_evaluation_v1'))
