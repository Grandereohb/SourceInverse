"""Reproduce the affine covariance failure and verify the square-root representation."""

import hashlib
import argparse
import shutil
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from run_metec_comparison import ROOT,OUT,AUDIT,read,save
from run_ia_affine_wind_pilot import calibration
from paper_sqrt_affine_wind import SquareRootAffineWindHistory,step_factors,exact_step_factors
from ia_metec_point_adapter import configure

DEST=OUT.parent/'metec_sqrt_flow_verification_20261028'


def run(destination=DEST):
    destination=Path(destination)
    torch.set_num_threads(1)
    identifier='20220512003'
    data=dict(np.load(OUT/'inputs'/identifier/'input.npz'))
    meta=read(OUT/'inputs'/identifier/'metadata.json')
    raw=pd.read_csv(AUDIT/'raw/input_data/ADED_data_clean_SAMPLE.csv')
    raw['timestamp']=pd.to_datetime(raw.time,utc=True)-pd.Timedelta(minutes=1)
    sensors=pd.read_csv(AUDIT/'raw/input_data/sensor_locations.csv').set_index('name')
    original,audit=calibration(data,meta,raw,sensors)
    history=SquareRootAffineWindHistory(original.edges,original.velocity,original.gradient,data['wind_xy'])
    local_error=0.
    for a,b,e in zip(history.gradient,history.velocity,history.direction):
        small=exact_step_factors(a,b,60.,e)
        large=step_factors(a,b,60.,e,32)
        for low,high in zip(small[2:],large[2:]):
            c0,c1=low@low.T,high@high.T
            local_error=max(local_error,float(np.linalg.norm(c0-c1)/max(np.linalg.norm(c1),1e-30)))
    if local_error>1e-10:
        raise FloatingPointError('Local positive quadrature is not accurate enough')
    files=[Path(__file__),ROOT/'paper_experiments/paper_sqrt_affine_wind.py',
        ROOT/'paper_experiments/paper_sqrt_gaussian.py',ROOT/'paper_experiments/metec_point_adapter.py',
        ROOT/'paper_experiments/paper_physical_beam_adapter.py',ROOT/'paper_experiments/paper_affine_wind.py']
    protocol=dict(event_id=identifier,selection='First frozen affine cohort numerical failure',
        source_truth_used=False,orders=[16,16],history_minutes=20,
        local_step_reference='Closed-form factor vs independent32-node positive quadrature',local_max_relative_covariance_change=local_error,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    if (destination/'protocol.json').exists() and read(destination/'protocol.json')!=protocol:
        raise ValueError('Changed verification; choose a new output directory')
    save(destination/'protocol.json',protocol)
    snapshot=destination/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in files:
        if not (snapshot/path.name).exists():
            shutil.copy2(path,snapshot/path.name)
    begun=time.perf_counter()
    print('SQRT_VERIFY_PLAN_START',identifier,flush=True)
    model,xyt,basis,response,observed=configure(data,meta['source_height_m'],device='cuda',
        time_order=16,release_order=16,history_minutes=20,transport_history=history)
    plan_seconds=time.perf_counter()-begun
    print('SQRT_VERIFY_PLAN_DONE',round(plan_seconds,1),flush=True)
    valid=response.plan.weighted_basis.abs().sum(-1)>0
    covariance=response.plan.parallel+response.plan.perpendicular
    xx,xy,yy=[covariance[...,index][valid] for index in range(3)]
    lost=int(((xx*yy-xy*xy)<=0).sum())
    initial=read(OUT/'fits'/identifier/'initialization.json')
    params=[np.array(s)[:4] for s in initial['starts']]
    center=(np.array(initial['bounds']['lower'])[:2]+np.array(initial['bounds']['upper'])[:2])/2
    params.extend(np.r_[center,perp*.006,contrast*.006] for perp,contrast in [(.1,0.),(.1,60.),(60.,0.),(60.,60.)])
    rows=[]
    for index,theta in enumerate(params):
        evaluation_start=time.perf_counter()
        matrix,jacobian=response.matrix_and_jacobian(model,0.,xyt,basis,model.xs.new_tensor(theta))
        if not bool(torch.isfinite(matrix).all()&torch.isfinite(jacobian).all()):
            raise FloatingPointError('Square-root response or derivative is nonfinite')
        rows.append(dict(parameter_set=index,matrix_finite=True,jacobian_finite=True,
            matrix_max=float(matrix.max()),jacobian_max=float(jacobian.abs().max()),
            seconds=time.perf_counter()-evaluation_start))
    save(destination/'result.json',dict(verified=True,event_id=identifier,
        reconstructed_isotropic_covariances_with_nonpositive_naive_determinant=lost,
        parameter_checks=rows,local_max_relative_covariance_change=local_error,
        total_seconds=time.perf_counter()-begun,
        plan_seconds=plan_seconds,
        physical_equations_unchanged=True,no_D_or_eigenvalue_clipping=True,
        source_truth_used=False,fit_performed=False,full_goal_achieved=False))
    print('SQRT_VERIFY_DONE',identifier,'naive_nonpositive',lost,'seconds',round(time.perf_counter()-begun,1),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEST)
    run(parser.parse_args().output)
