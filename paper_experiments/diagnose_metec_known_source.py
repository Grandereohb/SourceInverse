"""Evaluation-only oracle-location check of the anomalous release; never an inversion initializer."""

import gc
import hashlib
from pathlib import Path

import numpy as np
import torch

from run_metec_comparison import ROOT,OUT,read,save,load_input
from metec_point_adapter import xy_from_lon_lat,window_mean_release
from reduced_trust_region import fit_reduced_trust_region,ReducedResidual

DEST=OUT.parent/'metec_known_source_diagnostic_20261027'


def run():
    torch.set_num_threads(1)
    identifier='20220509006'
    orders=dict(time=32,release=32,history_minutes=40)
    path=DEST/identifier/'result.json'
    if path.exists():
        print('KNOWN_SOURCE_DIAGNOSTIC_ALREADY_SAVED',flush=True)
        return
    save(DEST/'protocol.json',dict(event_id=identifier,orders=orders,
        purpose='Oracle-location residual diagnosis only; truth may not initialize or select inversion solutions',
        selection='Frozen public event with largest previously recorded emission inflation',
        fitted='Diffusion and five nonnegative Q coefficients; source fixed at recorded truth for diagnosis',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    data,meta,model,xyt,basis,response,observed=load_input(identifier,'cuda',orders)
    truth=read(OUT/'inputs'/identifier/'truth.json')
    fixed=xy_from_lon_lat([truth['lon']],[truth['lat']],meta['origin'])[0]/100.
    initialization=read(OUT/'fits'/identifier/'initialization.json')
    lower=np.array(initialization['bounds']['lower'])
    upper=np.array(initialization['bounds']['upper'])

    def matrix(diffusion,jacobian):
        theta=model.xs.new_tensor(np.r_[fixed,diffusion])
        if jacobian:
            value,derivative=response.matrix_and_jacobian(model,0.,xyt,basis,theta)
            return value.cpu().numpy(),derivative[...,2:].cpu().numpy()
        return response(model,0.,xyt,basis,physical_parameters=theta).cpu().numpy()

    starts=[np.array(start)[2:4] for start in initialization['starts']]
    print('KNOWN_SOURCE_START',identifier,flush=True)
    fit=fit_reduced_trust_region(matrix,observed,starts,lower[2:4],upper[2:4],
        q_upper_bound=float(upper[4]),max_nfev=100,coordinate_scale=np.array([.03,.03]))
    fit.update(q_kg_h=3600*window_mean_release(fit['coefficients'],len(data['concentration'])*60.,data['release_window_s']),
        rmse_ppm=float(np.sqrt(fit['sse']/len(observed))))
    comparison={}
    for method in ['ia_vpdi','joint_nls_trf']:
        previous=read(OUT/'fits'/identifier/(method+'.json'))
        parallel,perpendicular=previous['diffusion_norm']
        theta=model.xs.new_tensor([*previous['source_norm'],perpendicular,parallel-perpendicular])
        high=response(model,0.,xyt,basis,physical_parameters=theta).cpu().numpy()
        reduced=ReducedResidual(lambda unused,jac,m=high:m,observed,q_upper_bound=float(upper[4]))
        residual=reduced.residual(np.zeros(1))
        prediction=high@np.asarray(previous['coefficients'])
        comparison[method]=dict(saved_parameter_high_order_sse=float(np.sum((prediction-observed)**2)),
            refitted_q_high_order_sse=float(residual@residual),
            location_error_m=float(np.linalg.norm((np.asarray(previous['source_norm'])-fixed)*100)),
            q_kg_h=3600*window_mean_release(reduced.solution['coefficients'],len(data['concentration'])*60.,data['release_window_s']))
    save(path,dict(oracle_location_fit=fit,existing_solutions=comparison,true_q_kg_h=truth['mean_q_kg_h'],
        inversion_parameters_not_modified=True,
        limitation='Two diffusion starts do not prove a global minimum even with known source coordinates'))
    print('KNOWN_SOURCE_DONE',fit['sse'],fit['q_kg_h'],flush=True)
    del model,response
    gc.collect()
    torch.cuda.empty_cache()


if __name__=='__main__':
    run()
