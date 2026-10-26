"""Physical observation adapter preparation and short real-input plumbing test.

This is not a completed Results4.3 inversion. Full quadrature/precision gates
and fixed-protocol fitting must precede any source-accuracy/mass-rate claims.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import run_paper_experiments as base
from paper_physical_beam_adapter import PhysicalBeamPlan, PhysicalBeamResponse

ROOT = base.ROOT/'result/paper_experiments_20261017'
DATA = ROOT/'controlled_release_preparation/source1_rel03'


def inputs(minutes=None,data_dir=DATA):
    observations = pd.read_csv(data_dir/'observations.csv', parse_dates=['time']).sort_values(['time','beam'])
    weather = pd.read_csv(data_dir/'raw_axis_weather.csv', parse_dates=['time']).sort_values('time')
    beams = pd.read_csv(data_dir/'beams.csv').sort_values('beam')
    if minutes is not None:
        weather = weather.iloc[:minutes]
        observations = observations[observations.time.isin(weather.time)]
    if (observations.duplicated(['time','beam']).any() or
            len(observations)!=len(weather)*len(beams) or
            not np.array_equal(observations.beam.to_numpy().reshape(-1,len(beams)),
                               np.broadcast_to(beams.beam.to_numpy(),(len(weather),len(beams))))):
        raise ValueError('Complete, unique, aligned beam observations required; do not drop rows silently')
    expected = pd.date_range(weather.time.iloc[0], periods=len(weather), freq='min')
    if not np.array_equal(expected.to_numpy(),weather.time.to_numpy()):
        raise ValueError('Do not bridge missing minutes as continuous wind history')
    return observations, weather, beams


def configure(height, minutes=None, device='cpu', dtype=torch.float64,
              beam_order=32, time_order=4, release_order=4, q_count=5,
              beam_solver='quadrature',data_dir=DATA):
    observations, weather, beams = inputs(minutes,data_dir)
    model = base.PINN(hidden=8,depth=1).to(device=device,dtype=dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in (model.xs,model.ys,model.rawD_perp,model.rawD_parallel_delta):
        parameter.requires_grad_(True)
    # Minimal experimental context; do not modify the production time/field setup.
    model.recurrent_d_scale_norm = 1.
    model.recurrent_d_parallel_min_norm = model.recurrent_d_perp_min_norm = .001
    edges = np.arange(len(weather)+1,dtype=float)*60.
    model.recurrent_times = torch.as_tensor(edges/60.,dtype=dtype,device=device)
    count = min(q_count,len(weather)+1)
    model.recurrent_integration_times = torch.unique(torch.cat((model.recurrent_times,
        torch.linspace(0.,float(len(weather)),count,dtype=dtype,device=device))),sorted=True)
    with torch.no_grad():
        model.xs.fill_(.55)
        model.ys.fill_(.55)
    base.ia.set_model_anisotropic_diffusion_norm(model,10.*60./10000.,3.*60./10000.)
    basis = base.vp.build_q_basis(model.recurrent_integration_times,count)
    wind = np.column_stack((-weather['WindY_m/s'],weather['WindX_m/s']))
    starts = beams[['start_x_m','start_y_m','start_z_m']].to_numpy()
    ends = beams[['end_x_m','end_y_m','end_z_m']].to_numpy()
    xyt = torch.as_tensor(np.column_stack((np.tile(ends[:,:2]/100.,(len(weather),1)),
        np.repeat(np.arange(len(weather),dtype=float),len(beams)))),dtype=dtype,device=device)
    plan = PhysicalBeamPlan(edges,wind,starts,ends,weather.Temp_K,weather.Press_Pa,
        model.recurrent_integration_times.detach().cpu().numpy()*60.,
        basis.detach().cpu().numpy(),height_m=height,beam_order=beam_order,
        time_order=time_order,release_order=release_order,device=device,dtype=dtype,beam_solver=beam_solver)
    response = PhysicalBeamResponse(plan,xyt,basis)
    return model,xyt,basis,response,observations.residual_ppm.to_numpy(copy=True),observations


def preflight(device,output,beam_solver='quadrature',data_dir=DATA):
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    protocol = dict(status='3-minute real-input plumbing only, not an accuracy or emission-rate result',
        heights_m=[.3,1.6], input_rows=21, dtype='float64', device=device,beam_solver=beam_solver,
        physical_units=dict(coordinates='m',wind='m/s',time='s in propagator, minutes in parameterization',
                            diffusion='m2/s',coefficients='kg/s',prediction='ppm'),
        assumptions=['Dz=D_transverse, no extra fitted vertical parameter',
            'reflecting ground, zero mean vertical advection, no methane decay',
            '2m-square frame represented by same-variance horizontal Gaussian',
            'measured T/P ideal-gas conversion, no humidity correction',
            'all valid prepared residuals retained, including negatives'],
        files={str(p.relative_to(base.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),Path(__file__).with_name('paper_physical_beam_adapter.py'),
                      data_dir/'observations.csv',data_dir/'raw_axis_weather.csv',data_dir/'beams.csv']})
    base.save_json(output/'protocol.json',protocol)
    for height in (.3,1.6):
        model,xyt,basis,response,observed,_ = configure(height,minutes=3,device=device,q_count=3,
            beam_solver=beam_solver,data_dir=data_dir)
        with patch.object(base.ia,'build_response_matrix',response):
            start = base.ia.score_all_data_candidate(model,0.,xyt,observed,basis,.55,.55,
                10.*60./10000.,3.*60./10000.)
            result = base.ia.refine_profiled_candidate(model,0.,xyt,observed,basis,start,
                (0.,1.1),(0.,1.1),steps=2,lr=.01,patience=3,min_delta=0.)
        if not result.success or not np.isfinite(result.best.prediction).all():
            raise RuntimeError('Physical source/NNLS plumbing failed')
        base.save_json(output/f'height_{height:.1f}.json',dict(height_m=height,
            q_units='kg/s by physical forward definition, NOT validated recovery',
            result=asdict(result), response_calls=response.calls))
    base.save_json(output/'PASS.json',dict(heights=2,minutes=3,steps=2,
        scope='finite physical responses and IA/NNLS integration only; no accuracy claim'))
    print('Physical real-input preflight passed both heights',flush=True)


def numerical_gate(device,output,beam_solver='quadrature',data_dir=DATA):
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    settings = [('coarse',32,4,4),('intermediate',64,8,8),('reference',64,16,16)]
    if beam_solver=='analytic':
        settings=[('coarse',2,8,8),('intermediate',2,16,16),('reference',2,32,32)]
    states = [(.25,.25,10.,3.),(.55,.55,10.,3.),(.85,.85,30.,10.)]
    protocol = dict(heights_m=[.3,1.6],settings=settings,fixed_states=states,beam_solver=beam_solver,
        threshold_relative_response_frobenius=.01,
        selection='cheapest listed configuration passing every fixed-state/height check',
        truth_used=False,model_units='physical Green response in ppm per kg/s',
        reference='finite high quadrature, NOT exact truth; unit tests cover kernel mass and endpoint regularity',
        adapter_sha256=hashlib.sha256(Path(__file__).with_name('paper_physical_beam_adapter.py').read_bytes()).hexdigest())
    base.save_json(output/'protocol.json',protocol)
    rows = []
    for height in (.3,1.6):
        values = {}
        for label,beam_order,time_order,release_order in settings:
            started = time.perf_counter()
            model,xyt,basis,response,_,_ = configure(height,device=device,
                beam_order=beam_order,time_order=time_order,release_order=release_order,
                beam_solver=beam_solver,data_dir=data_dir)
            for state_id,(x,y,dp,dq) in enumerate(states):
                with torch.no_grad():
                    model.xs.fill_(x)
                    model.ys.fill_(y)
                base.ia.set_model_anisotropic_diffusion_norm(model,dp*60./10000.,dq*60./10000.)
                matrix = response(model,0.,xyt,basis).detach().cpu().numpy()
                values[label,state_id] = matrix
                np.savez_compressed(output/f'height_{height:.1f}_{label}_state_{state_id}.npz',response=matrix)
            print('NUMERICAL',height,label,'seconds',round(time.perf_counter()-started,2),flush=True)
        for label,_,_,_ in settings[:-1]:
            for state_id in range(len(states)):
                error = values[label,state_id]-values['reference',state_id]
                denominator = float(np.linalg.norm(values['reference',state_id]))
                relative = float(np.linalg.norm(error)/max(denominator,np.finfo(float).tiny))
                rows.append(dict(height_m=height,setting=label,state_id=state_id,
                    relative_response_error=relative,maximum_absolute_response_error=float(np.abs(error).max()),
                    reference_norm=denominator,passed=bool(relative<=.01)))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'comparison.csv',index=False)
    passing = [label for label,_,_,_ in settings[:-1] if frame[frame.setting==label].passed.all()]
    base.save_json(output/'gate_result.json',dict(passed=bool(passing),
        selected=passing[0] if passing else None,
        interpretation='Numerical observation-model gate only; no fit or source accuracy claim'))
    if not passing:
        raise RuntimeError('Physical response quadrature gate failed; do not start real-data fitting')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--preflight',action='store_true')
    stage.add_argument('--numerical-gate',action='store_true')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--data-dir',type=Path,default=DATA)
    parser.add_argument('--analytic-beams',action='store_true')
    args = parser.parse_args()
    label=('controlled_release_adapter_analytic_' if args.analytic_beams else 'controlled_release_adapter_')
    output = args.output or ROOT/(label+('preflight_v3' if args.preflight else 'numerical_v1'))
    (preflight if args.preflight else numerical_gate)(args.device,output,
        'analytic' if args.analytic_beams else 'quadrature',args.data_dir)
