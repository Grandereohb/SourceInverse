"""Experimental measured-hour wind adapter with Gaussian release quadrature.

Uses the existing differentiable Gaussian parcel kernel without changing any
production source. Time is hours, velocity is converted from m/s, diffusivity
from m2/s. Source strength remains effective two-dimensional model units.
"""

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scipy.special import roots_legendre
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'pinn_source'))
import field
from models.pinn import PINN
import ia_vpdi as ia
import variable_projection as vp
from geo_utils import latlon_to_xy,xy_to_latlon
from transport_units import normalize_velocity_mps,normalize_diffusivity_m2s
from paper_fast_parcels import ParcelResponsePlan
from gaussian_parcels import RECURRENT_SOURCE_SCALE


@dataclass
class IndustrialContext:
    model: object
    xyt: torch.Tensor
    basis: torch.Tensor
    observations: np.ndarray
    observation_mask: np.ndarray
    stations_norm: np.ndarray
    length_m: float
    lon0: float
    lat0: float
    hours: int
    samples: int
    sigma_norm: float


def configure(arrays,device='cpu',dtype=torch.float32,samples=16):
    """Preserve all point observations and their same-hour wind, without filling."""
    lon,lat=np.asarray(arrays['longitude']),np.asarray(arrays['latitude'])
    sp,direction=np.asarray(arrays['wind_speed_mps']),np.asarray(arrays['wind_direction_deg'])
    observed=np.asarray(arrays['normalized_concentration'])
    mask=np.asarray(arrays['observation_mask'],dtype=bool)
    labels=np.asarray(arrays['labels'],dtype='datetime64[s]')
    n=len(sp)
    if (n<2 or samples<1 or int(samples)!=samples or len(lon)<1 or lon.shape!=lat.shape
            or observed.shape!=(n,len(lon)) or mask.shape!=observed.shape or not mask.any()
            or labels.shape!=(n,) or not np.all(np.diff(labels)==np.timedelta64(1,'h'))
            or not all(np.isfinite(v).all() for v in (lon,lat,sp,direction,observed[mask]))
            or np.any(sp<0) or np.any((direction<0)|(direction>360))):
        raise ValueError('Invalid hourly point/wind preparation; interpolation is not allowed')
    lon0,lat0=float(lon.mean()),float(lat.mean())
    xy=np.asarray([latlon_to_xy(x,y,lon0,lat0) for x,y in zip(lon,lat)])
    # A square contains the complete station envelope plus at least1km/side.
    length=(2*float(np.abs(xy).max())+2000.)/1.5
    stations=xy/length
    angle=np.deg2rad(direction)
    u=normalize_velocity_mps(-sp*np.sin(angle),1.,length)
    v=normalize_velocity_mps(-sp*np.cos(angle),1.,length)
    minimum=normalize_diffusivity_m2s(1.,1.,length)
    sigma=100./length
    model=PINN(hidden=8,depth=1).to(device=device,dtype=dtype)
    if getattr(model,'source_position_mode','single')!='single':
        raise ValueError('Industrial application requires the fixed single-source model')
    field.configure_recurrent_context(model,-.75,.75,-.75,.75,np.arange(n+1),
        np.r_[u,u[-1]],np.r_[v,v[-1]],d_min_norm=minimum,d_scale_norm=1.,
        anisotropic_diffusion=True,d_parallel_min_norm=minimum,d_perp_min_norm=minimum,
        decay_norm=0.,initial_release_dt=0.,nx=12,ny=12,source_mode='gaussian',
        source_sigma_norm=sigma,diffusion_solver='gaussian',transport_solver='gaussian_parcels')
    # Configure retains float32 wind attributes; restore exact external clocks
    # and wind for the requested dtype, particularly the float64 reference gate.
    cast=lambda a:torch.as_tensor(a,dtype=dtype,device=device)
    model.recurrent_times=cast(np.arange(n+1))
    model.recurrent_u=cast(np.r_[u,u[-1]])
    model.recurrent_v=cast(np.r_[v,v[-1]])
    model.recurrent_integration_times=torch.unique(torch.cat((model.recurrent_times,
        cast(np.linspace(0.,n,5)),cast(np.linspace(0.,n,6)))),sorted=True)
    ia.set_model_anisotropic_diffusion_norm(model,
        normalize_diffusivity_m2s(150.,1.,length),normalize_diffusivity_m2s(50.,1.,length))
    times=cast((np.arange(n*samples)+.5)/samples)
    locations=cast(stations)
    xyt=torch.cat((locations.repeat(len(times),1),times.repeat_interleave(len(lon))[:,None]),1)
    basis=vp.build_q_basis(model.recurrent_integration_times,5)
    return IndustrialContext(model,xyt,basis,observed[mask],mask,stations,length,lon0,lat0,n,int(samples),sigma)


def source_lonlat(context,source_norm):
    return xy_to_latlon(float(source_norm[0])*context.length_m,
                         float(source_norm[1])*context.length_m,context.lon0,context.lat0)


class GaussianReleasePlan(ParcelResponsePlan):
    """Same parcel evaluation/derivatives, different fixed release quadrature."""
    def __init__(self,model,sigma,xyt,q_times,basis,order=32):
        if int(order)!=order or order<2:
            raise ValueError('Integer Gaussian order >=2 required')
        cast=lambda a:torch.as_tensor(a,dtype=model.xs.dtype,device=model.xs.device)
        clock=cast(model.recurrent_times).reshape(-1)
        qt=cast(q_times).reshape(-1)
        queries=cast(xyt)
        basis=cast(basis)
        if (basis.ndim!=2 or basis.shape[0]!=len(qt) or basis.requires_grad or queries.requires_grad
                or not all(bool(torch.isfinite(a).all()) for a in (clock,qt,queries,basis))
                or not bool((clock.diff()>0).all()) or not bool((qt.diff()>0).all())
                or bool((queries[:,2]<clock[0]).any()) or bool((queries[:,2]>clock[-1]).any())):
            raise ValueError('Finite fixed clock/query/basis required')
        self.width=float(model.recurrent_source_sigma_norm)
        wind=torch.stack((cast(model.recurrent_u),cast(model.recurrent_v)),1)[:-1]
        speed=torch.linalg.vector_norm(wind,dim=1)
        orientation=wind/speed.clamp_min(1e-12)[:,None]
        parallel=torch.stack((orientation[:,0].square(),orientation[:,0]*orientation[:,1],
                              orientation[:,1].square()),1)
        parallel=torch.where((speed<=1e-12)[:,None],cast([.5,0.,.5]),parallel)
        perpendicular=cast([1.,0.,1.])-parallel
        nodes,weights=map(cast,roots_legendre(int(order)))
        breaks=torch.unique(torch.cat((clock,qt)),sorted=True)
        entries=[]
        with torch.no_grad():
            for end in torch.unique(queries[:,2],sorted=True):
                left=breaks[:-1][breaks[:-1]<end]
                right=torch.minimum(breaks[1:][:len(left)],end)
                release=((left+right)[:,None]*.5+(right-left)[:,None]*.5*nodes).flatten()
                quadrature=((right-left)[:,None]*.5*weights).flatten()
                high=torch.searchsorted(qt,release.contiguous()).clamp(1,len(qt)-1)
                alpha=(release-qt[high-1])/(qt[high]-qt[high-1])
                temporal=(1-alpha[:,None])*basis[high-1]+alpha[:,None]*basis[high]
                residence=(torch.minimum(clock[1:,None],end)-
                           torch.maximum(clock[:-1,None],release[None,:])).clamp_min(0.)
                indexes=torch.nonzero(queries[:,2]==end,as_tuple=False).flatten()
                weighted=(quadrature*torch.exp(-model.recurrent_decay_norm*(end-release)))[:,None]*temporal
                entries.append((indexes,queries[indexes,:2],residence.T@wind,
                                residence.T@parallel,residence.T@perpendicular,weighted))
            self.station_count=max(len(e[0]) for e in entries)
            self.release_count=max(len(e[2]) for e in entries)
            self.displacement=clock.new_zeros((len(entries),self.release_count,2))
            self.parallel=clock.new_zeros((len(entries),self.release_count,3))
            self.perpendicular=torch.zeros_like(self.parallel)
            self.weighted_basis=clock.new_zeros((len(entries),self.release_count,basis.shape[1]))
            self.xy=clock.new_zeros((len(entries),self.station_count,2))
            indexes_all,valid=[] ,[]
            for i,(indexes,xy,displacement,par,perp,weighted) in enumerate(entries):
                k=len(displacement)
                self.displacement[i,:k]=displacement
                self.parallel[i,:k]=par
                self.perpendicular[i,:k]=perp
                self.weighted_basis[i,:k]=weighted
                self.xy[i,:len(indexes)]=xy
                indexes_all.append(indexes)
                valid.append(torch.arange(len(indexes),device=clock.device)+i*self.station_count)
            self.output_indexes=torch.cat(indexes_all)
            self.valid_flat=torch.cat(valid)
            self.query_count=len(queries)
            self.dtype,self.device=model.xs.dtype,model.xs.device


class HourlyResponse:
    """An immutable physical context with a small cache of fixed Q bases."""
    def __init__(self,context,order=32):
        self.context=context
        self.order=int(order)
        self.plans={}
        self.calls=0

    def __call__(self,model,sigma,xyt,basis,track_gradients=False):
        if model is not self.context.model or sigma!=self.context.sigma_norm or xyt is not self.context.xyt:
            raise ValueError('Construct a separate response adapter for another fixed context')
        if basis.requires_grad:
            raise ValueError('Only fixed release bases may be cached')
        key=(tuple(basis.shape),basis.detach().cpu().numpy().tobytes())
        if key not in self.plans:
            if len(self.plans)>=8:
                self.plans.clear()
            self.plans[key]=GaussianReleasePlan(model,sigma,xyt,model.recurrent_integration_times,basis,self.order)
        self.calls+=1
        with torch.enable_grad() if track_gradients else torch.no_grad():
            response=self.plans[key].evaluate(model)
        c=self.context
        averaged=response.reshape(c.hours,c.samples,len(c.stations_norm),-1).mean(1)
        return averaged.reshape(-1,averaged.shape[-1])[torch.as_tensor(c.observation_mask.flatten(),device=model.xs.device)]


def numpy_hourly_reference(context,source_norm,d_parallel_m2s,d_perp_m2s,knots,*,order=64):
    """Independent prefix-integral NumPy reference, not a fitted comparator.

    Uses the exact piecewise-constant wind and tensor integrals at release and
    observation times; it does not call the production field or parcel kernel.
    """
    if not d_parallel_m2s>=d_perp_m2s>0:
        raise ValueError('Physical diffusivities must be ordered and positive')
    clock=context.model.recurrent_times.detach().cpu().double().numpy()
    wind=np.column_stack((context.model.recurrent_u.detach().cpu().double().numpy()[:-1],
                          context.model.recurrent_v.detach().cpu().double().numpy()[:-1]))
    speed=np.linalg.norm(wind,axis=1)
    unit=wind/np.maximum(speed[:,None],1e-12)
    par=np.stack((unit[:,0]**2,unit[:,0]*unit[:,1],unit[:,1]**2),axis=1)
    par[speed<=1e-12]=[.5,0.,.5]
    perp=np.array([1.,0.,1.])-par
    dp=normalize_diffusivity_m2s(d_parallel_m2s,1.,context.length_m)
    dq=normalize_diffusivity_m2s(d_perp_m2s,1.,context.length_m)
    rates=np.column_stack((wind,dp*par+dq*perp))
    prefix=np.vstack((np.zeros(5),np.cumsum(rates*np.diff(clock)[:,None],axis=0)))
    def integral(t):
        i=np.searchsorted(clock,t,side='right')-1
        i=np.clip(i,0,len(clock)-2)
        return prefix[i]+(np.asarray(t)-clock[i])[...,None]*rates[i]
    knots=np.asarray(knots,dtype=float)
    nodes,weights=roots_legendre(order)
    times=(np.arange(context.hours*context.samples)+.5)/context.samples
    out=[]
    for end in times:
        breaks=np.unique(np.r_[clock[clock<end],knots[knots<end],end])
        left,right=breaks[:-1],breaks[1:]
        release=((left+right)[:,None]*.5+(right-left)[:,None]*.5*nodes).ravel()
        w=((right-left)[:,None]*.5*weights).ravel()
        propagated=integral(end)-integral(release)
        dx=context.stations_norm[:,0,None]-source_norm[0]-propagated[:,0]
        dy=context.stations_norm[:,1,None]-source_norm[1]-propagated[:,1]
        xx=context.sigma_norm**2+2*propagated[:,2]
        xy=2*propagated[:,3]
        yy=context.sigma_norm**2+2*propagated[:,4]
        determinant=xx*yy-xy*xy
        exponent=(yy*dx**2+xx*dy**2-2*xy*dx*dy)/determinant
        kernel=np.exp(-.5*exponent)/(2*np.pi*np.sqrt(determinant))
        phi=np.stack([np.interp(release,knots,np.eye(len(knots))[j]) for j in range(len(knots))],axis=1)
        out.append(RECURRENT_SOURCE_SCALE*(kernel*w)@phi)
    response=np.asarray(out).reshape(context.hours,context.samples,len(context.stations_norm),len(knots)).mean(1)
    return response.reshape(-1,len(knots))[context.observation_mask.flatten()]


def numerical_preflight(preparation,output,device='cuda'):
    """Numerical probes only; no measured outcome is used to tune an inversion."""
    preparation,output=Path(preparation).resolve(),Path(output).resolve()
    manifest=json.loads((preparation/'PREPARATION_FINISHED.json').read_text(encoding='utf-8'))
    if (preparation/'SUPERSEDED.json').exists():
        raise ValueError('Superseded preparation is not eligible')
    for name,value in manifest['prepared_sha256'].items():
        if hashlib.sha256((preparation/name).read_bytes()).hexdigest()!=value:
            raise ValueError('Prepared input changed: '+name)
    cases=[]
    for path in sorted(preparation.glob('ind_*/input.json')):
        meta=json.loads(path.read_text(encoding='utf-8'))
        with np.load(path.with_suffix('.npz')) as arrays:
            cases.append((meta['case_id'],meta['n_hours'],float(arrays['wind_speed_mps'].max())))
    selected=list(manifest['representative_case_ids'])
    for case in (min(cases,key=lambda v:(-v[1],v[0]))[0],min(cases,key=lambda v:(-v[2],v[0]))[0]):
        if case not in selected:
            selected.append(case)
    files=[Path(__file__).resolve(),Path(__file__).with_name('paper_fast_parcels.py'),
           *sorted((ROOT/'pinn_source').rglob('*.py'))]
    protocol=dict(scope='Measured-wind numerical preflight only; no inversion or location truth',
        cases=selected,selection='Three predefined input representatives plus longest and fastest wind cases',
        source_width_m=100.,minimum_diffusivity_m2_s=1.,decay_per_hour=0.,
        wind='Unsmoothed measured hourly vector, from-direction, constant over its co-timed concentration hour',
        time='Hours, NOT full-window-normalized time; raw end-label assumption preserved',
        initial_field='Zero source contribution before first fitted interval, fixed pre-window background only',
        kernel='Existing differentiable Gaussian parcel kernel; GL release integral within wind and Q intervals',
        tested_release_orders=[16,32,64],numpy_reference_orders=[128,256],
        tested_hour_samples=[16,32,64],relative_gate=.0025,
        gate='Each numerical component <=0.25%; successful probes do not prove all fitted states accurate',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        prepared_manifest_sha256=hashlib.sha256((preparation/'PREPARATION_FINISHED.json').read_bytes()).hexdigest(),
        torch=torch.__version__,numpy=np.__version__,device=device)
    if output.exists():
        raise FileExistsError('Preserve previous numerical evidence; use a new version')
    output.mkdir(parents=True)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for source in files:
        target=output/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    torch.set_num_threads(1)
    records=[]
    started=time.perf_counter()
    for case_id in selected:
        print('PREFLIGHT_START',case_id,flush=True)
        with np.load(preparation/case_id/'input.npz') as loaded:
            arrays={name:loaded[name] for name in loaded.files}
        context=configure(arrays,device=device,samples=16)
        reference_contexts={s:configure(arrays,dtype=torch.float64,samples=s) for s in (16,32,64)}
        peak_hour,peak_station=np.unravel_index(np.nanargmax(arrays['normalized_concentration']),
                                                arrays['normalized_concentration'].shape)
        station=context.stations_norm[peak_station]
        angle=np.deg2rad(arrays['wind_direction_deg'][peak_hour])
        up=np.array([np.sin(angle),np.cos(angle)])
        limits=[((.74 if value>0 else -.74)-point)/value for point,value in zip(station,up) if abs(value)>1e-12]
        distance=min(4000./context.length_m,min(v for v in limits if v>0))
        probes=[('center',np.zeros(2)),('at_peak_station',station),
                ('upwind_1km',station+up*(1000./context.length_m)),
                ('far_upwind_within_search',station+up*distance)]
        callbacks={order:HourlyResponse(context,order) for order in (16,32,64)}
        case_records=[]
        if device=='cuda':
            torch.cuda.reset_peak_memory_stats()
        for label,source in probes:
            for dp,dq in ((150.,50.),(30.,1.01),(1.01,1.01)):
                model=context.model
                with torch.no_grad():
                    model.xs.fill_(float(source[0]))
                    model.ys.fill_(float(source[1]))
                ia.set_model_anisotropic_diffusion_norm(model,
                    normalize_diffusivity_m2s(dp,1.,context.length_m),
                    normalize_diffusivity_m2s(dq,1.,context.length_m))
                actual_d=tuple(float(v.detach())*context.length_m**2/3600.
                               for v in ia.model_anisotropic_diffusion_norm(model))
                references={s:numpy_hourly_reference(c,source,*actual_d,np.linspace(0.,c.hours,5),order=128)
                            for s,c in reference_contexts.items()}
                c=reference_contexts[64]
                refined_reference=numpy_hourly_reference(c,source,*actual_d,np.linspace(0.,c.hours,5),order=256)
                row=dict(case_id=case_id,probe=label,source_norm=source.tolist(),
                    diffusivity_m2_s=list(actual_d),n_hours=context.hours,n_stations=len(context.stations_norm),
                    reference_release_128_to_256=float(np.linalg.norm(references[64]-refined_reference)/max(np.linalg.norm(refined_reference),1e-12)),
                    temporal_16_to_32=float(np.linalg.norm(references[16]-references[32])/max(np.linalg.norm(references[32]),1e-12)),
                    temporal_32_to_64=float(np.linalg.norm(references[32]-references[64])/max(np.linalg.norm(references[64]),1e-12)))
                for order,callback in callbacks.items():
                    value=callback(model,context.sigma_norm,context.xyt,context.basis).cpu().numpy()
                    if not np.isfinite(value).all() or np.any(value<0):
                        raise FloatingPointError('Nonfinite/negative measured-wind response')
                    row[f'release_{order}_to_reference']=float(np.linalg.norm(value-references[16])/max(np.linalg.norm(references[16]),1e-12))
                case_records.append(row)
                print('PREFLIGHT_PROBE',case_id,label,dp,dq,
                    'release32',round(row['release_32_to_reference'],6),
                    'time16',round(row['temporal_16_to_32'],6),flush=True)
        case_report=dict(case_id=case_id,rows=case_records,
            peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()) if device=='cuda' else None)
        (output/f'{case_id}.json').write_text(json.dumps(case_report,indent=2),encoding='utf-8')
        records.extend(case_records)
    max_release={order:max(v[f'release_{order}_to_reference'] for v in records) for order in (16,32,64)}
    max_time={s:max(v[f'temporal_{s}_to_{2*s}'] for v in records) for s in (16,32)}
    release=next((v for v in (16,32,64) if max_release[v]<=.0025),None)
    samples=next((v for v in (16,32) if max_time[v]<=.0025),None)
    max_reference=max(v['reference_release_128_to_256'] for v in records)
    report=dict(cases=len(selected),state_probes=len(records),seconds=time.perf_counter()-started,
        max_release_relative_difference=max_release,max_temporal_relative_difference=max_time,
        max_reference_release_relative_difference=max_reference,
        recommended_release_order=release,recommended_hour_samples=samples,
        ready_for_smoke=release is not None and samples is not None and max_reference<=.0025,
        warning='Probe gates are not an accuracy proof for source location or all candidate states; per-fit replay still required')
    (output/'PREFLIGHT_FINISHED.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation',type=Path,default=ROOT/'result/paper_experiments_20261017/industrial_preparation_v2_flags')
    parser.add_argument('--output',type=Path,default=ROOT/'result/paper_experiments_20261017/industrial_transport_preflight_v1')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    numerical_preflight(args.preparation,args.output,args.device)
