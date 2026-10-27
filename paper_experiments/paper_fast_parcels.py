"""Experimental vectorized evaluation of the existing zero-initial-field puffs.

Only fixed wind/time/quadrature/basis factors are cached. Source position and
diffusivities remain differentiable. No production default or solver is changed.
"""

import hashlib
import math
from pathlib import Path
import sys
import time
import weakref

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pinn_source'))
from ia_vpdi import model_anisotropic_diffusion_norm
from gaussian_parcels import RECURRENT_SOURCE_SCALE


class ParcelResponsePlan:
    def __init__(self, model, sigma, xyt, q_times, basis, subdivisions=8):
        cast=lambda value:torch.as_tensor(value,dtype=model.xs.dtype,device=model.xs.device)
        if int(subdivisions)!=subdivisions or subdivisions<1:
            raise ValueError('Positive integer release subdivisions required')
        if getattr(model,'source_position_mode','single')!='single':
            raise ValueError('Only a single fixed source is supported')
        if getattr(model,'recurrent_source_mode','gaussian') not in ('gaussian','cell_gaussian'):
            raise ValueError('A finite-width Gaussian source is required')
        with torch.no_grad():
            times=cast(model.recurrent_times).reshape(-1)
            qt=cast(q_times).reshape(-1)
            basis=cast(basis)
            xyt=cast(xyt)
            wind=torch.stack((cast(model.recurrent_u),cast(model.recurrent_v)),1)[:-1]
            if (len(times)<2 or len(qt)<2 or basis.ndim!=2 or basis.shape[0]!=len(qt)
                or xyt.ndim!=2 or xyt.shape[1]!=3 or len(xyt)==0 or wind.shape!=(len(times)-1,2)
                or not all(bool(torch.isfinite(v).all()) for v in (times,qt,basis,xyt,wind))
                or not bool((times.diff()>0).all()) or not bool((qt.diff()>0).all())
                or bool(qt[0]>times[0]) or bool(qt[-1]<times[-1])
                or bool((xyt[:,2]<times[0]).any()) or bool((xyt[:,2]>times[-1]).any())):
                raise ValueError('Invalid fixed transport clock, query table or basis')
            if basis.requires_grad or xyt.requires_grad:
                raise ValueError('Cached queries and bases must be fixed')
            width=getattr(model,'recurrent_source_sigma_norm',None)
            self.width=max(float(sigma),1e-4) if width is None else float(width)
            if not math.isfinite(self.width) or self.width<=0:
                raise ValueError('Positive source width required')
            speed=torch.linalg.vector_norm(wind,dim=1)
            direction=wind/speed.clamp_min(1e-12)[:,None]
            parallel=torch.stack((direction[:,0]**2,direction[:,0]*direction[:,1],direction[:,1]**2),1)
            parallel=torch.where((speed<=1e-12)[:,None],cast([.5,0.,.5]),parallel)
            perpendicular=cast([1.,0.,1.])-parallel
            breaks=torch.unique(torch.cat((times,qt[(qt>times[0])&(qt<times[-1])])),sorted=True)
            fractions=torch.linspace(0.,1.,int(subdivisions)+1,device=times.device,dtype=times.dtype)
            entries=[]
            for end in torch.unique(xyt[:,2],sorted=True):
                left=breaks[:-1][breaks[:-1]<end]
                right=torch.minimum(breaks[1:][:len(left)],end)
                release=(left[:,None]+(right-left)[:,None]*fractions).reshape(-1)
                weights=((right-left)[:,None]/subdivisions).expand(-1,subdivisions+1).clone()
                weights[:,0]*=.5;weights[:,-1]*=.5
                release=torch.cat((release,times[:1]))
                weights=torch.cat((weights.reshape(-1),cast([0.])))
                high=torch.searchsorted(qt,release.contiguous()).clamp(1,len(qt)-1)
                alpha=(release-qt[high-1])/(qt[high]-qt[high-1])
                release_basis=(1-alpha[:,None])*basis[high-1]+alpha[:,None]*basis[high]
                residence=(torch.minimum(times[1:,None],end)-
                           torch.maximum(times[:-1,None],release[None,:])).clamp_min(0.)
                indexes=torch.nonzero(xyt[:,2]==end,as_tuple=False).flatten()
                weighted_basis=(weights*torch.exp(-max(float(model.recurrent_decay_norm),0.)*(end-release)))[:,None]*release_basis
                entries.append((indexes,xyt[indexes,:2],residence.T@wind,
                                residence.T@parallel,residence.T@perpendicular,weighted_basis))
            groups=len(entries)
            self.station_count=max(len(e[0]) for e in entries)
            self.release_count=max(len(e[2]) for e in entries)
            self.displacement=times.new_zeros((groups,self.release_count,2))
            self.parallel=times.new_zeros((groups,self.release_count,3))
            self.perpendicular=torch.zeros_like(self.parallel)
            self.weighted_basis=times.new_zeros((groups,self.release_count,basis.shape[1]))
            self.xy=times.new_zeros((groups,self.station_count,2))
            self.output_indexes=[]
            self.valid_flat=[]
            for i,(indexes,xy,displacement,par,perp,weighted) in enumerate(entries):
                count=len(displacement)
                self.displacement[i,:count]=displacement
                self.parallel[i,:count]=par
                self.perpendicular[i,:count]=perp
                self.weighted_basis[i,:count]=weighted
                self.xy[i,:len(indexes)]=xy
                self.output_indexes.append(indexes)
                self.valid_flat.append(torch.arange(len(indexes),device=times.device)+i*self.station_count)
            self.output_indexes=torch.cat(self.output_indexes)
            self.valid_flat=torch.cat(self.valid_flat)
            self.query_count=len(xyt)
            self.dtype,self.device=model.xs.dtype,model.xs.device

    def evaluate(self,model,max_kernel_elements=8_000_000,physical_parameters=None):
        if model.xs.dtype!=self.dtype or model.xs.device!=self.device:
            raise ValueError('Rebuild the plan after changing dtype/device')
        if physical_parameters is not None:
            source=physical_parameters[:2]
            dq=physical_parameters[2]
            dp=dq+physical_parameters[3]
        elif getattr(model,'recurrent_anisotropic_diffusion',False):
            dp,dq=model_anisotropic_diffusion_norm(model)
        else:
            dp=dq=model.recurrent_d_min_norm+model.D()*model.recurrent_d_scale_norm
        if physical_parameters is None:
            source=torch.stack((model.xs.reshape(()),model.ys.reshape(())))
        chunk=max(1,int(max_kernel_elements)//(self.station_count*self.release_count))
        pieces=[]
        for start in range(0,len(self.xy),chunk):
            stop=start+chunk
            covariance=2*(dp*self.parallel[start:stop]+dq*self.perpendicular[start:stop])
            xx=covariance[:,:,0]+self.width**2
            xy=covariance[:,:,1]
            yy=covariance[:,:,2]+self.width**2
            determinant=xx*yy-xy*xy
            if bool((determinant<=0).any()):
                raise FloatingPointError('Nonpositive cached puff covariance determinant')
            delta=self.xy[start:stop,:,None,:]-source-self.displacement[start:stop,None,:,:]
            dx,dy=delta[:,:,:,0],delta[:,:,:,1]
            exponent=(yy[:,None,:]*dx**2+xx[:,None,:]*dy**2-2*xy[:,None,:]*dx*dy)/determinant[:,None,:]
            kernel=torch.exp(-.5*exponent)/(2*math.pi*torch.sqrt(determinant[:,None,:]))
            pieces.append(torch.einsum('tsr,trb->tsb',kernel,self.weighted_basis[start:stop]))
        value=RECURRENT_SOURCE_SCALE*torch.cat(pieces).reshape(-1,self.weighted_basis.shape[-1])[self.valid_flat]
        output=value.new_zeros((self.query_count,value.shape[1]))
        return output.index_copy(0,self.output_indexes,value)


class CachedMeanResponse:
    """Bounded per-model cache with content-based fixed-context invalidation."""
    def __init__(self,samples=16,hours=6,stations=5,subdivisions=8):
        self.samples,self.hours,self.stations=int(samples),int(hours),int(stations)
        self.subdivisions=int(subdivisions)
        self._model=None
        self._plans={}
        self.build_seconds=0.
        self.calls=0

    def _plan(self, model, sigma, xyt, basis):
        if self._model is None or self._model() is not model:
            self._plans.clear()
            self._model=weakref.ref(model)
        digest=hashlib.sha256()
        for value in (model.recurrent_times,model.recurrent_u,model.recurrent_v,
                      model.recurrent_integration_times,xyt,basis):
            if isinstance(value,torch.Tensor):
                if value.requires_grad:
                    raise ValueError('Only fixed wind/clock/query/basis arrays may be cached')
                value=value.detach().cpu().numpy()
            value=np.asarray(value)
            digest.update(str((value.shape,value.dtype)).encode())
            digest.update(value.tobytes())
        digest.update(str((model.xs.dtype,model.xs.device,float(sigma),
            getattr(model,'recurrent_source_sigma_norm',None),model.recurrent_decay_norm)).encode())
        key=digest.digest()
        if key not in self._plans:
            if len(self._plans)>=8:
                self._plans.clear()
            started=time.perf_counter()
            self._plans[key]=ParcelResponsePlan(model,sigma,xyt,model.recurrent_integration_times,
                                                basis,self.subdivisions)
            self.build_seconds+=time.perf_counter()-started
        return self._plans[key]

    def __call__(self,model,sigma,xyt,basis,track_gradients=False,physical_parameters=None):
        plan = self._plan(model, sigma, xyt, basis)
        self.calls+=1
        with torch.enable_grad() if track_gradients else torch.no_grad():
            response=plan.evaluate(model,physical_parameters=physical_parameters)
        if response.shape[0]!=self.hours*self.samples*self.stations:
            raise ValueError('Mean operator dimensions do not match query rows')
        return response.reshape(self.hours,self.samples,self.stations,-1).mean(1).reshape(
            self.hours*self.stations,-1)

    def evaluate_population(self, model, sigma, xyt, basis, physical_parameters,
                            batch_size=4, max_kernel_elements=2_000_000, compute_dtype=None):
        """Same linear response, bounded particle/query batches; no optimizer change."""
        plan = self._plan(model, sigma, xyt, basis)
        dtype = plan.dtype if compute_dtype is None else compute_dtype
        if not hasattr(plan, '_population_arrays'):
            plan._population_arrays = {}
        if dtype not in plan._population_arrays:
            plan._population_arrays[dtype] = {name: getattr(plan, name).to(dtype=dtype)
                for name in ('parallel', 'perpendicular', 'xy', 'displacement', 'weighted_basis')}
        arrays = plan._population_arrays[dtype]
        parameters = torch.as_tensor(physical_parameters, dtype=dtype, device=plan.device)
        if parameters.ndim != 2 or parameters.shape[1] != 4 or batch_size < 1:
            raise ValueError('Population must have shape particles x four physical parameters')
        outputs = []
        with torch.no_grad():
            for start in range(0, len(parameters), batch_size):
                p = parameters[start:start+batch_size]
                pieces = []
                chunk = max(1, max_kernel_elements//(len(p)*plan.station_count*plan.release_count))
                for left in range(0, len(plan.xy), chunk):
                    right = left+chunk
                    covariance = 2*((p[:, 2]+p[:, 3])[:, None, None, None]*arrays['parallel'][None, left:right]
                        + p[:, 2, None, None, None]*arrays['perpendicular'][None, left:right])
                    xx = covariance[..., 0]+plan.width**2
                    xy = covariance[..., 1]
                    yy = covariance[..., 2]+plan.width**2
                    det = xx*yy-xy*xy
                    if bool((det <= 0).any()):
                        raise FloatingPointError('Nonpositive population covariance')
                    delta = (arrays['xy'][None, left:right, :, None, :]-p[:, None, None, None, :2]
                             -arrays['displacement'][None, left:right, None, :, :])
                    dx, dy = delta[..., 0], delta[..., 1]
                    exponent = (yy[:, :, None, :]*dx**2+xx[:, :, None, :]*dy**2
                        -2*xy[:, :, None, :]*dx*dy)/det[:, :, None, :]
                    kernel = torch.exp(-.5*exponent)/(2*math.pi*torch.sqrt(det[:, :, None, :]))
                    pieces.append(torch.einsum('ptsr,trb->ptsb', kernel, arrays['weighted_basis'][left:right]))
                value = RECURRENT_SOURCE_SCALE*torch.cat(pieces, dim=1).reshape(len(p), -1, basis.shape[1])[:, plan.valid_flat]
                output = value.new_zeros((len(p), plan.query_count, value.shape[-1]))
                output.index_copy_(1, plan.output_indexes, value)
                outputs.append(output.reshape(len(p), self.hours, self.samples, self.stations, -1)
                    .mean(2).reshape(len(p), self.hours*self.stations, -1))
        self.calls += len(parameters)
        return torch.cat(outputs)
