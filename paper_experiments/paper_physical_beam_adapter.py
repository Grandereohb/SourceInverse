"""Experimental mass-conserving3D beam adapter; production transport unchanged.

Horizontal diffusion follows the measured horizontal wind; vertical diffusivity
equals transverse diffusivity (no extra fitted parameter). Reflecting ground,
no mean vertical advection or methane decay. A2m-square source is approximated
by a horizontal Gaussian with the same per-axis variance1/3m2. Source height is
fixed per experiment, not selected by fit. Q coefficients have units kg/s.
"""

from contextlib import nullcontext
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
import torch

R_GAS = 8.31446261815324
M_CH4 = .01604246


def ppm_per_kg_m3(temperature_k, pressure_pa):
    if (not np.isfinite(np.asarray(temperature_k)).all() or not np.isfinite(np.asarray(pressure_pa)).all()
            or np.any(np.asarray(temperature_k) <= 0) or np.any(np.asarray(pressure_pa) <= 0)):
        raise ValueError('Finite positive absolute temperature and pressure required')
    return R_GAS * np.asarray(temperature_k) / (M_CH4 * np.asarray(pressure_pa)) * 1e6


def reflected_vertical_density(z, height, age, diffusivity):
    variance = 2*diffusivity*age
    return (torch.exp(-.5*(z-height)**2/variance) +
            torch.exp(-.5*(z+height)**2/variance))/torch.sqrt(2*math.pi*variance)


def horizontal_gaussian_density(dx, dy, xx, xy, yy):
    determinant = torch.as_tensor(xx*yy-xy*xy, dtype=dx.dtype, device=dx.device)
    exponent = (yy*dx**2+xx*dy**2-2*xy*dx*dy)/determinant
    return torch.exp(-.5*exponent)/(2*math.pi*torch.sqrt(determinant))


def horizontal_gaussian_beam_mean(dx,dy,lx,ly,xx,xy,yy):
    """Exact segment MEAN, with stable normal-CDF tail subtraction."""
    determinant=torch.as_tensor(xx*yy-xy*xy,dtype=dx.dtype,device=dx.device)
    ad=torch.as_tensor(yy*lx**2+xx*ly**2-2*xy*lx*ly,dtype=dx.dtype,device=dx.device)
    bn=yy*lx*dx+xx*ly*dy-xy*(lx*dy+ly*dx)
    denominator=torch.sqrt(ad*determinant)
    lower,upper=bn/denominator,(ad+bn)/denominator
    # Use survival probabilities on the positive tail; otherwise log-CDF.
    lo=torch.where(lower>=0,-upper,lower)
    hi=torch.where(lower>=0,-lower,upper)
    log_low,log_high=torch.special.log_ndtr(lo),torch.special.log_ndtr(hi)
    probability=torch.exp(log_high)*(-torch.expm1(log_low-log_high))
    perpendicular_exponent=(lx*dy-ly*dx)**2/ad
    return torch.exp(-.5*perpendicular_exponent)*probability/torch.sqrt(2*math.pi*ad)


def _integral_at(times, edges, values):
    times = np.asarray(times, dtype=float)
    index = np.minimum(np.searchsorted(edges, times, side='right')-1, len(values)-1)
    index = np.maximum(index, 0)
    cumulative = np.vstack((np.zeros((1, values.shape[1])),
                            np.cumsum(np.diff(edges)[:, None]*values, axis=0)))
    return cumulative[index]+(times-edges[index])[..., None]*values[index]


class PhysicalBeamPlan:
    """Fixed wind, clock and beam quadratures, differentiable source and D.

Wind edges are in seconds, velocities m/s, beam endpoints meters, atmosphere
per minute. Release quadrature uses sqrt(age), regularizing the t=release
vertical delta-source endpoint rather than introducing an artificial height.
"""
    def __init__(self, edges_s, velocity_xy, starts_m, ends_m, temperatures,
                 pressures, q_clock_s, q_basis, *, height_m, beam_order=32,
                 time_order=4, release_order=4, length_m=100., time_unit_s=60.,
                 device='cpu', dtype=torch.float64,beam_solver='quadrature'):
        edges, wind, starts, ends, clock, basis = [np.asarray(v, dtype=float) for v in
            (edges_s, velocity_xy, starts_m, ends_m, q_clock_s, q_basis)]
        if (edges.ndim != 1 or len(edges)<2 or edges[0]!=0 or np.any(np.diff(edges)<=0)
                or wind.shape!=(len(edges)-1,2) or starts.ndim!=2 or starts.shape[1]!=3
                or ends.shape!=starts.shape or clock.ndim!=1 or np.any(np.diff(clock)<=0)
                or clock[0]!=0 or clock[-1]<edges[-1] or basis.ndim!=2 or basis.shape[0]!=len(clock)
                or min(beam_order,time_order,release_order)<2 or min(length_m,time_unit_s)<=0
                or not np.isfinite([height_m,length_m,time_unit_s]).all()
                or height_m<0 or np.any(starts[:,2]<0) or np.any(ends[:,2]<0)
                or any(not np.isfinite(v).all() for v in (edges,wind,starts,ends,clock,basis))
                or np.any(np.linalg.norm(ends-starts,axis=1)==0)
                or beam_solver not in ('quadrature','analytic')):
            raise ValueError('Invalid finite physical beam/clock/transport specification')
        self.height, self.length, self.d_scale = float(height_m), float(length_m), length_m**2/time_unit_s
        self.minutes, self.beams = len(edges)-1, len(starts)
        self.time_order, self.beam_order = int(time_order), int(beam_order)
        self.beam_solver=beam_solver
        if beam_solver=='analytic' and not np.all(starts[:,2]==ends[:,2]):
            raise ValueError('Analytic horizontal beam mean requires equal endpoint heights')
        self.dtype, self.device = dtype, torch.device(device)
        concentration_scale = np.broadcast_to(ppm_per_kg_m3(temperatures, pressures), (self.minutes,))
        bnodes, bweights = leggauss(beam_order)
        tnodes, tweights = leggauss(time_order)
        rnodes, rweights = leggauss(release_order)
        beam_points = starts[:,None,:]+(bnodes[None,:,None]+1)*.5*(ends-starts)[:,None,:]
        if beam_solver=='analytic':
            beam_points=starts[:,None,:]
            self.beam_order=1
            bweights=np.array([2.])
        self.directions=torch.as_tensor((ends-starts)[:,:2],dtype=dtype,device=device)
        self.xy = torch.as_tensor(beam_points[...,:2].reshape(-1,2), dtype=dtype, device=device)
        self.device = self.xy.device
        self.z = torch.as_tensor(beam_points[...,2].reshape(-1), dtype=dtype, device=device)
        self.beam_weights = torch.as_tensor(bweights*.5, dtype=dtype, device=device)
        self.time_weights = torch.as_tensor(tweights*.5, dtype=dtype, device=device)
        speed = np.linalg.norm(wind, axis=1)
        direction = wind/np.maximum(speed[:,None], np.finfo(float).tiny)
        # At exact calm choose a documented x direction; isotropy is not imposed.
        direction[speed==0] = [1.,0.]
        parallel = np.column_stack((direction[:,0]**2, direction[:,0]*direction[:,1], direction[:,1]**2))
        perpendicular = np.array([1.,0.,1.])-parallel
        entries = []
        for minute in range(self.minutes):
            times = .5*(edges[minute]+edges[minute+1])+.5*(edges[minute+1]-edges[minute])*tnodes
            for end in times:
                breaks = np.unique(np.r_[edges[edges<end], clock[clock<end], end])
                roots_left, roots_right = np.sqrt(end-breaks[1:]), np.sqrt(end-breaks[:-1])
                roots = ((roots_right+roots_left)[:,None]*.5+
                    (roots_right-roots_left)[:,None]*.5*rnodes).ravel()
                age = roots**2
                release = end-age
                weights = ((roots_right-roots_left)[:,None]*.5*rweights).ravel()*2*roots
                release_basis = np.column_stack([np.interp(release,clock,basis[:,j]) for j in range(basis.shape[1])])
                entries.append((age,
                    _integral_at(end,edges,wind)-_integral_at(release,edges,wind),
                    _integral_at(end,edges,parallel)-_integral_at(release,edges,parallel),
                    _integral_at(end,edges,perpendicular)-_integral_at(release,edges,perpendicular),
                    weights[:,None]*release_basis*concentration_scale[minute]))
        count, maximum, columns = len(entries), max(len(v[0]) for v in entries), basis.shape[1]
        arrays = [np.ones((count,maximum)), np.zeros((count,maximum,2)),
                  np.zeros((count,maximum,3)), np.zeros((count,maximum,3)),
                  np.zeros((count,maximum,columns))]
        for i, values in enumerate(entries):
            for destination, value in zip(arrays, values):
                destination[i,:len(values[0])] = value
        self.age, self.displacement, self.parallel, self.perpendicular, self.weighted_basis = [
            torch.as_tensor(v,dtype=dtype,device=device) for v in arrays]
        self.groups, self.releases = count, maximum

    def evaluate(self, model, max_kernel_elements=2_000_000, physical_parameters=None):
        if model.xs.device!=self.device or model.xs.dtype!=self.dtype:
            raise ValueError('Physical plan and model device/dtype must match')
        if physical_parameters is None:
            parallel, transverse = model.anisotropic_D()
            scale = float(getattr(model,'recurrent_d_scale_norm',1.))
            dp = (float(getattr(model,'recurrent_d_parallel_min_norm',0.))+scale*parallel)*self.d_scale
            dq = (float(getattr(model,'recurrent_d_perp_min_norm',0.))+scale*transverse)*self.d_scale
            dp = torch.maximum(dp,dq)
            source = torch.stack((model.xs.reshape(()),model.ys.reshape(())))*self.length
        else:
            dq = physical_parameters[2]*self.d_scale
            dp = (physical_parameters[2]+physical_parameters[3])*self.d_scale
            source = physical_parameters[:2]*self.length
        chunk = max(1,int(max_kernel_elements)//(len(self.xy)*self.releases))
        responses = []
        for first in range(0,self.groups,chunk):
            last = min(first+chunk,self.groups)
            covariance = 2*(dp*self.parallel[first:last]+dq*self.perpendicular[first:last])
            # Same per-axis variance as the documented2m-square release frame.
            xx, xy, yy = covariance[:,:,0]+1/3, covariance[:,:,1], covariance[:,:,2]+1/3
            delta = self.xy[None,:,None,:]-source-self.displacement[first:last,None,:,:]
            dx,dy = delta[...,0],delta[...,1]
            if self.beam_solver=='analytic':
                horizontal=horizontal_gaussian_beam_mean(dx,dy,
                    self.directions[None,:,None,0],self.directions[None,:,None,1],
                    xx[:,None,:],xy[:,None,:],yy[:,None,:])
            else:
                horizontal = horizontal_gaussian_density(dx,dy,xx[:,None,:],xy[:,None,:],yy[:,None,:])
            vertical = reflected_vertical_density(self.z[None,:,None],self.height,
                                                  self.age[first:last,None,:],dq)
            responses.append(torch.einsum('gpr,grb->gpb',horizontal*vertical,self.weighted_basis[first:last]))
        response = torch.cat(responses).reshape(self.minutes,self.time_order,self.beams,self.beam_order,-1)
        return torch.einsum('mtbqk,t,q->mbk',response,self.time_weights,self.beam_weights).reshape(
            self.minutes*self.beams,-1)


class PhysicalBeamResponse:
    def __init__(self, plan, observation_xyt, q_basis):
        self.plan = plan
        self.xyt = torch.as_tensor(observation_xyt, dtype=plan.dtype, device=plan.device)
        self.basis = torch.as_tensor(q_basis, dtype=plan.dtype, device=plan.device)
        self.calls = 0

    def __call__(self, model, sigma_src, observation_xyt, q_basis, track_gradients=False,
                 physical_parameters=None):
        if not torch.equal(self.xyt,observation_xyt) or not torch.equal(self.basis,q_basis):
            raise ValueError('Physical beam adapter is bound to a fixed query/basis definition')
        self.calls += 1
        with nullcontext() if track_gradients else torch.no_grad():
            value = self.plan.evaluate(model,physical_parameters=physical_parameters)
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Nonfinite physical beam response')
        return value
