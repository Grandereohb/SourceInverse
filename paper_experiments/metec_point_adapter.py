"""Height-aware point observation adapter; reuses the tested physical transport core."""

from types import SimpleNamespace

import numpy as np
import torch

from paper_physical_beam_adapter import PhysicalBeamPlan, PhysicalBeamResponse, R_GAS, M_CH4


def xy_from_lon_lat(lon, lat, origin):
    return np.column_stack((np.deg2rad(np.asarray(lon)-origin[0])*6371000*np.cos(np.deg2rad(origin[1])),
                            np.deg2rad(np.asarray(lat)-origin[1])*6371000))


def hat_basis(clock, count=5):
    return np.column_stack([np.interp(clock,np.linspace(clock[0],clock[-1],count),np.eye(count)[j])
                            for j in range(count)])


def window_mean_release(coefficients,duration_s,window_s):
    values=np.asarray(coefficients,dtype=float)
    nodes=np.linspace(0.,duration_s,len(values))
    left,right=np.asarray(window_s,dtype=float)
    if values.ndim!=1 or len(values)<2 or not 0<=left<right<=duration_s:
        raise ValueError('A nonempty release window and hat coefficients are required')
    clock=np.r_[left,nodes[(nodes>left)&(nodes<right)],right]
    return float(np.trapezoid(np.interp(clock,nodes,values),clock)/(right-left))


class MaskedPointResponse(PhysicalBeamResponse):
    def __init__(self, plan, xyt, basis, indices):
        super().__init__(plan,xyt,basis)
        self.indices=torch.as_tensor(indices,dtype=torch.long,device=plan.device)

    def __call__(self,*args,**kwargs):
        return super().__call__(*args,**kwargs)[self.indices]

    def evaluate_population(self,*args,**kwargs):
        return super().evaluate_population(*args,**kwargs)[:,self.indices]

    @torch.no_grad()
    def matrix_and_jacobian(self,model,sigma,xyt,basis,physical_parameters):
        """Exact derivatives of the same point kernel; no finite differences or changed objective."""
        if not torch.equal(self.xyt,xyt) or not torch.equal(self.basis,basis):
            raise ValueError('Fixed point query and basis required')
        p=self.plan
        theta=physical_parameters
        dq=theta[2]*p.d_scale
        dp=(theta[2]+theta[3])*p.d_scale
        source=theta[:2]*p.length
        chunk=max(1,2_000_000//(len(p.xy)*p.releases))
        outputs=[[] for _ in range(5)]
        for first in range(0,p.groups,chunk):
            last=min(first+chunk,p.groups)
            par,per=p.parallel[first:last],p.perpendicular[first:last]
            covariance=2*(dp*par+dq*per)
            valid=p.weighted_basis[first:last].abs().sum(-1)>0
            xx=torch.where(valid,covariance[:,:,0]+p.source_variance_m2,1.)[:,None,:]
            xy=torch.where(valid,covariance[:,:,1],0.)[:,None,:]
            yy=torch.where(valid,covariance[:,:,2]+p.source_variance_m2,1.)[:,None,:]
            det=xx*yy-xy*xy
            delta=p.xy[None,:,None,:]-source-p.displacement[first:last,None,:,:]
            dx,dy=delta[...,0],delta[...,1]
            alpha,beta=(yy*dx-xy*dy)/det,(xx*dy-xy*dx)/det
            horizontal=torch.exp(-.5*(dx*alpha+dy*beta))/(2*np.pi*torch.sqrt(det))
            age=p.age[first:last,None,:]
            z=p.z[None,:,None]
            variance=2*dq*age
            near=torch.exp(-.5*(z-p.height)**2/variance)/torch.sqrt(2*np.pi*variance)
            reflected=torch.exp(-.5*(z+p.height)**2/variance)/torch.sqrt(2*np.pi*variance)
            vertical=near+reflected
            density=horizontal*vertical
            d_vertical=p.d_scale*(near*(-.5/dq+(z-p.height)**2/(4*dq*dq*age))+
                reflected*(-.5/dq+(z+p.height)**2/(4*dq*dq*age)))
            d_horizontal_iso=p.d_scale*age*(alpha**2+beta**2-(xx+yy)/det)
            a,b,c=par[:,None,:,0],par[:,None,:,1],par[:,None,:,2]
            d_horizontal_par=p.d_scale*(a*alpha**2+2*b*alpha*beta+c*beta**2-
                (yy*a-2*xy*b+xx*c)/det)
            fields=[density,density*alpha*p.length,density*beta*p.length,
                density*d_horizontal_iso+horizontal*d_vertical,density*d_horizontal_par]
            for store,field in zip(outputs,fields):
                store.append(torch.einsum('gpr,grb->gpb',field,p.weighted_basis[first:last]))
        def integrate(parts):
            values=torch.cat(parts).reshape(p.minutes,p.time_order,p.beams,1,-1)
            return torch.einsum('mtbqk,t,q->mbk',values,p.time_weights,p.beam_weights).reshape(p.minutes*p.beams,-1)[self.indices]
        self.calls+=1
        matrix=integrate(outputs[0])
        derivative=torch.stack([integrate(parts) for parts in outputs[1:]],dim=-1)
        if not torch.isfinite(matrix).all() or not torch.isfinite(derivative).all():
            raise FloatingPointError('Nonfinite analytic point derivative')
        return matrix,derivative


def configure(data, height_m, *, device='cpu', time_order=4, release_order=8, history_minutes=20):
    observations=data['concentration']
    count=len(observations)
    edges=np.arange(count+1)*60.
    clock=np.unique(np.r_[edges,np.linspace(0.,edges[-1],5)])
    basis_np=hat_basis(clock)
    xyz=np.column_stack((data['stations_xy'],data['station_heights']))
    # Match the author's fixed methane kg/m3-to-ppm factor without inventing weather observations.
    pressure=101325.
    temperature=1.524*pressure*M_CH4/R_GAS
    plan=PhysicalBeamPlan(edges,data['wind_xy'],xyz,xyz,temperature,pressure,clock,basis_np,
        height_m=height_m,device=device,dtype=torch.float64,beam_solver='point',
        time_order=time_order,release_order=release_order,source_variance_m2=0.,
        release_window_s=data.get('release_window_s'),
        max_age_s=history_minutes*60.)
    model=SimpleNamespace(xs=torch.zeros((),dtype=torch.float64,device=device))
    indices=np.flatnonzero(np.isfinite(observations.ravel()))
    xyt=torch.as_tensor(np.column_stack((np.tile(data['stations_xy']/100.,(count,1)),
        np.repeat(np.arange(count)+.5,len(xyz))))[indices],dtype=torch.float64,device=device)
    basis=torch.as_tensor(basis_np,dtype=torch.float64,device=device)
    return model,xyt,basis,MaskedPointResponse(plan,xyt,basis,indices),observations.ravel()[indices]
