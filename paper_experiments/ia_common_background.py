"""IA-only elimination of Q>=0 and one common nonnegative background per time.

The physical source response is unchanged. All valid station rows remain in
the likelihood. Background-free contrasts are used only when the corresponding
background is interior; a zero background retains the uncentered observations.
"""

import time
import numpy as np
from scipy.optimize import lsq_linear,minimize,least_squares


class TimeGroups:
    def __init__(self,groups):
        self.labels,self.rows=np.unique(np.asarray(groups),return_inverse=True)
        self.count=np.bincount(self.rows)

    def means(self,values):
        values=np.asarray(values,float)
        flat=values.reshape(len(self.rows),-1)
        result=np.column_stack([np.bincount(self.rows,weights=column,minlength=len(self.count))
            for column in flat.T])/self.count[:,None]
        return result.reshape((len(self.count),)+values.shape[1:])

    def project(self,values,interior):
        shape=(len(interior),)+(1,)*(np.ndim(values)-1)
        return values-(self.means(values)*np.asarray(interior).reshape(shape))[self.rows]


def solve_q_background(matrix,observed,groups,q_upper):
    matrix,observed=np.asarray(matrix,float),np.asarray(observed,float)
    all_free=np.ones(len(groups.count),bool)
    centered=groups.project(matrix,all_free)
    scale=np.linalg.norm(centered,axis=0)
    scale=np.maximum(scale,np.finfo(float).eps*np.linalg.norm(matrix,axis=0))
    scale[scale==0]=1.
    limit=np.broadcast_to(q_upper,scale.shape)*scale
    if np.any(limit<=0) or not np.isfinite(matrix).all() or not np.isfinite(observed).all():
        raise ValueError('Finite observations/response and positive Q bounds required')

    def evaluate(beta):
        coefficients=beta/scale
        background=np.maximum(groups.means(observed-matrix@coefficients),0.)
        residual=matrix@coefficients+background[groups.rows]-observed
        return residual,background

    interior=all_free
    beta=np.zeros(matrix.shape[1])
    seen=set()
    for _ in range(30):
        key=interior.tobytes()
        if key in seen:
            break
        seen.add(key)
        design=groups.project(matrix,interior)/scale
        target=groups.project(observed,interior)
        solution=lsq_linear(design,target,bounds=(0.,limit),method='bvls',tol=1e-12,max_iter=500)
        if not solution.success:
            raise RuntimeError('Conditional Q solve failed: '+str(solution.message))
        beta=solution.x
        residual,background=evaluate(beta)
        updated=background>0
        if np.array_equal(updated,interior):
            return dict(coefficients=beta/scale,background=background,residual=residual,
                active_mask=solution.active_mask,background_interior=updated,inner_kkt=solution.optimality)
        interior=updated
    # The collapsed problem remains convex when the background active set changes.
    def objective(value):
        residual,_=evaluate(value)
        return .5*float(residual@residual),(matrix.T@residual)/scale
    solution=minimize(objective,beta,jac=True,bounds=list(zip(np.zeros_like(limit),limit)),
        method='L-BFGS-B',options=dict(maxiter=500,ftol=1e-14,gtol=1e-10,maxls=40))
    beta=solution.x
    gradient=objective(beta)[1]
    kkt=float(np.max(np.abs(beta-np.clip(beta-gradient,0.,limit))))
    if kkt>1e-7:
        raise RuntimeError('Collapsed Q/background solve did not satisfy normalized KKT: '+str(kkt))
    residual,background=evaluate(beta)
    active=np.zeros(len(beta),int)
    tolerance=1e-10*max(1.,np.linalg.norm(beta))
    active[beta<tolerance]=-1
    active[limit-beta<tolerance]=1
    return dict(coefficients=beta/scale,background=background,residual=residual,
        active_mask=active,background_interior=background>0,inner_kkt=kkt)


class CommonBackgroundResidual:
    def __init__(self,response,observed,row_times,q_upper):
        self.response=response
        self.observed=np.asarray(observed,float)
        self.groups=TimeGroups(row_times)
        self.q_upper=q_upper
        self.theta=None
        self.best=None
        self.calls=0

    def residual(self,theta):
        theta=np.asarray(theta,float)
        if self.theta is not None and np.array_equal(theta,self.theta):
            return self.solution['residual'].copy()
        self.matrix=np.asarray(self.response(theta,False),float)
        self.calls+=1
        self.solution=solve_q_background(self.matrix,self.observed,self.groups,self.q_upper)
        self.theta=theta.copy()
        sse=float(self.solution['residual']@self.solution['residual'])
        if self.best is None or sse<self.best['sse']:
            source=self.matrix@self.solution['coefficients']
            self.best=dict(theta=theta.copy(),sse=sse,
                coefficients=self.solution['coefficients'].copy(),
                background=self.solution['background'].copy(),source_prediction=source,
                prediction=source+self.solution['background'][self.groups.rows],
                inner_kkt=float(self.solution['inner_kkt']))
        return self.solution['residual'].copy()

    def jacobian(self,theta):
        self.residual(theta)
        matrix,derivative=self.response(np.asarray(theta),True)
        self.calls+=1
        np.testing.assert_allclose(matrix,self.matrix,rtol=1e-8,atol=1e-12)
        project=lambda value:self.groups.project(value,self.solution['background_interior'])
        coefficients=self.solution['coefficients']
        jac=project(np.einsum('nkd,k->nd',derivative,coefficients))
        free=self.solution['active_mask']==0
        if np.any(free):
            nuisance=project(matrix[:,free])
            scale=np.linalg.norm(nuisance,axis=0)
            scale[scale==0]=1.
            nuisance=nuisance/scale
            u,s,vt=np.linalg.svd(nuisance,full_matrices=False)
            valid=s>1e-12*s[0] if len(s) and s[0]>0 else np.zeros_like(s,dtype=bool)
            inverse=(vt[valid].T/s[valid])@u[:,valid].T
            jac-=nuisance@(inverse@jac)
            mixed=np.einsum('nkd,n->kd',project(derivative[:,free,:])/scale[None,:,None],
                self.solution['residual'])
            jac-=inverse.T@mixed
        return jac


def fit_common_background(response,observed,row_times,starts,lower,upper,q_upper,max_nfev=250):
    begun=time.perf_counter()
    objective=CommonBackgroundResidual(response,observed,row_times,q_upper)
    scale=np.maximum(np.max(np.abs(starts),axis=0),1e-3)
    scale[:2]=1.
    runs=[]
    for start in starts:
        fit=least_squares(objective.residual,start,jac=objective.jacobian,bounds=(lower,upper),
            method='trf',loss='linear',x_scale=scale,max_nfev=max_nfev,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        runs.append(dict(success=bool(fit.success),nfev=int(fit.nfev),sse=float(fit.fun@fit.fun)))
    result=dict(objective.best)
    result.update(seconds=time.perf_counter()-begun,optimizer_runs=runs,response_calls=objective.calls,
        converged=runs[int(np.argmin([item['sse'] for item in runs]))]['success'],
        background_groups=len(objective.groups.count))
    return result
