"""Working capacity criterion accounting for correlated residual energy.

For a Gaussian residual with unit marginal variance and correlation C,
E[r'r]=tr(C), Var[r'r]=2tr(C'C). Moment matching gives energy degrees
of freedom nu=tr(C)^2/tr(C'C). Replacing independent sample size by nu
is an approximate capacity criterion, not an exact marginal likelihood.
The fitting residual and variable-projection equations remain unchanged.
"""

import numpy as np


def residual_energy_dof(times,stations,station_rho):
    times,stations,rho=np.asarray(times),np.asarray(stations),np.asarray(station_rho)
    if times.ndim!=1 or stations.shape!=times.shape or len(times)==0:
        raise ValueError('Matching nonempty observation minute/station indices required')
    total_square=0.
    for index in np.unique(stations):
        clock=times[stations==index]
        gaps=np.abs(clock[:,None]-clock[None,:])
        total_square+=float(np.sum(float(rho[index])**(2*gaps)))
    return float(len(times)**2/total_square)


def correlated_capacity_score(sse,n_obs,energy_dof,n_parameters):
    if (sse<0 or not np.isfinite(sse) or not n_obs>=energy_dof>n_parameters>=1):
        raise ValueError('Finite SSE and enough residual energy degrees of freedom required')
    return float(energy_dof*np.log(max(sse/n_obs,np.finfo(float).tiny))+
        n_parameters*np.log(energy_dof))
