"""Fixed, independently calibrated observation covariance for profiled inversion.

An irregularly sampled stationary AR(1) process is whitened exactly, station by
station. No event rows are withheld, and no source truth enters calibration.
"""

import numpy as np
import torch


def estimate_background_noise(background, max_rho=.98):
    values=np.asarray(background,dtype=float)
    if values.ndim!=2 or len(values)<3 or not 0<=max_rho<1:
        raise ValueError('Background time-by-station table and stationary rho bound required')
    sigma=[]
    rho=[]
    counts=[]
    for column in values.T:
        finite=np.isfinite(column)
        if finite.sum()<3:
            raise ValueError('At least three independently supplied background samples per station required')
        centered=column-np.mean(column[finite])
        deviation=float(np.std(column[finite],ddof=1))
        pairs=finite[:-1]&finite[1:]
        denominator=float(np.sum(centered[:-1][pairs]**2))
        coefficient=float(centered[:-1][pairs]@centered[1:][pairs]/denominator) if denominator else 0.
        if not np.isfinite(deviation) or deviation<=0:
            raise ValueError('Background variance is not estimable; supply measured noise instead')
        sigma.append(deviation)
        rho.append(np.clip(coefficient,0.,max_rho))
        counts.append(int(finite.sum()))
    return dict(station_sigma=sigma,station_rho=rho,background_counts=counts,
        rho_bound=max_rho,source='Pre-event observations only; centered stationary AR1, no source labels')


class StationAR1Whitening:
    def __init__(self,times,stations,sigma,rho):
        times=np.asarray(times,dtype=float)
        stations=np.asarray(stations,dtype=int)
        sigma=np.asarray(sigma,dtype=float)
        rho=np.asarray(rho,dtype=float)
        if (times.ndim!=1 or stations.shape!=times.shape or not len(times)
                or sigma.ndim!=1 or rho.shape!=sigma.shape or not np.isfinite(times).all()
                or not np.isfinite(sigma).all() or np.any(sigma<=0)
                or not np.isfinite(rho).all() or np.any(rho<0) or np.any(rho>=1)
                or stations.min()<0 or stations.max()>=len(sigma)):
            raise ValueError('Finite timestamps, valid stations and positive stationary noise required')
        previous=np.arange(len(times))
        current_weight=1/sigma[stations]
        previous_weight=np.zeros(len(times))
        for station in np.unique(stations):
            indexes=np.flatnonzero(stations==station)
            delta=np.diff(times[indexes])
            if np.any(delta<=0):
                raise ValueError('Each station must retain strictly increasing observation times')
            correlation=rho[station]**delta
            innovations=sigma[station]*np.sqrt(1-correlation**2)
            previous[indexes[1:]]=indexes[:-1]
            current_weight[indexes[1:]]=1/innovations
            previous_weight[indexes[1:]]=correlation/innovations
        self.previous=previous
        self.current_weight=current_weight
        self.previous_weight=previous_weight

    def numpy(self,values):
        values=np.asarray(values,dtype=float)
        shape=(len(self.previous),)+(1,)*(values.ndim-1)
        return self.current_weight.reshape(shape)*values-self.previous_weight.reshape(shape)*values[self.previous]

    def tensor(self,values):
        shape=(len(self.previous),)+(1,)*(values.ndim-1)
        current=torch.as_tensor(self.current_weight,device=values.device,dtype=values.dtype).reshape(shape)
        prior=torch.as_tensor(self.previous_weight,device=values.device,dtype=values.dtype).reshape(shape)
        indexes=torch.as_tensor(self.previous,device=values.device)
        return current*values-prior*values[indexes]


class CovarianceResponse:
    def __init__(self,response,whitening):
        self.response,self.whitening=response,whitening
        if hasattr(response,'matrix_and_jacobian'):
            self.matrix_and_jacobian=self._matrix_and_jacobian

    def __call__(self,*args,**kwargs):
        return self.whitening.tensor(self.response(*args,**kwargs))

    def _matrix_and_jacobian(self,*args,**kwargs):
        matrix,derivative=self.response.matrix_and_jacobian(*args,**kwargs)
        return self.whitening.tensor(matrix),self.whitening.tensor(derivative)
