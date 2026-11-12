"""Observation-wind ensemble, not a spatially resolved wind PDE.

Each member is the unchanged physical Green response driven by one measured
wind history. Source-distance weights are fixed by sensor geometry, not truth.
This represents unresolved local-wind uncertainty and preserves Q linearity.
"""

import numpy as np
import torch


class LocalWindEnsembleResponse:
    def __init__(self, members, wind_xy_m, *, length_m=100., distance_floor_m=1.):
        positions=np.asarray(wind_xy_m,dtype=float)
        if (not members or positions.shape!=(len(members),2) or
                not np.isfinite(positions).all() or min(length_m,distance_floor_m)<=0):
            raise ValueError('Finite wind sensor positions and positive physical scales required')
        self.members=list(members)
        self.positions=positions
        self.length=float(length_m)
        self.floor=float(distance_floor_m)
        self.calls=0

    def weights_and_derivative(self, theta):
        stations=theta.new_tensor(self.positions)
        delta=theta[:2]*self.length-stations
        distance=delta.square().sum(-1)+self.floor**2
        inverse=distance.reciprocal()
        weights=inverse/inverse.sum()
        log_derivative=-2*self.length*delta/distance[:,None]
        derivative=weights[:,None]*(log_derivative-(weights[:,None]*log_derivative).sum(0))
        return weights,derivative

    def __call__(self,model,sigma,xyt,basis,*,physical_parameters,**kwargs):
        weights,_=self.weights_and_derivative(physical_parameters)
        value=None
        for weight,member in zip(weights,self.members):
            contribution=weight*member(model,sigma,xyt,basis,
                physical_parameters=physical_parameters,**kwargs)
            value=contribution if value is None else value+contribution
        self.calls+=1
        return value

    @torch.no_grad()
    def matrix_and_jacobian(self,model,sigma,xyt,basis,physical_parameters):
        weights,weight_derivative=self.weights_and_derivative(physical_parameters)
        value=derivative=None
        for index,member in enumerate(self.members):
            matrix,jacobian=member.matrix_and_jacobian(model,sigma,xyt,basis,physical_parameters)
            contribution=weights[index]*matrix
            gradient=weights[index]*jacobian
            gradient[...,:2]+=matrix[...,None]*weight_derivative[index]
            value=contribution if value is None else value+contribution
            derivative=gradient if derivative is None else derivative+gradient
        self.calls+=1
        return value,derivative
