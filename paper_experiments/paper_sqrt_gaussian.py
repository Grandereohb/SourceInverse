"""Stable2D Gaussian density and covariance scores from diffusion factors."""

import math

import torch


def combine_diffusion_factors(parallel,perpendicular,dp,dq):
    """Analytic Givens reduction of two LOWER triangular covariance factors."""
    p,q=torch.sqrt(2*dp),torch.sqrt(2*dq)
    x,y=p*parallel[...,0,0],q*perpendicular[...,0,0]
    a=torch.hypot(x,y)
    cosine,sine=x/a,y/a
    pc,qf=p*parallel[...,1,0],q*perpendicular[...,1,0]
    c=cosine*pc+sine*qf
    remainder=sine*pc-cosine*qf
    d=torch.linalg.vector_norm(torch.stack((remainder,p*parallel[...,1,1],
        q*perpendicular[...,1,1]),dim=-1),dim=-1)
    return a,c,d


def gaussian_from_factors(dx,dy,parallel,perpendicular,dp,dq,*,derivatives=False):
    aa,cc,dd=combine_diffusion_factors(parallel,perpendicular,dp,dq)
    a,c,d=aa[:,None,:],cc[:,None,:],dd[:,None,:]
    z0=dx/a
    z1=(dy-c*z0)/d
    density=torch.exp(-.5*(z0*z0+z1*z1))/(2*math.pi*torch.abs(a*d))
    if not derivatives:
        return density
    beta=z1/d
    alpha=z0/a-(c/a)*beta
    scores=[]
    for factor in (parallel,perpendicular):
        whitening0=factor[...,0,:]/aa[...,None]
        whitening1=(factor[...,1,:]-cc[...,None]*whitening0)/dd[...,None]
        trace=(whitening0.square()+whitening1.square()).sum(-1)[:,None,:]
        projection=factor[:,None,:,0,:]*alpha[...,None]+factor[:,None,:,1,:]*beta[...,None]
        scores.append(projection.square().sum(-1)-trace)
    return density,alpha,beta,scores[0]+scores[1],scores[0]
