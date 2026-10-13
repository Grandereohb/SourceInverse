"""Opt-in full-space Gaussian parcels with linear source-time quadrature.

Coordinates, wind, diffusion and time use the configured normalized units.
The domain is unbounded: neither absorbing grid boundaries nor truncated-source
renormalization are reproduced. Hourly wind/tensor intervals remain unchanged.
"""

import math

import torch

from field import RECURRENT_INITIAL_RELEASE_FRACTION, RECURRENT_SOURCE_SCALE


def gaussian_parcel_response(model, sigma_src, observation_xyt, q_times, q_basis,
                             subdivisions=16, initial_release_fraction=None):
    if isinstance(subdivisions, bool) or int(subdivisions) != subdivisions or subdivisions < 1:
        raise ValueError("subdivisions must be a positive integer")
    cast = lambda value: torch.as_tensor(value, dtype=model.xs.dtype, device=model.xs.device)
    times = cast(model.recurrent_times).reshape(-1)
    qt = cast(q_times).reshape(-1)
    basis = cast(q_basis)
    xyt = cast(observation_xyt)
    if (times.numel() < 2 or qt.numel() < 2 or not bool(torch.all(times[1:] > times[:-1]))
            or not bool(torch.all(qt[1:] > qt[:-1]))):
        raise ValueError("Transport and Q times must be strictly increasing with at least two nodes")
    if basis.ndim != 2 or basis.shape[0] != qt.numel() or basis.shape[1] < 1:
        raise ValueError("Q basis rows must match Q times")
    if xyt.ndim != 2 or xyt.shape[1] != 3 or xyt.shape[0] == 0:
        raise ValueError("Observations must be a nonempty (n, 3) x/y/time table")
    if not all(bool(torch.isfinite(value).all()) for value in (times, qt, basis, xyt)):
        raise ValueError("Inputs must be finite")
    if bool(qt[0] > times[0]) or bool(qt[-1] < times[-1]):
        raise ValueError("Q time range must cover the transport time range")
    if bool((xyt[:, 2] < times[0]).any()) or bool((xyt[:, 2] > times[-1]).any()):
        raise ValueError("Observation times must lie within the transport time range")
    mode = getattr(model, "recurrent_source_mode", "gaussian")
    if getattr(model, "source_position_mode", "single") != "single":
        raise ValueError("Gaussian parcels currently require a single fixed source")
    if mode not in ("gaussian", "cell_gaussian"):
        raise ValueError("Gaussian parcels require a finite-width Gaussian source")
    width = getattr(model, "recurrent_source_sigma_norm", None)
    width = max(float(sigma_src), 1e-4) if width is None else float(width)
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Source width must be finite and positive")
    if getattr(model, "recurrent_anisotropic_diffusion", False):
        along, across = model.anisotropic_D()
        scale = model.recurrent_d_scale_norm
        across = model.recurrent_d_perp_min_norm + across*scale
        along = torch.maximum(model.recurrent_d_parallel_min_norm + along*scale, across)
    else:
        along = across = model.recurrent_d_min_norm + model.D()*model.recurrent_d_scale_norm
    wind = torch.stack((cast(model.recurrent_u), cast(model.recurrent_v)), dim=1)[:-1]
    speed = torch.linalg.vector_norm(wind, dim=1)
    direction = wind/torch.clamp(speed[:, None], min=1e-12)
    eye = torch.eye(2, dtype=model.xs.dtype, device=model.xs.device)
    tensors = across*eye + (along-across)*direction[:, :, None]*direction[:, None, :]
    tensors = torch.where((speed <= 1e-12)[:, None, None], .5*(along+across)*eye, tensors)
    fraction = (RECURRENT_INITIAL_RELEASE_FRACTION if initial_release_fraction is None
                else float(initial_release_fraction))
    if not math.isfinite(fraction) or fraction < 0:
        raise ValueError("Initial release fraction must be finite and nonnegative")
    if (not bool(torch.isfinite(tensors).all()) or not bool(torch.isfinite(wind).all())
            or not bool((across > 0).all())):
        raise ValueError("Finite wind and positive diffusion required")
    breaks = torch.unique(torch.cat((times, qt[(qt > times[0]) & (qt < times[-1])])), sorted=True)
    source = torch.stack((model.xs.reshape(()), model.ys.reshape(())))
    if not bool(torch.isfinite(source).all()) or not math.isfinite(float(model.recurrent_decay_norm)):
        raise ValueError("Source coordinates and decay must be finite")
    output = torch.zeros((len(xyt), basis.shape[1]), dtype=model.xs.dtype, device=model.xs.device)
    for end in torch.unique(xyt[:, 2], sorted=True):
        left = breaks[:-1][breaks[:-1] < end]
        right = torch.minimum(breaks[1:][:len(left)], end)
        fractions = torch.linspace(0., 1., int(subdivisions)+1, dtype=times.dtype, device=times.device)
        releases = (left[:, None] + (right-left)[:, None]*fractions).reshape(-1)
        weights = ((right-left)[:, None]/subdivisions).expand(-1, subdivisions+1).clone()
        weights[:, 0] *= .5
        weights[:, -1] *= .5
        releases = torch.cat((releases, times[:1]))
        weights = torch.cat((weights.reshape(-1), cast([fraction*model.recurrent_initial_release_dt])))
        high = torch.searchsorted(qt, releases.contiguous()).clamp(1, len(qt)-1)
        alpha = (releases-qt[high-1])/(qt[high]-qt[high-1])
        release_basis = (1-alpha[:, None])*basis[high-1] + alpha[:, None]*basis[high]
        residence = (torch.minimum(times[1:, None], end)-
                     torch.maximum(times[:-1, None], releases[None, :])).clamp(min=0.)
        displacement = residence.T @ wind
        covariance = width**2*eye + 2*torch.einsum("ir,ijk->rjk", residence, tensors)
        mask = xyt[:, 2] == end
        delta = xyt[mask, :2][:, None, :]-source-displacement[None, :, :]
        exponent = torch.einsum("nri,rij,nrj->nr", delta, torch.linalg.inv(covariance), delta)
        kernels = torch.exp(-.5*exponent)/(2*math.pi*torch.sqrt(torch.linalg.det(covariance)))
        attenuation = torch.exp(-max(float(model.recurrent_decay_norm), 0.)*(end-releases))
        output[mask] = RECURRENT_SOURCE_SCALE*(kernels*(weights*attenuation)[None, :]) @ release_basis
    return output


def gaussian_parcel_value_from_q(model, xyt, sigma_src, q_times, q_values, **kwargs):
    q = torch.as_tensor(q_values, dtype=model.xs.dtype, device=model.xs.device).reshape(-1, 1)
    return gaussian_parcel_response(model, sigma_src, xyt, q_times, q, **kwargs)
