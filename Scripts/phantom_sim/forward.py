"""Beer-Lambert forward model with Poisson photon noise."""
import numpy as np


def to_intensity(absorbance, i0):
    return i0 * np.exp(-np.asarray(absorbance, dtype=np.float64))


def add_poisson(counts, rng):
    return rng.poisson(np.clip(counts, 0, None)).astype(np.float64)


def to_neglog(counts, i0, eps=1.0):
    return -np.log(np.maximum(np.asarray(counts, dtype=np.float64), eps) / i0)


def simulate_absorbance_np(absorbance, i0, seed):
    rng = np.random.default_rng(seed)
    counts = add_poisson(to_intensity(absorbance, i0), rng)
    return to_neglog(counts, i0)


def simulate_projections(A, vol, i0, seed):
    """GPU path: forward-project vol (torch, cuda), apply Poisson, return -log tensor."""
    import torch
    from phantom_sim.geometry import make_projection_fn
    Projection = make_projection_fn()
    absorbance = Projection.apply(A, vol)                 # line integrals, (proj_shape)
    gen = torch.Generator(device="cuda").manual_seed(int(seed))
    counts = torch.poisson(i0 * torch.exp(-absorbance), generator=gen)
    return -torch.log(torch.clamp(counts, min=1.0) / i0)
