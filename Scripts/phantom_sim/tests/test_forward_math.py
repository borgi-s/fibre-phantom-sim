import numpy as np
from phantom_sim.forward import to_intensity, add_poisson, to_neglog, simulate_absorbance_np


def test_noise_free_roundtrip_recovers_absorbance():
    p = np.linspace(0.0, 2.0, 100)
    i0 = 1e12
    counts = to_intensity(p, i0)
    out = to_neglog(counts, i0)
    np.testing.assert_allclose(out, p, atol=1e-4)


def test_poisson_variance_matches_theory():
    p = np.full(200000, 0.5)
    i0 = 5000.0
    rng = np.random.default_rng(0)
    noisy = simulate_absorbance_np(p, i0, seed=0)
    # var(-log(N/i0)) approx 1/mean_counts for Poisson
    mean_counts = i0 * np.exp(-0.5)
    assert abs(noisy.var() - 1.0 / mean_counts) / (1.0 / mean_counts) < 0.1
    assert abs(noisy.mean() - 0.5) < 0.02


def test_higher_flux_lowers_noise():
    p = np.full(50000, 0.5)
    lo = simulate_absorbance_np(p, 1000.0, seed=1).var()
    hi = simulate_absorbance_np(p, 100000.0, seed=1).var()
    assert hi < lo
