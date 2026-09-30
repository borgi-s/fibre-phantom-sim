"""Torch-free tests for the diameter-sweep driver's pure helpers (exp9). The recon needs the cluster."""

import numpy as np
import pytest

from phantom_sim.run_plenoptic_diamsweep import (
    relax_move_sparse,
    worst_overlap_sparse,
    uniform_wander_config,
    draw_diameters_um,
    n_fibres_for_fvf,
    sparse_pack,
)
from phantom_sim.run_plenoptic_waypoint import (
    relax_move,
    worst_overlap,
    realized_angle_p90,
)


def _grid_bundle(nf=9, spacing=16.8):
    xs = (np.arange(nf) - (nf - 1) / 2) * spacing
    gx, gy = np.meshgrid(xs, xs)
    return np.stack([gx.ravel(), gy.ravel()])


def test_worst_overlap_sparse_matches_dense():
    rng = np.random.default_rng(0)
    p = rng.uniform(-60, 60, (2, 80))
    radii = rng.uniform(4, 7, 80)
    assert worst_overlap_sparse(p, radii) == pytest.approx(worst_overlap(p, radii))


def test_relax_sparse_matches_dense_update():
    rng = np.random.default_rng(1)
    p = _grid_bundle(7, 11.0) + rng.normal(0, 1.0, (2, 49))
    radii = np.full(49, 6.0)
    a = relax_move(p.copy(), radii, R=200.0, tol=0.1, iters=40)
    b = relax_move_sparse(p.copy(), radii, R=200.0, tol=0.1, iters=40)
    assert np.allclose(a, b, atol=1e-6)


def test_relax_sparse_resolves_and_confines():
    p = np.array([[0.0, 5.0, 95.0], [0.0, 0.0, 0.0]])
    radii = np.full(3, 6.0)
    q = relax_move_sparse(p, radii, R=50.0, tol=0.1, iters=400)
    assert worst_overlap_sparse(q, radii) <= 0.1 + 1e-3
    assert np.max(np.hypot(q[0], q[1]) + radii) <= 50.0 + 1e-6


def test_diameters_and_fibre_count():
    d = draw_diameters_um(20000, 14.0, 0.10, 2.0, seed=1)
    assert abs(d.mean() - 14.0) < 0.05
    assert d.min() >= 14.0 - 2.8 - 1e-9 and d.max() <= 14.0 + 2.8 + 1e-9
    n = n_fibres_for_fvf(0.6, 450.0, 20.0, 0.10)
    fvf = n * np.pi / 4 * 20.0**2 * 1.01 / (np.pi * 450.0**2)
    assert fvf == pytest.approx(0.6, abs=0.002)


def test_sparse_pack_dense_fvf_non_overlapping():
    R, D = 120.0, 12.0
    n = n_fibres_for_fvf(0.6, R, D, 0.10)
    radii = draw_diameters_um(n, D, 0.10, 2.0, seed=2) / 2.0
    p = sparse_pack(n, radii, R, tol=0.05, seed=0)
    assert worst_overlap_sparse(p, radii) <= 0.1
    assert np.max(np.hypot(p[0], p[1]) + radii) <= R + 1e-6


def test_uniform_wander_is_mildly_misaligned_and_clean():
    base = _grid_bundle(9, 16.8)
    N = base.shape[1]
    radii = draw_diameters_um(N, 12.0, 0.10, 2.0, seed=2) / 2.0
    cfg = uniform_wander_config(
        base, radii, depth=64, R=120.0, step=0.85, K=8, tol=0.3, seed=0
    )
    assert cfg.shape == (64, 2, N)
    for z in (0, 32, 63):
        assert worst_overlap_sparse(cfg[z], radii) <= 0.3 + 1e-2
    ang = realized_angle_p90(cfg, 2.0)
    assert 0.5 < np.median(ang) < 20.0
