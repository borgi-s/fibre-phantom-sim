"""Torch-free tests for the waypoint-wander driver's pure helpers (exp7 QC, independent per-fibre
waviness via re-packing + polydisperse radii). The reconstruction in main() needs the cluster."""

import numpy as np
import pytest

from phantom_sim.run_plenoptic_waypoint import (
    relax_move,
    worst_overlap,
    waypoint_wander_config,
    draw_radii,
    realized_angle_p90,
)
from phantom_sim.run_plenoptic_local import wavy_half_mask


def _grid_bundle(nf=9, spacing=16.8):
    xs = (np.arange(nf) - (nf - 1) / 2) * spacing
    gx, gy = np.meshgrid(xs, xs)
    return np.stack([gx.ravel(), gy.ravel()])  # (2, N)


def test_relax_move_resolves_overlap_to_tol():
    p = np.array([[0.0, 5.0], [0.0, 0.0]])  # two r=6 fibres 5 um apart -> overlap -7
    radii = np.full(2, 6.0)
    assert worst_overlap(p, radii) == pytest.approx(7.0)
    p2 = relax_move(p, radii, R=100.0, tol=0.1, iters=300)
    assert worst_overlap(p2, radii) <= 0.1 + 1e-3


def test_relax_move_confines_to_rope():
    p = np.array([[95.0], [0.0]])  # fibre surface would poke past R
    radii = np.array([6.0])
    p2 = relax_move(p, radii, R=50.0, tol=0.1, iters=100)
    assert np.hypot(p2[0, 0], p2[1, 0]) + 6.0 <= 50.0 + 1e-6


def test_draw_radii_distribution_bounds_and_mean():
    r = draw_radii(4000, mean_um=12.0, std_um=2.0, dmin_um=9.0, dmax_um=15.0, seed=1)
    d = 2.0 * r
    assert d.min() >= 9.0 - 1e-9 and d.max() <= 15.0 + 1e-9
    assert abs(d.mean() - 12.0) < 0.3  # centred on the mean
    assert d.std() > 1.0  # genuinely polydisperse


def test_waypoint_non_overlapping_in_rope_and_anchored():
    base = _grid_bundle()
    N = base.shape[1]
    radii = draw_radii(N, 12.0, 2.0, 9.0, 15.0, seed=2)
    cfg = waypoint_wander_config(
        base,
        radii,
        depth=64,
        R=120.0,
        voxel_um=2.0,
        step_calm=0.4,
        step_wavy=4.0,
        K=8,
        tol=0.3,
        seed=0,
    )
    assert cfg.shape == (64, 2, N)
    for z in (0, 32, 63):  # non-overlapping at every checked slice
        assert worst_overlap(cfg[z], radii) <= 0.3 + 1e-2
    reach = np.max(np.hypot(cfg[:, 0, :], cfg[:, 1, :]) + radii[None, :])
    assert reach <= 120.0 + 1e-6  # stays inside the rope
    assert np.allclose(
        cfg[0], relax_move(base.copy(), radii, 120.0, 0.3), atol=1e-6
    )  # z=0 = clean base


def test_waypoint_wavy_half_more_misaligned_than_calm():
    base = _grid_bundle(nf=11, spacing=16.8)
    N = base.shape[1]
    radii = draw_radii(N, 12.0, 2.0, 9.0, 15.0, seed=2)
    cfg = waypoint_wander_config(
        base,
        radii,
        depth=96,
        R=150.0,
        voxel_um=2.0,
        step_calm=0.35,
        step_wavy=6.0,
        K=10,
        tol=0.4,
        seed=0,
    )
    ang = realized_angle_p90(cfg, 2.0)
    bad = wavy_half_mask(base)
    assert np.mean(ang[bad]) > 2.0 * np.mean(
        ang[~bad]
    )  # wavy half clearly more misaligned


def test_waypoint_uniform_step_is_flat_control():
    base = _grid_bundle()
    N = base.shape[1]
    radii = draw_radii(N, 12.0, 2.0, 9.0, 15.0, seed=2)
    cfg = waypoint_wander_config(
        base,
        radii,
        depth=64,
        R=120.0,
        voxel_um=2.0,
        step_calm=0.35,
        step_wavy=0.35,
        K=8,
        tol=0.3,
        seed=0,
    )  # step_wavy == step_calm
    ang = realized_angle_p90(cfg, 2.0)
    bad = wavy_half_mask(base)
    assert (
        abs(np.mean(ang[bad]) - np.mean(ang[~bad])) < 1.5
    )  # both halves the same (defect-free)
