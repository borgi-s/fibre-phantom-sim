"""Torch-free tests for the localized-misalignment driver's pure helpers (exp7, QC demo).

The exp7 defect is modelled as coherent fibre WAVINESS (calm half vs wavy half); these tests cover
the shared-profile generation, the graded waviness config, the overlap safeguard, the wavy-half
mask, the region-corr reduction, and the metrics-meta schema. The forward projection and
reconstruction in main() need the cluster and are not imported.
"""

import numpy as np
import pytest

from phantom_sim.run_plenoptic_local import (
    shared_wave_profiles,
    graded_wave_config,
    wavy_half_mask,
    min_surface_gap,
    assert_no_overlap,
    max_new_overlap,
    region_corr,
    wave_metrics_meta,
)


def _grid_bundle(nx_f=9, ny_f=9, spacing=16.8, r=6.0):
    """A small non-overlapping bundle on a square grid centred on the origin. Returns (base, radii)."""
    xs = (np.arange(nx_f) - (nx_f - 1) / 2) * spacing
    ys = (np.arange(ny_f) - (ny_f - 1) / 2) * spacing
    gx, gy = np.meshgrid(xs, ys)
    base = np.stack([gx.ravel(), gy.ravel()])  # (2, N)
    return base, np.full(base.shape[1], r)


def test_shared_wave_profiles_ends_normalized_and_deterministic():
    u, g = shared_wave_profiles(256, 2.0, seed=1)
    assert u.shape == (256,) and g.shape == (256,)
    assert abs(u[0]) < 1e-9 and abs(u[-1]) < 1e-9  # ends on the pultrusion axis
    assert np.max(np.abs(np.gradient(u, 2.0))) == pytest.approx(
        1.0, rel=1e-6
    )  # max slope = 1
    assert g.min() >= 0.0 and g.max() <= 1.0  # envelope in [0, 1]
    u2, g2 = shared_wave_profiles(256, 2.0, seed=1)
    assert np.array_equal(u, u2) and np.array_equal(g, g2)  # deterministic in seed


def test_graded_wave_config_hits_target_peak_angles():
    base, _ = _grid_bundle()
    u, g = shared_wave_profiles(256, 2.0, seed=2)
    cfg, alpha_deg, peak = graded_wave_config(
        base,
        256,
        2.0,
        alpha_good_deg=3.5,
        alpha_bad_deg=30.0,
        u=u,
        g=g,
        transition_um=100.0,
    )
    assert cfg.shape == (256, 2, base.shape[1])
    x0 = base[0]
    calm = x0 < -60.0  # fully calm (past the transition)
    wavy = x0 > 60.0  # fully wavy
    assert peak[calm] == pytest.approx(3.5, abs=1e-3)
    assert peak[wavy] == pytest.approx(30.0, abs=1e-3)
    # x is what shears; y is untouched, and every fibre re-enters its axis at both ends.
    assert np.array_equal(cfg[:, 1, :], np.repeat(base[1][None, :], 256, axis=0))
    assert np.allclose(cfg[0, 0, :], x0, atol=1e-9)
    assert np.allclose(cfg[-1, 0, :], x0, atol=1e-9)


def test_graded_wave_config_2d_moves_y_and_hits_combined_target_peak():
    base, _ = _grid_bundle()
    u, g = shared_wave_profiles(256, 2.0, seed=2)
    u_y, g_y = shared_wave_profiles(256, 2.0, seed=3)  # independent y profile
    cfg, alpha_deg, peak = graded_wave_config(
        base,
        256,
        2.0,
        alpha_good_deg=3.5,
        alpha_bad_deg=30.0,
        u=u,
        g=g,
        transition_um=100.0,
        u_y=u_y,
        g_y=g_y,
    )
    x0 = base[0]
    calm = x0 < -60.0
    wavy = x0 > 60.0
    # The COMBINED x+y peak deviation from the fibre axis still hits the target angles.
    assert peak[calm] == pytest.approx(3.5, abs=1e-3)
    assert peak[wavy] == pytest.approx(30.0, abs=1e-3)
    # y is now genuinely displaced (isotropic wander), yet re-enters its axis at both ends.
    assert not np.allclose(cfg[:, 1, :], np.repeat(base[1][None, :], 256, axis=0))
    assert np.allclose(cfg[0, 1, :], base[1], atol=1e-9)
    assert np.allclose(cfg[-1, 1, :], base[1], atol=1e-9)
    assert np.allclose(cfg[0, 0, :], x0, atol=1e-9)  # x still re-enters too


def test_2d_coherent_waviness_can_be_made_non_overlapping():
    base, radii = _grid_bundle(spacing=16.8, r=6.0)
    u, g = shared_wave_profiles(256, 2.0, seed=3)
    u_y, g_y = shared_wave_profiles(256, 2.0, seed=4)
    transition = 100.0  # mirror main()'s auto-widen safeguard
    for _ in range(9):
        cfg, _, _ = graded_wave_config(
            base, 256, 2.0, 3.5, 30.0, u, g, transition_um=transition, u_y=u_y, g_y=g_y
        )
        if max_new_overlap(cfg, base, radii) <= 1.0:
            break
        transition *= 1.25
    assert (
        max_new_overlap(cfg, base, radii) <= 1.0
    )  # coherence holds in 2D -> no new collision


def test_graded_wave_config_uniform_is_flat_control():
    base, _ = _grid_bundle()
    u, g = shared_wave_profiles(128, 2.0, seed=5)
    _, _, peak = graded_wave_config(
        base, 128, 2.0, alpha_good_deg=3.5, alpha_bad_deg=3.5, u=u, g=g
    )
    # alpha_good == alpha_bad -> the whole cross-section has the same waviness (no defect gradient).
    assert peak == pytest.approx(3.5, abs=1e-3)


def test_wavy_half_mask_splits_on_x_center():
    base = np.array([[-30.0, -1.0, 1.0, 40.0], [0.0, 0.0, 0.0, 0.0]])
    assert list(wavy_half_mask(base)) == [False, False, True, True]
    assert list(wavy_half_mask(base, x_center_um=10.0)) == [False, False, False, True]


def test_coherent_waviness_stays_non_overlapping():
    base, radii = _grid_bundle(spacing=16.8, r=6.0)  # ~fvf 0.4 spacing
    u, g = shared_wave_profiles(256, 2.0, seed=3)
    cfg, _, _ = graded_wave_config(base, 256, 2.0, 3.5, 30.0, u, g, transition_um=100.0)
    # Coherent (shared u,g) + a wide transition -> neighbours move together -> no interpenetration.
    assert min_surface_gap(cfg, radii) >= -1e-6
    assert assert_no_overlap(cfg, radii) >= -1e-6


def test_assert_no_overlap_raises_on_interpenetration():
    # Two r=6 fibres 5 um apart centre-to-centre overlap (surface gap 5 - 12 = -7).
    cfg = np.zeros((2, 2, 2))
    cfg[:, 0, 1] = 5.0
    radii = np.array([6.0, 6.0])
    assert min_surface_gap(cfg, radii) == pytest.approx(-7.0)
    with pytest.raises(ValueError):
        assert_no_overlap(cfg, radii)


def test_max_new_overlap_grandfathers_base_contact_but_flags_new_collision():
    # Fibres 0 & 1 already interpenetrate in the base pack (centre dist 8, r=6 each -> gap -4);
    # fibre 2 sits well clear (x=40 -> gap +20 to fibre 1). This is the pre-existing packer contact.
    base = np.array([[0.0, 8.0, 40.0], [0.0, 0.0, 0.0]])
    radii = np.full(3, 6.0)
    # Re-extruding the base straight (no waviness) adds NO new overlap: the -4 contact is grandfathered.
    straight = np.repeat(base[None], 4, axis=0)  # (4, 2, 3)
    assert max_new_overlap(straight, base, radii) == pytest.approx(0.0, abs=1e-9)
    # Now drive the clear fibre 2 into fibre 1 (x 40 -> 11, dist 3 -> gap -9): a genuine NEW collision.
    moved = straight.copy()
    moved[:, 0, 2] = 11.0
    assert max_new_overlap(moved, base, radii) == pytest.approx(9.0, abs=1e-9)
    # The pre-existing 0&1 pair, left untouched, must NOT be what is reported (it stays grandfathered).
    only_base_contact = straight.copy()
    only_base_contact[:, 0, 1] = 8.5  # ease the base contact slightly (-3.5)
    assert max_new_overlap(only_base_contact, base, radii) <= 0.0


def test_region_corr_scalar_and_guards():
    obj = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    recon = 2.0 * obj + 1.0  # perfectly correlated
    full = np.ones_like(obj, dtype=bool)
    assert region_corr(recon, obj, full) == pytest.approx(1.0)
    tiny = np.zeros_like(obj, dtype=bool)
    tiny.flat[:4] = True  # < 8 voxels -> None
    assert region_corr(recon, obj, tiny) is None
    assert region_corr(recon, obj, None) is None
    const = np.zeros_like(obj, dtype=bool)
    const[0] = True
    assert region_corr(np.ones_like(obj), obj, const) is None  # zero variance in recon


def test_wave_metrics_meta_schema():
    from types import SimpleNamespace

    args = SimpleNamespace(
        smoke=False,
        alpha_good=3.5,
        alpha_bad=30.0,
        wave_u_sigma_um=14.0,
        wave_g_sigma_um=55.0,
        x_center_um=0.0,
        wave_seed=0,
        wave_x_only=False,
        wave_seed_y_offset=1,
        overlap_tol_um=1.0,
        voxel_um=2.0,
        domain_radius_um=120.0,
        fvf=0.4,
        r_mean_um=6.0,
        mu_fibre=1.0,
        mu_matrix=0.0,
        seed=0,
        sod_mm=129.24,
        sdd_mm=1171.57,
        detector_pixel_um=55.0,
        source_range_um=22000.0,
        det_range_um=177400.0,
        n_grid=21,
        n_iter=200,
        lr=0.05,
        tv_weight=0.5,
    )
    meta = wave_metrics_meta(
        args,
        n_fibres=160,
        n_bad=80,
        max_half_angle=9.66,
        transition_used_um=100.0,
        nz=256,
        ny=384,
        nx=384,
        pad_t=64,
        pad_d=32,
    )
    assert meta["model"] == "waviness"
    assert meta["alpha_bad_deg"] == 30.0
    assert meta["alpha_good_deg"] == 3.5
    assert meta["n_bad"] == 80
    assert meta["transition_um"] == 100.0
    assert meta["overlap_tol_um"] == 1.0
    assert meta["max_half_angle_deg"] == 9.66
    assert meta["n_grid"] == 21
    assert meta["wave_x_only"] is False
    assert meta["wave_seed_y"] == 1  # wave_seed 0 + offset 1
    args.wave_x_only = True
    assert (
        wave_metrics_meta(args, 160, 80, 9.66, 100.0, 256, 384, 384, 64, 32)[
            "wave_seed_y"
        ]
        is None
    )
