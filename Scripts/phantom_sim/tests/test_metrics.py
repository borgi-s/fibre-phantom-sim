import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from phantom_sim.metrics import (
    register, psnr, directional_sharpness, resolution_along_across, corr_with_gt,
)


def test_corr_with_gt_is_scale_invariant_and_bounded():
    g = np.random.default_rng(0).random((10, 10, 10))
    assert corr_with_gt(g, g) == pytest.approx(1.0)
    assert corr_with_gt(3.0 * g + 5.0, g) == pytest.approx(1.0)   # gain+offset invariant
    assert corr_with_gt(-g, g) == pytest.approx(-1.0)             # anti-correlated
    assert corr_with_gt(np.zeros_like(g), g) == 0.0              # degenerate -> 0, no nan


def test_register_recovers_gain_and_offset():
    rng = np.random.default_rng(0)
    gt = rng.random((20, 20, 20)).astype(np.float32)
    recon = 3.0 * gt + 5.0
    out = register(recon, gt)
    np.testing.assert_allclose(out, gt, atol=1e-4)


def test_psnr_perfect_is_high():
    gt = np.random.default_rng(1).random((16, 16, 16)).astype(np.float32)
    assert psnr(gt, gt) > 80.0


def test_directional_sharpness_drops_with_blur():
    v = np.zeros((40, 40, 40), np.float32)
    v[10:30, 10:30, 10:30] = 1.0
    blurred = gaussian_filter(v, sigma=(4.0, 0.0, 0.0))   # blur only along axis 0
    s0 = directional_sharpness(blurred, axis=0)
    s1 = directional_sharpness(blurred, axis=1)
    assert s0 < s1                                        # axis 0 is blurrier


def test_resolution_along_across_ratio_reflects_anisotropy():
    v = np.zeros((60, 60, 60), np.float32)
    v[10:50, 25:35, 25:35] = 1.0                          # fibre along axis 0
    blurred = gaussian_filter(v, sigma=(3.0, 1.0, 1.0))
    res = resolution_along_across(blurred, fibre_axis=0)
    assert set(res) == {"along", "across_mean", "ratio"}
    assert res["ratio"] > 0
    assert res["along"] < res["across_mean"]
