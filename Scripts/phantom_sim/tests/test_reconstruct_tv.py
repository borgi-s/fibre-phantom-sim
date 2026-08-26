"""This-box (torch CPU, no astra) tests for the TV penalty shape.

The reference PlenoXFiber solver uses a squared-L2 gradient penalty along the fibre
axis only. These tests pin that shape: squared (not L1) magnitude, and independent
per-axis weighting, so the reconstruct loss cannot silently revert to L1 or isotropic.
"""
import pytest

torch = pytest.importorskip("torch")

from phantom_sim.reconstruct import _tv_isotropic_aniso


def _ramp_along_axis0(n=4, side=3):
    # value == index along dim 0, constant across the transverse plane:
    # first difference along dim 0 is exactly 1 everywhere, and 0 on the other axes.
    return torch.arange(n, dtype=torch.float32).reshape(n, 1, 1).repeat(1, side, side)


def test_tv_is_squared_not_l1():
    v = _ramp_along_axis0()
    tv1 = float(_tv_isotropic_aniso(v, (1.0, 0.0, 0.0)))
    assert abs(tv1 - 1.0) < 1e-6                    # mean(1**2) == 1
    # doubling the gradient must QUADRUPLE a squared penalty (L1 would only double it)
    tv2 = float(_tv_isotropic_aniso(2 * v, (1.0, 0.0, 0.0)))
    assert abs(tv2 - 4.0) < 1e-6


def test_tv_per_axis_weighting_is_independent():
    v = _ramp_along_axis0()                          # varies only along dim 0
    assert abs(float(_tv_isotropic_aniso(v, (0.0, 1.0, 0.0)))) < 1e-6
    assert abs(float(_tv_isotropic_aniso(v, (0.0, 0.0, 1.0)))) < 1e-6
    a = float(_tv_isotropic_aniso(v, (0.5, 0.0, 0.0)))
    b = float(_tv_isotropic_aniso(v, (1.0, 0.0, 0.0)))
    assert abs(b - 2.0 * a) < 1e-6                    # weight scales the term linearly
