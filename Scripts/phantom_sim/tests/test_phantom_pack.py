import numpy as np
import pytest

torch = pytest.importorskip("torch")
from phantom_sim.phantom import pack_bundle, voxelize_config


def test_pack_bundle_shapes():
    cfg, radii = pack_bundle(domain_radius_um=70.0, fvf=0.4, r_mean_um=3.5,
                             n_slices=8, misalignment="very low", iters=20, seed=1)
    Z, two, N = cfg.shape
    assert two == 2
    assert radii.shape == (N,)
    assert N > 5
    assert np.isfinite(cfg).all()


def test_pack_then_voxelize_produces_contrast():
    cfg, radii = pack_bundle(70.0, 0.4, 3.5, n_slices=8, iters=20, seed=1)
    vol = voxelize_config(cfg, radii, (100, 100), voxel_um=2.0,
                          mu_fibre=0.30, mu_matrix=0.28)
    assert vol.min() == pytest.approx(0.28, abs=1e-6)
    assert vol.max() == pytest.approx(0.30, abs=1e-6)
    assert (vol > 0.29).mean() > 0.05   # some fibre voxels present
