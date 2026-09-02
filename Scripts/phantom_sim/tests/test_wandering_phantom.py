import numpy as np
from phantom_sim.phantom import upsample_config


def test_upsample_config_endpoints_and_midpoint():
    # two control slices, N=1 fibre, moving from x=0 to x=10 (index 0 = x)
    cfg = np.zeros((2, 2, 1))
    cfg[0, 0, 0] = 0.0
    cfg[1, 0, 0] = 10.0
    out = upsample_config(cfg, 5)
    assert out.shape == (5, 2, 1)
    assert np.isclose(out[0, 0, 0], 0.0)
    assert np.isclose(out[-1, 0, 0], 10.0)
    assert np.isclose(out[2, 0, 0], 5.0)  # midpoint of a linear ramp


def test_upsample_config_identity_when_depth_equals_zc():
    cfg = np.random.default_rng(0).standard_normal((7, 2, 4))
    out = upsample_config(cfg, 7)
    assert np.allclose(out, cfg)
