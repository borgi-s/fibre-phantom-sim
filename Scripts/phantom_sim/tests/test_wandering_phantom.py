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


from phantom_sim.phantom import chunk_gt, pair_misorientation


def _straight_two_fibre_cfg(zc=8, N=2):
    # two fibres at fixed positions in every control slice: a perfectly straight bundle
    cfg = np.zeros((zc, 2, N))
    cfg[:, 0, 0] = -20.0  # fibre 0 at x=-20 um
    cfg[:, 0, 1] = 20.0  # fibre 1 at x=+20 um
    return cfg


def test_chunk_gt_shape_and_content():
    cfg = _straight_two_fibre_cfg()
    radii = np.array([6.0, 6.0])
    v = chunk_gt(
        cfg,
        radii,
        chunk_idx=0,
        n_chunks=8,
        chunk_depth=4,
        transverse_shape=(64, 64),
        voxel_um=2.0,
        mu_fibre=1.0,
        mu_matrix=0.0,
    )
    assert v.shape == (4, 64, 64)
    assert v.max() > 0.5  # the two discs are present
    assert v.min() == 0.0  # void background


def test_pair_misorientation_straight_bundle_is_one():
    cfg = _straight_two_fibre_cfg()
    radii = np.array([6.0, 6.0])
    c = pair_misorientation(
        cfg,
        radii,
        0,
        7,
        n_chunks=8,
        chunk_depth=4,
        transverse_shape=(64, 64),
        voxel_um=2.0,
        mu_fibre=1.0,
        mu_matrix=0.0,
    )
    assert c > 0.999  # straight bundle: every chunk cross-section identical
