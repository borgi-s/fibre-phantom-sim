import numpy as np
import pytest

pytest.importorskip("phantom_rs")
from phantom_sim.phantom import voxelize_config


def _random_config(rng, Z, N, half_width_um):
    cfg = rng.uniform(-half_width_um, half_width_um, size=(Z, 2, N))
    radii = rng.uniform(2.0, 9.0, size=N)
    return cfg, radii


@pytest.mark.parametrize("supersample", [1, 2, 4])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rust_matches_python_bit_for_bit(supersample, axis):
    rng = np.random.default_rng(supersample * 10 + axis)
    # half width beyond the grid edge so some discs are clipped or fully outside;
    # dense enough that discs overlap and the np.maximum path is exercised
    cfg, radii = _random_config(rng, Z=7, N=60, half_width_um=70.0)
    kw = dict(transverse_shape=(53, 61), voxel_um=2.0, axis=axis,
              mu_fibre=0.30, mu_matrix=0.28, supersample=supersample)
    py = voxelize_config(cfg, radii, backend="python", **kw)
    rs = voxelize_config(cfg, radii, backend="rust", **kw)
    assert rs.dtype == py.dtype == np.float32
    assert rs.shape == py.shape
    assert np.array_equal(rs, py)


def test_unknown_backend_rejected():
    cfg, radii = _random_config(np.random.default_rng(0), 2, 3, 10.0)
    with pytest.raises(ValueError):
        voxelize_config(cfg, radii, (8, 8), 2.0, backend="cuda")
