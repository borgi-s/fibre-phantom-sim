import numpy as np
from phantom_sim.geometry import (
    build_ct_vectors, build_plenoptic_geometry, plenoptic_grid,
    _vol_geom_args, _vol_geom_args_windowed,
)


def test_ct_vectors_at_zero_angle():
    v = build_ct_vectors(np.array([0.0]), vol_z0_pix=100.0, detector_z=900.0,
                         pixel_size_x=1.0, pixel_size_y=1.0)
    assert v.shape == (1, 12)
    np.testing.assert_allclose(v[0, 0:3], [0.0, -100.0, 0.0], atol=1e-9)   # source
    np.testing.assert_allclose(v[0, 3:6], [0.0, 900.0, 0.0], atol=1e-9)    # det centre
    np.testing.assert_allclose(v[0, 6:9], [1.0, 0.0, 0.0], atol=1e-9)      # u dir
    np.testing.assert_allclose(v[0, 9:12], [0.0, 0.0, 1.0], atol=1e-9)     # v dir (along z)


def test_ct_rotation_axis_is_z():
    v = build_ct_vectors(np.linspace(0, 2 * np.pi, 8, endpoint=False),
                         100.0, 900.0, 1.0, 1.0)
    assert np.allclose(v[:, 2], 0.0)    # source z fixed
    assert np.allclose(v[:, 5], 0.0)    # det z fixed
    assert np.allclose(v[:, 11], 1.0)   # v dir always along z


def test_plenoptic_geometry_z_planes():
    src, det = plenoptic_grid(source_range=1000.0, det_range=8000.0, n=21)
    g = build_plenoptic_geometry(src, det, vol_z0_pix=100.0, detector_z=900.0,
                                 pixel_size_x=1.0, pixel_size_y=1.0)
    assert g.shape == (441, 12)
    assert np.allclose(g[:, 2], -100.0)          # all sources at z = -vol_z0_pix
    assert np.allclose(g[:, 5], 900.0 - 100.0)   # all detectors at detector_z - vol_z0_pix


def test_vol_geom_args_put_special_axis_on_astra_slices():
    # Array order is (nz, ny, nx) with dim 0 = the special axis (CT rotation / plenoptic
    # beam), which must land on ASTRA's Z (slices) so astra.geom_size(vol_geom) == vol.shape.
    # ASTRA create_vol_geom takes (GridRowCount=Y, GridColCount=X, GridSliceCount=Z).
    assert _vol_geom_args((24, 48, 48)) == (48, 48, 24)
    assert _vol_geom_args((10, 30, 40)) == (30, 40, 10)


def test_vol_geom_args_windowed_offsets_beam_axis_only():
    args = _vol_geom_args_windowed((24, 48, 40), vol_z_offset_pix=5)
    # rows=ny=48, cols=nx=40, slices=nz=24
    assert args[:3] == (48, 40, 24)
    # window is (minX, maxX, minY, maxY, minZ, maxZ): X from nx=40, Y from ny=48,
    # Z from nz=24 shifted by the beam-axis offset +5
    assert args[3:] == (-20.0, 20.0, -24.0, 24.0, -12.0 + 5, 12.0 + 5)


def test_vol_geom_args_windowed_zero_offset_is_symmetric():
    args = _vol_geom_args_windowed((16, 16, 16))
    assert args == (16, 16, 16, -8.0, 8.0, -8.0, 8.0, -8.0, 8.0)
