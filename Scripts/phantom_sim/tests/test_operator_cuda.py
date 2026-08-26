import numpy as np
import pytest

pytest.importorskip("astra")
torch = pytest.importorskip("torch")


@pytest.mark.cuda
def test_ct_forward_backproject_roundtrip_shapes():
    from phantom_sim.geometry import build_ct_vectors, make_ct_projector, XrayOperator
    angles = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    vecs = build_ct_vectors(angles, 100.0, 900.0, 1.0, 1.0)
    pid, _ = make_ct_projector((64, 64, 32), vecs, sinogram_size=[32, 64, len(angles)])
    A = XrayOperator(pid)
    vol = torch.ones(A.vol_shape, dtype=torch.float32, device="cuda")
    proj = A(vol)
    back = A.T(proj)
    assert proj.shape == tuple(A.proj_shape)
    assert back.shape == tuple(A.vol_shape)
    assert float(proj.sum()) > 0


@pytest.mark.cuda
def test_ct_operator_vol_shape_matches_array_order():
    # Guard the driver's real usage: a phantom is built as (nz, ny, nx) with dim 0 the
    # rotation axis and fed straight to the operator. astra.geom_size(vol_geom) must
    # equal that array shape or direct_FP3D fails to link the volume (regression: the
    # verbatim-lifted create_vol_geom(*vol_shape) put the special axis on the wrong dim).
    from phantom_sim.geometry import build_ct_vectors, make_ct_projector, XrayOperator
    nz, ny, nx = 24, 48, 40
    angles = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    vecs = build_ct_vectors(angles, 100.0, 900.0, 1.0, 1.0)
    pid, _ = make_ct_projector((nz, ny, nx), vecs, sinogram_size=[nz, nx, len(angles)])
    A = XrayOperator(pid)
    assert tuple(A.vol_shape) == (nz, ny, nx)
    vol = torch.ones((nz, ny, nx), dtype=torch.float32, device="cuda")  # links only if order is right
    proj = A(vol)
    assert float(proj.sum()) > 0


@pytest.mark.cuda
def test_plenoptic_operator_vol_shape_matches_array_order():
    from phantom_sim.geometry import (
        build_plenoptic_geometry, make_plenoptic_projector, plenoptic_grid, XrayOperator,
    )
    nz, ny, nx = 24, 48, 40
    src, det = plenoptic_grid(1000.0, 8000.0, n=3)
    geom = build_plenoptic_geometry(src, det, vol_z0_pix=100.0, detector_z=900.0,
                                    pixel_size_x=1.0, pixel_size_y=1.0)
    pid, _ = make_plenoptic_projector((nz, ny, nx), geom, sinogram_size=[ny, nx, 9])
    A = XrayOperator(pid)
    assert tuple(A.vol_shape) == (nz, ny, nx)
    vol = torch.ones((nz, ny, nx), dtype=torch.float32, device="cuda")
    proj = A(vol)
    assert float(proj.sum()) > 0
