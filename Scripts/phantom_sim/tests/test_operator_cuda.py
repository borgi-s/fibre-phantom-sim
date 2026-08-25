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
