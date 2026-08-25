import numpy as np
import pytest

pytest.importorskip("astra")
torch = pytest.importorskip("torch")


@pytest.mark.cuda
def test_cold_recon_improves_and_warm_start_starts_high():
    from phantom_sim.phantom import voxelize_config
    from phantom_sim.tests.fixtures.make_synth_config import straight_bundle
    from phantom_sim.geometry import build_ct_vectors, make_ct_projector, XrayOperator
    from phantom_sim.forward import simulate_projections
    from phantom_sim.reconstruct import reconstruct, first_crossing
    from phantom_sim.metrics import psnr

    cfg, radii = straight_bundle(Z=32, n_side=3, spacing_um=24.0, radius_um=6.0)
    vol = voxelize_config(cfg, radii, (64, 64), voxel_um=2.0, axis=0,
                          mu_fibre=0.30, mu_matrix=0.28)  # (32,64,64)
    angles = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    vecs = build_ct_vectors(angles, 100.0, 900.0, 1.0, 1.0)
    pid, _ = make_ct_projector(vol.shape, vecs, sinogram_size=[vol.shape[0], 64, len(angles)])
    A = XrayOperator(pid)
    volt = torch.tensor(vol, device="cuda")
    proj = simulate_projections(A, volt, i0=5e4, seed=0)

    recon, hist = reconstruct(A, proj, n_iter=60, lr=1e-2, tv_weights=(1e-4, 1e-4, 1e-4),
                              gt=vol, log_every=10)
    assert hist["metric"][-1] > hist["metric"][0]
    assert hist["metric"][-1] > 15.0

    warm, hist_w = reconstruct(A, proj, n_iter=10, x_init=vol, gt=vol, log_every=1)
    assert hist_w["metric"][0] > 25.0
