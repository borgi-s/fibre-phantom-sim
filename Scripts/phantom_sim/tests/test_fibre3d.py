"""Tests for the torch-free 3D fibre-tube illustrations (this box, synthetic configs).

Each test feeds a tiny hand-built packer configuration (Z, 2, N) and asserts the figure
function writes a self-contained interactive Plotly HTML. No packing, no torch, no GPU.
"""
import numpy as np


def _toy_straight_cfg(n=8):
    """A 2-slice straight bundle: n fibres on a line, identical at both control slices."""
    cfg = np.zeros((2, 2, n), dtype=np.float64)
    cfg[:, 0, :] = np.linspace(-100.0, 100.0, n)   # x spread in um
    cfg[:, 1, :] = 0.0                              # y
    radii = np.full(n, 6.0)
    return cfg, radii


def _toy_wander_cfg(zc=8, n=8):
    """A wandering bundle: fibres drift transversely down the control slices."""
    cfg = np.zeros((zc, 2, n), dtype=np.float64)
    base = np.linspace(-100.0, 100.0, n)
    for z in range(zc):
        cfg[z, 0, :] = base + 8.0 * z            # x drifts with depth
        cfg[z, 1, :] = 5.0 * np.sin(z / 2.0)     # y wanders
    radii = np.full(n, 6.0)
    return cfg, radii


def test_exp1_orientation_html_writes(tmp_path):
    from phantom_sim.fibre3d import exp1_orientation_html

    cfg, radii = _toy_straight_cfg()
    out = tmp_path / "exp1.html"
    ret = exp1_orientation_html(cfg, radii, str(out), depth_um=400.0, n_show=8)
    assert ret == str(out)
    txt = out.read_text(encoding="utf-8").lower()
    assert out.stat().st_size > 0 and "<html" in txt
    assert "plotly" in txt


def test_exp2_zpositions_html_writes(tmp_path):
    from phantom_sim.fibre3d import exp2_zpositions_html

    cfg, radii = _toy_straight_cfg()
    out = tmp_path / "exp2.html"
    ret = exp2_zpositions_html(cfg, radii, str(out), depth_um=400.0, n_positions=4, n_show=8)
    assert ret == str(out)
    assert out.stat().st_size > 0 and "<html" in out.read_text(encoding="utf-8").lower()


def test_exp3_chunks_html_writes(tmp_path):
    from phantom_sim.fibre3d import exp3_chunks_html

    cfg, radii = _toy_wander_cfg()
    out = tmp_path / "exp3_A.html"
    ret = exp3_chunks_html(cfg, radii, str(out), n_chunks=8, highlight=[0, 1, 2, 3], n_show=8)
    assert ret == str(out)
    assert out.stat().st_size > 0 and "<html" in out.read_text(encoding="utf-8").lower()
