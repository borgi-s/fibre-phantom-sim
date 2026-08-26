"""Task 10 analysis/figure tests: torch-free, this-box, synthetic inputs (no cluster).

Each test exercises one public analyze.py function against hand-built inputs that
match the exact JSON schemas the two cluster drivers write (ct_metrics.json and
pleno_metrics.json), so the figure code is proven without any GPU run.
"""
import json

import numpy as np

from phantom_sim.analyze import (
    load_volume,
    slice_montage,
    exp1_figure,
    exp2_figures,
    volume_plotly_html,
)


def _fake_history(metrics):
    """A convergence history dict shaped exactly like reconstruct.new_history()."""
    n = len(metrics)
    return {
        "iter": list(range(n)),
        "time": [float(i) * 0.5 for i in range(n)],
        "data_loss": [1.0 / (i + 1) for i in range(n)],
        "tv_loss": [0.1] * n,
        "metric": [float(m) for m in metrics],
    }


def _fake_pleno_metrics():
    """A pleno_metrics.json dict matching run_plenoptic_zpos.assemble_metrics()."""
    def by_N(base):
        out = []
        for n in range(1, 5):
            hist = _fake_history([base + 3.0 * k for k in range(n + 1)])
            out.append({
                "N": n,
                "psnr": base + 2.0 * n,
                "ssim": 0.6 + 0.05 * n,
                "resolution": {"along": 0.02, "across_mean": 0.04, "ratio": 0.5},
                "iters_to_threshold": n if n >= 2 else None,
                "time_to_threshold": float(n) * 0.5 if n >= 2 else None,
                "history": hist,
            })
        return out

    return {
        "cold_joint": {"by_N": by_N(20.0)},
        "warm_sequential": {
            "by_N": by_N(22.0),
            "cumulative_iters": 800,
            "cumulative_time": 42.0,
        },
        "meta": {"voxel_um": 2.0, "target_psnr": 28.0},
    }


def test_load_volume_roundtrips_saved_npy(tmp_path):
    vol = np.random.default_rng(0).random((5, 7, 9)).astype(np.float32)
    p = tmp_path / "v.npy"
    np.save(p, vol)
    loaded = load_volume(str(p))
    assert loaded.shape == (5, 7, 9)
    np.testing.assert_array_equal(np.asarray(loaded), vol)


def test_slice_montage_writes_png(tmp_path):
    vol = np.random.default_rng(0).random((20, 30, 30)).astype(np.float32)
    out = tmp_path / "montage.png"
    slice_montage(vol, str(out), fibre_axis=0)
    assert out.exists() and out.stat().st_size > 0


def test_exp1_figure_writes_png(tmp_path):
    m = {"axis0": {"psnr": 30.0, "ssim": 0.9, "resolution": {"ratio": 1.2}},
         "axis1": {"psnr": 24.0, "ssim": 0.8, "resolution": {"ratio": 2.1}}}
    j = tmp_path / "ct_metrics.json"
    j.write_text(json.dumps(m), encoding="utf-8")
    out = tmp_path / "exp1.png"
    exp1_figure(str(j), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_exp2_figures_writes_multiple_pngs(tmp_path):
    j = tmp_path / "pleno_metrics.json"
    j.write_text(json.dumps(_fake_pleno_metrics()), encoding="utf-8")
    paths = exp2_figures(str(j), str(tmp_path))
    assert len(paths) >= 2
    for p in paths:
        assert p.endswith(".png")
        assert (tmp_path / p).exists() if not p.startswith(str(tmp_path)) else True
    # every returned path points at a non-empty file
    from pathlib import Path
    for p in paths:
        fp = Path(p)
        assert fp.exists() and fp.stat().st_size > 0


def test_volume_plotly_html_writes_html(tmp_path):
    vol = np.zeros((24, 24, 24), np.float32)
    vol[:, 8:16, 8:16] = 1.0  # a bright bar so the isosurface has something to draw
    out = tmp_path / "vol3d.html"
    volume_plotly_html(vol, str(out))
    assert out.exists() and out.stat().st_size > 0
    assert "<html" in out.read_text(encoding="utf-8").lower()
