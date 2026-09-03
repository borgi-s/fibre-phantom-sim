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
    exp2_corr_convergence,
    stage_boundaries,
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


def _fake_corr_history(corrs, log_every=10):
    """History shaped like reconstruct.new_history(), carrying a per-iteration corr trace."""
    n = len(corrs)
    return {
        "iter": [i * log_every for i in range(n)],
        "time": [float(i) * 0.5 for i in range(n)],
        "data_loss": [1.0 / (i + 1) for i in range(n)],
        "tv_loss": [0.1] * n,
        "metric": [10.0 + i for i in range(n)],   # psnr, still logged
        "corr": [float(c) for c in corrs],
    }


def _fake_pleno_metrics_corr():
    """pleno_metrics.json with per-iteration corr traces + meta.reconstruction, as the
    corr-thresholding driver writes. Warm starts higher than cold (warm-started)."""
    def by_N(start):
        out = []
        for n in range(1, 5):
            corrs = [min(start + 0.05 * k, 0.99) for k in range(5)]
            out.append({
                "N": n,
                "corr_with_gt": corrs[-1],
                "psnr": 12.0 + n,
                "ssim": 0.6,
                "resolution": {"along": 0.02, "across_mean": 0.04, "ratio": 0.5},
                "iters_to_threshold": 20,
                "time_to_threshold": 1.0,
                "history": _fake_corr_history(corrs),
            })
        return out

    return {
        "cold_joint": {"by_N": by_N(0.70)},
        "warm_sequential": {"by_N": by_N(0.80),
                            "cumulative_iters": 200, "cumulative_time": 20.0},
        "meta": {"voxel_um": 2.0, "target_psnr": 28.0,
                 "reconstruction": {"n_iter": 50, "log_every": 10, "target_corr": 0.90}},
    }


def test_stage_boundaries_marks_zpos_insertions():
    # stages 2,3,4 begin at 1x,2x,3x the per-stage iteration budget
    assert stage_boundaries(50, n_stages=4) == [50, 100, 150]
    assert stage_boundaries(50, n_stages=1) == []


def test_exp2_corr_convergence_writes_png(tmp_path):
    j = tmp_path / "pleno_metrics.json"
    j.write_text(json.dumps(_fake_pleno_metrics_corr()), encoding="utf-8")
    out = tmp_path / "exp2_corr_convergence.png"
    ret = exp2_corr_convergence(str(j), str(out))
    assert ret == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_volume_plotly_html_writes_html(tmp_path):
    vol = np.zeros((24, 24, 24), np.float32)
    vol[:, 8:16, 8:16] = 1.0  # a bright bar so the isosurface has something to draw
    out = tmp_path / "vol3d.html"
    volume_plotly_html(vol, str(out))
    assert out.exists() and out.stat().st_size > 0
    assert "<html" in out.read_text(encoding="utf-8").lower()


def _cold_solve(chunk):
    """A from-scratch chunk solve: climbs from ~0 to just past C0 over 4 log-points.

    History span is 31 iterations (last iter 30, +1) and 3.0 s, the compute a from-scratch
    solve spends; matches how _sequential_chain accounts a segment.
    """
    return {
        "chunk": chunk,
        "corr_with_gt": 0.951,
        "iters_to_threshold": 30,
        "time_to_threshold": 2.4,
        "history": {
            "iter": [0, 10, 20, 30],
            "time": [0.0, 1.0, 2.0, 3.0],
            "corr": [0.05, 0.60, 0.88, 0.951],
        },
    }


def _warm_solve(chunk):
    """A warm-started chunk solve: starts already near C0, crosses in 2 log-points.

    History span is 11 iterations (last iter 10, +1) and 1.0 s, far less than a from-scratch
    solve, so the warm arm's totals come in well below the cold arm's.
    """
    return {
        "chunk": chunk,
        "corr_with_gt": 0.952,
        "iters_to_threshold": 10,
        "time_to_threshold": 0.8,
        "history": {
            "iter": [0, 10],
            "time": [0.0, 1.0],
            "corr": [0.93, 0.952],
        },
    }


def _fake_chunkseq_metrics():
    """chunkseq_metrics.json shape (schema in run_plenoptic_chunkseq.py): anchor (chunk 0,
    full solve) + cold solves for every target chunk + two warm orderings A (adjacent) and
    B (every-second), each carrying the previous->new pair_misorientation. Warm solves are
    cheap (11 iters), cold solves full (31 iters); chunk 0 in each warm arm IS the anchor.
    """
    return {
        "meta": {"C0": 0.95, "orderings": {"A": [0, 1, 2, 3], "B": [0, 2, 4, 6]}},
        "anchor": _cold_solve(0),
        "cold": [_cold_solve(c) for c in (1, 2, 3, 4, 6)],
        "runs": {
            "A": {
                "warm": [_cold_solve(0), _warm_solve(1), _warm_solve(2), _warm_solve(3)],
                "pair_misorientation": [0.965, 0.963, 0.957],
                "cumulative_iters": 64, "cumulative_time": 6.0,
            },
            "B": {
                "warm": [_cold_solve(0), _warm_solve(2), _warm_solve(4), _warm_solve(6)],
                "pair_misorientation": [0.920, 0.915, 0.902],
                "cumulative_iters": 64, "cumulative_time": 6.0,
            },
        },
    }


def test_exp3_numbers_totals_and_speedup(tmp_path):
    from phantom_sim.analyze import exp3_numbers

    p = tmp_path / "chunkseq_metrics.json"
    p.write_text(json.dumps(_fake_chunkseq_metrics()), encoding="utf-8")
    nums = exp3_numbers(str(p))

    a = nums["orderings"]["A"]
    # Anchor step is the shared seed: no similarity, no saving.
    assert a["steps"][0]["similarity"] is None
    assert a["steps"][0]["speedup"] == 1.0
    # Every warm step is cheaper than the same chunk from scratch, and carries its neighbour sim.
    assert a["steps"][1]["similarity"] == 0.965
    for st in a["steps"][1:]:
        assert st["warm_iters"] < st["cold_iters"]
        assert st["speedup"] > 1.0
    # Totals: warm 31 + 11 + 11 + 11 = 64 iters vs cold 31 * 4 = 124 iters (segment-span
    # accounting identical to the sawtooth's end-lines).
    assert a["total"]["warm_iters"] == 64
    assert a["total"]["cold_iters"] == 124
    assert abs(a["total"]["speedup"] - 124 / 64) < 1e-9
    # Misorientation erosion is visible across orderings: B (more-different) has a lower
    # mean neighbour similarity than A (adjacent).
    assert nums["orderings"]["B"]["mean_similarity"] < a["mean_similarity"]


def test_exp3_sawtooth_writes_png(tmp_path):
    import os
    from phantom_sim.analyze import exp3_sawtooth

    p = tmp_path / "chunkseq_metrics.json"
    p.write_text(json.dumps(_fake_chunkseq_metrics()), encoding="utf-8")
    out = tmp_path / "exp3_sawtooth.png"
    ret = exp3_sawtooth(str(p), str(out))
    assert ret == str(out)
    assert os.path.exists(ret) and os.path.getsize(ret) > 0
    # The every-second ordering B renders through the same function.
    out_b = tmp_path / "exp3_sawtooth_B.png"
    ret_b = exp3_sawtooth(str(p), str(out_b), ordering="B")
    assert os.path.exists(ret_b) and os.path.getsize(ret_b) > 0


def test_exp3_table_writes_png(tmp_path):
    import os
    from phantom_sim.analyze import exp3_table

    p = tmp_path / "chunkseq_metrics.json"
    p.write_text(json.dumps(_fake_chunkseq_metrics()), encoding="utf-8")
    out = tmp_path / "exp3_numbers_table.png"
    ret = exp3_table(str(p), str(out))
    assert ret == str(out)
    assert os.path.exists(ret) and os.path.getsize(ret) > 0
