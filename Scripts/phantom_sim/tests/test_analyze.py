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
            out.append(
                {
                    "N": n,
                    "psnr": base + 2.0 * n,
                    "ssim": 0.6 + 0.05 * n,
                    "resolution": {"along": 0.02, "across_mean": 0.04, "ratio": 0.5},
                    "iters_to_threshold": n if n >= 2 else None,
                    "time_to_threshold": float(n) * 0.5 if n >= 2 else None,
                    "history": hist,
                }
            )
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
    m = {
        "axis0": {"psnr": 30.0, "ssim": 0.9, "resolution": {"ratio": 1.2}},
        "axis1": {"psnr": 24.0, "ssim": 0.8, "resolution": {"ratio": 2.1}},
    }
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
        "metric": [10.0 + i for i in range(n)],  # psnr, still logged
        "corr": [float(c) for c in corrs],
    }


def _fake_pleno_metrics_corr():
    """pleno_metrics.json with per-iteration corr traces + meta.reconstruction, as the
    corr-thresholding driver writes. Warm starts higher than cold (warm-started)."""

    def by_N(start):
        out = []
        for n in range(1, 5):
            corrs = [min(start + 0.05 * k, 0.99) for k in range(5)]
            out.append(
                {
                    "N": n,
                    "corr_with_gt": corrs[-1],
                    "psnr": 12.0 + n,
                    "ssim": 0.6,
                    "resolution": {"along": 0.02, "across_mean": 0.04, "ratio": 0.5},
                    "iters_to_threshold": 20,
                    "time_to_threshold": 1.0,
                    "history": _fake_corr_history(corrs),
                }
            )
        return out

    return {
        "cold_joint": {"by_N": by_N(0.70)},
        "warm_sequential": {
            "by_N": by_N(0.80),
            "cumulative_iters": 200,
            "cumulative_time": 20.0,
        },
        "meta": {
            "voxel_um": 2.0,
            "target_psnr": 28.0,
            "reconstruction": {"n_iter": 50, "log_every": 10, "target_corr": 0.90},
        },
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
                "warm": [
                    _cold_solve(0),
                    _warm_solve(1),
                    _warm_solve(2),
                    _warm_solve(3),
                ],
                "pair_misorientation": [0.965, 0.963, 0.957],
                "cumulative_iters": 64,
                "cumulative_time": 6.0,
            },
            "B": {
                "warm": [
                    _cold_solve(0),
                    _warm_solve(2),
                    _warm_solve(4),
                    _warm_solve(6),
                ],
                "pair_misorientation": [0.920, 0.915, 0.902],
                "cumulative_iters": 64,
                "cumulative_time": 6.0,
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


def _warm_solve_span(chunk, span):
    """A warm-started chunk solve whose iteration segment span is exactly `span`.

    Segment span is (last iter value + 1), matching _sequential_chain / exp3_numbers, so the
    last logged iteration is span - 1. A larger span means a costlier warm solve, i.e. less
    inherited from the previous chunk, which is how a higher-misorientation level is emulated.
    """
    iters = list(range(0, span, 10))
    n = len(iters)
    return {
        "chunk": chunk,
        "corr_with_gt": 0.952,
        "iters_to_threshold": iters[-1],
        "time_to_threshold": 0.8,
        "history": {
            "iter": iters,
            "time": [float(i) * 0.1 for i in range(n)],
            "corr": [0.90 + 0.005 * k for k in range(n)],
        },
    }


def _fake_level(misalignment, warm_span):
    """One misalignment level's chunkseq_metrics.json, with cold solves fixed at a 31-iter
    span and warm solves at `warm_span`, so a larger warm_span emulates a wavier bundle whose
    warm start saves less. Chunk 0 in each warm arm is the shared anchor."""
    return {
        "meta": {
            "misalignment": misalignment,
            "C0": 0.95,
            "orderings": {"A": [0, 1, 2, 3], "B": [0, 2, 4, 6]},
        },
        "anchor": _cold_solve(0),
        "cold": [_cold_solve(c) for c in (1, 2, 3, 4, 6)],
        "runs": {
            "A": {
                "warm": [_cold_solve(0)]
                + [_warm_solve_span(c, warm_span) for c in (1, 2, 3)],
                "pair_misorientation": [0.965, 0.963, 0.957],
            },
            "B": {
                "warm": [_cold_solve(0)]
                + [_warm_solve_span(c, warm_span) for c in (2, 4, 6)],
                "pair_misorientation": [0.920, 0.915, 0.902],
            },
        },
    }


def _write_ladder_levels(tmp_path):
    """Write three chunkseq files in ascending misorientation and return their paths in order."""
    paths = []
    for label, span in [("very low", 11), ("moderate", 21), ("high", 31)]:
        p = tmp_path / ("cs_%s.json" % label.replace(" ", "_"))
        p.write_text(json.dumps(_fake_level(label, span)), encoding="utf-8")
        paths.append(str(p))
    return paths


def test_exp3_ladder_stacks_levels_by_misorientation(tmp_path):
    from phantom_sim.analyze import exp3_ladder

    ladder = exp3_ladder(_write_ladder_levels(tmp_path))
    levels = ladder["levels"]
    # Levels preserve the caller's ascending-misorientation order and carry their labels.
    assert [lv["label"] for lv in levels] == ["very low", "moderate", "high"]
    # Cold compute is the same set of from-scratch solves at every level: 4 chunks * 31 = 124.
    assert levels[0]["orderings"]["A"]["cold_iters"] == 124
    # The warm-start speed-up erodes as fibres get more misoriented (warm solves cost more).
    a_speedups = [lv["orderings"]["A"]["speedup"] for lv in levels]
    assert a_speedups[0] > a_speedups[1] > a_speedups[2]
    # At the highest level the warm span equals the cold span, so there is no speed-up left.
    assert abs(a_speedups[2] - 1.0) < 1e-9


def test_exp3_ladder_figure_writes_png(tmp_path):
    import os
    from phantom_sim.analyze import exp3_ladder_figure

    out = tmp_path / "exp3_ladder.png"
    ret = exp3_ladder_figure(_write_ladder_levels(tmp_path), str(out))
    assert ret == str(out)
    assert os.path.exists(ret) and os.path.getsize(ret) > 0


def _fake_tilt_metrics(max_half_angle=9.66):
    """tilt_metrics.json shape (schema in run_plenoptic_tilt.assemble_tilt_metrics): corr holds
    high up to the array half-angle, then collapses as the fibres tilt beyond what the array
    samples."""
    data = [(0, 0.95), (4, 0.94), (8, 0.90), (10, 0.60), (12, 0.35), (20, 0.15)]
    sweep = [
        {
            "theta_deg": float(t),
            "corr_with_gt": float(c),
            "psnr": 17.0,
            "ssim": 0.9,
            "resolution": {"along": 0.01, "across_mean": 0.1, "ratio": 0.1},
        }
        for t, c in data
    ]
    return {"meta": {"max_half_angle_deg": max_half_angle}, "sweep": sweep}


def test_exp4_threshold_finds_dropoff(tmp_path):
    from phantom_sim.analyze import exp4_threshold_deg

    p = tmp_path / "tilt_metrics.json"
    p.write_text(json.dumps(_fake_tilt_metrics()), encoding="utf-8")
    # corr0 = 0.95, target = 0.475, crossed between 10 deg (0.60) and 12 deg (0.35) -> 11.0.
    thr = exp4_threshold_deg(str(p), drop_frac=0.5)
    assert abs(thr - 11.0) < 0.2


def test_exp4_threshold_none_when_no_dropoff(tmp_path):
    from phantom_sim.analyze import exp4_threshold_deg

    m = {
        "meta": {"max_half_angle_deg": 9.66},
        "sweep": [
            {
                "theta_deg": float(t),
                "corr_with_gt": 0.95,
                "psnr": 17.0,
                "ssim": 0.9,
                "resolution": {"along": 0.01, "across_mean": 0.1, "ratio": 0.1},
            }
            for t in (0, 5, 10)
        ],
    }
    p = tmp_path / "tilt_metrics.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    assert exp4_threshold_deg(str(p), drop_frac=0.5) is None


def test_exp4_bandwidth_figure_writes_png(tmp_path):
    import os
    from phantom_sim.analyze import exp4_bandwidth_figure

    p = tmp_path / "tilt_metrics.json"
    p.write_text(json.dumps(_fake_tilt_metrics()), encoding="utf-8")
    out = tmp_path / "exp4_bandwidth.png"
    ret = exp4_bandwidth_figure(str(p), str(out))
    assert ret == str(out)
    assert os.path.exists(ret) and os.path.getsize(ret) > 0


def _fake_psf_metrics():
    """psf_metrics.json shape (schema in run_plenoptic_psf.assemble_psf_metrics): a wide z
    point-spread and a narrow transverse one, built with the driver's own helpers so the
    profile/MTF fields are realistic."""
    from phantom_sim.run_plenoptic_psf import psf_entry, fwhm, mtf

    xs = np.arange(41)
    sigmas = {"z": 6.0, "y": 2.0, "x": 2.0}
    entries = []
    for name in ("z", "y", "x"):
        prof = np.exp(-0.5 * ((xs - 20) / sigmas[name]) ** 2)
        w = fwhm(prof, 2.0)
        f, mm = mtf(prof, 2.0)
        entries.append(psf_entry(name, w, prof, f, mm))
    fwz = entries[0]["fwhm_um"]
    fwt = 0.5 * (entries[1]["fwhm_um"] + entries[2]["fwhm_um"])
    meta = {
        "voxel_um": 2.0,
        "fwhm_z_um": fwz,
        "fwhm_transverse_um": fwt,
        "z_to_transverse_ratio": fwz / fwt,
    }
    return {"meta": meta, "psf": entries}


def test_exp5_psf_figure_writes_png(tmp_path):
    import os
    from phantom_sim.analyze import exp5_psf_figure

    p = tmp_path / "psf_metrics.json"
    p.write_text(json.dumps(_fake_psf_metrics()), encoding="utf-8")
    out = tmp_path / "exp5_psf.png"
    ret = exp5_psf_figure(str(p), str(out))
    assert ret == str(out)
    assert os.path.exists(ret) and os.path.getsize(ret) > 0


def _fake_aperture_level(half_angle):
    """One array aperture's tilt_metrics.json (schema in run_plenoptic_tilt): corr holds ~0.95
    until the fibre tilt reaches the array half-angle, then drops steeply, so the 50%-drop
    threshold lands just past half_angle and moves outward as the aperture grows."""
    thetas = [0, 5, 10, 15, 20, 25, 30, 35]
    sweep = []
    for t in thetas:
        c = 0.95 if t <= half_angle else max(0.05, 0.95 - 0.2 * (t - half_angle))
        sweep.append(
            {
                "theta_deg": float(t),
                "corr_with_gt": float(c),
                "psnr": 20.0,
                "ssim": 0.95,
                "resolution": {"along": 0.01, "across_mean": 0.1, "ratio": 0.1},
            }
        )
    return {"meta": {"max_half_angle_deg": float(half_angle)}, "sweep": sweep}


def _write_aperture_levels(tmp_path):
    """Three enlarged-array runs, ascending half-angle, whose collapse moves out with aperture."""
    paths = []
    for ha in (8.0, 16.0, 24.0):
        p = tmp_path / ("aperture_ha%02d.json" % ha)
        p.write_text(json.dumps(_fake_aperture_level(ha)), encoding="utf-8")
        paths.append(str(p))
    return paths


def test_exp6_aperture_threshold_grows_with_half_angle(tmp_path):
    from phantom_sim.analyze import exp6_aperture

    red = exp6_aperture(_write_aperture_levels(tmp_path))
    levels = red["levels"]
    # Levels preserve the caller's ascending-aperture order and carry their half-angles.
    assert [lv["half_angle_deg"] for lv in levels] == [8.0, 16.0, 24.0]
    thr = [lv["threshold_deg"] for lv in levels]
    assert all(t is not None for t in thr)
    # A larger array (bigger half-angle) tolerates more tilt before the reconstruction fails.
    assert thr[0] < thr[1] < thr[2]


def test_exp6_aperture_figure_writes_png(tmp_path):
    import os
    from phantom_sim.analyze import exp6_aperture_figure

    out = tmp_path / "exp6_aperture.png"
    ret = exp6_aperture_figure(_write_aperture_levels(tmp_path), str(out))
    assert ret == str(out)
    assert os.path.exists(ret) and os.path.getsize(ret) > 0


# ----------------------------------------------------------------------------------
# Experiment 7: ground-truth-free localized-misalignment QC map
# ----------------------------------------------------------------------------------


def _synth_bundle(shape=(40, 48, 48), theta_deg=25.0, seed=0):
    """Straight rods along axis 0 (the optical axis). Rods whose x-centre is in the right half
    shear in x with z by theta_deg (a localized tilted patch, as the exp7 driver builds); those
    voxels form the bad mask. Returns (vol float32, bad bool). scipy-only, matches the real data's
    fibre-axis-0 geometry so the structure-tensor convention is exercised end to end."""
    from scipy.ndimage import gaussian_filter

    nz, ny, nx = shape
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, np.float32)
    bad = np.zeros(shape, bool)
    zc = (nz - 1) / 2.0
    for y0 in range(4, ny - 4, 6):
        for x0 in range(4, nx - 4, 6):
            if abs(x0 - nx // 2) < 5:
                continue  # empty gap so aligned rods are not contaminated by the tilted patch
            tilted = x0 >= nx // 2
            th = np.radians(theta_deg) if tilted else 0.0
            for z in range(nz):
                xx = int(round(x0 + np.tan(th) * (z - zc)))
                if 0 <= xx < nx:
                    vol[z, y0, xx] = 1.0
                    if tilted:
                        bad[z, y0, xx] = True
    vol = gaussian_filter(vol, 0.8)
    # A touch of noise: real reconstructions are textured, so the structure tensor is
    # non-degenerate. (Perfectly clean rods make the transverse eigenvalues exactly equal and the
    # eigenvector NaN.) Keep it well below the blurred-rod peak (~0.12) so orientation stays clean.
    vol = vol + rng.normal(0.0, 0.004, size=shape).astype(np.float32)
    return vol.astype(np.float32), bad


def test_defect_auc_orders_and_handles_ties_and_empty():
    import math
    from phantom_sim.analyze import _defect_auc

    assert _defect_auc([3, 4, 5], [0, 1, 2]) == 1.0  # defect scores all above normal
    assert _defect_auc([0, 1, 2], [3, 4, 5]) == 0.0  # fully reversed
    assert (
        _defect_auc([0, 1, 2], [0, 1, 2]) == 0.5
    )  # identical pools -> chance, via ties
    assert math.isnan(_defect_auc([], [1.0, 2.0]))  # empty pool -> NaN


def test_exp7_qc_score_keys_and_separation():
    from phantom_sim.analyze import exp7_qc_score

    dev = np.full((4, 4, 4), np.nan, np.float32)
    fg = np.zeros((4, 4, 4), bool)
    bad = np.zeros((4, 4, 4), bool)
    fg[:2] = True
    dev[:2] = 1.0  # aligned region: low deviation
    fg[2:] = True
    dev[2:] = 20.0  # patch: high deviation
    bad[2:] = True

    s = exp7_qc_score(dev, fg, bad)
    for k in ("auc", "bad_mean", "bad_p90", "good_mean", "n_bad", "n_good"):
        assert k in s
    assert s["auc"] > 0.99
    assert s["bad_mean"] > s["good_mean"]
    assert s["n_bad"] == 32 and s["n_good"] == 32


def test_orientation_deviation_map_flags_tilted_patch(tmp_path):
    import pytest

    pytest.importorskip("structure_tensor")
    from phantom_sim.analyze import orientation_deviation_map

    # An all-axial bundle reads ~0 deg everywhere.
    vol0, _ = _synth_bundle(theta_deg=0.0)
    dev0, fg0 = orientation_deviation_map(vol0, sigma=0.5, rho=2.5, optical_axis=0)
    assert np.nanmean(dev0[fg0]) < 5.0

    # A 25 deg tilted patch lights up locally while the aligned region stays flat.
    vol, bad = _synth_bundle(theta_deg=25.0)
    dev, fg = orientation_deviation_map(vol, sigma=0.5, rho=2.5, optical_axis=0)
    good_m = float(np.nanmean(dev[fg & ~bad]))
    patch_m = float(np.nanmean(dev[fg & bad]))
    assert good_m < 8.0  # aligned region stays near-axial
    assert patch_m > 12.0  # the tilted patch reads a large deviation
    assert patch_m > good_m + 6.0  # and is clearly separated from the aligned region


def _write_exp7_dir(dirp, theta):
    import json as _json

    dirp.mkdir()
    vol, bad = _synth_bundle(theta_deg=theta)
    np.save(str(dirp / "local_recon.npy"), vol)
    np.save(str(dirp / "bad_mask.npy"), bad.astype(np.uint8))
    (dirp / "local_metrics.json").write_text(
        _json.dumps(
            {"meta": {"alpha_bad_deg": float(theta), "max_half_angle_deg": 9.66}}
        ),
        encoding="utf-8",
    )
    return str(dirp)


def test_exp7_reduce_ranks_control_below_tilted(tmp_path):
    import pytest

    pytest.importorskip("structure_tensor")
    from phantom_sim.analyze import exp7_reduce

    d0 = _write_exp7_dir(tmp_path / "tb00", 0.0)
    d1 = _write_exp7_dir(tmp_path / "tb25", 25.0)
    levels = exp7_reduce([d0, d1])["levels"]
    assert [lv["misalignment_deg"] for lv in levels] == [0.0, 25.0]
    # The untilted control is a flat map (AUC ~0.5); the tilted patch localizes far better.
    assert levels[1]["auc"] > levels[0]["auc"]
    assert levels[1]["bad_p90"] > levels[0]["bad_p90"]


def test_exp7_figure_writes_png(tmp_path):
    import os
    import pytest

    pytest.importorskip("structure_tensor")
    from phantom_sim.analyze import exp7_figure

    d0 = _write_exp7_dir(tmp_path / "tb00", 0.0)
    d1 = _write_exp7_dir(tmp_path / "tb25", 25.0)
    out = tmp_path / "exp7.png"
    ret = exp7_figure([d0, d1], str(out))
    assert ret == str(out)
    assert os.path.exists(ret) and os.path.getsize(ret) > 0
