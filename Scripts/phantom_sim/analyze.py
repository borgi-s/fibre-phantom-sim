"""Task 10: torch-free analysis and figures for the fibre phantom study (this box).

Reads the two cluster drivers' outputs (float32 `.npy` recon volumes plus their
`ct_metrics.json` / `pleno_metrics.json` sidecars, whose schemas are pinned in
run_ct_orientation.py and run_plenoptic_zpos.py) and renders the study figures. No
torch, no astra, no CUDA: pure numpy + matplotlib (Agg) + plotly, so every function
runs and is unit-tested on this box.

Public API:
  - load_volume(path) -> np.ndarray                       (torch-free memmap load)
  - slice_montage(vol, out_png, fibre_axis) -> None       (three orthogonal mid-slices)
  - exp1_figure(metrics_json, out_png) -> None            (rotation vs beam bars)
  - exp2_figures(metrics_json, out_dir) -> list[str]      (fidelity/convergence/threshold)
  - exp2_corr_convergence(metrics_json, out_png) -> str   (corr-vs-iter, warm vs cold, z-lines)
  - exp3_numbers(metrics_json) -> dict                    (warm-vs-scratch cost numbers)
  - exp3_sawtooth(metrics_json, out_png) -> str           (chunk-sequence sawtooth, warm vs scratch)
  - exp3_table(metrics_json, out_png) -> str              (concrete cost/speed-up table PNG)
  - exp3_ladder(level_paths) -> dict                      (warm-start savings stacked by level)
  - exp3_ladder_figure(level_paths, out_png) -> str       (speed-up + saved compute vs misorientation)
  - exp4_threshold_deg(metrics_json) -> float|None        (angular bandwidth threshold, deg)
  - exp4_bandwidth_figure(metrics_json, out_png) -> str    (corr vs tilt, half-angle + threshold)
  - exp5_psf_figure(metrics_json, out_png) -> str          (z vs transverse point-spread + MTF)
  - exp5b_psf_sweep_figure(metrics_json, out_png) -> str    (z/transverse FWHM vs array half-angle)
  - exp5b_anisotropy_fit(theta_deg, ratio) -> dict           (a cot(theta)^b fit of dz/dx vs aperture)
  - exp6_aperture(level_paths) -> dict                     (bandwidth threshold vs array aperture)
  - exp6_aperture_figure(level_paths, out_png) -> str      (corr-vs-tilt, threshold-vs-aperture, corr-vs-aperture)
  - orientation_deviation_map(vol, ...) -> (dev, fg)       (ground-truth-free structure-tensor QC map)
  - exp7_qc_score(dev, fg, bad_mask) -> dict               (defect-localization score vs the GT mask)
  - exp7_reduce(level_dirs, ...) -> dict                   (per-tilt QC numbers across the exp7 ladder)
  - exp7_figure(level_dirs, out_png, ...) -> str           (deviation-map localization + score ladder)
  - exp7_deviation_html(recon_path, out_html, ...) -> str  (interactive 3D of the flagged patch)
  - volume_plotly_html(vol, out_html, ...) -> None        (interactive 3D isosurface)
  - main(argv) -> None                                     (build every figure for a run)

The exp7 functions need the DTU/QIM ``structure_tensor`` package (scipy-based); that import is
deferred into the functions that use it, so the rest of analyze.py stays numpy + matplotlib +
plotly only. The library returns the principal orientation vector in (x, y, z) component order,
i.e. reversed from numpy's (axis0, axis1, axis2); the optical axis is numpy axis 0 (the fibre
axis the drivers voxelize along), so its component is ``vec[2]`` (calibrated against tb00: an
all-axial bundle gives vec ~ (0, 0, 1)).

Figures are written with a modest DPI so PNGs stay well under the 5 MB SendUserFile cap.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------


def load_volume(path):
    """Load a float32 recon volume torch-free, memory-mapped (read-only).

    Returns a numpy memmap; wrap in np.asarray(...) if you need it fully in RAM.
    """
    return np.load(str(path), mmap_mode="r")


def _read_json(path):
    with open(str(path), "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------
# Slice montage (works for any recon volume)
# ----------------------------------------------------------------------------------


def slice_montage(vol, out_png, fibre_axis=0, dpi=110):
    """Three orthogonal mid-plane slices of a volume, saved as one PNG.

    The panel whose slicing axis is `fibre_axis` is the cross-fibre (transverse) view;
    the other two cut along the fibres. Titles name each cutting axis so the along vs
    across geometry is unambiguous.
    """
    vol = np.asarray(vol, dtype=np.float32)
    labels = {fibre_axis: "cross-fibre (perp to axis %d)" % fibre_axis}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmin = float(np.percentile(vol, 1.0))
    vmax = float(np.percentile(vol, 99.0))
    for ax_idx, ax in zip(range(3), axes):
        mid = vol.shape[ax_idx] // 2
        sl = np.take(vol, mid, axis=ax_idx)
        ax.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        title = labels.get(ax_idx, "along-fibre (cut axis %d)" % ax_idx)
        ax.set_title("axis %d = %d  %s" % (ax_idx, mid, title), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------
# Experiment 1: CT rotation-axis vs beam-axis
# ----------------------------------------------------------------------------------


def exp1_figure(metrics_json, out_png, dpi=110):
    """Grouped bars comparing axis0 (rotation-aligned) vs axis1 (beam/transverse).

    Reads ct_metrics.json = {"axis0": {psnr, ssim, resolution:{ratio}}, "axis1": {...}}.
    Three panels: PSNR (dB), SSIM, and along/across resolution ratio.
    """
    m = _read_json(metrics_json)
    a0, a1 = m["axis0"], m["axis1"]
    labels = ["rotation-aligned\n(axis 0)", "beam/transverse\n(axis 1)"]
    panels = [
        ("PSNR (dB)", [a0["psnr"], a1["psnr"]]),
        ("SSIM", [a0["ssim"], a1["ssim"]]),
        (
            "resolution ratio\n(along / across)",
            [a0["resolution"]["ratio"], a1["resolution"]["ratio"]],
        ),
    ]
    colours = ["#2c7fb8", "#d95f0e"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, (title, vals) in zip(axes, panels):
        ax.bar([0, 1], vals, color=colours, width=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=10)
        for xi, v in enumerate(vals):
            ax.text(xi, v, "%.3g" % v, ha="center", va="bottom", fontsize=9)
        ax.margins(y=0.15)
    fig.suptitle(
        "Experiment 1: CT fibre orientation vs reconstruction fidelity", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------
# Experiment 2: plenoptic z-position count and warm vs cold
# ----------------------------------------------------------------------------------


def _by_N(strategy_block):
    entries = sorted(strategy_block["by_N"], key=lambda e: e["N"])
    Ns = [e["N"] for e in entries]
    return entries, Ns


def exp2_figures(metrics_json, out_dir, dpi=110):
    """Build the Experiment 2 figure set. Returns the list of PNG paths written.

    Reads pleno_metrics.json (schema in run_plenoptic_zpos.py):
      cold_joint.by_N[*] and warm_sequential.by_N[*] carry N, psnr, ssim,
      resolution, iters_to_threshold, time_to_threshold, history; warm_sequential
      also carries cumulative_iters / cumulative_time.

    Figures:
      1. fidelity vs N: final PSNR and SSIM, cold vs warm.
      2. convergence curves: PSNR vs iteration for the N=max stage, cold vs warm.
      3. cost to threshold: iters_to_threshold and time_to_threshold vs N, cold vs warm.
    """
    m = _read_json(metrics_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = m.get("meta", {}).get("target_psnr")

    cold, Ns = _by_N(m["cold_joint"])
    warm, _ = _by_N(m["warm_sequential"])
    written = []

    # --- Figure 1: fidelity vs N (PSNR + SSIM) -------------------------------------
    fig, (axp, axs) = plt.subplots(1, 2, figsize=(11, 4))
    axp.plot(Ns, [e["psnr"] for e in cold], "o-", color="#2c7fb8", label="cold joint")
    axp.plot(
        Ns, [e["psnr"] for e in warm], "s-", color="#d95f0e", label="warm sequential"
    )
    if target is not None:
        axp.axhline(
            target, color="grey", ls="--", lw=1, label="target %.0f dB" % target
        )
    axp.set_xlabel("number of z-positions N")
    axp.set_ylabel("final PSNR (dB)")
    axp.set_title("Fidelity vs z-position count")
    axp.set_xticks(Ns)
    axp.legend(fontsize=8)
    axs.plot(Ns, [e["ssim"] for e in cold], "o-", color="#2c7fb8", label="cold joint")
    axs.plot(
        Ns, [e["ssim"] for e in warm], "s-", color="#d95f0e", label="warm sequential"
    )
    axs.set_xlabel("number of z-positions N")
    axs.set_ylabel("final SSIM")
    axs.set_title("Structural similarity vs N")
    axs.set_xticks(Ns)
    axs.legend(fontsize=8)
    fig.tight_layout()
    p1 = out_dir / "exp2_fidelity_vs_N.png"
    fig.savefig(str(p1), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p1))

    # --- Figure 2: convergence curves at the largest N -----------------------------
    n_max = Ns[-1]
    cold_last = next(e for e in cold if e["N"] == n_max)
    warm_last = next(e for e in warm if e["N"] == n_max)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ch, wh = cold_last["history"], warm_last["history"]
    if ch.get("metric"):
        ax.plot(ch["iter"], ch["metric"], "o-", color="#2c7fb8", label="cold joint")
    if wh.get("metric"):
        ax.plot(
            wh["iter"], wh["metric"], "s-", color="#d95f0e", label="warm sequential"
        )
    if target is not None:
        ax.axhline(target, color="grey", ls="--", lw=1, label="target %.0f dB" % target)
    ax.set_xlabel("iteration")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Convergence at N=%d: cold vs warm" % n_max)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p2 = out_dir / "exp2_convergence_Nmax.png"
    fig.savefig(str(p2), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p2))

    # --- Figure 3: cost to threshold vs N ------------------------------------------
    def _clean(vals):
        return [np.nan if v is None else float(v) for v in vals]

    fig, (axi, axt) = plt.subplots(1, 2, figsize=(11, 4))
    axi.plot(
        Ns,
        _clean([e["iters_to_threshold"] for e in cold]),
        "o-",
        color="#2c7fb8",
        label="cold joint",
    )
    axi.plot(
        Ns,
        _clean([e["iters_to_threshold"] for e in warm]),
        "s-",
        color="#d95f0e",
        label="warm sequential (per stage)",
    )
    axi.set_xlabel("number of z-positions N")
    axi.set_ylabel("iterations to threshold")
    ttl = "Iterations to reach target"
    if target is not None:
        ttl += " (%.0f dB)" % target
    axi.set_title(ttl)
    axi.set_xticks(Ns)
    axi.legend(fontsize=8)
    axt.plot(
        Ns,
        _clean([e["time_to_threshold"] for e in cold]),
        "o-",
        color="#2c7fb8",
        label="cold joint",
    )
    axt.plot(
        Ns,
        _clean([e["time_to_threshold"] for e in warm]),
        "s-",
        color="#d95f0e",
        label="warm sequential (per stage)",
    )
    axt.set_xlabel("number of z-positions N")
    axt.set_ylabel("wall-clock to threshold (s)")
    axt.set_title("Time to reach target")
    axt.set_xticks(Ns)
    axt.legend(fontsize=8)
    cum_i = m["warm_sequential"].get("cumulative_iters")
    cum_t = m["warm_sequential"].get("cumulative_time")
    if cum_i is not None and cum_t is not None:
        fig.suptitle(
            "warm-sequential total compute over the 1..%d chain: "
            "%d iters, %.1f s" % (n_max, cum_i, cum_t),
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        fig.tight_layout()
    p3 = out_dir / "exp2_cost_to_threshold.png"
    fig.savefig(str(p3), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    written.append(str(p3))

    return written


# ----------------------------------------------------------------------------------
# Experiment 2: warm-start vs cold-joint convergence to a corr-with-GT threshold
# ----------------------------------------------------------------------------------


def stage_boundaries(n_iter, n_stages=4):
    """Cumulative-iteration x-positions where a new z-position enters the warm chain.

    The warm chain runs one solve per N in 1..n_stages, each of n_iter iterations, drawn
    on one continuous axis; stages 2..n_stages therefore begin at 1x, 2x, ... the per-stage
    budget. These are the "a new z-position was added here" marker lines. Empty for a
    single-stage chain.
    """
    return [k * n_iter for k in range(1, n_stages)]


def _sequential_chain(entries, xkey, thr_key, target_corr):
    """Lay one arm's stages end to end on a single cumulative axis in `xkey` units.

    Returns (xs, ys, seg_starts, crosses, total): one polyline over all stages (warm reads
    as a mostly-rising chain with small dips where a new z-position enters; cold sawtooths,
    dropping to ~0 at each independent restart), the x where each stage begins, the
    (x, target_corr) threshold-crossing points, and the total compute the arm spent to reach
    the 4-position result. `xkey` is "iter" or "time"; the per-stage span is iterations-run
    (last+1) or seconds-elapsed (last) accordingly, so both arms' variable-length (early-
    stopped) stages place honestly and the two totals are directly comparable.
    """
    xs, ys, seg_starts, crosses = [], [], [], []
    off = 0.0
    for e in entries:
        h = e["history"]
        loc = h.get(xkey) or []
        cr = h.get("corr") or []
        if not loc:
            continue
        seg_starts.append(off)
        xs.extend(off + v for v in loc)
        ys.extend(cr)
        thr = e.get(thr_key)
        if thr is not None and target_corr is not None:
            crosses.append((off + thr, target_corr))
        off += (loc[-1] + 1) if xkey == "iter" else loc[-1]
    return xs, ys, seg_starts, crosses, off


def exp2_corr_convergence(metrics_json, out_png, dpi=130):
    """corr-with-GT vs compute for Experiment 2, warm-sequential vs cold-joint, as two
    sequential chains laid end to end on one axis.

    Reads pleno_metrics.json (schema in run_plenoptic_zpos.py). Warm inherits the previous
    position's volume, so its chain stays high across all four stages with small bumps where
    each new z-position enters; cold restarts every solve from scratch, so it sawtooths down
    to ~0 and re-climbs each stage. The two chains END at different x, and that gap is the
    total compute the warm chain saves to reach the same 4-position result. Two panels: x =
    iterations (the fair count) and x = wall-clock seconds (the real time). Vertical marks:
    warm segment starts along the top, cold along the bottom, plus a solid end line per arm;
    dots/squares sit at each stage's threshold crossing. Returns the PNG path written.
    """
    m = _read_json(metrics_json)
    recon = m.get("meta", {}).get("reconstruction", {})
    cold = sorted(m["cold_joint"]["by_N"], key=lambda e: e["N"])
    warm = sorted(m["warm_sequential"]["by_N"], key=lambda e: e["N"])
    target_corr = recon.get("target_corr")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    warm_col, cold_col = "#d95f0e", "#2c7fb8"
    panels = [
        ("iter", "iters_to_threshold", "iteration"),
        ("time", "time_to_threshold", "wall-clock (s)"),
    ]

    for ax, (xkey, thr_key, xlabel) in zip(axes, panels):
        cx, cy, c_starts, c_cross, c_end = _sequential_chain(
            cold, xkey, thr_key, target_corr
        )
        wx, wy, w_starts, w_cross, w_end = _sequential_chain(
            warm, xkey, thr_key, target_corr
        )
        ax.plot(
            cx, cy, "--", color=cold_col, lw=1.4, label="cold joint (restart per N)"
        )
        ax.plot(wx, wy, "-", color=warm_col, lw=1.9, label="warm sequential (chained)")
        if c_cross:
            ax.plot(
                [p[0] for p in c_cross],
                [p[1] for p in c_cross],
                "s",
                color=cold_col,
                ms=6,
            )
        if w_cross:
            ax.plot(
                [p[0] for p in w_cross],
                [p[1] for p in w_cross],
                "o",
                color=warm_col,
                ms=6,
            )
        if target_corr is not None:
            ax.axhline(
                target_corr,
                color="k",
                ls="-.",
                lw=0.8,
                label="target corr %.2f" % target_corr,
            )
        ymin, ymax = ax.get_ylim()
        for i, s in enumerate(w_starts):  # warm z-position insertions, along the top
            ax.axvline(s, color=warm_col, ls=":", lw=0.7, alpha=0.5)
            ax.text(
                s,
                ymax,
                " N=%d" % (i + 1),
                color=warm_col,
                fontsize=7,
                va="top",
                ha="left",
            )
        for i, s in enumerate(c_starts):  # cold independent restarts, along the bottom
            ax.text(
                s,
                ymin,
                " N=%d" % (i + 1),
                color=cold_col,
                fontsize=7,
                va="bottom",
                ha="left",
            )
        ax.axvline(w_end, color=warm_col, lw=1.2)  # compute each arm needed to finish
        ax.axvline(c_end, color=cold_col, lw=1.2)
        ax.set_title(
            "%s to the 4-position result: warm %.0f vs cold %.0f (save %.0f)"
            % (xlabel, w_end, c_end, c_end - w_end),
            fontsize=9,
        )
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("corr-with-GT (object crop)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Experiment 2: warm-start vs cold-joint, corr-with-GT vs compute", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


# ----------------------------------------------------------------------------------
# Experiment 3: chunk-sequence warm-start cost vs fibre misorientation
# ----------------------------------------------------------------------------------


def _segment_span(history, xkey):
    """Compute one solve actually spent along `xkey`, matching _sequential_chain's accounting.

    For iterations the span is last_iter + 1 (the count of iterations run); for wall-clock it
    is the last elapsed second. Returns 0 for an empty history so a missing solve adds nothing.
    """
    loc = (history or {}).get(xkey) or []
    if not loc:
        return 0
    return (loc[-1] + 1) if xkey == "iter" else loc[-1]


def exp3_numbers(metrics_json):
    """Concrete warm-vs-from-scratch cost numbers for the chunk-sequence study.

    Reads chunkseq_metrics.json (schema in run_plenoptic_chunkseq.py). For each ordering
    (A adjacent 0,1,2,3; B every-second 0,2,4,6) it pairs every warm chunk solve with the
    from-scratch (cold) solve of the same target chunk and reports the compute each spent
    (iterations run and wall-clock minutes) plus the warm speed-up. Chunk 0 is the shared
    anchor, identical on both arms. Cost is the segment span actually run, so per-ordering
    totals equal the sawtooth end-lines exactly. Returns a plain dict (no figure), which the
    table renderer and any caption text can format.
    """
    m = _read_json(metrics_json)
    C0 = m["meta"]["C0"]
    orderings = m["meta"]["orderings"]
    anchor = m["anchor"]
    cold_by_chunk = {e["chunk"]: e for e in m["cold"]}
    cold_by_chunk[anchor["chunk"]] = anchor  # chunk 0 from-scratch IS the anchor solve

    out = {"C0": C0, "orderings": {}}
    for name, chunks in orderings.items():
        run = m["runs"][name]
        warm = run["warm"]
        mis = run.get("pair_misorientation", [])
        steps = []
        wi_tot = wt_tot = ci_tot = ct_tot = 0
        for k, w in enumerate(warm):
            chunk = w["chunk"]
            wi = _segment_span(w.get("history"), "iter")
            wt = _segment_span(w.get("history"), "time")
            cold_entry = cold_by_chunk.get(chunk, {})
            ci = _segment_span(cold_entry.get("history"), "iter")
            ct = _segment_span(cold_entry.get("history"), "time")
            wi_tot += wi
            wt_tot += wt
            ci_tot += ci
            ct_tot += ct
            if k == 0:
                label, sim, speed = "chunk %d (seed)" % chunk, None, 1.0
            else:
                label = "%d → %d" % (warm[k - 1]["chunk"], chunk)
                sim = mis[k - 1] if k - 1 < len(mis) else None
                speed = (float(ci) / wi) if wi else float("nan")
            steps.append(
                {
                    "label": label,
                    "chunk": chunk,
                    "similarity": sim,
                    "cold_iters": ci,
                    "cold_min": ct / 60.0,
                    "warm_iters": wi,
                    "warm_min": wt / 60.0,
                    "speedup": speed,
                }
            )
        valid_mis = [x for x in mis if x is not None]
        out["orderings"][name] = {
            "ordering": chunks,
            "steps": steps,
            "total": {
                "cold_iters": ci_tot,
                "cold_min": ct_tot / 60.0,
                "warm_iters": wi_tot,
                "warm_min": wt_tot / 60.0,
                "speedup": (float(ci_tot) / wi_tot) if wi_tot else float("nan"),
            },
            "mean_similarity": (sum(valid_mis) / len(valid_mis)) if valid_mis else None,
        }
    return out


def exp3_sawtooth(metrics_json, out_png, ordering="A", dpi=130):
    """Corr-with-GT vs compute for the chunk sequence, warm-start vs from-scratch, as the
    same end-to-end sawtooth as Experiment 2.

    Reads chunkseq_metrics.json. The warm arm reconstructs a genuinely new volume at each
    chunk by inheriting the previous chunk's recon, so it stays high with shallow dips where
    each new chunk enters; the from-scratch arm restarts every solve, sawtoothing to ~0 and
    re-climbing. The two arms end at different x, and that gap is the compute the warm chain
    saves to reach the same set of chunk reconstructions. Two panels: x = iterations (the fair
    count) and x = wall-clock seconds. Chunk 0 is the shared anchor. `ordering` selects the
    chain (A adjacent, B every-second). Returns the PNG path written.
    """
    m = _read_json(metrics_json)
    C0 = m["meta"]["C0"]
    chunks = m["meta"]["orderings"][ordering]
    anchor = m["anchor"]
    cold_by_chunk = {e["chunk"]: e for e in m["cold"]}
    cold_by_chunk[anchor["chunk"]] = anchor
    warm_arm = m["runs"][ordering]["warm"]
    cold_arm = [cold_by_chunk[c] for c in chunks]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    warm_col, cold_col = "#d95f0e", "#2c7fb8"
    panels = [
        ("iter", "iters_to_threshold", "iteration"),
        ("time", "time_to_threshold", "wall-clock (s)"),
    ]

    for ax, (xkey, thr_key, xlabel) in zip(axes, panels):
        cx, cy, c_starts, c_cross, c_end = _sequential_chain(
            cold_arm, xkey, thr_key, C0
        )
        wx, wy, w_starts, w_cross, w_end = _sequential_chain(
            warm_arm, xkey, thr_key, C0
        )
        ax.plot(
            cx,
            cy,
            "--",
            color=cold_col,
            lw=1.4,
            label="from scratch (restart per chunk)",
        )
        ax.plot(wx, wy, "-", color=warm_col, lw=1.9, label="warm start (chained)")
        if c_cross:
            ax.plot(
                [p[0] for p in c_cross],
                [p[1] for p in c_cross],
                "s",
                color=cold_col,
                ms=6,
            )
        if w_cross:
            ax.plot(
                [p[0] for p in w_cross],
                [p[1] for p in w_cross],
                "o",
                color=warm_col,
                ms=6,
            )
        ax.axhline(C0, color="k", ls="-.", lw=0.8, label="target quality %.3f" % C0)
        ymin, ymax = ax.get_ylim()
        for i, s in enumerate(w_starts):  # warm chunk insertions, along the top
            ax.axvline(s, color=warm_col, ls=":", lw=0.7, alpha=0.5)
            ax.text(
                s,
                ymax,
                " c%d" % chunks[i],
                color=warm_col,
                fontsize=7,
                va="top",
                ha="left",
            )
        for i, s in enumerate(c_starts):  # from-scratch restarts, along the bottom
            ax.text(
                s,
                ymin,
                " c%d" % chunks[i],
                color=cold_col,
                fontsize=7,
                va="bottom",
                ha="left",
            )
        ax.axvline(w_end, color=warm_col, lw=1.2)
        ax.axvline(c_end, color=cold_col, lw=1.2)
        ax.set_title(
            "%s for %d chunks: warm %.0f vs scratch %.0f (save %.0f)"
            % (xlabel, len(chunks), w_end, c_end, c_end - w_end),
            fontsize=9,
        )
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("corr-with-GT (object crop)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Experiment 3: warm-start across a wandering bundle (ordering %s), "
        "corr-with-GT vs compute" % ordering,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


def exp3_table(metrics_json, out_png, dpi=130):
    """Render the concrete warm-vs-from-scratch numbers as a two-block table PNG.

    Top block: per-chunk cost for ordering A (adjacent) with the from-scratch cost of the same
    chunk and the warm speed-up, ending in an all-chunks total. Bottom block: the two orderings
    compared, so the misorientation nuance reads at a glance (more-different neighbours save
    slightly less). Numbers come from exp3_numbers, so the totals match the sawtooth. Returns
    the PNG path written.
    """
    nums = exp3_numbers(metrics_json)
    A = nums["orderings"]["A"]

    def fmt_sim(s):
        return "-" if s is None else "%.3f" % s

    def fmt_im(it, mn):
        return "%d  (%.1f)" % (round(it), mn)

    def fmt_x(s):
        return "%.1fx" % s

    col = [
        "Step",
        "New-vol\nsimilarity",
        "From scratch\niters (min)",
        "Warm start\niters (min)",
        "Speed-up",
    ]
    rows = [
        [
            st["label"],
            fmt_sim(st["similarity"]),
            fmt_im(st["cold_iters"], st["cold_min"]),
            fmt_im(st["warm_iters"], st["warm_min"]),
            fmt_x(st["speedup"]),
        ]
        for st in A["steps"]
    ]
    tot = A["total"]
    rows.append(
        [
            "TOTAL (%d chunks)" % len(A["ordering"]),
            "-",
            fmt_im(tot["cold_iters"], tot["cold_min"]),
            fmt_im(tot["warm_iters"], tot["warm_min"]),
            fmt_x(tot["speedup"]),
        ]
    )

    pretty = {"A": "A: adjacent (0,1,2,3)", "B": "B: every-second (0,2,4,6)"}
    col2 = [
        "Chain",
        "Mean new-vol\nsimilarity",
        "Warm total\niters (min)",
        "Speed-up\nvs scratch",
    ]
    rows2 = []
    for name in ("A", "B"):
        o = nums["orderings"][name]
        t = o["total"]
        rows2.append(
            [
                pretty.get(name, name),
                "%.3f" % o["mean_similarity"]
                if o["mean_similarity"] is not None
                else "-",
                fmt_im(t["warm_iters"], t["warm_min"]),
                fmt_x(t["speedup"]),
            ]
        )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(11, 5),
        gridspec_kw={"height_ratios": [len(rows) + 1, len(rows2) + 1]},
    )
    for ax in (ax1, ax2):
        ax.axis("off")

    t1 = ax1.table(cellText=rows, colLabels=col, loc="center", cellLoc="center")
    t1.auto_set_font_size(False)
    t1.set_fontsize(9)
    t1.scale(1, 1.6)
    ax1.set_title(
        "Warm-start vs from-scratch: reconstructing a wandering bundle chunk by chunk",
        fontsize=11,
        pad=12,
    )

    t2 = ax2.table(cellText=rows2, colLabels=col2, loc="center", cellLoc="center")
    t2.auto_set_font_size(False)
    t2.set_fontsize(9)
    t2.scale(1, 1.6)
    ax2.set_title("More-different neighbours save slightly less", fontsize=10, pad=8)

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


# ----------------------------------------------------------------------------------
# Experiment 3 ladder: warm-start savings across misorientation levels
# ----------------------------------------------------------------------------------


def exp3_ladder(level_paths):
    """Stack per-misalignment chunk-sequence runs into one warm-start savings ladder.

    `level_paths` is an ordered sequence of chunkseq_metrics.json paths, ascending in fibre
    misorientation (for example very low, moderate, high). For each file and each ordering
    (A adjacent, B every-second) this reuses exp3_numbers to get the warm-vs-from-scratch
    totals, and reports the warm compute, the from-scratch compute, the warm speed-up, and the
    mean neighbour misorientation. Returns {"levels": [...]}, one entry per file in the given
    order, each carrying its misalignment label and per-ordering totals, so the
    savings-versus-misorientation trend reads straight off. No new compute: pure reduction of
    the metrics files already on disk.
    """
    levels = []
    for path in level_paths:
        nums = exp3_numbers(path)
        label = _read_json(path)["meta"].get("misalignment", "")
        orderings = {}
        for name, o in nums["orderings"].items():
            t = o["total"]
            orderings[name] = {
                "warm_iters": t["warm_iters"],
                "cold_iters": t["cold_iters"],
                "warm_min": t["warm_min"],
                "cold_min": t["cold_min"],
                "speedup": t["speedup"],
                "mean_similarity": o["mean_similarity"],
            }
        levels.append({"label": label, "orderings": orderings})
    return {"levels": levels}


def exp3_ladder_figure(level_paths, out_png, dpi=130):
    """Two-panel warm-start savings ladder across misorientation levels. Returns the PNG path.

    Left: warm-start speed-up versus fibre misorientation level, one line per ordering
    (A adjacent, B every-second), with a unity reference line. Right: absolute compute saved
    (from-scratch minus warm iterations) versus level. `level_paths` must be ordered ascending
    in misorientation; the x-axis labels come from each file's meta.misalignment.
    """
    ladder = exp3_ladder(level_paths)
    levels = ladder["levels"]
    labels = [lv["label"] for lv in levels]
    x = list(range(len(levels)))
    styles = (
        ("A", "#d95f0e", "o", "A: adjacent"),
        ("B", "#2c7fb8", "s", "B: every-second"),
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, col, mk, lbl in styles:
        sp = [lv["orderings"][name]["speedup"] for lv in levels]
        ax1.plot(x, sp, mk + "-", color=col, label=lbl)
        for xi, v in zip(x, sp):
            ax1.text(xi, v, " %.2fx" % v, fontsize=8, va="bottom", color=col)
    ax1.axhline(1.0, color="grey", ls="--", lw=1, label="no speed-up")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("fibre misorientation level")
    ax1.set_ylabel("warm-start speed-up (x)")
    ax1.set_title("Warm-start compute saving vs misorientation")
    ax1.legend(fontsize=8)
    ax1.margins(y=0.15)

    for name, col, mk, lbl in styles:
        saved = [
            lv["orderings"][name]["cold_iters"] - lv["orderings"][name]["warm_iters"]
            for lv in levels
        ]
        ax2.plot(x, saved, mk + "-", color=col, label=lbl)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("fibre misorientation level")
    ax2.set_ylabel("iterations saved (from scratch minus warm)")
    ax2.set_title("Absolute compute saved vs misorientation")
    ax2.legend(fontsize=8)
    ax2.margins(y=0.15)

    fig.suptitle(
        "Experiment 3 ladder: warm-start benefit across misorientation levels",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


# ----------------------------------------------------------------------------------
# Experiment 4: angular bandwidth from the tilt sweep
# ----------------------------------------------------------------------------------


def exp4_threshold_deg(metrics_json, drop_frac=0.5):
    """Tilt angle at which the reconstruction correlation first falls below drop_frac of its
    straight-bundle value, linearly interpolated. Returns None if it never drops that far.

    Reads tilt_metrics.json (schema in run_plenoptic_tilt.assemble_tilt_metrics). This is the
    angular bandwidth: the largest fibre tilt the plenoptic reconstruction still captures.
    """
    m = _read_json(metrics_json)
    sweep = sorted(m.get("sweep", []), key=lambda e: e["theta_deg"])
    if not sweep:
        return None
    target = drop_frac * sweep[0]["corr_with_gt"]
    prev = sweep[0]
    for e in sweep[1:]:
        if e["corr_with_gt"] < target <= prev["corr_with_gt"]:
            t0, c0 = prev["theta_deg"], prev["corr_with_gt"]
            t1, c1 = e["theta_deg"], e["corr_with_gt"]
            if c0 == c1:
                return float(t1)
            return float(t0 + (c0 - target) / (c0 - c1) * (t1 - t0))
        prev = e
    return None


def exp4_bandwidth_figure(metrics_json, out_png, dpi=130, drop_frac=0.5):
    """Correlation-with-ground-truth vs fibre tilt, with the array max half-angle and the
    reconstruction threshold marked. Returns the PNG path.

    The correlation holds while the fibre tilt stays within the angles the plenoptic array
    samples, then collapses; the threshold (a drop to `drop_frac` of the straight-bundle value)
    versus the array half-angle is the angular bandwidth of the plenoptic reconstruction.
    """
    m = _read_json(metrics_json)
    sweep = sorted(m["sweep"], key=lambda e: e["theta_deg"])
    theta = [e["theta_deg"] for e in sweep]
    corr = [e["corr_with_gt"] for e in sweep]
    half = m.get("meta", {}).get("max_half_angle_deg")
    thr = exp4_threshold_deg(metrics_json, drop_frac=drop_frac)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(theta, corr, "o-", color="#2c7fb8", label="corr with ground truth")
    if half is not None:
        ax.axvline(
            half,
            color="#d95f0e",
            ls="--",
            lw=1.5,
            label="array max half-angle %.1f deg" % half,
        )
    if thr is not None:
        ax.axvline(
            thr, color="k", ls=":", lw=1.5, label="recon threshold %.1f deg" % thr
        )
    ax.set_xlabel("fibre tilt from optical axis (deg)")
    ax.set_ylabel("corr with ground truth")
    ax.set_title("Angular bandwidth: plenoptic reconstruction vs fibre tilt")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


# ----------------------------------------------------------------------------------
# Experiment 5: point-spread / MTF, longitudinal vs transverse resolution
# ----------------------------------------------------------------------------------


def exp5_psf_figure(metrics_json, out_png, dpi=130):
    """Point-spread profiles and MTF, along z versus transverse. Returns the PNG path.

    Reads psf_metrics.json (schema in run_plenoptic_psf.assemble_psf_metrics). Left: the
    normalised reconstruction response through the centre along z (the fibre axis) and along a
    transverse axis, with the FWHM of each. Right: the corresponding MTF curves. The z response
    is far broader than the transverse one, which is the poor longitudinal resolution that blurs
    defects varying quickly along z.
    """
    m = _read_json(metrics_json)
    by = {e["axis"]: e for e in m["psf"]}
    voxel_um = m.get("meta", {}).get("voxel_um", 1.0)
    z = by["z"]
    tr = by.get("x") or by.get("y")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for e, lbl, col in (
        (z, "along z (fibre axis)", "#d95f0e"),
        (tr, "transverse", "#2c7fb8"),
    ):
        prof = np.asarray(e["profile"], dtype=np.float64)
        base, pk = prof.min(), prof.max()
        norm = (prof - base) / (pk - base) if pk > base else prof * 0.0
        off = (np.arange(len(prof)) - (len(prof) // 2)) * voxel_um
        ax1.plot(
            off, norm, "-", color=col, label="%s (FWHM %.0f um)" % (lbl, e["fwhm_um"])
        )
    ax1.axhline(0.5, color="grey", ls=":", lw=1)
    ax1.set_xlabel("offset from centre (um)")
    ax1.set_ylabel("normalised response")
    ax1.set_title("Point-spread profile: z vs transverse")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    for e, lbl, col in ((z, "along z", "#d95f0e"), (tr, "transverse", "#2c7fb8")):
        f = np.asarray(e["freqs_inv_um"], dtype=np.float64)
        mm = np.asarray(e["mtf"], dtype=np.float64)
        ax2.plot(f, mm, "-", color=col, label=lbl)
    ax2.set_xlabel("spatial frequency (cycles/um)")
    ax2.set_ylabel("MTF")
    ax2.set_title("MTF: z vs transverse")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    ratio = m.get("meta", {}).get("z_to_transverse_ratio")
    if ratio:
        fig.suptitle(
            "Longitudinal vs transverse resolution (z is %.1fx coarser)" % ratio,
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


def exp5b_anisotropy_fit(theta_deg, ratio):
    """Least-squares fit of the anisotropy ratio to r(theta) = a * cot(theta)**b.

    Numpy only: the model is linear in log space (log r = log a + b * log cot theta), so it is a
    straight polyfit on (log cot theta, log r). b = 1 is the pure parallax law dz ~ cot(theta) with
    a theta-independent transverse FWHM; b < 1 means the transverse FWHM itself grows with the
    aperture, softening the fall-off. Returns a dict with a, b, r2 (in linear space), the mean and
    max relative residual (%), and theta_iso_deg, the half-angle where the fit crosses r = 1
    (isotropic resolution), which is atan(a**(1/b)).
    """
    th = np.asarray(theta_deg, dtype=np.float64)
    r = np.asarray(ratio, dtype=np.float64)
    ok = (th > 0) & (th < 90) & (r > 0)
    th, r = th[ok], r[ok]
    if th.size < 3:
        return {}
    x = np.log(1.0 / np.tan(np.radians(th)))
    b, log_a = np.polyfit(x, np.log(r), 1)
    a = float(np.exp(log_a))
    b = float(b)
    pred = a * (1.0 / np.tan(np.radians(th))) ** b
    ss_res = float(np.sum((r - pred) ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    rel = 100.0 * np.abs(pred - r) / r
    theta_iso = math.degrees(math.atan(a ** (1.0 / b))) if b != 0.0 else float("nan")
    return {
        "a": a,
        "b": b,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rel_mean_pct": float(rel.mean()),
        "rel_max_pct": float(rel.max()),
        "theta_iso_deg": theta_iso,
    }


def exp5b_psf_sweep_figure(metrics_json, out_png, dpi=130):
    """Longitudinal and transverse FWHM vs array max half-angle, with the parallax law overlaid.
    Returns the PNG path.

    Reads psf_sweep_metrics.json (schema in run_plenoptic_psf_sweep.assemble_psf_sweep_metrics).
    Left: measured z FWHM and transverse FWHM against the half-angle, with the geometry
    prediction dz = dz_ref * tan(theta_ref)/tan(theta) anchored at the run's own current-array
    control point (so it passes through the measured anchor). Right: the anisotropy ratio
    dz/dx (measured markers) with the isotropic line and a least-squares fit
    r = a cot(theta)^b (exp5b_anisotropy_fit), annotated with the half-angle where the fit reaches
    isotropy. The prediction is parallax-limited depth resolution; the measured points test
    whether the plenoptic reconstruction follows cot(theta) and where it reaches isotropy.
    """
    m = _read_json(metrics_json)
    sweep = sorted(m["sweep"], key=lambda e: e["theta_deg"])
    theta = np.array([e["theta_deg"] for e in sweep], dtype=np.float64)
    dz = np.array([e["fwhm_z_um"] for e in sweep], dtype=np.float64)
    dx = np.array([e["fwhm_transverse_um"] for e in sweep], dtype=np.float64)
    ratio = np.array([e["z_to_transverse_ratio"] for e in sweep], dtype=np.float64)

    # Anchor the parallax law at the current-array control point (closest to 9.66 deg).
    cur = math.degrees(
        math.atan(
            (
                m.get("meta", {}).get("source_range_um", 22000.0)
                + m.get("meta", {}).get("current_det_range_um", 177400.0)
            )
            / (m.get("meta", {}).get("sdd_mm", 1171.57) * 1000.0)
        )
    )
    i_anchor = int(np.argmin(np.abs(theta - cur)))
    dz_ref, th_ref = dz[i_anchor], theta[i_anchor]
    th_dense = np.linspace(max(2.0, theta.min() - 1), theta.max() + 1, 300)
    dz_pred = dz_ref * math.tan(math.radians(th_ref)) / np.tan(np.radians(th_dense))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax1.plot(
        th_dense,
        dz_pred,
        "-",
        color="#d95f0e",
        lw=1.3,
        alpha=0.6,
        label="parallax law  dz = dz0 tan(theta0)/tan(theta)",
    )
    ax1.plot(theta, dz, "o-", color="#d95f0e", label="measured z FWHM")
    ax1.plot(theta, dx, "s-", color="#2c7fb8", label="measured transverse FWHM")
    ax1.scatter(
        [th_ref],
        [dz_ref],
        s=90,
        marker="*",
        color="k",
        zorder=6,
        label="current array (%.1f deg)" % th_ref,
    )
    ax1.set_xlabel("array max half-angle theta (deg)")
    ax1.set_ylabel("FWHM (um)")
    ax1.set_title("Longitudinal vs transverse resolution vs aperture")
    ax1.set_ylim(0, max(dz.max(), 50) * 1.05)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(theta, ratio, "o", color="#6a0dad", ms=6, label="measured dz/dx")
    fit = exp5b_anisotropy_fit(theta, ratio)
    if fit:
        th_fit = np.linspace(theta.min() * 0.9, theta.max() * 1.05, 400)
        ax2.plot(
            th_fit,
            fit["a"] * (1.0 / np.tan(np.radians(th_fit))) ** fit["b"],
            "-",
            color="#6a0dad",
            lw=1.5,
            alpha=0.75,
            label="fit  %.2f cot(theta)^%.2f  (R2 = %.3f)"
            % (fit["a"], fit["b"], fit["r2"]),
        )
        ax2.plot(
            theta,
            1.0 / np.tan(np.radians(theta)),
            "--",
            color="grey",
            lw=1.1,
            alpha=0.8,
            label="pure parallax  cot(theta)",
        )
        if np.isfinite(fit["theta_iso_deg"]):
            ax2.axvline(
                fit["theta_iso_deg"], color="#6a0dad", ls="-.", lw=1.0, alpha=0.5
            )
            ax2.annotate(
                "isotropy at %.0f deg" % fit["theta_iso_deg"],
                xy=(fit["theta_iso_deg"], 1.0),
                xytext=(-6, 18),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color="#6a0dad",
            )
    ax2.axhline(1.0, color="k", ls=":", lw=1.3, label="isotropic (dz = dx)")
    ax2.set_xlabel("array max half-angle theta (deg)")
    ax2.set_ylabel("anisotropy ratio  dz / dx")
    ax2.set_ylim(0, max(ratio.max() * 1.1, 1.2))
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_title("Anisotropy vs aperture")

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


# ----------------------------------------------------------------------------------
# Experiment 6: angular bandwidth vs array aperture (the enlarged-array sweep)
# ----------------------------------------------------------------------------------


def _corr_at_tilt(thetas, corr, tilt_deg):
    """Correlation linearly interpolated at tilt_deg; clamps to the endpoints out of range."""
    return float(
        np.interp(
            float(tilt_deg),
            np.asarray(thetas, dtype=np.float64),
            np.asarray(corr, dtype=np.float64),
        )
    )


def exp6_aperture(level_paths, drop_frac=0.9):
    """Reduce tilt sweeps taken at several array apertures into a bandwidth-vs-aperture summary.

    `level_paths` is an ordered sequence of tilt_metrics.json paths, ascending in the array's
    max ray half-angle (the enlarged-array runs of exp6). For each file this reads the array
    half-angle (meta.max_half_angle_deg) and the corr-vs-tilt sweep, and reuses
    exp4_threshold_deg for the onset angle. The default drop_frac=0.9 (corr falls to 90% of the
    straight-bundle value, a 10% drop) is the deviation angle at which the reconstruction starts to
    fail: exp4 showed the corr declines gently, so the 50% collapse lands
    several times past the geometric half-angle (and off the frame), while the 10% onset knee
    sits near it and stays inside the feasible tilt range. Returns {"levels": [...]} in the given
    order, each with half_angle_deg, thetas, corr, and threshold_deg (None if the sweep never
    drops that far). Pure reduction, no new compute; the trend it tests is that the onset moves
    outward as the array grows (the bandwidth is extended by enlarging the aperture).
    """
    levels = []
    for path in level_paths:
        m = _read_json(path)
        sweep = sorted(m["sweep"], key=lambda e: e["theta_deg"])
        levels.append(
            {
                "half_angle_deg": float(m.get("meta", {}).get("max_half_angle_deg")),
                "thetas": [float(e["theta_deg"]) for e in sweep],
                "corr": [float(e["corr_with_gt"]) for e in sweep],
                "threshold_deg": exp4_threshold_deg(path, drop_frac=drop_frac),
            }
        )
    return {"levels": levels}


def exp6_aperture_figure(
    level_paths, out_png, ref_tilt_deg=20.0, drop_frac=0.9, dpi=130
):
    """Three-panel enlarged-array result: does spreading the array recover misaligned fibres?

    `level_paths` must be ordered ascending in array half-angle. Left: corr-with-ground-truth
    versus fibre tilt, one curve per array aperture (coloured by half-angle), with each array's
    max half-angle drawn as a matching dotted vertical line, so the onset visibly moves outward
    as the array grows. Middle: the reconstruction onset angle (from exp4_threshold_deg, a 10%
    corr drop by default) versus the array half-angle, against the y = x diagonal reference; a
    line parallel to and above it means the bandwidth is extended proportionally to the aperture.
    Right: corr at a fixed "highly misaligned" tilt (ref_tilt_deg) versus the array half-angle,
    the intuitive "a wider array recovers the tilted fibres" curve. Returns the PNG path.
    """
    red = exp6_aperture(level_paths, drop_frac=drop_frac)
    levels = red["levels"]
    cols = plt.cm.viridis(np.linspace(0.12, 0.85, max(len(levels), 1)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

    # Left: corr vs tilt, one curve per aperture, with each array's half-angle marked.
    for lv, c in zip(levels, cols):
        ax1.plot(
            lv["thetas"],
            lv["corr"],
            "o-",
            color=c,
            label="half-angle %.0f deg" % lv["half_angle_deg"],
        )
        ax1.axvline(lv["half_angle_deg"], color=c, ls=":", lw=1)
    ax1.set_xlabel("fibre tilt from optical axis (deg)")
    ax1.set_ylabel("corr with ground truth")
    ax1.set_title("Reconstruction vs tilt, per array aperture")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Middle: threshold vs half-angle against y = x (the bandwidth-is-the-aperture test).
    ha = [lv["half_angle_deg"] for lv in levels]
    lo, hi = (min(ha), max(ha)) if ha else (0.0, 1.0)
    ax2.plot([lo, hi], [lo, hi], "k--", lw=1, label="threshold = half-angle")
    got = [
        (lv["half_angle_deg"], lv["threshold_deg"])
        for lv in levels
        if lv["threshold_deg"] is not None
    ]
    if got:
        ax2.plot(
            [h for h, _ in got],
            [t for _, t in got],
            "o-",
            color="#d95f0e",
            label="recon threshold",
        )
    for lv in (
        levels
    ):  # apertures whose sweep never collapsed in range: arrow up off the top point
        if lv["threshold_deg"] is None and lv["thetas"]:
            top = max(lv["thetas"])
            ax2.annotate(
                "",
                xy=(lv["half_angle_deg"], top + 4),
                xytext=(lv["half_angle_deg"], top),
                arrowprops=dict(arrowstyle="->", color="grey"),
            )
    ax2.set_xlabel("array max half-angle (deg)")
    ax2.set_ylabel("onset tilt: 10% corr drop (deg)")
    ax2.set_title("Angular bandwidth vs array aperture")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Right: corr at a fixed high tilt vs half-angle (recovering the misaligned fibres).
    corr_ref = [_corr_at_tilt(lv["thetas"], lv["corr"], ref_tilt_deg) for lv in levels]
    ax3.plot(ha, corr_ref, "s-", color="#2c7fb8")
    ax3.axvline(
        ref_tilt_deg,
        color="grey",
        ls=":",
        lw=1,
        label="array half-angle = tilt (%.0f deg)" % ref_tilt_deg,
    )
    ax3.set_xlabel("array max half-angle (deg)")
    ax3.set_ylabel("corr at %.0f deg fibre tilt" % ref_tilt_deg)
    ax3.set_title("Recovering %.0f deg-misaligned fibres" % ref_tilt_deg)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    fig.suptitle(
        "Experiment 6: a larger array extends the angular bandwidth", fontsize=12
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


# ----------------------------------------------------------------------------------
# Experiment 7: localized-misalignment QC map (ground-truth-free structure tensor)
# ----------------------------------------------------------------------------------


def _defect_auc(pos, neg):
    """Area under the ROC of a defect score: P(score(defect) > score(normal)), ties at 0.5.

    `pos` are the scores on ground-truth defect voxels, `neg` on normal voxels; a higher score
    means "more defect-like", so this is the probability the map ranks a random defect voxel
    above a random normal one. Non-finite entries are dropped. Returns NaN if either pool is
    empty. Vectorised via searchsorted (O(n log n)), exact including ties.
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    n_p, n_n = pos.size, neg.size
    if n_p == 0 or n_n == 0:
        return float("nan")
    neg_sorted = np.sort(neg)
    less = np.searchsorted(neg_sorted, pos, side="left").sum()
    lesseq = np.searchsorted(neg_sorted, pos, side="right").sum()
    wins = float(less) + 0.5 * float(lesseq - less)
    return wins / (float(n_p) * float(n_n))


def orientation_deviation_map(
    vol, sigma=0.5, rho=2.5, optical_axis=0, fg_frac=0.5, fg_pctl=99.0
):
    """Ground-truth-free QC map: local fibre-orientation deviation from the optical axis (deg).

    Reuses the established structure-tensor workflow (Structure_tensor_test.ipynb): build the 3D
    structure tensor at noise scale `sigma` and integration scale `rho`, take the principal fibre
    orientation (eigenvector of the smallest eigenvalue), and measure its angle away from the
    optical axis. A well-aligned bundle reads ~0 deg; a locally misaligned patch lights up near
    its true tilt, with no ground truth used. Deviation is defined only on reconstructed fibre
    voxels (the foreground `fg = vol > fg_frac * percentile(vol, fg_pctl)`); off-foreground voxels
    are NaN.

    `optical_axis` is a numpy axis (0 = the fibre axis the drivers voxelize along). The
    ``structure_tensor`` library returns the orientation vector in (x, y, z) order, reversed from
    numpy axes, so numpy axis a maps to vector component ``2 - a``. Returns (dev, fg): dev is a
    float32 volume in degrees (NaN off-foreground), fg the boolean foreground mask.
    """
    import warnings
    import structure_tensor as st

    vol = np.ascontiguousarray(vol, dtype=np.float32)
    fg = vol > (fg_frac * float(np.percentile(vol, fg_pctl)))
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )  # background voxels divide-by-zero in vec normalisation
        S = st.structure_tensor_3d(vol, sigma, rho)
        _, vec = st.eig_special_3d(S, eigenvalue_order="asc")
    comp = np.abs(np.clip(vec[2 - int(optical_axis)], -1.0, 1.0))
    dev = np.degrees(np.arccos(comp)).astype(np.float32)
    dev[~fg] = np.nan
    return dev, fg


def exp7_qc_score(dev, fg, bad_mask):
    """Score a deviation QC map against the ground-truth bad-fibre mask (localization readout).

    Compares the deviation inside the GT misaligned patch against the aligned region, both
    restricted to reconstructed fibre voxels (`fg`). Returns a dict with the ROC-AUC of the map as
    a defect detector (high deviation = defect), plus the mean / 90th-percentile / std deviation
    in each region and the voxel counts. AUC ~0.5 with a near-zero bad-vs-good gap means a flat
    map (the all-aligned control); AUC -> 1 means the patch is cleanly localized.
    """
    bad = np.asarray(bad_mask, dtype=bool)
    fg = np.asarray(fg, dtype=bool)
    pos = dev[bad & fg]
    neg = dev[fg & ~bad]
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]

    def _stats(a):
        if a.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        return (float(np.mean(a)), float(np.percentile(a, 90.0)), float(np.std(a)))

    bad_mean, bad_p90, bad_std = _stats(pos)
    good_mean, good_p90, good_std = _stats(neg)
    return {
        "auc": _defect_auc(pos, neg),
        "bad_mean": bad_mean,
        "bad_p90": bad_p90,
        "bad_std": bad_std,
        "good_mean": good_mean,
        "good_p90": good_p90,
        "good_std": good_std,
        "n_bad": int(pos.size),
        "n_good": int(neg.size),
    }


def _load_exp7_dir(d):
    """Load one exp7 tag directory: (recon volume, bad-fibre mask, local_metrics.json meta)."""
    d = Path(d)
    vol = np.asarray(load_volume(str(d / "local_recon.npy")), dtype=np.float32)
    bad = np.asarray(load_volume(str(d / "bad_mask.npy"))).astype(bool)
    meta = (
        _read_json(d / "local_metrics.json").get("meta", {})
        if (d / "local_metrics.json").exists()
        else {}
    )
    return vol, bad, meta


def _exp7_analyze_dir(d, sigma, rho):
    """Full per-tag analysis: (vol, bad, meta, dev, fg, score) for one exp7 directory."""
    vol, bad, meta = _load_exp7_dir(d)
    dev, fg = orientation_deviation_map(vol, sigma=sigma, rho=rho, optical_axis=0)
    score = exp7_qc_score(dev, fg, bad)
    return vol, bad, meta, dev, fg, score


def _exp7_misalignment_deg(meta):
    """The defect misalignment angle from a local_metrics.json meta: the wavy-half peak angle
    (`alpha_bad_deg`) for the waviness model, falling back to the old rigid-shear `theta_bad_deg`."""
    return float(meta.get("alpha_bad_deg", meta.get("theta_bad_deg", float("nan"))))


def exp7_reduce(level_dirs, sigma=0.5, rho=2.5):
    """Reduce the exp7 misalignment ladder to per-level QC numbers (no figure, no heavy arrays kept).

    `level_dirs` is a sequence of tag directories (each with local_recon.npy, bad_mask.npy and
    local_metrics.json). For each it computes the ground-truth-free deviation map and scores it
    against the wavy-half mask, returning {"levels": [...]} in the given order, each entry carrying
    its misalignment angle plus the exp7_qc_score fields. Ascending misalignment reads the trend
    straight off: the defect-free control stays flat (AUC ~0.5), wavy halves localize (AUC -> 1).
    """
    levels = []
    for d in level_dirs:
        vol, bad, meta, dev, fg, score = _exp7_analyze_dir(d, sigma, rho)
        entry = {
            "misalignment_deg": _exp7_misalignment_deg(meta),
            "max_half_angle_deg": meta.get("max_half_angle_deg"),
        }
        entry.update(score)
        levels.append(entry)
    return {"levels": levels}


def _exp7_patch_slice(vol, dev, bad, axis=1):
    """A representative side-slice through the patch centroid: (recon 2D, dev 2D, bad 2D).

    Cuts the volume at the bad-patch centroid along `axis` (default 1 = y, which the shear leaves
    intact, so the tilt shows as a diagonal band in the returned z-x plane). Falls back to the
    mid-plane when the mask is empty.
    """
    if bad.any():
        c = int(round(float(np.mean(np.where(bad)[axis]))))
    else:
        c = vol.shape[axis] // 2
    return (
        np.take(vol, c, axis=axis),
        np.take(dev, c, axis=axis),
        np.take(bad, c, axis=axis),
    )


def exp7_figure(level_dirs, out_png, sigma=0.5, rho=2.5, dpi=130):
    """Localized-misalignment QC demonstrator figure across the exp7 tilt ladder. Returns the path.

    Top row: for each tag, a side-slice of the reconstruction (grey) through the bad-patch
    centroid with the ground-truth-free deviation map overlaid (hot = large tilt) and the GT patch
    outlined; the control stays dark, and the patch lights up more as the true tilt grows. Bottom
    row: (left) mean and 90th-percentile recovered deviation inside the patch vs the aligned
    region against the true tilt, with the y = x line (the p90 tracks the true tilt while the mean
    saturates past onset as the patch smears); (right) the localization AUC vs tilt, flat-map
    reference at 0.5. `level_dirs` should be ordered ascending in theta_bad.
    """
    tags = list(level_dirs)
    recon_sl, dev_sl, bad_sl, thetas, scores, half_angle = [], [], [], [], [], None
    for d in tags:
        vol, bad, meta, dev, fg, score = _exp7_analyze_dir(d, sigma, rho)
        rs, ds, bs = _exp7_patch_slice(vol, dev, bad, axis=1)
        recon_sl.append(rs)
        dev_sl.append(ds)
        bad_sl.append(bs)
        thetas.append(_exp7_misalignment_deg(meta))
        scores.append(score)
        if half_angle is None and meta.get("max_half_angle_deg") is not None:
            half_angle = float(meta["max_half_angle_deg"])
        del vol, dev, fg, bad

    n = len(tags)
    dev_vmax = max(30.0, max((t for t in thetas if np.isfinite(t)), default=30.0))
    fig = plt.figure(figsize=(3.4 * max(n, 1), 8.2))
    gs = fig.add_gridspec(
        2, max(n, 2), height_ratios=[1.25, 1.0], hspace=0.28, wspace=0.12
    )

    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        rs = np.asarray(recon_sl[i], dtype=np.float32)
        vmin, vmax = np.percentile(rs, 1.0), np.percentile(rs, 99.5)
        ax.imshow(rs, cmap="gray", vmin=vmin, vmax=vmax, aspect="equal")
        overlay = np.ma.masked_invalid(dev_sl[i])
        im = ax.imshow(
            overlay, cmap="inferno", vmin=0.0, vmax=dev_vmax, alpha=0.75, aspect="equal"
        )
        if np.asarray(bad_sl[i]).any():
            ax.contour(
                np.asarray(bad_sl[i], dtype=float),
                levels=[0.5],
                colors="cyan",
                linewidths=0.7,
            )
        ttl = "alpha_bad = %.1f deg" % thetas[i]
        ax.set_title("%s\nAUC = %.3f" % (ttl, scores[i]["auc"]), fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    cax = fig.add_axes([0.92, 0.55, 0.015, 0.33])
    fig.colorbar(im, cax=cax).set_label("orientation deviation (deg)", fontsize=8)

    axL = fig.add_subplot(gs[1, : max(n, 2) // 2])
    axR = fig.add_subplot(gs[1, max(n, 2) // 2 :])
    th = np.asarray(thetas, dtype=float)
    bad_mean = [s["bad_mean"] for s in scores]
    bad_p90 = [s["bad_p90"] for s in scores]
    good_mean = [s["good_mean"] for s in scores]
    lo, hi = float(np.nanmin(th)), float(np.nanmax(th))
    axL.plot([lo, hi], [lo, hi], "k--", lw=1, label="recovered = imposed angle")
    axL.plot(th, bad_p90, "o-", color="#d95f0e", label="wavy half, 90th pct")
    axL.plot(th, bad_mean, "s-", color="#f0a860", label="wavy half, mean")
    axL.plot(th, good_mean, "^-", color="#2c7fb8", label="calm half, mean")
    if half_angle is not None:
        axL.axvline(
            half_angle,
            color="grey",
            ls=":",
            lw=1.2,
            label="array half-angle %.1f deg" % half_angle,
        )
    axL.set_xlabel("wavy-half angle alpha_bad (deg)")
    axL.set_ylabel("recovered orientation deviation (deg)")
    axL.set_title("Recovered waviness vs ground truth")
    axL.legend(fontsize=7.5)
    axL.grid(alpha=0.3)

    axR.plot(th, [s["auc"] for s in scores], "o-", color="#762a83")
    axR.axhline(0.5, color="grey", ls="--", lw=1, label="flat map (no localization)")
    for xi, s in zip(th, scores):
        axR.text(xi, s["auc"], " %.3f" % s["auc"], fontsize=7.5, va="bottom")
    axR.set_ylim(0.4, 1.03)
    axR.set_xlabel("wavy-half angle alpha_bad (deg)")
    axR.set_ylabel("defect-localization AUC")
    axR.set_title("Ground-truth-free localization")
    axR.legend(fontsize=7.5)
    axR.grid(alpha=0.3)

    fig.suptitle(
        "Experiment 7: localized misalignment as a ground-truth-free QC signal",
        fontsize=13,
    )
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


def exp7_deviation_html(
    recon_path, out_html, sigma=0.5, rho=2.5, dev_min_deg=6.0, max_points=40000, seed=0
):
    """Interactive 3D scatter of the flagged (high-deviation) fibre voxels. Returns the HTML path.

    Computes the ground-truth-free deviation map for one reconstruction and plots the foreground
    voxels whose deviation exceeds `dev_min_deg`, coloured by deviation, so the misaligned patch
    stands out as a coloured blob inside the otherwise-empty aligned bundle (the phone-friendly
    "where is the defect" view). Voxels are randomly subsampled to at most `max_points`.
    """
    import plotly.graph_objects as go

    vol = np.asarray(load_volume(str(recon_path)), dtype=np.float32)
    dev, fg = orientation_deviation_map(vol, sigma=sigma, rho=rho, optical_axis=0)
    flag = fg & np.isfinite(dev) & (dev >= dev_min_deg)
    zz, yy, xx = np.where(flag)
    vals = dev[flag]
    if zz.size > max_points:
        rng = np.random.default_rng(seed)
        sel = rng.choice(zz.size, size=max_points, replace=False)
        zz, yy, xx, vals = zz[sel], yy[sel], xx[sel], vals[sel]
    fig = go.Figure(
        data=go.Scatter3d(
            x=xx,
            y=yy,
            z=zz,
            mode="markers",
            marker=dict(
                size=1.6,
                color=vals,
                colorscale="Inferno",
                cmin=0.0,
                colorbar=dict(title="dev (deg)"),
                opacity=0.65,
            ),
        )
    )
    fig.update_layout(
        title="exp7 QC: fibre voxels tilted > %.0f deg from the optical axis"
        % dev_min_deg,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z (optical axis 0)"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)
    return str(out_html)


# ----------------------------------------------------------------------------------
# Interactive 3D view (mobile workflow)
# ----------------------------------------------------------------------------------


def volume_plotly_html(vol, out_html, max_dim=96, iso_percentile=90.0):
    """Write a self-contained interactive Plotly isosurface HTML for one recon volume.

    The volume is downsampled so the largest axis is at most `max_dim` voxels (keeps the
    HTML small enough for the phone workflow), and the isosurface level is placed at the
    `iso_percentile` intensity percentile.
    """
    import plotly.graph_objects as go

    vol = np.asarray(vol, dtype=np.float32)
    step = tuple(max(1, s // max_dim) for s in vol.shape)
    small = vol[:: step[0], :: step[1], :: step[2]]
    zz, yy, xx = np.mgrid[0 : small.shape[0], 0 : small.shape[1], 0 : small.shape[2]]
    level = float(np.percentile(small, iso_percentile))
    lo = float(small.min())
    hi = float(small.max())
    if hi <= lo:
        hi = lo + 1.0
    fig = go.Figure(
        data=go.Isosurface(
            x=xx.ravel(),
            y=yy.ravel(),
            z=zz.ravel(),
            value=small.ravel(),
            isomin=level,
            isomax=hi,
            surface_count=2,
            opacity=0.4,
            colorscale="Viridis",
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )
    fig.update_layout(
        title="Reconstructed volume (isosurface at %.1f pct)" % iso_percentile,
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z (fibre axis 0)"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)


# ----------------------------------------------------------------------------------
# CLI: build every figure for a results directory
# ----------------------------------------------------------------------------------


def build_parser():
    p = argparse.ArgumentParser(
        description="Build all Experiment 1/2 figures from a returned results directory."
    )
    p.add_argument(
        "--results",
        required=True,
        help="Directory holding ct_metrics.json and/or pleno_metrics.json plus "
        "the recon .npy volumes.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Figure output directory (default: <results>/figures).",
    )
    p.add_argument(
        "--plotly-volume",
        default=None,
        help="Optional recon .npy to render as an interactive 3D HTML.",
    )
    p.add_argument(
        "--aperture-parent",
        default=None,
        help="Parent dir holding several tilt-sweep runs at different array apertures "
        "(each in its own subdir with tilt_metrics.json); builds the exp6 "
        "angular-bandwidth-vs-aperture figure across them.",
    )
    p.add_argument(
        "--exp7-parent",
        default=None,
        help="Parent dir holding the exp7 tilt-ladder tag runs (each subdir with "
        "local_recon.npy, bad_mask.npy, local_metrics.json); builds the exp7 "
        "localized-misalignment QC figure across them, ascending in theta_bad.",
    )
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    results = Path(args.results)
    out_dir = Path(args.out) if args.out else results / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    ct_json = results / "ct_metrics.json"
    if ct_json.exists():
        out = out_dir / "exp1_ct_orientation.png"
        exp1_figure(str(ct_json), str(out))
        made.append(str(out))
        for name in ["ct_axis0.npy", "ct_axis1.npy"]:
            vp = results / name
            if vp.exists():
                mp = out_dir / (name.replace(".npy", "_montage.png"))
                slice_montage(load_volume(str(vp)), str(mp), fibre_axis=0)
                made.append(str(mp))

    pleno_json = results / "pleno_metrics.json"
    if pleno_json.exists():
        made.extend(exp2_figures(str(pleno_json), str(out_dir)))
        made.append(
            exp2_corr_convergence(
                str(pleno_json), str(out_dir / "exp2_corr_convergence.png")
            )
        )

    chunkseq_json = results / "chunkseq_metrics.json"
    if chunkseq_json.exists():
        for od in ("A", "B"):
            made.append(
                exp3_sawtooth(
                    str(chunkseq_json),
                    str(out_dir / ("exp3_sawtooth_%s.png" % od)),
                    ordering=od,
                )
            )
        made.append(
            exp3_table(str(chunkseq_json), str(out_dir / "exp3_numbers_table.png"))
        )

    tilt_json = results / "tilt_metrics.json"
    if tilt_json.exists():
        made.append(
            exp4_bandwidth_figure(str(tilt_json), str(out_dir / "exp4_bandwidth.png"))
        )

    psf_json = results / "psf_metrics.json"
    if psf_json.exists():
        made.append(exp5_psf_figure(str(psf_json), str(out_dir / "exp5_psf.png")))

    psf_sweep_json = results / "psf_sweep_metrics.json"
    if psf_sweep_json.exists():
        made.append(
            exp5b_psf_sweep_figure(
                str(psf_sweep_json), str(out_dir / "exp5b_psf_sweep.png")
            )
        )

    if args.aperture_parent:
        parent = Path(args.aperture_parent)
        metric_paths = sorted(
            parent.glob("*/tilt_metrics.json"),
            key=lambda mp: (
                _read_json(str(mp)).get("meta", {}).get("max_half_angle_deg", 0.0)
            ),
        )
        if metric_paths:
            made.append(
                exp6_aperture_figure(
                    [str(mp) for mp in metric_paths], str(out_dir / "exp6_aperture.png")
                )
            )
        else:
            print("No */tilt_metrics.json under %s for the aperture figure" % parent)

    if args.exp7_parent:
        parent = Path(args.exp7_parent)
        tag_dirs = sorted(
            (p.parent for p in parent.glob("*/local_metrics.json")),
            key=lambda d: _exp7_misalignment_deg(
                _read_json(d / "local_metrics.json").get("meta", {})
            ),
        )
        if tag_dirs:
            made.append(
                exp7_figure(
                    [str(d) for d in tag_dirs],
                    str(out_dir / "exp7_localmisalign_qc.png"),
                )
            )
        else:
            print("No */local_metrics.json under %s for the exp7 figure" % parent)

    if args.plotly_volume:
        vp = Path(args.plotly_volume)
        out = out_dir / (vp.stem + "_3d.html")
        volume_plotly_html(load_volume(str(vp)), str(out))
        made.append(str(out))

    for pth in made:
        print("Wrote %s" % pth)
    if not made:
        print("No ct_metrics.json or pleno_metrics.json found under %s" % results)


if __name__ == "__main__":
    main()
