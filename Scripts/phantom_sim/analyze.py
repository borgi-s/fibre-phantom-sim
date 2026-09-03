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
  - volume_plotly_html(vol, out_html, ...) -> None        (interactive 3D isosurface)
  - main(argv) -> None                                     (build every figure for a run)

Figures are written with a modest DPI so PNGs stay well under the 5 MB SendUserFile cap.
"""
import argparse
import json
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
        ("resolution ratio\n(along / across)",
         [a0["resolution"]["ratio"], a1["resolution"]["ratio"]]),
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
    fig.suptitle("Experiment 1: CT fibre orientation vs reconstruction fidelity",
                 fontsize=12)
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
    axp.plot(Ns, [e["psnr"] for e in warm], "s-", color="#d95f0e", label="warm sequential")
    if target is not None:
        axp.axhline(target, color="grey", ls="--", lw=1, label="target %.0f dB" % target)
    axp.set_xlabel("number of z-positions N")
    axp.set_ylabel("final PSNR (dB)")
    axp.set_title("Fidelity vs z-position count")
    axp.set_xticks(Ns)
    axp.legend(fontsize=8)
    axs.plot(Ns, [e["ssim"] for e in cold], "o-", color="#2c7fb8", label="cold joint")
    axs.plot(Ns, [e["ssim"] for e in warm], "s-", color="#d95f0e", label="warm sequential")
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
        ax.plot(wh["iter"], wh["metric"], "s-", color="#d95f0e", label="warm sequential")
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
    axi.plot(Ns, _clean([e["iters_to_threshold"] for e in cold]), "o-",
             color="#2c7fb8", label="cold joint")
    axi.plot(Ns, _clean([e["iters_to_threshold"] for e in warm]), "s-",
             color="#d95f0e", label="warm sequential (per stage)")
    axi.set_xlabel("number of z-positions N")
    axi.set_ylabel("iterations to threshold")
    ttl = "Iterations to reach target"
    if target is not None:
        ttl += " (%.0f dB)" % target
    axi.set_title(ttl)
    axi.set_xticks(Ns)
    axi.legend(fontsize=8)
    axt.plot(Ns, _clean([e["time_to_threshold"] for e in cold]), "o-",
             color="#2c7fb8", label="cold joint")
    axt.plot(Ns, _clean([e["time_to_threshold"] for e in warm]), "s-",
             color="#d95f0e", label="warm sequential (per stage)")
    axt.set_xlabel("number of z-positions N")
    axt.set_ylabel("wall-clock to threshold (s)")
    axt.set_title("Time to reach target")
    axt.set_xticks(Ns)
    axt.legend(fontsize=8)
    cum_i = m["warm_sequential"].get("cumulative_iters")
    cum_t = m["warm_sequential"].get("cumulative_time")
    if cum_i is not None and cum_t is not None:
        fig.suptitle("warm-sequential total compute over the 1..%d chain: "
                     "%d iters, %.1f s" % (n_max, cum_i, cum_t), fontsize=10)
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
    panels = [("iter", "iters_to_threshold", "iteration"),
              ("time", "time_to_threshold", "wall-clock (s)")]

    for ax, (xkey, thr_key, xlabel) in zip(axes, panels):
        cx, cy, c_starts, c_cross, c_end = _sequential_chain(cold, xkey, thr_key, target_corr)
        wx, wy, w_starts, w_cross, w_end = _sequential_chain(warm, xkey, thr_key, target_corr)
        ax.plot(cx, cy, "--", color=cold_col, lw=1.4, label="cold joint (restart per N)")
        ax.plot(wx, wy, "-", color=warm_col, lw=1.9, label="warm sequential (chained)")
        if c_cross:
            ax.plot([p[0] for p in c_cross], [p[1] for p in c_cross], "s", color=cold_col, ms=6)
        if w_cross:
            ax.plot([p[0] for p in w_cross], [p[1] for p in w_cross], "o", color=warm_col, ms=6)
        if target_corr is not None:
            ax.axhline(target_corr, color="k", ls="-.", lw=0.8,
                       label="target corr %.2f" % target_corr)
        ymin, ymax = ax.get_ylim()
        for i, s in enumerate(w_starts):  # warm z-position insertions, along the top
            ax.axvline(s, color=warm_col, ls=":", lw=0.7, alpha=0.5)
            ax.text(s, ymax, " N=%d" % (i + 1), color=warm_col, fontsize=7, va="top", ha="left")
        for i, s in enumerate(c_starts):  # cold independent restarts, along the bottom
            ax.text(s, ymin, " N=%d" % (i + 1), color=cold_col, fontsize=7, va="bottom", ha="left")
        ax.axvline(w_end, color=warm_col, lw=1.2)   # compute each arm needed to finish
        ax.axvline(c_end, color=cold_col, lw=1.2)
        ax.set_title("%s to the 4-position result: warm %.0f vs cold %.0f (save %.0f)"
                     % (xlabel, w_end, c_end, c_end - w_end), fontsize=9)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("corr-with-GT (object crop)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Experiment 2: warm-start vs cold-joint, corr-with-GT vs compute", fontsize=12)
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
            steps.append({
                "label": label, "chunk": chunk, "similarity": sim,
                "cold_iters": ci, "cold_min": ct / 60.0,
                "warm_iters": wi, "warm_min": wt / 60.0, "speedup": speed,
            })
        valid_mis = [x for x in mis if x is not None]
        out["orderings"][name] = {
            "ordering": chunks,
            "steps": steps,
            "total": {
                "cold_iters": ci_tot, "cold_min": ct_tot / 60.0,
                "warm_iters": wi_tot, "warm_min": wt_tot / 60.0,
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
    panels = [("iter", "iters_to_threshold", "iteration"),
              ("time", "time_to_threshold", "wall-clock (s)")]

    for ax, (xkey, thr_key, xlabel) in zip(axes, panels):
        cx, cy, c_starts, c_cross, c_end = _sequential_chain(cold_arm, xkey, thr_key, C0)
        wx, wy, w_starts, w_cross, w_end = _sequential_chain(warm_arm, xkey, thr_key, C0)
        ax.plot(cx, cy, "--", color=cold_col, lw=1.4, label="from scratch (restart per chunk)")
        ax.plot(wx, wy, "-", color=warm_col, lw=1.9, label="warm start (chained)")
        if c_cross:
            ax.plot([p[0] for p in c_cross], [p[1] for p in c_cross], "s", color=cold_col, ms=6)
        if w_cross:
            ax.plot([p[0] for p in w_cross], [p[1] for p in w_cross], "o", color=warm_col, ms=6)
        ax.axhline(C0, color="k", ls="-.", lw=0.8, label="target quality %.3f" % C0)
        ymin, ymax = ax.get_ylim()
        for i, s in enumerate(w_starts):  # warm chunk insertions, along the top
            ax.axvline(s, color=warm_col, ls=":", lw=0.7, alpha=0.5)
            ax.text(s, ymax, " c%d" % chunks[i], color=warm_col, fontsize=7, va="top", ha="left")
        for i, s in enumerate(c_starts):  # from-scratch restarts, along the bottom
            ax.text(s, ymin, " c%d" % chunks[i], color=cold_col, fontsize=7, va="bottom",
                    ha="left")
        ax.axvline(w_end, color=warm_col, lw=1.2)
        ax.axvline(c_end, color=cold_col, lw=1.2)
        ax.set_title("%s for %d chunks: warm %.0f vs scratch %.0f (save %.0f)"
                     % (xlabel, len(chunks), w_end, c_end, c_end - w_end), fontsize=9)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("corr-with-GT (object crop)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Experiment 3: warm-start across a wandering bundle (ordering %s), "
                 "corr-with-GT vs compute" % ordering, fontsize=12)
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

    col = ["Step", "New-vol\nsimilarity", "From scratch\niters (min)",
           "Warm start\niters (min)", "Speed-up"]
    rows = [[st["label"], fmt_sim(st["similarity"]),
             fmt_im(st["cold_iters"], st["cold_min"]),
             fmt_im(st["warm_iters"], st["warm_min"]), fmt_x(st["speedup"])]
            for st in A["steps"]]
    tot = A["total"]
    rows.append(["TOTAL (%d chunks)" % len(A["ordering"]), "-",
                 fmt_im(tot["cold_iters"], tot["cold_min"]),
                 fmt_im(tot["warm_iters"], tot["warm_min"]), fmt_x(tot["speedup"])])

    pretty = {"A": "A: adjacent (0,1,2,3)", "B": "B: every-second (0,2,4,6)"}
    col2 = ["Chain", "Mean new-vol\nsimilarity", "Warm total\niters (min)", "Speed-up\nvs scratch"]
    rows2 = []
    for name in ("A", "B"):
        o = nums["orderings"][name]
        t = o["total"]
        rows2.append([pretty.get(name, name),
                      "%.3f" % o["mean_similarity"] if o["mean_similarity"] is not None else "-",
                      fmt_im(t["warm_iters"], t["warm_min"]), fmt_x(t["speedup"])])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 5),
        gridspec_kw={"height_ratios": [len(rows) + 1, len(rows2) + 1]})
    for ax in (ax1, ax2):
        ax.axis("off")

    t1 = ax1.table(cellText=rows, colLabels=col, loc="center", cellLoc="center")
    t1.auto_set_font_size(False)
    t1.set_fontsize(9)
    t1.scale(1, 1.6)
    ax1.set_title("Warm-start vs from-scratch: reconstructing a wandering bundle chunk by chunk",
                  fontsize=11, pad=12)

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
    small = vol[::step[0], ::step[1], ::step[2]]
    zz, yy, xx = np.mgrid[0:small.shape[0], 0:small.shape[1], 0:small.shape[2]]
    level = float(np.percentile(small, iso_percentile))
    lo = float(small.min())
    hi = float(small.max())
    if hi <= lo:
        hi = lo + 1.0
    fig = go.Figure(data=go.Isosurface(
        x=xx.ravel(), y=yy.ravel(), z=zz.ravel(),
        value=small.ravel(),
        isomin=level, isomax=hi,
        surface_count=2, opacity=0.4, colorscale="Viridis",
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))
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
        description="Build all Experiment 1/2 figures from a returned results directory.")
    p.add_argument("--results", required=True,
                   help="Directory holding ct_metrics.json and/or pleno_metrics.json plus "
                        "the recon .npy volumes.")
    p.add_argument("--out", default=None,
                   help="Figure output directory (default: <results>/figures).")
    p.add_argument("--plotly-volume", default=None,
                   help="Optional recon .npy to render as an interactive 3D HTML.")
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
        made.append(exp2_corr_convergence(
            str(pleno_json), str(out_dir / "exp2_corr_convergence.png")))

    chunkseq_json = results / "chunkseq_metrics.json"
    if chunkseq_json.exists():
        for od in ("A", "B"):
            made.append(exp3_sawtooth(
                str(chunkseq_json), str(out_dir / ("exp3_sawtooth_%s.png" % od)), ordering=od))
        made.append(exp3_table(str(chunkseq_json), str(out_dir / "exp3_numbers_table.png")))

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
