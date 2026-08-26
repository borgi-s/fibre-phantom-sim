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
