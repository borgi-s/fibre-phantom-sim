"""CLI: cross-rung summary of the exp9 diameter sweep: measured vs true diameter per method.

Reads ``<parent>/D*/diam/diam_summary.json`` (+ ``metrics.json``) and writes to ``<parent>/figures``:
    diam_recovery.png   measured vs true mean diameter, recon (solid) and ideal GT (dashed), per method
    diam_bias.png       recon - true and recon - GT(method) bias vs true diameter, plus recon fidelity
    diam_sweep_table.csv / diam_sweep_table.json
Torch-free; runs on the laptop or a CPU node.

Usage:
    python -m phantom_sim.diam_sweep_summary --parent phantom_sim_results/exp9_diamsweep
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# (label, method key, path into that method's summary dict)
ESTIMATORS = [
    ("M1 chord fit 2R", "M1", ("chord_fit", "median")),
    ("M1 max width", "M1", ("max_width", "median")),
    ("M2 disk fit (all)", "M2", ("fit_shared_all", "median")),
    ("M2 half level (all)", "M2", ("half_level_all", "median")),
    ("M3 minor axis", "M3", ("by_frac", "0.5", "minor", "median")),
    ("M3 equiv. area", "M3", ("by_frac", "0.5", "equiv", "median")),
    ("M4 grey peak", "M4", ("grey_peak_interp_um",)),
]


def _get(d, path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return None if d is None else float(d)


def collect(parent):
    rows = []
    for summ in sorted(Path(parent).glob("D*/diam/diam_summary.json")):
        with open(summ, encoding="utf-8") as f:
            s = json.load(f)
        mpath = summ.parent.parent / "metrics.json"
        res = {}
        if mpath.exists():
            with open(mpath, encoding="utf-8") as f:
                res = json.load(f).get("result", {})
        row = {
            "rung": summ.parent.parent.name,
            "d_nominal_um": s["meta"]["diam_mean_um"],
            "d_true_um": s["truth_mean_diam_um"],
            "n_fibres": s["meta"]["n_fibres"],
            "fvf_eff": s["meta"]["fvf_eff"],
            "angle_p90_median_deg": s["meta"]["angle_p90_median_deg"],
            "corr_rope": res.get("corr_rope"),
            "corr_all": res.get("corr_all"),
        }
        for label, m, path in ESTIMATORS:
            for src in ("recon", "gt"):
                r = s["results"].get(f"{src}/{m}")
                val = _get(r, path) if r else None
                if val is None and m == "M4" and r:
                    val = _get(r, ("grey_peak_um",))
                row[f"{src}|{label}"] = val
        m2 = s["results"].get("recon/M2") or {}
        row["recon_M2_sigma_um"] = m2.get("sigma_shared_um")
        rows.append(row)
    rows.sort(key=lambda r: r["d_nominal_um"])
    return rows


def figures(rows, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    dt = np.array([r["d_true_um"] for r in rows])
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    lo, hi = 0.8 * dt.min() - 2, 1.1 * dt.max() + 2
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=1, label="identity")
    for k, (label, _, _) in enumerate(ESTIMATORS):
        rec = np.array(
            [
                np.nan if r[f"recon|{label}"] is None else r[f"recon|{label}"]
                for r in rows
            ]
        )
        gt = np.array(
            [np.nan if r[f"gt|{label}"] is None else r[f"gt|{label}"] for r in rows]
        )
        ax.plot(dt, rec, "-o", color=cmap(k), ms=5, label=f"{label} (recon)")
        ax.plot(dt, gt, "--", color=cmap(k), lw=1, alpha=0.7)
    ax.set_xlabel("true mean fibre diameter (um)")
    ax.set_ylabel("measured median diameter (um)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title(
        "exp9 diameter sweep: recovery per method\nsolid = plenoptic recon, dashed = same method on ideal GT"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "diam_recovery.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for k, (label, _, _) in enumerate(ESTIMATORS):
        rec = np.array(
            [
                np.nan if r[f"recon|{label}"] is None else r[f"recon|{label}"]
                for r in rows
            ]
        )
        gt = np.array(
            [np.nan if r[f"gt|{label}"] is None else r[f"gt|{label}"] for r in rows]
        )
        axes[0].plot(dt, rec - dt, "-o", color=cmap(k), ms=4, label=label)
        axes[1].plot(dt, rec - gt, "-o", color=cmap(k), ms=4, label=label)
    axes[0].set_title("recon minus TRUE diameter (total bias)")
    axes[1].set_title(
        "recon minus same method on GT\n(bias added by the plenoptic recon)"
    )
    for a in axes[:2]:
        a.axhline(0, color="0.5", lw=1)
        a.set_xlabel("true mean diameter (um)")
        a.set_ylabel("bias (um)")
        a.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    corr = [r["corr_rope"] for r in rows]
    axes[2].plot(
        dt,
        [np.nan if c is None else c for c in corr],
        "-o",
        color="k",
        label="corr(recon, GT) in rope",
    )
    axes[2].set_xlabel("true mean diameter (um)")
    axes[2].set_ylabel("correlation")
    ax2 = axes[2].twinx()
    ax2.plot(
        dt,
        [r["angle_p90_median_deg"] for r in rows],
        "s--",
        color="tab:orange",
        label="misalignment p90 (deg)",
    )
    ax2.set_ylabel("median per-fibre p90 tilt (deg)", color="tab:orange")
    axes[2].set_title("recon fidelity and realised misalignment")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "diam_bias.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--parent", required=True)
    args = p.parse_args(argv)
    parent = Path(args.parent)
    rows = collect(parent)
    if not rows:
        raise SystemExit(f"no D*/diam/diam_summary.json under {parent}")
    fig_dir = parent / "figures"
    figures(rows, fig_dir)
    with open(fig_dir / "diam_sweep_table.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(fig_dir / "diam_sweep_table.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {fig_dir} ({len(rows)} rungs)")


if __name__ == "__main__":
    main()
