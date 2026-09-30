# -*- coding: utf-8 -*-
"""Analyze the exp8 real-data dose-reduction sweep.

Reads dose_metrics.json produced by run_plenoptic_dose.py (a list of
{f, seed, corr_all, corr_fibre, final_loss}), averages the fibre correlation over
seeds at each dose fraction, plots correlation vs dose, marks where it crosses a set of
acceptability thresholds, and prints the corresponding exposure headroom:
    Z% = f * 100 , exposure T = baseline_s * f , reduction N = 1 / f.

Laptop-side (no torch / astra needed): numpy + matplotlib only. Pass an ABSOLUTE
--results path; a relative ..\\ path makes savefig raise Errno 22 on Windows.

Usage:
    python analyze_dose.py --results C:\\...\\DataFiles\\phantom_sim\\exp8_dose
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter, FuncFormatter


THRESHOLDS = [0.99, 0.95, 0.90]  # corr-with-full-dose acceptability levels to report


def load_metrics(results_dir):
    path = os.path.join(results_dir, "dose_metrics.json")
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    return rows


def group_by_dose(rows):
    """Return sorted doses and per-dose (mean, std, n) of corr_fibre and corr_all."""
    by_f = defaultdict(lambda: {"fibre": [], "all": []})
    for r in rows:
        by_f[float(r["f"])]["fibre"].append(float(r["corr_fibre"]))
        by_f[float(r["f"])]["all"].append(float(r["corr_all"]))
    doses = sorted(by_f)  # ascending
    stats = {}
    for f in doses:
        fib = np.array(by_f[f]["fibre"])
        all = np.array(by_f[f]["all"])
        stats[f] = dict(
            fibre_mean=float(fib.mean()),
            fibre_std=float(fib.std()),
            all_mean=float(all.mean()),
            n=len(fib),
        )
    return doses, stats


def lowest_acceptable(doses, stats, thr):
    """Lowest f whose mean fibre corr is still >= thr, contiguously from full dose.

    Walk doses from high to low; stop at the first that drops below thr. Returns the
    lowest f still meeting thr (or None if even full dose fails, which cannot happen for
    f=1 self-corr=1.0).
    """
    ok = None
    for f in sorted(doses, reverse=True):
        if stats[f]["fibre_mean"] >= thr:
            ok = f
        else:
            break
    return ok


def main():
    ap = argparse.ArgumentParser(description="Analyze the exp8 dose-reduction sweep.")
    ap.add_argument(
        "--results",
        required=True,
        help="ABSOLUTE path to the exp8_dose folder holding dose_metrics.json",
    )
    ap.add_argument(
        "--baseline-s",
        type=float,
        default=60.0,
        help="full-dose exposure in seconds (default 60)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="output PNG path (default: <results>/dose_corr_vs_dose.png)",
    )
    args = ap.parse_args()

    results_dir = os.path.abspath(args.results)
    rows = load_metrics(results_dir)
    doses, stats = group_by_dose(rows)

    print(f"\ndose sweep from {results_dir}")
    print(
        f"{'f':>6}  {'dose%':>6}  {'corr_fibre':>11}  {'+/- std':>8}  {'corr_all':>9}  {'seeds':>5}"
    )
    for f in sorted(doses, reverse=True):
        s = stats[f]
        print(
            f"{f:6g}  {100 * f:6.1f}  {s['fibre_mean']:11.4f}  {s['fibre_std']:8.4f}  "
            f"{s['all_mean']:9.4f}  {s['n']:5d}"
        )

    print(f"\nexposure headroom (baseline {args.baseline_s:g} s):")
    for thr in THRESHOLDS:
        f_min = lowest_acceptable(doses, stats, thr)
        if f_min is None or f_min >= 1.0:
            print(f"  corr_fibre >= {thr:.2f}:  not reached below full dose")
            continue
        floored = " (floor of tested range)" if f_min <= min(doses) else ""
        print(
            f"  corr_fibre >= {thr:.2f}:  down to f = {f_min:g}  ->  "
            f"Z = {100 * f_min:g}% dose,  T = {args.baseline_s * f_min:g} s,  "
            f"N = {1.0 / f_min:.0f}x reduction{floored}"
        )

    # ---- plot ------------------------------------------------------------------
    f_arr = np.array(sorted(doses))
    fib = np.array([stats[f]["fibre_mean"] for f in f_arr])
    err = np.array([stats[f]["fibre_std"] for f in f_arr])
    all = np.array([stats[f]["all_mean"] for f in f_arr])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        f_arr,
        fib,
        yerr=err,
        marker="o",
        color="#d95f0e",
        capsize=3,
        label="fibre-region corr (vs full dose)",
    )
    ax.plot(
        f_arr, all, marker="s", color="#2c7fb8", alpha=0.5, label="whole-volume corr"
    )
    for thr in THRESHOLDS:
        ax.axhline(thr, color="grey", ls=":", lw=1)
        ax.text(
            f_arr.min(),
            thr,
            f" {thr:.2f}",
            va="bottom",
            ha="left",
            fontsize=8,
            color="grey",
        )
    ax.set_xscale("log")
    # Label only the measured doses (major ticks, plain numbers), but keep the full log
    # minor grid unlabelled so the axis still visibly reads as logarithmic.
    ax.xaxis.set_major_locator(FixedLocator(f_arr))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100)
    )
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("dose fraction f  (exposure / full)")
    ax.set_ylabel("correlation with full-dose reconstruction")
    ax.set_title("Real-data plenoptic dose reduction (260703, Zpos1)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)
    fig.tight_layout()

    out = (
        os.path.abspath(args.out)
        if args.out
        else os.path.join(results_dir, "dose_corr_vs_dose.png")
    )
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nsaved {out}")

    # ---- second plot: same curve with the x-axis in exposure SECONDS -----------
    # For a real-time-feasibility argument the natural axis is exposure
    # time T = baseline_s * f (full dose = baseline_s). Curves, error bars and threshold
    # lines are identical; only the x mapping changes. Written next to the dose figure
    # with an "_exposure" suffix so both refresh together on every analysis run.
    t_arr = args.baseline_s * f_arr

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.errorbar(
        t_arr,
        fib,
        yerr=err,
        marker="o",
        color="#d95f0e",
        capsize=3,
        label="fibre-region corr (vs full dose)",
    )
    ax2.plot(
        t_arr, all, marker="s", color="#2c7fb8", alpha=0.5, label="whole-volume corr"
    )
    for thr in THRESHOLDS:
        ax2.axhline(thr, color="grey", ls=":", lw=1)
        ax2.text(
            t_arr.min(),
            thr,
            f" {thr:.2f}",
            va="bottom",
            ha="left",
            fontsize=8,
            color="grey",
        )
    ax2.set_xscale("log")
    # Label only the measured exposures (T = baseline_s * f), keep the log minor grid.
    ax2.xaxis.set_major_locator(FixedLocator(t_arr))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    ax2.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100)
    )
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel(f"exposure time (s; full dose = {args.baseline_s:g} s)")
    ax2.set_ylabel("correlation with full-dose reconstruction")
    ax2.set_title("Real-data plenoptic dose reduction (260703, Zpos1)")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, which="major", alpha=0.35)
    ax2.grid(True, which="minor", alpha=0.15)
    fig2.tight_layout()

    base, ext = os.path.splitext(out)
    out2 = base + "_exposure" + ext
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    print(f"saved {out2}")


if __name__ == "__main__":
    main()
