"""CLI: run the M1-M4 fibre-diameter methods on one diameter-sweep rung (exp9), recon AND ideal GT.

Torch-free (numpy/scipy/scikit-image/matplotlib); run it as its own process after the recon so no
torch import precedes scipy (DTU_HPC_Guide B8.1). Writes, under ``<run>/diam/``:
    recon/M1..M4/*   figures + per-fibre tables for the reconstruction
    gt/M1..M4/*      the same on the ideal (unblurred, anti-aliased) ground truth
    diam_summary.json  every method's summary dict + the rung meta

Usage:
    python -m phantom_sim.diam_analysis --run phantom_sim_results/exp9_diamsweep/D20 [--methods M1,M2] [--fast]
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from phantom_sim.diam_methods.common import load_run, save_json

METHODS = {
    "M1": "phantom_sim.diam_methods.m1_longitudinal",
    "M2": "phantom_sim.diam_methods.m2_radial_fit",
    "M3": "phantom_sim.diam_methods.m3_local_ellipse",
    "M4": "phantom_sim.diam_methods.m4_granulometry",
}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run",
        required=True,
        help="Rung directory written by run_plenoptic_diamsweep.",
    )
    p.add_argument("--methods", default="M1,M2,M3,M4")
    p.add_argument(
        "--sources",
        default="recon,gt",
        help="Which volumes to analyse: recon, gt or both.",
    )
    p.add_argument("--fast", action="store_true", help="Tiny test run of every method.")
    p.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Process pool size (default: LSF slots or cpu_count).",
    )
    return p


def main(argv=None):
    import importlib

    args = build_parser().parse_args(argv)
    run_dir = Path(args.run)
    data = load_run(run_dir)
    meta, truth = data["meta"], data["truth"]
    voxel_um = float(meta["voxel_um"])
    n_workers = (
        args.n_workers or int(os.environ.get("LSB_DJOB_NUMPROC", 0)) or os.cpu_count()
    )
    tag = f"D{meta['diam_mean_um']:g}"
    out_root = run_dir / "diam"
    summary = {
        "meta": meta,
        "truth_mean_diam_um": truth.mean_diam_um,
        "results": {},
        "errors": {},
    }
    prev = out_root / "diam_summary.json"
    if prev.exists():  # merge: re-running a subset keeps other methods
        with open(prev, encoding="utf-8") as f:
            old = json.load(f)
        summary["results"].update(old.get("results", {}))
        summary["errors"].update(old.get("errors", {}))
    for src in [s.strip() for s in args.sources.split(",") if s.strip()]:
        vol = data[src]
        for m in [s.strip() for s in args.methods.split(",") if s.strip()]:
            key = f"{src}/{m}"
            t0 = time.time()
            print(f"[{tag}] {key} ...", flush=True)
            try:
                mod = importlib.import_module(METHODS[m])
                kw = {}
                if (
                    m == "M1"
                ):  # sim matrix/fibre levels are known exactly (the real-data port hard-coded them too)
                    kw["levels"] = (float(meta["mu_matrix"]), float(meta["mu_fibre"]))
                res = mod.run(
                    vol,
                    voxel_um,
                    out_root / src / m,
                    label=f"{tag} {src}",
                    truth=truth,
                    fast=args.fast,
                    n_workers=n_workers,
                    **kw,
                )
                res["runtime_s"] = time.time() - t0
                summary["results"][key] = res
                summary["errors"].pop(key, None)
                print(f"[{tag}] {key} done in {res['runtime_s']:.0f}s", flush=True)
            except (
                Exception
            ) as e:  # keep going: one failing method must not lose the rest
                summary["errors"][key] = (
                    f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                )
                print(f"[{tag}] {key} FAILED: {e}", flush=True)
            save_json(summary, out_root / "diam_summary.json")
    print(
        f"Saved {out_root / 'diam_summary.json'} ({len(summary['errors'])} errors)",
        flush=True,
    )


if __name__ == "__main__":
    main()
