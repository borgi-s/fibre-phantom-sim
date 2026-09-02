"""CLI: chunk-sequence plenoptic study. Warm-start compute saving vs fibre misorientation.

Packs one long wandering bundle (fibres wander along dim 0), cuts it into n_chunks equal
chunks, and reconstructs a selected sequence of chunks from a single plenoptic capture
each, warm-starting each chunk from the previous chunk's result. Two orderings:
  A (adjacent, small step):     chunks 0,1,2,3
  B (every-second, large step): chunks 0,2,4,6
Anchor chunk 0 runs the full n_iter and defines C0 = its corr-with-GT; every other solve
(cold from scratch, and warm chained) early-stops es_extra iters after corr reaches C0.
The metric iters_to_threshold is the first iteration corr crosses C0 - es_tol.

Module-level code is stdlib + numpy + import-safe helpers only, so this module imports on a
box without torch/astra. `main` defers every torch/astra/CUDA import to its body.

Memory-frugal: one chunk's padded volume + one projector resident on the GPU at a time;
projection blocks are CPU-offloaded (4-position mode); the warm chain carries the previous
chunk's padded recon on CPU. Every projector is freed and the cache emptied after each solve.

See docs/superpowers/specs/2026-09-02-plenoptic-warmstart-misorientation-design.md.

Usage (cluster, CUDA interpreter):
    python -m phantom_sim.run_plenoptic_chunkseq --smoke --out /tmp/chunkseq_smoke
    python -m phantom_sim.run_plenoptic_chunkseq --out phantom_sim_results/exp3_chunkseq
"""
import argparse
import json
from pathlib import Path

import numpy as np

from phantom_sim.reconstruct import first_crossing

ORDERINGS = {"A": [0, 1, 2, 3], "B": [0, 2, 4, 6]}


def cold_chunks(orderings, anchor=0):
    """Sorted unique non-anchor chunks needed by the cold arm across all orderings."""
    s = set()
    for seq in orderings.values():
        s.update(seq)
    s.discard(anchor)
    return sorted(s)


def chunkseq_entry(chunk, corr_val, psnr_val, ssim_val, resolution, history, bar):
    """One (arm, chunk) metrics entry. iters/time_to_threshold are the first crossing of
    `bar` (= C0 - es_tol) in the corr trace; null if never crossed. Pure, no torch."""
    iters_thr, time_thr = first_crossing(history, bar, key="corr")
    return {
        "chunk": int(chunk),
        "corr_with_gt": float(corr_val),
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "resolution": resolution,
        "iters_to_threshold": iters_thr,
        "time_to_threshold": time_thr,
        "history": history,
    }


def assemble_chunkseq_metrics(anchor_entry, cold_entries, runs, C0, meta):
    """Pure schema assembly for chunkseq_metrics.json. `runs` maps ordering name ->
    {warm:[entry...], pair_misorientation:[float...], cumulative_iters:int,
    cumulative_time:float}. No torch/astra."""
    return {
        "meta": {**meta, "C0": float(C0)},
        "anchor": anchor_entry,
        "cold": cold_entries,
        "runs": runs,
    }


def build_parser():
    p = argparse.ArgumentParser(
        description="Chunk-sequence plenoptic warm-start vs misorientation study.")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="Tiny bundle/grid/iters for a fast cluster smoke test.")
    p.add_argument("--misalignment", default="very low")
    p.add_argument("--z-ctrl", type=int, default=64)
    p.add_argument("--n-chunks", type=int, default=8)
    p.add_argument("--chunk-depth", type=int, default=512)
    p.add_argument("--positions", type=int, choices=[1, 4], default=1,
                   help="Plenoptic z-positions imaged per chunk (1 default; 4 = multiview).")
    p.add_argument("--domain-radius-um", type=float, default=300.0)
    p.add_argument("--fvf", type=float, default=0.4)
    p.add_argument("--r-mean-um", type=float, default=6.0)
    p.add_argument("--ny", type=int, default=300)
    p.add_argument("--nx", type=int, default=300)
    p.add_argument("--voxel-um", type=float, default=2.0)
    p.add_argument("--mu-fibre", type=float, default=1.0)
    p.add_argument("--mu-matrix", type=float, default=0.0)
    p.add_argument("--iters", type=int, default=200, help="Packing optimisation iters.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sods", type=float, nargs="+", default=[129.24, 129.62, 130.01, 130.39])
    p.add_argument("--sdd-mm", type=float, default=1171.57)
    p.add_argument("--detector-pixel-um", type=float, default=55.0)
    p.add_argument("--source-range-um", type=float, default=22000.0)
    p.add_argument("--det-range-um", type=float, default=177400.0)
    p.add_argument("--n-grid", type=int, default=21)
    p.add_argument("--super-sampling", type=int, default=2)
    p.add_argument("--pad-transverse-vox", type=int, default=175)
    p.add_argument("--pad-depth-vox", type=int, default=100)
    p.add_argument("--n-iter", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--tv-weight", type=float, default=0.5)
    p.add_argument("--es-tol", type=float, default=1e-3)
    p.add_argument("--es-extra", type=int, default=15)
    p.add_argument("--log-every", type=int, default=10)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke:
        z_ctrl, n_chunks, chunk_depth = 8, 8, 8
        ny = nx = 48
        n_grid, n_iter, log_every = 3, 6, 1
        pad_t, pad_d = 4, 2
        pack_iters = 10
        domain_radius = 40.0  # shrink the domain so fibres fit the tiny smoke frame
        r_mean = 6.0
    else:
        z_ctrl, n_chunks, chunk_depth = args.z_ctrl, args.n_chunks, args.chunk_depth
        ny, nx = args.ny, args.nx
        n_grid, n_iter, log_every = args.n_grid, args.n_iter, args.log_every
        pad_t, pad_d = args.pad_transverse_vox, args.pad_depth_vox
        pack_iters = args.iters
        domain_radius = args.domain_radius_um
        r_mean = args.r_mean_um

    voxel_um = args.voxel_um
    es_tol, es_extra = args.es_tol, args.es_extra
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from phantom_sim.phantom import pack_bundle, chunk_gt, pair_misorientation
    import torch
    from phantom_sim.geometry import (
        build_plenoptic_geometry, make_plenoptic_projector, plenoptic_grid, XrayOperator,
        delete_projector,
    )
    from phantom_sim.reconstruct import reconstruct
    from phantom_sim.metrics import psnr, ssim, register, resolution_along_across, corr_with_gt

    # 1. Pack the wandering bundle once (control slices).
    print(f"Packing wandering bundle: misalignment='{args.misalignment}' z_ctrl={z_ctrl} ...")
    cfg_ctrl, radii = pack_bundle(
        domain_radius_um=domain_radius, fvf=args.fvf, r_mean_um=r_mean, r_sigma_um=0.0,
        n_slices=z_ctrl, misalignment=args.misalignment, iters=pack_iters, seed=args.seed)
    print(f"  {radii.shape[0]} fibres")

    tv_weights = (args.tv_weight, 0.0, 0.0)
    positions = args.positions
    sods = args.sods[:positions]

    # 2. Geometry (shared by all chunks; only the object changes between chunks).
    detector_z = (args.sdd_mm * 1000.0) / voxel_um
    pixel_size = args.detector_pixel_um / voxel_um
    src, det = plenoptic_grid(args.source_range_um / voxel_um,
                              args.det_range_um / voxel_um, n=n_grid)
    num_imgs = n_grid * n_grid
    padded_shape = (chunk_depth + 2 * pad_d, ny + 2 * pad_t, nx + 2 * pad_t)
    obj_sl = (slice(pad_d, pad_d + chunk_depth),
              slice(pad_t, pad_t + ny), slice(pad_t, pad_t + nx))
    sinogram_size_1 = [padded_shape[1], padded_shape[2], num_imgs]

    def voxelise_padded_chunk(c):
        obj = chunk_gt(cfg_ctrl, radii, c, n_chunks, chunk_depth, (ny, nx),
                       voxel_um, args.mu_fibre, args.mu_matrix, supersample=4)
        vol = np.pad(obj, ((pad_d, pad_d), (pad_t, pad_t), (pad_t, pad_t)))
        return obj, vol

    def forward_blocks(vol):
        """Clean forward projection of a padded chunk at each of `positions` SODs.
        Returns (list of CPU projection tensors, list of geometry vector tables)."""
        volt = torch.tensor(vol, device="cuda")
        blocks, geoms = [], []
        for sod in sods:
            vol_z0 = (sod * 1000.0) / voxel_um
            geom = build_plenoptic_geometry(src, det, vol_z0, detector_z, pixel_size, pixel_size)
            pid, _ = make_plenoptic_projector(vol.shape, geom, sinogram_size=sinogram_size_1,
                                              super_sampling=args.super_sampling)
            A = XrayOperator(pid)
            proj = A(volt).detach().cpu()
            blocks.append(proj)
            geoms.append(geom)
            delete_projector(pid)
            torch.cuda.empty_cache()
        del volt
        torch.cuda.empty_cache()
        return blocks, geoms

    def build_operator(geoms):
        geom_n = np.vstack(geoms)
        pid, _ = make_plenoptic_projector(
            padded_shape, geom_n,
            sinogram_size=[padded_shape[1], padded_shape[2], positions * num_imgs],
            super_sampling=args.super_sampling)
        return XrayOperator(pid)

    def solve(c, x_init, es_bar, full_run):
        """Voxelise chunk c, project, reconstruct. es_bar None + full_run True = anchor
        (runs the full n_iter). x_init is the previous chunk's PADDED recon (or None).
        Returns (recon_padded_np, recon_obj_np, history, corr, psnr, ssim, res); the padded
        recon is what the warm chain carries forward (matches the proven zpos driver, so the
        void margin the solver reconstructed is preserved rather than zeroed)."""
        obj, vol = voxelise_padded_chunk(c)
        blocks, geoms = forward_blocks(vol)
        proj_n = torch.cat(blocks, dim=1).to("cuda")
        A_n = build_operator(geoms)

        def corr_metric(recon_padded_np):
            return corr_with_gt(np.ascontiguousarray(recon_padded_np[obj_sl]), obj)

        recon, history = reconstruct(
            A_n, proj_n, mask=None, n_iter=n_iter, lr=args.lr, tv_weights=tv_weights,
            x_init=x_init, log_every=log_every, gt=vol, target_psnr=None, lower_clamp=0.0,
            plateau_scheduler=True, metric_fn=corr_metric,
            early_stop=(not full_run), es_tol=es_tol, es_extra_iters=es_extra,
            es_floor=None, es_bar=es_bar)
        recon_obj = np.ascontiguousarray(recon[obj_sl])
        c_val = corr_with_gt(recon_obj, obj)
        p_val = psnr(recon_obj, obj)
        s_val = ssim(recon_obj, obj)
        res = resolution_along_across(register(recon_obj, obj), fibre_axis=0)
        delete_projector(A_n.projector_id)
        del proj_n, A_n
        torch.cuda.empty_cache()
        return recon, recon_obj, history, c_val, p_val, s_val, res

    def save(arm, c, recon_obj, entry):
        np.save(out_dir / f"chunkseq_{arm}_c{c}.npy", recon_obj.astype(np.float32))
        with open(out_dir / f"chunkseq_{arm}_c{c}.meta.json", "w", encoding="utf-8") as f:
            json.dump({"arm": arm, "chunk": int(c), "metrics": entry}, f, indent=2)

    def stage_iters(h):
        return (h["iter"][-1] + 1) if h.get("iter") else 0

    def stage_time(h):
        return h["time"][-1] if h.get("time") else 0.0

    # 3. Anchor: chunk 0 from scratch, full n_iter, defines C0. Keep its PADDED recon
    #    (recon0_pad) for the warm chains; score/save the cropped object (recon0_obj).
    recon0_pad, recon0_obj, h0, C0, p0, s0, r0 = solve(0, x_init=None, es_bar=None, full_run=True)
    bar = C0 - es_tol
    anchor_entry = chunkseq_entry(0, C0, p0, s0, r0, h0, bar)
    save("anchor", 0, recon0_obj, anchor_entry)
    print(f"  anchor chunk 0: C0={C0:.4f} (bar={bar:.4f}) iters={stage_iters(h0)}")

    # 4. Cold arm: each union chunk from scratch, early-stop at C0 (padded recon unused).
    cold_entries = []
    for c in cold_chunks(ORDERINGS, anchor=0):
        _, rec_obj, h, cc, pp, ss, rr = solve(c, x_init=None, es_bar=bar, full_run=False)
        e = chunkseq_entry(c, cc, pp, ss, rr, h, bar)
        cold_entries.append(e)
        save("cold", c, rec_obj, e)
        print(f"  cold chunk {c}: corr={cc:.4f} iters_to_C0={e['iters_to_threshold']}")

    # 5. Warm chains: chunk k warm-started from chunk k-1's PADDED recon.
    runs = {}
    for name, seq in ORDERINGS.items():
        warm_entries = [anchor_entry]  # chunk 0 shared
        pair_mis = []
        cum_i, cum_t = stage_iters(h0), stage_time(h0)
        prev_padded = recon0_pad
        for k in range(1, len(seq)):
            c = seq[k]
            rec_pad, rec_obj, h, cc, pp, ss, rr = solve(
                c, x_init=prev_padded, es_bar=bar, full_run=False)
            e = chunkseq_entry(c, cc, pp, ss, rr, h, bar)
            warm_entries.append(e)
            save(f"warm_{name}", c, rec_obj, e)
            pair_mis.append(pair_misorientation(cfg_ctrl, radii, seq[k - 1], c, n_chunks,
                                                 chunk_depth, (ny, nx), voxel_um,
                                                 args.mu_fibre, args.mu_matrix))
            cum_i += stage_iters(h)
            cum_t += stage_time(h)
            prev_padded = rec_pad
            print(f"  warm {name} chunk {c}: corr={cc:.4f} iters_to_C0={e['iters_to_threshold']} "
                  f"mis={pair_mis[-1]:.3f}")
        runs[name] = {"warm": warm_entries, "pair_misorientation": pair_mis,
                      "cumulative_iters": int(cum_i), "cumulative_time": float(cum_t)}

    meta = {
        "smoke": bool(args.smoke), "misalignment": args.misalignment, "z_ctrl": z_ctrl,
        "n_chunks": n_chunks, "chunk_depth": chunk_depth, "positions": positions,
        "orderings": ORDERINGS, "n_fibres": int(radii.shape[0]), "voxel_um": voxel_um,
        "mu_fibre": args.mu_fibre, "mu_matrix": args.mu_matrix, "fvf": args.fvf,
        "r_mean_um": r_mean, "domain_radius_um": domain_radius, "seed": args.seed,
        "sods_mm": sods, "sdd_mm": args.sdd_mm, "n_grid": n_grid, "n_iter": n_iter,
        "lr": args.lr, "tv_weight": args.tv_weight, "es_tol": es_tol, "es_extra": es_extra,
        "pad_transverse_vox": pad_t, "pad_depth_vox": pad_d,
    }
    metrics = assemble_chunkseq_metrics(anchor_entry, cold_entries, runs, C0, meta)
    with open(out_dir / "chunkseq_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {out_dir / 'chunkseq_metrics.json'}  C0={C0:.4f} bar={bar:.4f}")
    for name in ORDERINGS:
        warm = runs[name]["warm"]
        iters = [e["iters_to_threshold"] for e in warm[1:]]
        print(f"  run {name}: warm iters-to-C0 per step = {iters}  "
              f"misorientation = {[round(x, 3) for x in runs[name]['pair_misorientation']]}")


if __name__ == "__main__":
    main()
