"""CLI: Experiment 2, plenoptic z-position count and warm-start vs cold-joint solves.

Packs ONE fibre bundle, voxelises it ONCE along axis=0 (beam axis: fibres run parallel
to dim 0, the plenoptic geometry's near-Z beam direction for every source/detector cell
in every z-position's grid). Builds FOUR z-position plenoptic geometries (four SODs,
Zpos1..4, each a full n x n source/detector grid via `plenoptic_grid`; the shared
source-to-detector rail distance `--sdd-mm` is fixed, only the sample moves along the
beam between z-positions). Forward-projects the phantom ONCE per z-position at flux
`i0` to get that position's own noisy projection block (kept alongside its geometry
vectors).

Two reconstruction strategies, both driven by the SAME shared iterative TV solver
(`reconstruct.py`), compared over N = 1..4 z-positions:
  - cold_joint: for each N, an INDEPENDENT solve from x_init=None over the concatenated
    blocks 1..N (does adding z-positions improve fidelity, on its own).
  - warm_sequential: a chained solve -- x=None; for each N, warm-start from the
    previous N's own result, concatenate blocks 1..N, solve, and carry the result
    forward. Tracks CUMULATIVE iterations/wall-time actually spent across the WHOLE
    1..4 chain (`cumulative_iters` / `cumulative_time`), since reaching a warm N=4
    result requires first running N=1, N=2, N=3 -- the fair-compute number for "does
    warm-starting reach the 4-position result for less total compute than 4
    independent cold solves."

Concatenation axes (load-bearing -- get this wrong and shapes silently mismatch):
geometry vectors are a flat (num_vectors, 12) table, so blocks 1..N stack with plain
np.vstack (axis 0). Projection DATA follows ASTRA's 3D cone_vec convention (det_rows,
num_vectors, det_cols) -- XrayOperator.proj_shape is astra.geom_size(proj_geom) in that
same order -- so the matching torch concatenation axis for the projection TENSORS is
dim=1, not dim=0.

pleno_metrics.json schema (Task 10 and the cuda smoke test both depend on this):
{
  "cold_joint": {"by_N": [ {N, psnr, ssim, resolution, iters_to_threshold,
                             time_to_threshold, history}, ... x4 ]},
  "warm_sequential": {"by_N": [ ... x4, same per-stage fields ],
                       "cumulative_iters": int, "cumulative_time": float},
  "meta": { phantom + geometry + solver params, mu values, voxel_um, i0, target_psnr,
            the 4 z-position SODs, ... }
}
`iters_to_threshold`/`time_to_threshold` are `first_crossing(history, target_psnr,
key="metric")`; both are `null` in the json when the threshold was never crossed.
`psnr`/`ssim` register internally; `resolution_along_across` is computed on a
separately-registered copy of the recon (beam-axis fibres are along array dim 0).

This-box vs cluster split: `simulate_projections` and `reconstruct` require a CUDA GPU
with ASTRA's cuda3d projector, absent on this box. Every astra/torch-cuda-touching
import is deferred to inside `main`, so `import phantom_sim.run_plenoptic_zpos`
succeeds on this box without astra or a CUDA torch build; module-level imports are
stdlib + numpy + the import-safe `phantom_sim.phantom` and
`phantom_sim.reconstruct.first_crossing` (pack_bundle's own torch import is itself
lazy, inside the function body; reconstruct.py's torch import is lazy too, inside
`reconstruct()`/`_tv_isotropic_aniso()`, so importing just `first_crossing` needs
neither astra nor torch).

Usage (cluster, using its CUDA-enabled interpreter):
    python -m phantom_sim.run_plenoptic_zpos --out DataFiles/phantom_sim/exp2
    python -m phantom_sim.run_plenoptic_zpos --smoke --out /tmp/exp2_smoke

Outputs under --out:
    pleno_cold_joint_z1..4.npy / .meta.json        cold_joint recon volumes + meta
    pleno_warm_sequential_z1..4.npy / .meta.json    warm_sequential recon volumes + meta
    pleno_metrics.json                              combined schema above
"""
import argparse
import json
from pathlib import Path

import numpy as np

from phantom_sim.phantom import make_straight_fibre_phantom
from phantom_sim.reconstruct import first_crossing


def write_meta(path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def assemble_stage_entry(n, corr_val, psnr_val, ssim_val, resolution, history, target_psnr):
    """Pure metrics-assembly for one (strategy, N) stage.

    No astra/torch here: corr_val, psnr_val, ssim_val, resolution and history are already-
    computed plain python/numpy values, so this is directly unit-testable on this box
    with fake inputs (no GPU needed). corr_with_gt is the headline fidelity metric.
    """
    iters_thr, time_thr = first_crossing(history, target_psnr, key="metric")
    return {
        "N": int(n),
        "corr_with_gt": float(corr_val),
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "resolution": resolution,
        "iters_to_threshold": iters_thr,
        "time_to_threshold": time_thr,
        "history": history,
    }


def assemble_metrics(cold_entries, warm_entries, cumulative_iters, cumulative_time, meta):
    """Pure schema assembly for the combined pleno_metrics.json dict. No astra/torch."""
    return {
        "cold_joint": {"by_N": cold_entries},
        "warm_sequential": {
            "by_N": warm_entries,
            "cumulative_iters": int(cumulative_iters),
            "cumulative_time": float(cumulative_time),
        },
        "meta": meta,
    }


def build_parser():
    p = argparse.ArgumentParser(
        description="Experiment 2: plenoptic z-position count and warm-start vs "
                    "cold-joint solves.")
    p.add_argument("--out", required=True, help="Output directory (created if missing).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny volume, small plenoptic grid, few packing/recon iters, for "
                        "a fast cluster smoke test. Overrides the study-scale defaults "
                        "below (geometry constants are NOT shrunk, only grid/volume/iters).")

    # packing params (see phantom.pack_bundle); study-scale defaults match
    # make_phantom.py and run_ct_orientation.py (Experiment 1), so both experiments
    # phantom the same way.
    p.add_argument("--domain-radius-um", type=float, default=100.0,
                   help="Packing domain (bundle cross-section) radius, um.")
    p.add_argument("--fvf", type=float, default=0.3,
                   help="Fibre volume fraction as a 0-1 fraction.")
    p.add_argument("--r-mean-um", type=float, default=6.0,
                   help="Mean fibre radius, um (glass, ~12um diameter = 6um radius; "
                        "typical E-glass filament 5-13um). At 2um voxel this is a 6-voxel "
                        "disc, which anti-aliased renders as a clean circle.")
    p.add_argument("--depth", type=int, default=512,
                   help="Fibre-axis (beam-axis) length in voxels; deep volume so line "
                        "integrals accumulate strong along-fibre contrast.")
    p.add_argument("--iters", type=int, default=200, help="2D packing optimisation iterations.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base RNG seed for packing and per-z-position Poisson noise "
                        "(z-position i draws with seed + i, for independent noise "
                        "realisations across the 4 positions).")

    # voxelisation params; the packing is voxelised ONCE, along axis=0 (beam axis:
    # fibres run parallel to the plenoptic geometry's near-Z beam direction, dim 0, for
    # every source/detector cell in every z-position's grid). voxel-um is the global
    # recon_pixel_size constraint (2.0 um throughout the study).
    p.add_argument("--ny", type=int, default=300, help="Transverse grid height, voxels.")
    p.add_argument("--nx", type=int, default=300, help="Transverse grid width, voxels.")
    p.add_argument("--voxel-um", type=float, default=2.0, help="Voxel size, um.")
    p.add_argument("--mu-fibre", type=float, default=6.2e-4,
                   help="Fibre (glass) per-voxel absorbance (mu times voxel size); ASTRA "
                        "sums voxel values over unit-length rays, so this must be per-voxel "
                        "absorbance, NOT mu in 1/cm, or line integrals saturate the beam.")
    p.add_argument("--mu-matrix", type=float, default=3.5e-4,
                   help="Matrix (resin) per-voxel absorbance; see --mu-fibre. Reference "
                        "PlenoXFiber scale (matrix 3.5e-4, fibre 6.2e-4, ~75 percent "
                        "contrast); override with calibrated 12 keV values as per-voxel "
                        "absorbance.")

    # plenoptic geometry params: study-scale defaults match the calibrated 260507 pin
    # calibration (GEOM_MODE A, Scripts/plenoptic-NewAstra-Al12thr_Sep05_combined_SRC.py
    # lines 640-664): SOD 129.24 mm, SDD 1171.57 mm, source/detector grid range 22 mm /
    # 177.4 mm, 55 um Timepix pixel, 21 x 21 = 441 positions per z-position.
    p.add_argument("--sods", type=float, nargs=4, default=[129.24, 129.62, 130.01, 130.39],
                   metavar=("SOD1", "SOD2", "SOD3", "SOD4"),
                   help="Source-to-sample distance per z-position (Zpos1..4), mm. The "
                        "source-to-detector rail (--sdd-mm) is fixed; only the sample "
                        "moves along the beam between z-positions. Default steps ~0.384 "
                        "mm from Zpos1 (129.24 mm, the calibrated GEOM_MODE A value), "
                        "matching the z_pos_offset=192 voxel (2 um voxel) Zpos2 offset "
                        "noted in plenoptic-NewAstra-Al12thr_Sep05_combined_SRC.py:847, "
                        "extrapolated uniformly for Zpos3/4 (no repo record of their real "
                        "offsets). Override with the true calibrated Zpos1..4 values for "
                        "the study run.")
    p.add_argument("--sdd-mm", type=float, default=1171.57,
                   help="Source-to-detector rail distance, mm (fixed across all "
                        "z-positions; only the sample moves).")
    p.add_argument("--detector-pixel-um", type=float, default=55.0,
                   help="Physical detector pixel size, um.")
    p.add_argument("--source-range-um", type=float, default=22000.0,
                   help="Plenoptic source grid half-range, um (22 mm calibrated default).")
    p.add_argument("--det-range-um", type=float, default=177400.0,
                   help="Plenoptic detector grid half-range, um (177.4 mm calibrated "
                        "default).")
    p.add_argument("--n-grid", type=int, default=21,
                   help="Plenoptic source/detector grid side length (n x n positions "
                        "per z-position); full study uses 21 (441 positions).")
    p.add_argument("--super-sampling", type=int, default=2,
                   help="ASTRA voxel/detector supersampling factor.")

    # forward model / reconstruction, shared identically across both strategies and all N
    p.add_argument("--i0", type=float, default=0.0,
                   help="0 = clean line integrals (no noise); >0 = Poisson photon noise at "
                        "that incident flux, drawn per z-position (the SNR dial).")
    p.add_argument("--n-iter", type=int, default=200,
                   help="TV solver iterations per stage/solve.")
    p.add_argument("--lr", type=float, default=1e-1, help="Adam learning rate.")
    p.add_argument("--tv-weight", type=float, default=0.5,
                   help="Squared-L2 gradient penalty weight along the fibre/beam axis "
                        "(dim 0) only; transverse axes unpenalised, matching the reference "
                        "along-fibre smoothing. May need one GPU tuning pass.")
    p.add_argument("--target-psnr", type=float, default=28.0,
                   help="PSNR (dB) threshold used for iters_to_threshold / "
                        "time_to_threshold (first_crossing).")
    p.add_argument("--log-every", type=int, default=10,
                   help="Convergence-history logging interval, iterations.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.smoke:
        domain_radius_um = 40.0
        r_mean_um = 6.0
        depth = 32
        iters = 20
        ny = nx = 64
        n_grid = 3
        n_iter = 8
        log_every = 1
    else:
        domain_radius_um = args.domain_radius_um
        r_mean_um = args.r_mean_um
        depth = args.depth
        iters = args.iters
        ny = args.ny
        nx = args.nx
        n_grid = args.n_grid
        n_iter = args.n_iter
        log_every = args.log_every

    voxel_um = args.voxel_um
    sods_mm = list(args.sods)
    sdd_mm = args.sdd_mm
    target_psnr = args.target_psnr

    # One deep straight bundle: fibres run along the beam axis (array dim 0).
    print(f"Building deep straight bundle ({depth}, {ny}, {nx}) ...")
    vol, radii = make_straight_fibre_phantom(
        nz=depth, ny=ny, nx=nx, voxel_um=voxel_um, domain_radius_um=domain_radius_um,
        fvf=args.fvf, r_mean_um=r_mean_um, mu_fibre=args.mu_fibre, mu_matrix=args.mu_matrix,
        iters=iters, seed=args.seed)
    print(f"  {radii.shape[0]} fibres, mu in [{vol.min():.2e}, {vol.max():.2e}]")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deferred to here: everything below touches astra and/or CUDA torch.
    import torch
    from phantom_sim.geometry import (
        build_plenoptic_geometry, make_plenoptic_projector, plenoptic_grid, XrayOperator,
    )
    from phantom_sim.forward import simulate_projections
    from phantom_sim.reconstruct import reconstruct
    from phantom_sim.metrics import psnr, ssim, register, resolution_along_across, corr_with_gt

    use_noise = bool(args.i0 and args.i0 > 0)
    forward_desc = f"poisson_i0_{args.i0:g}" if use_noise else "clean_line_integrals"
    volt = torch.tensor(vol, device="cuda")

    detector_z = (sdd_mm * 1000.0) / voxel_um
    pixel_size = args.detector_pixel_um / voxel_um
    source_range_pix = args.source_range_um / voxel_um
    det_range_pix = args.det_range_um / voxel_um
    src, det = plenoptic_grid(source_range_pix, det_range_pix, n=n_grid)
    num_imgs = n_grid * n_grid

    # Detector panel spans the two dims transverse to the beam. Beam axis is dim 0 (Z),
    # so the panel covers (ny, nx) = vol.shape[1], vol.shape[2] -- unlike the CT driver
    # (run_ct_orientation.py), whose ROTATING beam makes the fixed rotation axis (dim 0)
    # one of the panel's own axes instead.
    sinogram_size_1 = [vol.shape[1], vol.shape[2], num_imgs]

    geom_blocks = []
    proj_blocks = []
    vol_z0_list = []
    for i, sod_mm in enumerate(sods_mm):
        vol_z0_pix = (sod_mm * 1000.0) / voxel_um
        vol_z0_list.append(vol_z0_pix)
        geom_i = build_plenoptic_geometry(src, det, vol_z0_pix, detector_z, pixel_size, pixel_size)
        pid_i, _ = make_plenoptic_projector(
            vol.shape, geom_i, sinogram_size=sinogram_size_1, super_sampling=args.super_sampling)
        A_i = XrayOperator(pid_i)
        if use_noise:
            proj_i = simulate_projections(A_i, volt, i0=args.i0, seed=args.seed + i)
        else:
            proj_i = A_i(volt).detach()
        geom_blocks.append(geom_i)
        proj_blocks.append(proj_i)

    def build_stage_operator(n):
        # Geometry vectors are a flat (num_vectors, 12) table: vstack (default axis 0)
        # concatenates positions 1..n into one longer table. Projection DATA follows
        # ASTRA's 3D cone_vec convention (det_rows, num_vectors, det_cols) --
        # XrayOperator's proj_shape is astra.geom_size(proj_geom) in that same order --
        # so the matching concatenation axis for the projection TENSORS is dim=1, NOT
        # dim=0.
        geom_n = np.vstack(geom_blocks[:n])
        proj_n = torch.cat(proj_blocks[:n], dim=1)
        sinogram_size_n = [vol.shape[1], vol.shape[2], n * num_imgs]
        pid_n, _ = make_plenoptic_projector(
            vol.shape, geom_n, sinogram_size=sinogram_size_n, super_sampling=args.super_sampling)
        return XrayOperator(pid_n), proj_n

    # Fibres run along the beam axis = array dim 0, so the strong squared-L2 TV goes
    # on dim 0 only; the transverse axes stay unpenalised so fibre cross-sections stay
    # sharp. Non-negativity (lower_clamp=0.0) stops the plenoptic missing-wedge
    # null-space from blowing up, which is what wrecked the warm chain before.
    tv_weights = (args.tv_weight, 0.0, 0.0)

    def geometry_meta():
        return {
            "sods_mm": sods_mm,
            "sdd_mm": sdd_mm,
            "detector_pixel_um": args.detector_pixel_um,
            "source_range_um": args.source_range_um,
            "det_range_um": args.det_range_um,
            "n_grid": n_grid,
            "num_imgs_per_zpos": num_imgs,
            "vol_z0_pix": vol_z0_list,
            "detector_z_pix": detector_z,
            "pixel_size_pix": pixel_size,
            "super_sampling": args.super_sampling,
        }

    def save_stage(strategy, n, recon, entry):
        npy_path = out_dir / f"pleno_{strategy}_z{n}.npy"
        meta_path = out_dir / f"pleno_{strategy}_z{n}.meta.json"
        np.save(npy_path, recon.astype(np.float32))
        meta = {
            "strategy": strategy,
            "N": n,
            "smoke": bool(args.smoke),
            "shape": list(vol.shape),
            "dtype": "float32",
            "voxel_um": voxel_um,
            "mu_fibre": args.mu_fibre,
            "mu_matrix": args.mu_matrix,
            "n_fibres": int(radii.shape[0]),
            "packing": {
                "domain_radius_um": domain_radius_um,
                "fvf": args.fvf,
                "r_mean_um": r_mean_um,
                "depth": depth,
                "iters": iters,
                "seed": args.seed,
            },
            "geometry": geometry_meta(),
            "reconstruction": {
                "forward": forward_desc,
                "i0": args.i0,
                "n_iter": n_iter,
                "lr": args.lr,
                "tv_weight": args.tv_weight,
                "target_psnr": target_psnr,
                "log_every": log_every,
            },
            "metrics": entry,
        }
        write_meta(meta_path, meta)
        print(f"Saved {npy_path}")
        print(f"Saved {meta_path}")

    # --- cold_joint: N independent solves from x_init=None -----------------------------
    cold_entries = []
    for n in range(1, 5):
        A_n, proj_n = build_stage_operator(n)
        recon, history = reconstruct(
            A_n, proj_n, mask=None, n_iter=n_iter, lr=args.lr, tv_weights=tv_weights,
            x_init=None, log_every=log_every, gt=vol, target_psnr=target_psnr,
            lower_clamp=0.0,
        )
        c = corr_with_gt(recon, vol)
        p = psnr(recon, vol)
        s = ssim(recon, vol)
        recon_reg = register(recon, vol)
        res = resolution_along_across(recon_reg, fibre_axis=0)
        entry = assemble_stage_entry(n, c, p, s, res, history, target_psnr)
        cold_entries.append(entry)
        save_stage("cold_joint", n, recon, entry)
        print(f"  cold_joint N={n}: corr_with_gt={c:.4f} psnr={p:.2f} ssim={s:.4f}")

    # --- warm_sequential: chain x_init forward, track cumulative compute ---------------
    warm_entries = []
    x = None
    cumulative_iters = 0
    cumulative_time = 0.0
    for n in range(1, 5):
        A_n, proj_n = build_stage_operator(n)
        recon, history = reconstruct(
            A_n, proj_n, mask=None, n_iter=n_iter, lr=args.lr, tv_weights=tv_weights,
            x_init=x, log_every=log_every, gt=vol, target_psnr=target_psnr,
            lower_clamp=0.0,
        )
        x = recon
        cumulative_iters += n_iter
        cumulative_time += history["time"][-1] if history["time"] else 0.0
        c = corr_with_gt(recon, vol)
        p = psnr(recon, vol)
        s = ssim(recon, vol)
        recon_reg = register(recon, vol)
        res = resolution_along_across(recon_reg, fibre_axis=0)
        entry = assemble_stage_entry(n, c, p, s, res, history, target_psnr)
        warm_entries.append(entry)
        save_stage("warm_sequential", n, recon, entry)
        print(f"  warm_sequential N={n}: corr_with_gt={c:.4f} psnr={p:.2f} ssim={s:.4f}")

    meta = {
        "smoke": bool(args.smoke),
        "shape": list(vol.shape),
        "voxel_um": voxel_um,
        "mu_fibre": args.mu_fibre,
        "mu_matrix": args.mu_matrix,
        "n_fibres": int(radii.shape[0]),
        "packing": {
            "domain_radius_um": domain_radius_um,
            "fvf": args.fvf,
            "r_mean_um": r_mean_um,
            "depth": depth,
            "iters": iters,
            "seed": args.seed,
        },
        "geometry": geometry_meta(),
        "reconstruction": {
            "forward": forward_desc,
            "i0": args.i0,
            "n_iter": n_iter,
            "lr": args.lr,
            "tv_weight": args.tv_weight,
            "target_psnr": target_psnr,
            "log_every": log_every,
        },
    }
    metrics = assemble_metrics(cold_entries, warm_entries, cumulative_iters, cumulative_time, meta)
    metrics_path = out_dir / "pleno_metrics.json"
    write_meta(metrics_path, metrics)
    print(f"Saved {metrics_path}  (forward: {forward_desc})")
    print("HEADLINE corr-with-GT vs N (1..4):")
    print("  cold_joint:      " + "  ".join(f"{e['corr_with_gt']:.4f}" for e in cold_entries))
    print("  warm_sequential: " + "  ".join(f"{e['corr_with_gt']:.4f}" for e in warm_entries))
    print(f"  warm cumulative compute: {cumulative_iters} iters, {cumulative_time:.1f}s "
          f"(vs {4 * n_iter} iters for four independent cold solves)")


if __name__ == "__main__":
    main()
