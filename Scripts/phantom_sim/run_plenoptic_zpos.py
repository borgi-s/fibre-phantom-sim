"""CLI: Experiment 2, plenoptic z-position count and warm-start vs cold-joint solves.

Packs ONE fibre bundle, voxelises it ONCE along axis=0 (beam axis: fibres run parallel
to dim 0, the plenoptic geometry's near-Z beam direction for every source/detector cell
in every z-position's grid). Builds FOUR z-position plenoptic geometries (four SODs,
Zpos1..4, each a full n x n source/detector grid via `plenoptic_grid`; the shared
source-to-detector rail distance `--sdd-mm` is fixed, only the sample moves along the
beam between z-positions). Forward-projects the phantom ONCE per z-position at flux
`i0` to get that position's own noisy projection block (kept alongside its geometry
vectors).

Phantom + solver match Experiment 1 (the reference vedrana recipe), so CT and plenoptic
reconstruct the SAME object: a frame-filling bundle (`--domain-radius-um` 300 at ny=nx=300,
`--fvf` 0.4) voxelised as fill-fraction in [0,1] against a ZERO background (`--mu-matrix 0`,
`--mu-fibre 1`); void recon margin via `--pad-transverse-vox` / `--pad-depth-vox` (scored on
the object crop); solver lr 0.05, 300 iters, ReduceLROnPlateau. Switch to the physical
glass-in-resin study with `--mu-matrix 3.5e-4 --mu-fibre 6.2e-4`. NOTE Exp 2 runs 8 solves
(cold x4 + warm x4), so the padded volume multiplies memory/time versus Exp 1's 2 solves.

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
`iters_to_threshold`/`time_to_threshold` are `first_crossing(history, target_corr,
key="corr")` -- the first iteration/wall-clock at which the per-iteration corr-with-GT
trace (`history["corr"]`, scored on the object crop) reaches `target_corr`; both are
`null` when the bar was never crossed. The speed threshold is on corr, NOT the padded-
frame PSNR, which the void margin pins far below any meaningful bar; PSNR is still logged
per iteration as `history["metric"]` for the convergence curve. Because the full corr
trace is saved, the reported bar can be re-chosen post-hoc without rerunning.
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
    # resume a run whose cold arm finished but whose warm arm was killed:
    python -m phantom_sim.run_plenoptic_zpos --warm-only --out phantom_sim_results/exp2_match

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


def load_cold_entries(out_dir):
    """Load the four cold_joint stage entries from a prior run's saved metas.

    Enables --warm-only resume: a completed cold arm leaves
    pleno_cold_joint_z{1..4}.meta.json, each carrying its per-stage "metrics"
    entry (the assemble_stage_entry dict). This lets the warm arm be run on its own
    and merged into the final pleno_metrics.json, instead of recomputing the
    identical cold solves. Pure/torch-free, so it is unit-testable on a box without
    astra or CUDA. Raises FileNotFoundError if any of the four cold metas is missing.
    """
    out_dir = Path(out_dir)
    entries = []
    for n in range(1, 5):
        meta_path = out_dir / f"pleno_cold_joint_z{n}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"--warm-only needs the cold arm's saved metas; missing {meta_path}. "
                f"Run the full driver (without --warm-only) first.")
        with open(meta_path, "r", encoding="utf-8") as f:
            entries.append(json.load(f)["metrics"])
    return entries


def assemble_stage_entry(n, corr_val, psnr_val, ssim_val, resolution, history, target_corr):
    """Pure metrics-assembly for one (strategy, N) stage.

    No astra/torch here: corr_val, psnr_val, ssim_val, resolution and history are already-
    computed plain python/numpy values, so this is directly unit-testable on this box
    with fake inputs (no GPU needed). corr_with_gt is the headline fidelity metric, and the
    speed threshold (iters_to_threshold / time_to_threshold) is the first iteration/wall-
    clock at which the per-iteration corr-with-GT trace (history["corr"]) reaches
    target_corr. Both are null when the bar was never crossed. corr is used, not the
    padded-frame PSNR, because the void recon margin pins PSNR far below any meaningful bar.
    """
    iters_thr, time_thr = first_crossing(history, target_corr, key="corr")
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


def assemble_metrics(cold_entries, warm_entries, cold_cum_iters, cold_cum_time,
                     warm_cum_iters, warm_cum_time, meta):
    """Pure schema assembly for the combined pleno_metrics.json dict. No astra/torch.

    Both arms carry cumulative_iters/cumulative_time (total compute actually spent), so the
    'does the warm chain reach the 4-position result for less total compute than four cold
    solves' question is one subtraction. Meaningful only under early stopping; with a fixed
    budget both totals are 4*n_iter by construction.
    """
    return {
        "cold_joint": {
            "by_N": cold_entries,
            "cumulative_iters": int(cold_cum_iters),
            "cumulative_time": float(cold_cum_time),
        },
        "warm_sequential": {
            "by_N": warm_entries,
            "cumulative_iters": int(warm_cum_iters),
            "cumulative_time": float(warm_cum_time),
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
    p.add_argument("--warm-only", dest="warm_only", action="store_true",
                   help="Resume: skip the cold_joint arm and load its four stage entries "
                        "from the pleno_cold_joint_z{1..4}.meta.json already in --out "
                        "(saved by a prior run), then compute only the warm_sequential arm "
                        "and write the combined pleno_metrics.json. Uses ~half the compute "
                        "when a completed cold arm is already on disk. The phantom and "
                        "forward projections are rebuilt (deterministic under --seed), so "
                        "the warm arm matches the cold arm's object exactly.")

    # packing params (see phantom.pack_bundle); study-scale defaults match
    # make_phantom.py and run_ct_orientation.py (Experiment 1), so both experiments
    # phantom the same way.
    p.add_argument("--domain-radius-um", type=float, default=300.0,
                   help="Packing domain (bundle cross-section) radius, um. Default 300 fills "
                        "the ny=nx=300 transverse frame, matching Experiment 1 so the CT and "
                        "plenoptic comparisons run on the SAME object. Keep near ny/2 * voxel_um.")
    p.add_argument("--fvf", type=float, default=0.4,
                   help="Fibre volume fraction as a 0-1 fraction (reference vedrana phantom 0.4).")
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
    p.add_argument("--mu-fibre", type=float, default=1.0,
                   help="Fibre value. Default 1.0 with --mu-matrix 0 reproduces the reference "
                        "[0,1] fill-fraction phantom (fibres in void), matching Experiment 1. "
                        "For the physical glass-in-resin study use 6.2e-4 (per-voxel "
                        "absorbance; ASTRA sums voxel values over unit-length rays, so this "
                        "must be per-voxel absorbance, NOT mu in 1/cm).")
    p.add_argument("--mu-matrix", type=float, default=0.0,
                   help="Background (matrix) value. Default 0.0 = fibres in void (reference "
                        "recipe; a zero background also makes the recon padding seamless). For "
                        "glass-in-resin use 3.5e-4 (per-voxel absorbance).")

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
    p.add_argument("--n-iter", type=int, default=300,
                   help="TV solver iterations per stage/solve (reference 300).")
    p.add_argument("--lr", type=float, default=5e-2, help="Adam learning rate (reference 0.05).")
    p.add_argument("--tv-weight", type=float, default=0.5,
                   help="Squared-L2 gradient penalty weight along the fibre/beam axis "
                        "(dim 0) only; transverse axes unpenalised, matching the reference "
                        "along-fibre smoothing. May need one GPU tuning pass.")
    p.add_argument("--pad-transverse-vox", type=int, default=175,
                   help="Void margin (voxels per side) added to each transverse recon axis, "
                        "matching Experiment 1 (reference pads the 300-wide object into a "
                        "650-wide frame). Seamless only with a void background (--mu-matrix 0); "
                        "metrics score the object crop only. NOTE: Exp 2 runs 8 solves "
                        "(cold x4 + warm x4), so a large pad multiplies memory/time; dial it "
                        "down (or --n-iter) for a first pass if the A100 is tight.")
    p.add_argument("--pad-depth-vox", type=int, default=100,
                   help="Void margin (voxels per side) added to the fibre/beam recon axis "
                        "(reference pads 600 -> 800).")
    p.add_argument("--plateau-scheduler", dest="plateau_scheduler", action="store_true",
                   default=True, help="ReduceLROnPlateau (factor 0.5, patience 10), the "
                                      "reference solver (default on).")
    p.add_argument("--no-plateau-scheduler", dest="plateau_scheduler", action="store_false",
                   help="Disable the LR scheduler (fixed lr).")
    p.add_argument("--target-psnr", type=float, default=28.0,
                   help="PSNR (dB) still logged per iteration as history['metric'] for the "
                        "convergence curve, but NO LONGER the speed threshold (padded-frame "
                        "PSNR is pinned low by the void margin).")
    p.add_argument("--target-corr", type=float, default=0.90,
                   help="corr-with-GT threshold (on the object crop) for the Experiment 2 "
                        "speed comparison: iters_to_threshold / time_to_threshold are the "
                        "first iteration/wall-clock at which the per-iteration corr trace "
                        "reaches this. corr climbs to ~0.95, so 0.90 (~95%% of converged) is "
                        "a bar both arms reach. Because the full corr trace is saved in the "
                        "history, the reported bar can be re-chosen post-hoc without rerunning.")
    p.add_argument("--log-every", type=int, default=10,
                   help="Convergence-history logging interval, iterations.")
    p.add_argument("--early-stop", dest="early_stop", action="store_true", default=True,
                   help="Stop each stage once its corr-with-GT trace plateaus (+ a short "
                        "tail), instead of always running --n-iter. --n-iter becomes the hard "
                        "cap. Warm stages may not stop until corr recovers to the previous "
                        "position's fidelity (floor = prev_final_corr - --es-tol), so a warm "
                        "restart transient cannot stop a stage below the inherited level. Both "
                        "arms early-stop, so the iteration counts compare fairly. Default on.")
    p.add_argument("--no-early-stop", dest="early_stop", action="store_false",
                   help="Run every stage to the full --n-iter (fixed budget; the old behaviour).")
    p.add_argument("--es-tol", type=float, default=1e-3,
                   help="Early-stop plateau tolerance: a stage is plateaued when its corr "
                        "spread over the last (patience+1) logged points is below this. Also "
                        "the warm floor margin (prev_final_corr - es_tol). Default 1e-3.")
    p.add_argument("--es-extra", type=int, default=15,
                   help="Iterations to run after the plateau is first detected before stopping "
                        "(the '10-20 more iterations' tail). Default 15.")
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
        pad_transverse = 4
        pad_depth = 2
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
        pad_transverse = args.pad_transverse_vox
        pad_depth = args.pad_depth_vox

    voxel_um = args.voxel_um
    sods_mm = list(args.sods)
    sdd_mm = args.sdd_mm
    target_psnr = args.target_psnr
    target_corr = args.target_corr

    # One deep straight bundle: fibres run along the beam axis (array dim 0).
    print(f"Building deep straight bundle ({depth}, {ny}, {nx}) ...")
    vol_obj, radii = make_straight_fibre_phantom(
        nz=depth, ny=ny, nx=nx, voxel_um=voxel_um, domain_radius_um=domain_radius_um,
        fvf=args.fvf, r_mean_um=r_mean_um, mu_fibre=args.mu_fibre, mu_matrix=args.mu_matrix,
        iters=iters, seed=args.seed)
    print(f"  {radii.shape[0]} fibres, values in [{vol_obj.min():.2e}, {vol_obj.max():.2e}]")

    # Void margin around the object (reference pads 300->650 transverse, 600->800 beam),
    # so the object sits off the reconstruction boundary. Seamless only because the
    # background is void (mu_matrix 0). The fibre/beam axis is dim 0 (single orientation,
    # no moveaxis), so the crop back to the object region is a plain slice.
    vol = np.pad(vol_obj, ((pad_depth, pad_depth), (pad_transverse, pad_transverse),
                           (pad_transverse, pad_transverse)))
    obj_sl = (slice(pad_depth, pad_depth + depth),
              slice(pad_transverse, pad_transverse + ny),
              slice(pad_transverse, pad_transverse + nx))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deferred to here: everything below touches astra and/or CUDA torch.
    import torch
    from phantom_sim.geometry import (
        build_plenoptic_geometry, make_plenoptic_projector, plenoptic_grid, XrayOperator,
        delete_projector,
    )
    from phantom_sim.forward import simulate_projections
    from phantom_sim.reconstruct import reconstruct
    from phantom_sim.metrics import psnr, ssim, register, resolution_along_across, corr_with_gt

    # Per-iteration fidelity for the speed threshold: corr-with-GT on the object crop only
    # (the solver logs this into history["corr"] every log step, sharing the one GPU->CPU
    # copy with the PSNR metric). Scored on the crop, not the padded frame, so it matches the
    # final headline corr and the void margin cannot inflate it.
    def corr_metric(recon_padded_np):
        return corr_with_gt(np.ascontiguousarray(recon_padded_np[obj_sl]), vol_obj)

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
        # Park each projection block in CPU RAM. Keeping all four resident on the GPU costs
        # ~n_grid^2 * det_rows * det_cols * 4 z-positions of float32 (~3 GB at study scale),
        # which is what tips a 32 GB card over at N=3/N=4; only the concatenated proj_n for
        # the stage being solved needs to be on the GPU (staged in build_stage_operator).
        # This moves memory, not math: the .cpu()/.to(cuda) round-trip is exact float32, so
        # results are bit-identical to keeping the blocks resident.
        proj_blocks.append(proj_i.cpu())
        del proj_i
        # The per-position projector is only needed to make this block's projection; free
        # it now so four forward projectors do not sit resident through the eight solves.
        delete_projector(pid_i)
        torch.cuda.empty_cache()

    # The phantom-on-GPU is only needed for the forward projections above; the solves work
    # from proj_blocks + a fresh x, so release it before reconstructing.
    del volt
    torch.cuda.empty_cache()

    def build_stage_operator(n):
        # Geometry vectors are a flat (num_vectors, 12) table: vstack (default axis 0)
        # concatenates positions 1..n into one longer table. Projection DATA follows
        # ASTRA's 3D cone_vec convention (det_rows, num_vectors, det_cols) --
        # XrayOperator's proj_shape is astra.geom_size(proj_geom) in that same order --
        # so the matching concatenation axis for the projection TENSORS is dim=1, NOT
        # dim=0.
        geom_n = np.vstack(geom_blocks[:n])
        # proj_blocks live in CPU RAM; concatenate there (no GPU transient) and stage only
        # this one contiguous proj_n onto the GPU. free_gpu_stage() drops it after the solve.
        proj_n = torch.cat(proj_blocks[:n], dim=1).to("cuda")
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
            "object_shape": list(vol_obj.shape),
            "recon_padded_shape": list(vol.shape),
            "saved_shape": list(recon.shape),
            "pad_transverse_vox": pad_transverse,
            "pad_depth_vox": pad_depth,
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
                "plateau_scheduler": bool(args.plateau_scheduler),
                "target_psnr": target_psnr,
                "target_corr": target_corr,
                "early_stop": bool(args.early_stop),
                "es_tol": args.es_tol,
                "es_extra": args.es_extra,
                "log_every": log_every,
            },
            "metrics": entry,
        }
        write_meta(meta_path, meta)
        print(f"Saved {npy_path}")
        print(f"Saved {meta_path}")

    def stage_iters(history):
        return (history["iter"][-1] + 1) if history.get("iter") else 0

    def free_gpu_stage(A_n, proj_n):
        # Free this stage's projector + concatenated projection tensor so the eight solves
        # do not accumulate dead GPU allocations (OOM insurance on a 32 GB V100).
        delete_projector(A_n.projector_id)
        del proj_n
        torch.cuda.empty_cache()

    es_kwargs = dict(early_stop=args.early_stop, es_tol=args.es_tol, es_extra_iters=args.es_extra)

    # --- cold_joint: N independent solves from x_init=None -----------------------------
    # --warm-only resumes a run whose cold arm already finished: reuse its saved
    # per-stage entries (and the .npy volumes already on disk) instead of recomputing.
    if args.warm_only:
        cold_entries = load_cold_entries(out_dir)
        print(f"--warm-only: loaded {len(cold_entries)} cold_joint entries from {out_dir} "
              f"(cold arm not recomputed)")
    else:
        cold_entries = []
        for n in range(1, 5):
            A_n, proj_n = build_stage_operator(n)
            recon, history = reconstruct(
                A_n, proj_n, mask=None, n_iter=n_iter, lr=args.lr, tv_weights=tv_weights,
                x_init=None, log_every=log_every, gt=vol, target_psnr=target_psnr,
                lower_clamp=0.0, plateau_scheduler=args.plateau_scheduler,
                metric_fn=corr_metric, es_floor=None, **es_kwargs,
            )
            # Score on the object region only (crop the void margin off), vs the unpadded GT.
            recon_obj = np.ascontiguousarray(recon[obj_sl])
            c = corr_with_gt(recon_obj, vol_obj)
            p = psnr(recon_obj, vol_obj)
            s = ssim(recon_obj, vol_obj)
            res = resolution_along_across(register(recon_obj, vol_obj), fibre_axis=0)
            entry = assemble_stage_entry(n, c, p, s, res, history, target_corr)
            cold_entries.append(entry)
            save_stage("cold_joint", n, recon_obj, entry)
            print(f"  cold_joint N={n}: iters={stage_iters(history)} "
                  f"corr_with_gt={c:.4f} psnr={p:.2f} ssim={s:.4f}")
            free_gpu_stage(A_n, proj_n)
            del A_n, recon, recon_obj

    # Total compute actually spent on the cold arm (its four independent solves); works
    # whether the entries were computed above or loaded via --warm-only.
    cold_cum_iters = sum(stage_iters(e["history"]) for e in cold_entries)
    cold_cum_time = sum((e["history"]["time"][-1] if e["history"].get("time") else 0.0)
                        for e in cold_entries)

    # --- warm_sequential: chain x_init forward, track cumulative compute ---------------
    warm_entries = []
    x = None
    prev_corr = None  # previous position's final corr -> this stage's early-stop floor
    cumulative_iters = 0
    cumulative_time = 0.0
    for n in range(1, 5):
        A_n, proj_n = build_stage_operator(n)
        # N=1 is from scratch (no floor); N>=2 must not stop before corr recovers to the
        # previous position's fidelity minus the tolerance margin.
        es_floor = None if prev_corr is None else prev_corr - args.es_tol
        recon, history = reconstruct(
            A_n, proj_n, mask=None, n_iter=n_iter, lr=args.lr, tv_weights=tv_weights,
            x_init=x, log_every=log_every, gt=vol, target_psnr=target_psnr,
            lower_clamp=0.0, plateau_scheduler=args.plateau_scheduler,
            metric_fn=corr_metric, es_floor=es_floor, **es_kwargs,
        )
        x = recon  # carry the PADDED recon forward (matches the geometry); crop only to score
        cumulative_iters += stage_iters(history)
        cumulative_time += history["time"][-1] if history["time"] else 0.0
        recon_obj = np.ascontiguousarray(recon[obj_sl])
        c = corr_with_gt(recon_obj, vol_obj)
        p = psnr(recon_obj, vol_obj)
        s = ssim(recon_obj, vol_obj)
        res = resolution_along_across(register(recon_obj, vol_obj), fibre_axis=0)
        prev_corr = c
        entry = assemble_stage_entry(n, c, p, s, res, history, target_corr)
        warm_entries.append(entry)
        save_stage("warm_sequential", n, recon_obj, entry)
        print(f"  warm_sequential N={n}: iters={stage_iters(history)} "
              f"corr_with_gt={c:.4f} psnr={p:.2f} ssim={s:.4f}")
        free_gpu_stage(A_n, proj_n)
        del A_n, recon_obj

    meta = {
        "smoke": bool(args.smoke),
        "object_shape": list(vol_obj.shape),
        "recon_padded_shape": list(vol.shape),
        "pad_transverse_vox": pad_transverse,
        "pad_depth_vox": pad_depth,
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
            "target_corr": target_corr,
            "early_stop": bool(args.early_stop),
            "es_tol": args.es_tol,
            "es_extra": args.es_extra,
            "log_every": log_every,
        },
    }
    metrics = assemble_metrics(cold_entries, warm_entries, cold_cum_iters, cold_cum_time,
                               cumulative_iters, cumulative_time, meta)
    metrics_path = out_dir / "pleno_metrics.json"
    write_meta(metrics_path, metrics)
    print(f"Saved {metrics_path}  (forward: {forward_desc})")
    print("HEADLINE corr-with-GT vs N (1..4):")
    print("  cold_joint:      " + "  ".join(f"{e['corr_with_gt']:.4f}" for e in cold_entries))
    print("  warm_sequential: " + "  ".join(f"{e['corr_with_gt']:.4f}" for e in warm_entries))
    print(f"  TOTAL compute to the 4-position result: "
          f"warm {cumulative_iters} iters / {cumulative_time:.1f}s  vs  "
          f"cold {cold_cum_iters} iters / {cold_cum_time:.1f}s "
          f"({'early-stop' if args.early_stop else 'fixed budget'})")


if __name__ == "__main__":
    main()
