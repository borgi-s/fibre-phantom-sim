"""CLI: fibre-DIAMETER sweep for the plenoptic resolution argument (exp9).

How small a fibre can the plenoptic setup resolve and size correctly? One rung per mean diameter
(20/18/16/14/12/10 um): a polydisperse bundle at the real sample's volume fraction (~0.60, CT
interior Otsu) and width (0.9 mm rope in a 1.02 mm frame), with a small UNIFORM misalignment at the
exp7 a08 level (independent momentum waypoint wander, ~8 deg p90), reconstructed with the same
single-position plenoptic geometry. The M1-M4 diameter methods then run on both the recon and the
ideal ground truth (``phantom_sim.diam_analysis``).

Differences from run_plenoptic_waypoint: (1) the re-pack uses a NEIGHBOUR LIST (scipy cKDTree) so
~5000 fibres stay cheap (the dense (N,N) version is O(N^2) memory and time); (2) the wander step is
uniform over the bundle (no calm/wavy split); (3) diameters scale with the rung, N(D, cv*D) clipped
at +/- clip_sigma; (4) the ideal GT volume is saved (uint8 coverage) with the fibre config, so the
analysis needs no re-voxelisation; (5) ``--phantom-only`` builds and saves the phantom without the
GPU (laptop preflight / step calibration).

Module-level code is stdlib + numpy only; scipy/torch/astra imports are deferred (scipy BEFORE
torch, see DTU_HPC_Guide B8.1).

Usage (cluster, CUDA interpreter):
    python -m phantom_sim.run_plenoptic_diamsweep --out phantom_sim_results/exp9_diamsweep/D20 --diam-mean-um 20
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from phantom_sim.run_plenoptic_waypoint import _catmull_rom, realized_angle_p90


def _pairs(p, cutoff):
    from scipy.spatial import cKDTree

    return cKDTree(p.T).query_pairs(float(cutoff), output_type="ndarray")


def worst_overlap_sparse(p, radii, cutoff=None):
    """Worst surface interpenetration (um) over neighbouring pairs; same value as the dense version."""
    if cutoff is None:
        cutoff = 2.0 * float(radii.max())
    ij = _pairs(p, cutoff)
    if len(ij) == 0:
        return 0.0
    i, j = ij[:, 0], ij[:, 1]
    dist = np.hypot(p[0, i] - p[0, j], p[1, i] - p[1, j])
    return float(np.clip(radii[i] + radii[j] - dist, 0.0, None).max())


def relax_move_sparse(p, radii, R, tol, iters=250):
    """Neighbour-list version of run_plenoptic_waypoint.relax_move (same update rule and stop test):
    confine every fibre surface inside the rope of radius R and push overlapping pairs apart until
    the surface gap is >= -tol. p (2, N) um."""
    p = np.array(p, dtype=np.float64, copy=True)
    lim = R - radii
    cutoff = 2.0 * float(radii.max())
    for _ in range(iters):
        rad = np.hypot(p[0], p[1])
        over = np.clip(rad - lim, 0.0, None)
        if float(over.max()) > 1e-9:
            p = p - over * (p / (rad + 1e-9))
        ij = _pairs(p, cutoff)
        pen_max = 0.0
        if len(ij):
            i, j = ij[:, 0], ij[:, 1]
            d = p[:, i] - p[:, j]
            dist = np.hypot(d[0], d[1])
            pen = np.clip((radii[i] + radii[j] - dist) - tol, 0.0, None)
            pen_max = float(pen.max())
            if pen_max > 1e-3:
                push = 0.6 * (d / (dist + 1e-9)) * pen
                np.add.at(p[0], i, push[0])
                np.add.at(p[1], i, push[1])
                np.add.at(p[0], j, -push[0])
                np.add.at(p[1], j, -push[1])
        if pen_max <= 1e-3 and float(over.max()) <= 1e-9:
            break
    return p


def uniform_wander_config(
    base_centers,
    radii,
    depth,
    R,
    step,
    K,
    revert=0.14,
    persistence=0.74,
    tol=0.05,
    seed=0,
):
    """Independent per-fibre momentum waypoint wander with a UNIFORM kick (the exp7 model with
    step_calm == step_wavy), re-packed with the neighbour list. Returns cfg (depth, 2, N) um."""
    rng = np.random.default_rng(int(seed))
    base = np.asarray(base_centers, float)
    N = base.shape[1]
    ctrl = np.empty((K, 2, N))
    ctrl[0] = relax_move_sparse(base, radii, R, tol)
    p = ctrl[0].copy()
    vel = np.zeros((2, N))
    for k in range(1, K):
        vel = persistence * vel + step * rng.standard_normal((2, N))
        p_prev = p
        p = relax_move_sparse(p + vel - revert * (p - base), radii, R, tol)
        vel = p - p_prev
        ctrl[k] = p
    cfg = _catmull_rom(ctrl, depth)
    for z in range(depth):
        if worst_overlap_sparse(cfg[z], radii) > tol:
            cfg[z] = relax_move_sparse(cfg[z], radii, R, tol, iters=120)
    return cfg


def draw_diameters_um(n, mean_um, cv, clip_sigma, seed):
    """Per-fibre DIAMETERS ~ N(mean, cv*mean) clipped to mean +/- clip_sigma*std."""
    rng = np.random.default_rng(int(seed))
    sd = cv * mean_um
    return np.clip(
        rng.normal(mean_um, sd, n), mean_um - clip_sigma * sd, mean_um + clip_sigma * sd
    )


def n_fibres_for_fvf(fvf, R, diam_mean_um, cv):
    """Fibre count whose mean area fills fraction fvf of a rope of radius R (E[d^2] = m^2 (1+cv^2))."""
    mean_area = np.pi / 4.0 * diam_mean_um**2 * (1.0 + cv**2)
    return int(round(fvf * np.pi * R**2 / mean_area))


def sparse_pack(n, radii, R, tol, seed, iters=4000):
    """Random-sequential start inside the rope + neighbour-list relaxation to a non-overlapping pack.
    Fallback for when the vendored fibre-pack is too slow for thousands of fibres."""
    rng = np.random.default_rng(int(seed))
    rad = (R - radii) * np.sqrt(rng.uniform(0.0, 1.0, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    p = np.stack([rad * np.cos(th), rad * np.sin(th)])
    return relax_move_sparse(p, radii, R, tol, iters=iters)


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", required=True, help="Output directory for this rung.")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny fast problem (still needs CUDA unless --phantom-only).",
    )
    p.add_argument(
        "--phantom-only",
        action="store_true",
        help="Build + save the phantom, skip the GPU recon.",
    )
    # Diameters
    p.add_argument("--diam-mean-um", type=float, required=True)
    p.add_argument(
        "--diam-cv",
        type=float,
        default=0.10,
        help="Diameter std / mean.",
    )
    p.add_argument("--diam-clip-sigma", type=float, default=2.0)
    p.add_argument("--diam-seed", type=int, default=3)
    # Bundle
    p.add_argument(
        "--fvf",
        type=float,
        default=0.60,
        help="Fibre fraction of the rope (CT interior Otsu 0.606).",
    )
    p.add_argument(
        "--domain-radius-um",
        type=float,
        default=450.0,
        help="Rope radius (0.9 mm sample).",
    )
    p.add_argument(
        "--pack",
        choices=("fibrepack", "sparse"),
        default="sparse",
        help="Straight base pack: vendored fibre-pack (torch, uniform r) or neighbour-list relax.",
    )
    p.add_argument("--pack-iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    # Uniform misalignment (exp7 a08 level)
    p.add_argument(
        "--step",
        type=float,
        default=0.85,
        help="Per-waypoint kick (um), uniform (a08 = 0.85).",
    )
    p.add_argument("--waypoint-spacing-um", type=float, default=15.0)
    p.add_argument("--persistence", type=float, default=0.74)
    p.add_argument("--revert", type=float, default=0.14)
    p.add_argument("--overlap-tol-um", type=float, default=0.05)
    p.add_argument("--wave-seed", type=int, default=0)
    # Grid / geometry / recon (as run_plenoptic_waypoint)
    p.add_argument("--nz", type=int, default=256)
    p.add_argument("--ny", type=int, default=512)
    p.add_argument("--nx", type=int, default=512)
    p.add_argument("--voxel-um", type=float, default=2.0)
    p.add_argument("--mu-fibre", type=float, default=1.0)
    p.add_argument("--mu-matrix", type=float, default=0.0)
    p.add_argument("--sod-mm", type=float, default=129.24)
    p.add_argument("--sdd-mm", type=float, default=1171.57)
    p.add_argument("--detector-pixel-um", type=float, default=55.0)
    p.add_argument("--source-range-um", type=float, default=22000.0)
    p.add_argument("--det-range-um", type=float, default=177400.0)
    p.add_argument("--n-grid", type=int, default=21)
    p.add_argument("--super-sampling", type=int, default=2)
    p.add_argument("--pad-transverse-vox", type=int, default=64)
    p.add_argument("--pad-depth-vox", type=int, default=32)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--tv-weight", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=10)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    import scipy.optimize  # noqa: F401  (import scipy before torch: libstdc++ CXXABI clash on DTU HPC)
    import scipy.spatial  # noqa: F401

    t0 = time.time()
    if args.smoke:
        nz, ny, nx, R = 32, 128, 128, 110.0
        n_grid, n_iter, log_every, pad_t, pad_d = 3, 6, 1, 8, 2
    else:
        nz, ny, nx, R = args.nz, args.ny, args.nx, args.domain_radius_um
        n_grid, n_iter, log_every = args.n_grid, args.n_iter, args.log_every
        pad_t, pad_d = args.pad_transverse_vox, args.pad_depth_vox
    voxel_um = args.voxel_um
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if R + 0.0 > (min(nx, ny) / 2.0) * voxel_um:
        raise SystemExit(
            f"rope radius {R} um does not fit the {nx}x{ny} frame at {voxel_um} um"
        )

    from phantom_sim.phantom import voxelize_config

    # 1. Straight base pack at the target fvf, then polydisperse diameters N(D, cv D).
    D = args.diam_mean_um
    n = n_fibres_for_fvf(args.fvf, R, D, args.diam_cv)
    diam = draw_diameters_um(n, D, args.diam_cv, args.diam_clip_sigma, args.diam_seed)
    radii = diam / 2.0
    print(
        f"[D{D:g}] {n} fibres, d={diam.mean():.2f}+/-{diam.std():.2f} um, target fvf {args.fvf}, "
        f"rope R={R} um, pack={args.pack}",
        flush=True,
    )
    if args.pack == "fibrepack":
        from phantom_sim.phantom import pack_bundle

        cfg0, _ = pack_bundle(
            domain_radius_um=R,
            fvf=args.fvf,
            r_mean_um=D / 2.0,
            r_sigma_um=0.0,
            n_slices=2,
            misalignment="none",
            iters=args.pack_iters,
            seed=args.seed,
        )
        base = cfg0[0]
        if base.shape[1] != n:
            n = int(base.shape[1])
            diam = draw_diameters_um(
                n, D, args.diam_cv, args.diam_clip_sigma, args.diam_seed
            )
            radii = diam / 2.0
            print(f"  fibre-pack returned {n} fibres; redrew diameters", flush=True)
    else:
        base = sparse_pack(n, radii, R, args.overlap_tol_um, seed=args.seed)
    t_pack = time.time() - t0
    print(
        f"  base pack done in {t_pack:.0f}s, worst overlap {worst_overlap_sparse(base, radii):.3f} um",
        flush=True,
    )

    # 2. Uniform a08-level waypoint wander.
    K = max(4, int(round(nz * voxel_um / args.waypoint_spacing_um)) + 1)
    cfg = uniform_wander_config(
        base,
        radii,
        nz,
        R,
        args.step,
        K,
        revert=args.revert,
        persistence=args.persistence,
        tol=args.overlap_tol_um,
        seed=args.wave_seed,
    )
    ang = realized_angle_p90(cfg, voxel_um)
    worst = max(worst_overlap_sparse(cfg[z], radii) for z in range(nz))
    fvf_eff = float(np.sum(radii**2) / R**2)
    t_wander = time.time() - t0 - t_pack
    print(
        f"  wander K={K}: p90 angle median {np.median(ang):.1f} deg (mean {ang.mean():.1f}, "
        f"p90-of-p90 {np.percentile(ang, 90):.1f}); worst overlap {worst:.3f} um; fvf_eff {fvf_eff:.3f}; "
        f"{t_wander:.0f}s",
        flush=True,
    )

    # 3. Ideal GT volume (anti-aliased coverage) + fibre truth.
    obj = voxelize_config(
        cfg,
        radii,
        (ny, nx),
        voxel_um=voxel_um,
        axis=0,
        mu_fibre=args.mu_fibre,
        mu_matrix=args.mu_matrix,
        supersample=4,
    )
    cover = (obj - args.mu_matrix) / (args.mu_fibre - args.mu_matrix)
    np.save(
        out_dir / "gt_vol_u8.npy",
        np.clip(np.round(cover * 255.0), 0, 255).astype(np.uint8),
    )
    np.savez(
        out_dir / "fibre_gt.npz",
        cfg=cfg.astype(np.float32),
        radii=radii,
        base=base,
        angle_p90_deg=ang,
    )
    t_vox = time.time() - t0 - t_pack - t_wander
    print(
        f"  voxelised {obj.shape} in {t_vox:.0f}s; GT fill {float((cover > 0.5).mean()):.3f}",
        flush=True,
    )

    meta = {
        "exp": "exp9_diamsweep",
        "smoke": bool(args.smoke),
        "phantom_only": bool(args.phantom_only),
        "diam_mean_um": D,
        "diam_cv": args.diam_cv,
        "diam_clip_sigma": args.diam_clip_sigma,
        "diam_seed": args.diam_seed,
        "diam_realized_mean_um": float(diam.mean()),
        "diam_realized_std_um": float(diam.std()),
        "n_fibres": int(n),
        "fvf": args.fvf,
        "fvf_eff": fvf_eff,
        "domain_radius_um": float(R),
        "pack": args.pack,
        "seed": args.seed,
        "step": args.step,
        "waypoint_spacing_um": args.waypoint_spacing_um,
        "n_waypoints": int(K),
        "persistence": args.persistence,
        "revert": args.revert,
        "overlap_tol_um": args.overlap_tol_um,
        "worst_overlap_um": float(worst),
        "wave_seed": args.wave_seed,
        "angle_p90_median_deg": float(np.median(ang)),
        "angle_p90_mean_deg": float(ang.mean()),
        "nz": nz,
        "ny": ny,
        "nx": nx,
        "voxel_um": voxel_um,
        "mu_fibre": args.mu_fibre,
        "mu_matrix": args.mu_matrix,
        "sod_mm": args.sod_mm,
        "sdd_mm": args.sdd_mm,
        "detector_pixel_um": args.detector_pixel_um,
        "source_range_um": args.source_range_um,
        "det_range_um": args.det_range_um,
        "n_grid": n_grid,
        "n_iter": n_iter,
        "lr": args.lr,
        "tv_weight": args.tv_weight,
        "pad_transverse_vox": pad_t,
        "pad_depth_vox": pad_d,
        "t_pack_s": t_pack,
        "t_wander_s": t_wander,
        "t_voxelise_s": t_vox,
    }
    result = {}
    if not args.phantom_only:
        import torch
        from phantom_sim.geometry import (
            build_plenoptic_geometry,
            make_plenoptic_projector,
            plenoptic_grid,
            XrayOperator,
            delete_projector,
        )
        from phantom_sim.reconstruct import reconstruct
        from phantom_sim.metrics import psnr, corr_with_gt
        from phantom_sim.run_plenoptic_tilt import max_plenoptic_half_angle_deg

        t1 = time.time()
        detector_z = (args.sdd_mm * 1000.0) / voxel_um
        pixel_size = args.detector_pixel_um / voxel_um
        src, det = plenoptic_grid(
            args.source_range_um / voxel_um, args.det_range_um / voxel_um, n=n_grid
        )
        num_imgs = n_grid * n_grid
        vol_z0 = (args.sod_mm * 1000.0) / voxel_um
        padded_shape = (nz + 2 * pad_d, ny + 2 * pad_t, nx + 2 * pad_t)
        obj_sl = (
            slice(pad_d, pad_d + nz),
            slice(pad_t, pad_t + ny),
            slice(pad_t, pad_t + nx),
        )
        sinogram_size = [padded_shape[1], padded_shape[2], num_imgs]
        vol = np.pad(obj, ((pad_d, pad_d), (pad_t, pad_t), (pad_t, pad_t)))
        geom = build_plenoptic_geometry(
            src, det, vol_z0, detector_z, pixel_size, pixel_size
        )
        pid, _ = make_plenoptic_projector(
            padded_shape,
            geom,
            sinogram_size=sinogram_size,
            super_sampling=args.super_sampling,
        )
        A = XrayOperator(pid)
        volt = torch.tensor(vol, device="cuda")
        proj = A(volt).detach()
        del volt
        torch.cuda.empty_cache()

        def corr_metric(recon_padded_np):
            return corr_with_gt(np.ascontiguousarray(recon_padded_np[obj_sl]), obj)

        recon, _ = reconstruct(
            A,
            proj,
            mask=None,
            n_iter=n_iter,
            lr=args.lr,
            tv_weights=(args.tv_weight, 0.0, 0.0),
            x_init=None,
            log_every=log_every,
            gt=vol,
            target_psnr=None,
            lower_clamp=0.0,
            plateau_scheduler=True,
            metric_fn=corr_metric,
            early_stop=False,
        )
        recon_obj = np.ascontiguousarray(recon[obj_sl]).astype(np.float32)
        delete_projector(A.projector_id)
        del proj, A
        torch.cuda.empty_cache()
        np.save(out_dir / "recon.npy", recon_obj)
        fibre_zone = np.zeros(obj.shape, dtype=bool)
        rr, cc = np.mgrid[0:ny, 0:nx]
        inside = (
            np.hypot((cc - (nx - 1) / 2.0) * voxel_um, (rr - (ny - 1) / 2.0) * voxel_um)
            <= R
        )
        fibre_zone[:] = inside[None]
        result = {
            "corr_all": float(corr_with_gt(recon_obj, obj)),
            "corr_rope": float(
                np.corrcoef(recon_obj[fibre_zone], obj[fibre_zone])[0, 1]
            ),
            "psnr": float(psnr(recon_obj, obj)),
            "max_half_angle_deg": float(
                max_plenoptic_half_angle_deg(
                    args.source_range_um, args.det_range_um, args.sdd_mm * 1000.0
                )
            ),
            "t_recon_s": time.time() - t1,
        }
        print(
            f"  recon done in {result['t_recon_s']:.0f}s: corr_all {result['corr_all']:.4f}, "
            f"corr_rope {result['corr_rope']:.4f}",
            flush=True,
        )
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "result": result}, f, indent=2)
    print(f"Saved {out_dir} ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
