"""CLI: localized-misalignment plenoptic study, WAYPOINT wander model (exp7, QC demonstrator).

Follow-up to run_plenoptic_local. The coherent-shared-profile waviness there keeps the pack dense
by moving every fibre together, which forbids genuinely INDEPENDENT per-fibre waviness. This driver
instead lets each fibre wander on its own path and keeps the pack non-overlapping by RE-PACKING
(moving fibres) rather than shrinking them, so fvf is preserved and the fibres take a realistic
POLYDISPERSE diameter distribution.

Model: pack a straight fvf-target cross-section, draw per-fibre diameters ~ N(mean, std), then grow
the fibres through K waypoints spaced `--waypoint-spacing-um` along z. At each waypoint every fibre
takes a step with MOMENTUM (it keeps most of its previous direction plus a random kick, and turns
only when a collision redirects it -> long smooth directional sweeps, not jitter), mildly reverting
toward its base position; overlaps are resolved by moving the offenders apart (vectorised), and the
fibre stays inside the rope. The step amplitude is graded across x from a calm half (`--step-calm`)
to a wavy half (`--step-wavy`), so the wavy half is the localised defect. Waypoints are cubic-spline
interpolated to full depth and every slice is relaxed to <= `--overlap-tol-um`.

Module-level code is stdlib + numpy only (imports on a box without torch); `main` defers every
torch/astra/CUDA import to its body.

Usage (cluster, CUDA interpreter):
    python -m phantom_sim.run_plenoptic_waypoint --out phantom_sim_results/exp7_wp/a30 --step-wavy 5.0
"""

import argparse
import json
from pathlib import Path

import numpy as np

from phantom_sim.run_plenoptic_tilt import max_plenoptic_half_angle_deg
from phantom_sim.run_plenoptic_local import (
    wavy_half_mask,
    region_corr,
)


def relax_move(p, radii, R, tol, iters=250):
    """Push all overlapping pairs apart (vectorised 'move the offenders') and confine centres so the
    fibre surface stays inside the rope of radius R. Resolves to absolute surface gap >= -tol. p (2,N).
    Confinement and overlap are BOTH enforced each iteration, and the loop stops only when neither is
    violated (so a fibre nudged outside the rope with no collision is still pulled back)."""
    rr = radii[:, None] + radii[None, :]
    lim = R - radii
    for _ in range(iters):
        rad = np.hypot(p[0], p[1])
        over = np.clip(rad - lim, 0.0, None)
        if float(over.max()) > 1e-9:
            p = p - over * (p / (rad + 1e-9))  # confine surface inside the rope
        d = p[:, :, None] - p[:, None, :]
        dist = np.hypot(d[0], d[1])
        np.fill_diagonal(dist, np.inf)
        pen = np.clip(-(dist - rr) - tol, 0.0, None)
        if pen.max() <= 1e-3 and float(over.max()) <= 1e-9:
            break
        p = p + 0.6 * ((d / (dist[None] + 1e-9)) * pen[None]).sum(axis=2)
    return p


def worst_overlap(p, radii):
    rr = radii[:, None] + radii[None, :]
    d = p[:, :, None] - p[:, None, :]
    dist = np.hypot(d[0], d[1])
    np.fill_diagonal(dist, np.inf)
    return float(np.clip(-(dist - rr), 0.0, None).max())


def _catmull_rom(Y, depth):
    """Smooth (C1) interpolation of K control points Y (K,2,N) at positions linspace(0,depth-1,K)
    onto 0..depth-1. Pure numpy Catmull-Rom (avoids scipy, whose highspy needs a newer libstdc++
    than some cluster envs ship)."""
    K = Y.shape[0]
    xc = np.linspace(0.0, depth - 1, K)
    xq = np.arange(depth)
    seg = np.clip(np.searchsorted(xc, xq, side="right") - 1, 0, K - 2)  # (depth,)
    u = ((xq - xc[seg]) / (xc[seg + 1] - xc[seg]))[:, None, None]
    p0, p1 = Y[np.clip(seg - 1, 0, K - 1)], Y[seg]
    p2, p3 = Y[np.clip(seg + 1, 0, K - 1)], Y[np.clip(seg + 2, 0, K - 1)]
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * u
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u**2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * u**3
    )


def waypoint_wander_config(
    base_centers,
    radii,
    depth,
    R,
    voxel_um,
    step_calm,
    step_wavy,
    K,
    x_center_um=0.0,
    transition_um=100.0,
    revert=0.14,
    persistence=0.74,
    tol=0.05,
    seed=0,
):
    """Return cfg (depth,2,N) um. Independent per-fibre wander via K non-overlapping waypoints
    (momentum-driven random walk, re-packed at each waypoint) + smooth interpolation."""
    rng = np.random.default_rng(int(seed))
    base = np.asarray(base_centers, float)
    N = base.shape[1]
    x0 = base[0]
    t = np.clip((x0 - x_center_um) / transition_um + 0.5, 0.0, 1.0)
    grade = t * t * (3.0 - 2.0 * t)
    step = (
        step_calm + (step_wavy - step_calm) * grade
    )  # (N,) per-fibre kick size, graded across x

    ctrl = np.empty((K, 2, N))
    ctrl[0] = relax_move(
        base.copy(), radii, R, tol
    )  # clean the base pack to the (tight) tol once
    p = ctrl[0].copy()
    vel = np.zeros((2, N))
    for k in range(1, K):
        vel = persistence * vel + step[None, :] * rng.standard_normal(
            (2, N)
        )  # momentum + random kick
        p_prev = p
        p = relax_move(p + vel - revert * (p - base), radii, R, tol)
        vel = p - p_prev  # conflict-aware momentum (turns on collision)
        ctrl[k] = p
    cfg = _catmull_rom(ctrl, depth)
    for z in range(depth):  # spline can nudge midpoints into contact
        if worst_overlap(cfg[z], radii) > tol:
            cfg[z] = relax_move(cfg[z], radii, R, tol, iters=120)
    return cfg


def draw_radii(n, mean_um, std_um, dmin_um, dmax_um, seed):
    """Per-fibre radii from a normal diameter distribution N(mean, std), clipped to [dmin, dmax]."""
    rng = np.random.default_rng(int(seed))
    return np.clip(
        rng.normal(mean_um / 2.0, std_um / 2.0, n), dmin_um / 2.0, dmax_um / 2.0
    )


def realized_angle_p90(cfg, voxel_um):
    """Per-fibre 90th-percentile deviation angle from the fibre axis (deg), from the smooth path."""
    sl = np.gradient(cfg, voxel_um, axis=0)
    return np.degrees(
        np.arctan(np.percentile(np.hypot(sl[:, 0, :], sl[:, 1, :]), 90, axis=0))
    )


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out", required=True, help="Output directory for recon + masks + truth."
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny fast problem (still needs CUDA to recon).",
    )
    # Waypoint wander
    p.add_argument(
        "--step-calm",
        type=float,
        default=0.35,
        help="Per-waypoint kick (um) in the calm half.",
    )
    p.add_argument(
        "--step-wavy",
        type=float,
        default=5.0,
        help="Per-waypoint kick (um) in the wavy half "
        "(the severity knob; graded smoothly across x). Set equal to --step-calm for the control.",
    )
    p.add_argument(
        "--waypoint-spacing-um",
        type=float,
        default=15.0,
        help="z spacing between wander waypoints (10-20 um). K = depth/spacing + 1.",
    )
    p.add_argument(
        "--persistence",
        type=float,
        default=0.74,
        help="Wander momentum in [0,1): fraction of "
        "the previous step's direction carried into the next (higher = longer smoother sweeps).",
    )
    p.add_argument(
        "--revert",
        type=float,
        default=0.14,
        help="Mean-reversion toward the base position "
        "(spring; higher = the wander oscillates rather than drifts).",
    )
    p.add_argument(
        "--overlap-tol-um",
        type=float,
        default=0.05,
        help="Max allowed surface interpenetration "
        "(um) after re-packing at every slice.",
    )
    p.add_argument(
        "--x-center-um",
        type=float,
        default=0.0,
        help="x where the calm/wavy split is centred.",
    )
    p.add_argument(
        "--transition-um",
        type=float,
        default=100.0,
        help="Smoothstep width of the calm->wavy grade.",
    )
    p.add_argument(
        "--wave-seed",
        type=int,
        default=0,
        help="Seed for the wander (keep fixed across a ladder).",
    )
    # Polydisperse diameters
    p.add_argument("--diam-mean-um", type=float, default=12.0)
    p.add_argument("--diam-std-um", type=float, default=2.0)
    p.add_argument("--diam-min-um", type=float, default=9.0)
    p.add_argument("--diam-max-um", type=float, default=15.0)
    p.add_argument("--diam-seed", type=int, default=3)
    # Phantom / geometry / recon (mirrors run_plenoptic_local)
    p.add_argument("--nz", type=int, default=256)
    p.add_argument("--ny", type=int, default=256)
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--domain-radius-um", type=float, default=215.0)
    p.add_argument("--fvf", type=float, default=0.4)
    p.add_argument(
        "--r-mean-um",
        type=float,
        default=6.0,
        help="Radius for the initial STRAIGHT pack "
        "(positions only; per-fibre diameters are drawn separately).",
    )
    p.add_argument("--voxel-um", type=float, default=2.0)
    p.add_argument("--mu-fibre", type=float, default=1.0)
    p.add_argument("--mu-matrix", type=float, default=0.0)
    p.add_argument("--pack-iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
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
    if args.smoke:
        nz, ny, nx = 16, 64, 64
        n_grid, n_iter, log_every, pad_t, pad_d, pack_iters, domain_radius = (
            3,
            6,
            1,
            8,
            2,
            10,
            55.0,
        )
    else:
        nz, ny, nx = args.nz, args.ny, args.nx
        n_grid, n_iter, log_every = args.n_grid, args.n_iter, args.log_every
        pad_t, pad_d, pack_iters, domain_radius = (
            args.pad_transverse_vox,
            args.pad_depth_vox,
            args.pack_iters,
            args.domain_radius_um,
        )
    voxel_um = args.voxel_um
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from phantom_sim.phantom import pack_bundle, voxelize_config
    import torch
    from phantom_sim.geometry import (
        build_plenoptic_geometry,
        make_plenoptic_projector,
        plenoptic_grid,
        XrayOperator,
        delete_projector,
    )
    from phantom_sim.reconstruct import reconstruct
    from phantom_sim.metrics import (
        psnr,
        ssim,
        register,
        resolution_along_across,
        corr_with_gt,
    )

    # 1. Pack one straight cross-section for the fibre POSITIONS (uniform r; diameters drawn next).
    print(f"Packing straight cross-section (fvf={args.fvf}, r={args.r_mean_um} um) ...")
    cfg0, _ = pack_bundle(
        domain_radius_um=domain_radius,
        fvf=args.fvf,
        r_mean_um=args.r_mean_um,
        r_sigma_um=0.0,
        n_slices=2,
        misalignment="none",
        iters=pack_iters,
        seed=args.seed,
    )
    base_centers = cfg0[0]
    n_fibres = int(base_centers.shape[1])
    radii = draw_radii(
        n_fibres,
        args.diam_mean_um,
        args.diam_std_um,
        args.diam_min_um,
        args.diam_max_um,
        args.diam_seed,
    )

    # 2. Independent momentum waypoint wander, re-packed to the tight tol (fvf preserved, no shrink).
    K = max(4, int(round(nz * voxel_um / args.waypoint_spacing_um)) + 1)
    print(
        f"Waypoint wander: {n_fibres} fibres, K={K} waypoints (~{args.waypoint_spacing_um:.0f} um), "
        f"step_wavy={args.step_wavy}, tol={args.overlap_tol_um} um ..."
    )
    cfg = waypoint_wander_config(
        base_centers,
        radii,
        nz,
        domain_radius,
        voxel_um,
        args.step_calm,
        args.step_wavy,
        K,
        x_center_um=args.x_center_um,
        transition_um=args.transition_um,
        revert=args.revert,
        persistence=args.persistence,
        tol=args.overlap_tol_um,
        seed=args.wave_seed,
    )
    ang = realized_angle_p90(cfg, voxel_um)
    bad = wavy_half_mask(base_centers, x_center_um=args.x_center_um)
    n_bad = int(bad.sum())
    worst = max(worst_overlap(cfg[z], radii) for z in range(nz))
    fvf_eff = float(np.sum(radii**2) / domain_radius**2)
    print(
        f"  wavy-half angle p90 mean={np.mean(ang[bad]):.1f} deg, calm={np.mean(ang[~bad]):.1f} deg; "
        f"worst overlap {worst:.3f} um; diameter {2 * radii.mean():.1f}+/-{2 * radii.std():.1f} um; fvf {fvf_eff:.3f}"
    )

    # 3. Geometry.
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
    max_half_angle = max_plenoptic_half_angle_deg(
        args.source_range_um, args.det_range_um, args.sdd_mm * 1000.0
    )
    print(f"  plenoptic max ray half-angle = {max_half_angle:.2f} deg")
    for axis, npix in (("x", nx), ("y", ny)):
        reach = float(np.max(np.abs(cfg[:, 0 if axis == "x" else 1, :]))) + float(
            radii.max()
        )
        if reach > (npix / 2.0) * voxel_um:
            print(
                f"  WARNING: a fibre reaches |{axis}|={reach:.0f} um, near the frame half-size "
                f"{(npix / 2.0) * voxel_um:.0f} um; widen --n{axis}."
            )

    # 4. Voxelize the full object (per-fibre radii) and the wavy-half fibres for the GT mask.
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
    if n_bad > 0:
        bad_obj = voxelize_config(
            cfg[:, :, bad],
            radii[bad],
            (ny, nx),
            voxel_um=voxel_um,
            axis=0,
            mu_fibre=1.0,
            mu_matrix=0.0,
            supersample=4,
        )
        bad_mask = bad_obj > (0.5 * args.mu_fibre)
    else:
        bad_mask = np.zeros_like(obj, dtype=bool)
    good_mask = (obj > (0.5 * args.mu_fibre)) & (~bad_mask)

    # 5. Single reconstruction.
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
    recon_obj = np.ascontiguousarray(recon[obj_sl])
    delete_projector(A.projector_id)
    del proj, A
    torch.cuda.empty_cache()

    # 6. Metrics.
    c_all = corr_with_gt(recon_obj, obj)
    c_bad = region_corr(recon_obj, obj, bad_mask)
    c_good = region_corr(recon_obj, obj, good_mask)
    res = resolution_along_across(register(recon_obj, obj), fibre_axis=0)
    print(
        f"  corr overall={c_all:.4f}  corr(wavy half)={c_bad}  corr(calm half)={c_good}"
    )

    # 7. Save recon + GT wavy-half mask + the FULL fibre config and radii (exact GT rebuild on laptop).
    np.save(out_dir / "local_recon.npy", recon_obj.astype(np.float32))
    np.save(out_dir / "bad_mask.npy", bad_mask.astype(np.uint8))
    np.savez(
        out_dir / "wave_gt.npz",
        cfg=cfg.astype(np.float32),
        radii=radii,
        base_x=base_centers[0],
        base_y=base_centers[1],
        bad=bad,
        angle_p90_deg=ang,
    )
    meta = {
        "smoke": bool(args.smoke),
        "model": "waypoint",
        "step_calm": float(args.step_calm),
        "step_wavy": float(args.step_wavy),
        "waypoint_spacing_um": float(args.waypoint_spacing_um),
        "n_waypoints": int(K),
        "persistence": float(args.persistence),
        "revert": float(args.revert),
        "overlap_tol_um": float(args.overlap_tol_um),
        "worst_overlap_um": float(worst),
        "x_center_um": float(args.x_center_um),
        "transition_um": float(args.transition_um),
        "wave_seed": int(args.wave_seed),
        "diam_mean_um": float(args.diam_mean_um),
        "diam_std_um": float(args.diam_std_um),
        "diam_min_um": float(args.diam_min_um),
        "diam_max_um": float(args.diam_max_um),
        "diam_realized_mean_um": float(2 * radii.mean()),
        "diam_realized_std_um": float(2 * radii.std()),
        "diam_seed": int(args.diam_seed),
        "fvf_eff": fvf_eff,
        "max_half_angle_deg": float(max_half_angle),
        "n_fibres": int(n_fibres),
        "n_bad": int(n_bad),
        "nz": nz,
        "ny": ny,
        "nx": nx,
        "voxel_um": args.voxel_um,
        "domain_radius_um": args.domain_radius_um,
        "fvf": args.fvf,
        "r_mean_um": args.r_mean_um,
        "mu_fibre": args.mu_fibre,
        "mu_matrix": args.mu_matrix,
        "seed": args.seed,
        "sod_mm": args.sod_mm,
        "sdd_mm": args.sdd_mm,
        "detector_pixel_um": args.detector_pixel_um,
        "source_range_um": args.source_range_um,
        "det_range_um": args.det_range_um,
        "n_grid": args.n_grid,
        "n_iter": args.n_iter,
        "lr": args.lr,
        "tv_weight": args.tv_weight,
        "pad_transverse_vox": pad_t,
        "pad_depth_vox": pad_d,
    }
    try:
        ssim_val = float(ssim(recon_obj, obj))
    except Exception as e:  # skimage may be broken in some cluster envs
        print(f"  (ssim unavailable, saving null: {e})")
        ssim_val = None
    result = {
        "corr_all": float(c_all),
        "corr_bad_region": c_bad,
        "corr_good_region": c_good,
        "psnr": float(psnr(recon_obj, obj)),
        "ssim": ssim_val,
        "resolution": res,
        "peak_deg_wavy_mean": float(np.mean(ang[bad])) if n_bad else None,
        "peak_deg_calm_mean": float(np.mean(ang[~bad])) if (n_fibres - n_bad) else None,
    }
    with open(out_dir / "local_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "result": result}, f, indent=2)
    print(
        f"Saved {out_dir / 'local_metrics.json'}  (recon + bad_mask + wave_gt[cfg,radii] alongside)"
    )


if __name__ == "__main__":
    main()
