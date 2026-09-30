"""CLI: localized-misalignment plenoptic study (exp7, QC demonstrator, follow-up to exp4).

exp4 tilts the WHOLE bundle uniformly and shows reconstruction quality collapses once fibres
tilt past the array's max ray half-angle. exp7 asks the quality-control question: if only PART of
an otherwise aligned bundle is misaligned (as a manufacturing defect would be), does the
reconstruction degrade LOCALLY, so the defect can be spatially localized from the reconstruction
alone (no ground truth)?

The misalignment is modelled as realistic fibre WAVINESS (not a rigid tilt): the cross-section is
split smoothly across x into a calm half (small natural waviness, alpha_good, mostly straight) and
a wavy half (large waviness up to alpha_bad, mostly wavy). Each fibre's x-coordinate undulates
with depth as x(z) = x0 + A(x0) * g(z) * u(z), where u(z) is a shared, band-limited, non-sinusoidal
undulation (tapered to zero at both ends so every fibre re-enters its pultrusion axis) and g(z) an
intermittency envelope (straight runs vs wavy runs). The waviness amplitude A is graded smoothly
across x; both halves share the same straight/wavy pattern and differ only in amplitude. Because
every fibre shares g(z)*u(z) (coherent waviness), neighbours move together and stay
non-overlapping; the driver ASSERTS this
before reconstructing (the rigid-shear model it replaces had no such safeguard and swept fibres
through their neighbours). One plenoptic capture, one reconstruction. The reconstruction and a
ground-truth "wavy-half" mask are saved; the ground-truth-free structure-tensor orientation QC map
is built on the laptop after pull-back and scored against the mask.

Module-level code is stdlib + numpy only, so this imports on a box without torch/astra; `main`
defers every torch/astra/CUDA import to its body (same contract as run_plenoptic_tilt).

Usage (cluster, CUDA interpreter):
    python -m phantom_sim.run_plenoptic_local --smoke --out /tmp/local_smoke
    python -m phantom_sim.run_plenoptic_local --out phantom_sim_results/exp7_local/wb30 --alpha-bad 30
"""

import argparse
import json
from pathlib import Path

import numpy as np

from phantom_sim.run_plenoptic_tilt import max_plenoptic_half_angle_deg


def _gauss1d(x, sigma):
    """1D Gaussian smoothing, reflect-padded. numpy-only (no scipy) so this module stays importable
    with numpy alone."""
    x = np.asarray(x, dtype=np.float64)
    if sigma <= 0:
        return x.copy()
    radius = max(int(4.0 * sigma), 1)
    t = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    xp = np.pad(x, radius, mode="reflect")
    return np.convolve(xp, k, mode="valid")


def shared_wave_profiles(
    depth, voxel_um, seed=0, u_sigma_um=14.0, g_sigma_um=55.0, taper_frac=0.08
):
    """Two shared 1D profiles for the coherent waviness model, along the fibre axis (`depth` slices).

    u(z): a band-limited random undulation (genuinely non-sinusoidal: filtered white noise), tapered
    to zero at both ends so every fibre re-enters its pultrusion axis, and normalised so its steepest
    slope is 1 (max |du/dz| = 1, u in um). g(z): an intermittency envelope in [0, 1] (lower-frequency
    filtered noise, smoothstepped) that is high on wavy runs and low on straight runs. Both are SHARED
    by every fibre (coherent waviness), which is what keeps neighbours from colliding. Deterministic in
    `seed`.
    """
    rng = np.random.default_rng(int(seed))
    u = _gauss1d(rng.standard_normal(depth), u_sigma_um / voxel_um)
    taper = max(int(taper_frac * depth), 4)
    win = np.ones(depth)
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, taper)))
    win[:taper] = ramp
    win[-taper:] = ramp[::-1]
    u = u * win
    u = u - np.linspace(u[0], u[-1], depth)  # exactly zero at both ends
    u = u / (np.max(np.abs(np.gradient(u, voxel_um))) + 1e-12)  # max |du/dz| = 1
    g = _gauss1d(rng.standard_normal(depth), g_sigma_um / voxel_um)
    g = (g - g.min()) / (np.ptp(g) + 1e-12)
    g = g * g * (3.0 - 2.0 * g)  # smoothstep -> clearer runs
    return u, g


def graded_wave_config(
    base_centers,
    depth,
    voxel_um,
    alpha_good_deg,
    alpha_bad_deg,
    u,
    g,
    transition_um=100.0,
    x_center_um=0.0,
    u_y=None,
    g_y=None,
):
    """Coherent, intermittent waviness with the AMPLITUDE graded smoothly across x from the calm
    half (peak angle alpha_good) to the wavy half (peak angle alpha_bad). Returns
    (cfg (depth, 2, N) in um, alpha_target_deg (N), peak_angle_deg (N)).

    base_centers: (2, N) packed fibre centres in um, index 0 = x, 1 = y. Every fibre shears in x by
    A(x0) * s_x(z), where s_x(z) = g(z) * u(z) is a SHARED, non-sinusoidal, intermittent profile
    (straight runs where g is low, wavy runs where g is high) and A(x0) is a smoothstep ramp across
    x. When `u_y`/`g_y` are given, the fibre ALSO shears in y by A(x0) * s_y(z) with s_y = g_y * u_y
    a SECOND, independent shared profile (isotropic 3D wander); leave them None for the original
    x-only model (y untouched). The amplitude is a SINGLE global normalisation of the COMBINED slope
    magnitude sqrt(s_x'^2 + s_y'^2) (never a per-fibre division, which would blow up for a fibre
    whose shape happens to be flat) so the peak local orientation angle AWAY FROM THE FIBRE AXIS is
    alpha_good on the calm side and alpha_bad on the wavy side; both halves share the same
    straight/wavy pattern and differ only in amplitude. Because all fibres share s_x (and s_y),
    neighbours move together and stay non-overlapping. A uniform angle (alpha_good == alpha_bad)
    gives a defect-free control whose deviation map stays flat. The transition is a smoothstep of
    width `transition_um` centred at `x_center_um`.
    """
    base = np.asarray(base_centers, dtype=np.float64)  # (2, N)
    x0 = base[0]
    t = np.clip((x0 - x_center_um) / transition_um + 0.5, 0.0, 1.0)
    s_grade = t * t * (3.0 - 2.0 * t)  # smoothstep: 0 calm -> 1 wavy
    shape_x = g * u  # (depth,) shared intermittent profile
    slope_x = np.gradient(shape_x, voxel_um)
    if u_y is not None:
        shape_y = g_y * u_y  # second, independent shared profile
        slope_y = np.gradient(shape_y, voxel_um)
        max_slope = (
            float(np.max(np.hypot(slope_x, slope_y))) + 1e-12
        )  # combined x+y peak slope
    else:
        shape_y = None
        max_slope = float(np.max(np.abs(slope_x))) + 1e-12
    tan_good, tan_bad = (
        np.tan(np.radians(alpha_good_deg)),
        np.tan(np.radians(alpha_bad_deg)),
    )
    tan_target = (
        tan_good + (tan_bad - tan_good) * s_grade
    )  # (N,), interpolate in tan-space
    amp = tan_target / max_slope  # (N,), bounded (global normalisation)
    cfg = np.repeat(base[None, :, :], depth, axis=0)  # (depth, 2, N)
    disp_x = amp[None, :] * shape_x[:, None]  # (depth, N), um
    cfg[:, 0, :] = cfg[:, 0, :] + disp_x
    grad2 = (
        np.gradient(disp_x, voxel_um, axis=0) ** 2
    )  # (depth, N) axial slope^2, x part
    if shape_y is not None:
        disp_y = amp[None, :] * shape_y[:, None]
        cfg[:, 1, :] = cfg[:, 1, :] + disp_y
        grad2 = grad2 + np.gradient(disp_y, voxel_um, axis=0) ** 2
    alpha_deg = np.degrees(np.arctan(tan_target))
    peak = np.degrees(
        np.arctan(np.max(np.sqrt(grad2), axis=0))
    )  # peak COMBINED deviation from axis
    return cfg, alpha_deg, peak


def wavy_half_mask(base_centers, x_center_um=0.0):
    """The defect (wavy) half: fibres whose packed centre has x0 > x_center_um. Returns bool (N,)."""
    return np.asarray(base_centers, dtype=np.float64)[0] > x_center_um


def min_surface_gap(cfg, radii):
    """Minimum fibre surface-to-surface gap over all slices (um): min over pairs and z of
    (centre distance - r_i - r_j). Negative means interpenetration. cfg (depth, 2, N)."""
    r = np.asarray(radii, dtype=np.float64)
    rr = r[:, None] + r[None, :]
    m = np.inf
    for z in range(cfg.shape[0]):
        p = cfg[z].T  # (N, 2)
        d = p[:, None, :] - p[None, :, :]
        dist = np.hypot(d[..., 0], d[..., 1]) - rr
        np.fill_diagonal(dist, np.inf)
        mz = float(dist.min())
        if mz < m:
            m = mz
    return m


def assert_no_overlap(cfg, radii, tol_um=1e-6):
    """Raise if any two fibres interpenetrate at any slice (the safeguard the shear model lacked).
    Touching (surface gap == 0) is allowed. Returns the minimum surface gap (um) when safe."""
    gap = min_surface_gap(cfg, radii)
    if gap < -tol_um:
        raise ValueError(
            "fibre overlap: min surface gap %.3f um < 0. Widen --transition-um or lower "
            "--alpha-bad." % gap
        )
    return gap


def max_new_overlap(cfg, base_centers, radii):
    """Worst NEW interpenetration (um) the wavy config adds beyond the straight base pack, per pair.

    The fibre packer routinely leaves benign near-contacts in the straight cross-section (pairs a
    fraction of a voxel into each other); under union (np.maximum) voxelisation these are cosmetic.
    So the overlap safeguard must not punish them, only genuine collisions the waviness creates.

    For every fibre pair, compare its minimum surface gap over all wavy slices against the gap it
    already had in the base pack, with each pair's baseline CLAMPED at 0 so a pre-existing overlap is
    grandfathered (it may stay as bad, not get worse). Returns max over pairs of
    (clamped_base_gap - wavy_gap): <= 0 means no pair is worse than its own baseline, and a positive
    value is exactly how much deeper the waviness drove some pair than it is allowed to. This is the
    per-pair discriminator a single scalar ``min_surface_gap`` cannot provide -- it separates a
    grandfathered pack contact from a real new sweep. cfg (depth, 2, N); base_centers (2, N)."""
    r = np.asarray(radii, dtype=np.float64)
    rr = r[:, None] + r[None, :]
    b = np.asarray(base_centers, dtype=np.float64)
    d0 = b[:, :, None] - b[:, None, :]
    base_gap = np.hypot(d0[0], d0[1]) - rr  # (N, N) straight-pack surface gaps
    np.fill_diagonal(base_gap, np.inf)
    allowed = np.minimum(base_gap, 0.0)  # grandfather pre-existing overlaps
    worst = np.full_like(base_gap, np.inf)
    for z in range(cfg.shape[0]):
        p = cfg[z].T  # (N, 2)
        d = p[:, None, :] - p[None, :, :]
        worst = np.minimum(worst, np.hypot(d[..., 0], d[..., 1]) - rr)
    np.fill_diagonal(worst, np.inf)
    return float(np.max(allowed - worst))


def region_corr(recon_obj, obj, mask3d):
    """Pearson correlation of recon vs ground truth over a boolean voxel mask (a scalar QC readout).
    Returns None if the mask selects too few voxels or has no variance."""
    if mask3d is None or int(mask3d.sum()) < 8:
        return None
    a = recon_obj[mask3d].ravel().astype(np.float64)
    b = obj[mask3d].ravel().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def wave_metrics_meta(
    args, n_fibres, n_bad, max_half_angle, transition_used_um, nz, ny, nx, pad_t, pad_d
):
    """Pure schema assembly for local_metrics.json meta. No torch/astra."""
    return {
        "smoke": bool(args.smoke),
        "model": "waviness",
        "alpha_good_deg": float(args.alpha_good),
        "alpha_bad_deg": float(args.alpha_bad),
        "wave_u_sigma_um": float(args.wave_u_sigma_um),
        "wave_g_sigma_um": float(args.wave_g_sigma_um),
        "transition_um": float(transition_used_um),
        "x_center_um": float(args.x_center_um),
        "overlap_tol_um": float(args.overlap_tol_um),
        "wave_seed": int(args.wave_seed),
        "wave_x_only": bool(args.wave_x_only),
        "wave_seed_y": None
        if args.wave_x_only
        else int(args.wave_seed + args.wave_seed_y_offset),
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


def build_parser():
    p = argparse.ArgumentParser(
        description="Localized-misalignment plenoptic study (exp7): a calm half and a wavy half of "
        "an otherwise aligned bundle, modelled as coherent fibre waviness, for the QC / "
        "defect-localization demonstrator."
    )
    p.add_argument("--out", required=True)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny bundle/grid/iters for a fast cluster smoke test.",
    )
    # Waviness specification (the exp7-specific knobs).
    p.add_argument(
        "--alpha-good",
        type=float,
        default=3.5,
        help="Peak local waviness angle (deg) of the calm half (natural waviness). "
        "Default 3.5.",
    )
    p.add_argument(
        "--alpha-bad",
        type=float,
        default=30.0,
        help="Peak local waviness angle (deg) of the wavy half (the defect). Set equal "
        "to --alpha-good for the defect-free control. Default 30 (>> the ~9.66 deg "
        "array half-angle).",
    )
    p.add_argument(
        "--wave-u-sigma-um",
        type=float,
        default=14.0,
        help="Smoothing scale of the shared undulation u(z) (sets its wavelength band).",
    )
    p.add_argument(
        "--wave-g-sigma-um",
        type=float,
        default=55.0,
        help="Smoothing scale of the intermittency envelope g(z) (straight/wavy run length).",
    )
    p.add_argument(
        "--transition-um",
        type=float,
        default=100.0,
        help="Smoothstep transition width across x from calm to wavy. Auto-widened if a "
        "config would add new overlap beyond --overlap-tol-um.",
    )
    p.add_argument(
        "--overlap-tol-um",
        type=float,
        default=1.0,
        help="Max NEW interpenetration (um) the waviness may add beyond the base pack, "
        "per pair (pre-existing packer contacts are grandfathered). Default 1.0 "
        "(~half a 2 um voxel): forbids any real sweep while ignoring cosmetic, "
        "sub-voxel pack contacts. The transition is auto-widened until met.",
    )
    p.add_argument(
        "--x-center-um",
        type=float,
        default=0.0,
        help="x where the calm/wavy transition is centred (wavy half is x0 > this).",
    )
    p.add_argument(
        "--wave-seed",
        type=int,
        default=0,
        help="Seed for the shared u(z)/g(z) profiles (keep fixed across a ladder so only "
        "alpha_bad changes).",
    )
    p.add_argument(
        "--wave-x-only",
        action="store_true",
        help="Disable the independent y-waviness profile and use the original x-only "
        "model (default is isotropic x+y wander).",
    )
    p.add_argument(
        "--wave-seed-y-offset",
        type=int,
        default=1,
        help="Offset added to --wave-seed for the independent y-waviness profile "
        "(so the y undulation is uncorrelated with x). Default 1.",
    )
    # Phantom / geometry / recon (mirrors run_plenoptic_tilt defaults).
    p.add_argument("--nz", type=int, default=256)
    p.add_argument("--ny", type=int, default=384)
    p.add_argument("--nx", type=int, default=384)
    p.add_argument("--domain-radius-um", type=float, default=120.0)
    p.add_argument("--fvf", type=float, default=0.4)
    p.add_argument("--r-mean-um", type=float, default=6.0)
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
        nz, ny, nx = 16, 48, 48
        n_grid, n_iter, log_every = 3, 6, 1
        pad_t, pad_d = 6, 2
        pack_iters = 10
        domain_radius = 40.0
    else:
        nz, ny, nx = args.nz, args.ny, args.nx
        n_grid, n_iter, log_every = args.n_grid, args.n_iter, args.log_every
        pad_t, pad_d = args.pad_transverse_vox, args.pad_depth_vox
        pack_iters = args.pack_iters
        domain_radius = args.domain_radius_um
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

    # 1. Pack one straight cross-section (identical seed => same fibres as exp4).
    print(f"Packing straight cross-section (fvf={args.fvf}, r={args.r_mean_um} um) ...")
    cfg0, radii = pack_bundle(
        domain_radius_um=domain_radius,
        fvf=args.fvf,
        r_mean_um=args.r_mean_um,
        r_sigma_um=0.0,
        n_slices=2,
        misalignment="none",
        iters=pack_iters,
        seed=args.seed,
    )
    base_centers = cfg0[0]  # (2, N)
    n_fibres = int(radii.shape[0])

    # 2. Coherent graded waviness: calm half (alpha_good) vs wavy half (alpha_bad), overlap-safe.
    #    Independent x and y shared profiles (isotropic 3D wander) unless --wave-x-only.
    u, g = shared_wave_profiles(
        nz,
        voxel_um,
        seed=args.wave_seed,
        u_sigma_um=args.wave_u_sigma_um,
        g_sigma_um=args.wave_g_sigma_um,
    )
    if args.wave_x_only:
        u_y = g_y = None
    else:
        u_y, g_y = shared_wave_profiles(
            nz,
            voxel_um,
            seed=args.wave_seed + args.wave_seed_y_offset,
            u_sigma_um=args.wave_u_sigma_um,
            g_sigma_um=args.wave_g_sigma_um,
        )
    # The packer routinely leaves benign near-contacts in the straight cross-section (fibres a
    # fraction of a voxel into each other); under union voxelisation these are cosmetic. The
    # safeguard is therefore PER-PAIR and relative: the waviness may not drive any pair more than
    # --overlap-tol-um deeper than it already sat in the base pack (pre-existing overlaps
    # grandfathered). A single scalar min gap conflates a grandfathered pack contact with a real
    # new collision; max_new_overlap separates them. Auto-widen the transition until safe.
    base_gap = min_surface_gap(base_centers[None, :, :], radii)
    if base_gap < -1e-6:
        print(
            f"  NOTE: base pack min surface gap {base_gap:.3f} um (pre-existing packer contacts, "
            f"grandfathered); the safeguard only forbids NEW waviness collisions deeper than "
            f"{args.overlap_tol_um:.2f} um."
        )
    transition = args.transition_um
    for _attempt in range(9):
        cfg, alpha_deg, peak_deg = graded_wave_config(
            base_centers,
            nz,
            voxel_um,
            args.alpha_good,
            args.alpha_bad,
            u,
            g,
            transition_um=transition,
            x_center_um=args.x_center_um,
            u_y=u_y,
            g_y=g_y,
        )
        excess = max_new_overlap(cfg, base_centers, radii)
        if excess <= args.overlap_tol_um:
            break
        new_t = transition * 1.25
        print(
            f"  waviness adds {excess:.3f} um new interpenetration (> tol {args.overlap_tol_um:.2f} "
            f"um) at transition {transition:.0f} um; widening to {new_t:.0f} um"
        )
        transition = new_t
    else:
        raise ValueError(
            "waviness still adds > %.2f um of new interpenetration after widening the "
            "transition to %.0f um; lower --alpha-bad or raise --overlap-tol-um."
            % (args.overlap_tol_um, transition)
        )
    gap = min_surface_gap(cfg, radii)

    bad = wavy_half_mask(base_centers, x_center_um=args.x_center_um)
    n_bad = int(bad.sum())
    print(
        f"  {n_fibres} fibres; wavy half = {n_bad} fibres (alpha_bad={args.alpha_bad} deg), "
        f"calm half alpha_good={args.alpha_good} deg; transition {transition:.0f} um, "
        f"worst new overlap {excess:.3f} um, min surface gap {gap:.3f} um (base {base_gap:.3f} um)"
    )

    # 3. Geometry (single plenoptic capture).
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

    # FOV check: waviness is a bounded oscillation, but flag it if a fibre could leave the frame.
    max_abs_x = float(np.max(np.abs(cfg[:, 0, :]))) + float(radii.max())
    max_abs_y = float(np.max(np.abs(cfg[:, 1, :]))) + float(radii.max())
    if max_abs_x > (nx / 2.0) * voxel_um:
        print(
            f"  WARNING: a wavy fibre reaches |x|={max_abs_x:.0f} um, near the frame half-width "
            f"{(nx / 2.0) * voxel_um:.0f} um; widen --nx."
        )
    if max_abs_y > (ny / 2.0) * voxel_um:
        print(
            f"  WARNING: a wavy fibre reaches |y|={max_abs_y:.0f} um, near the frame half-height "
            f"{(ny / 2.0) * voxel_um:.0f} um; widen --ny."
        )

    # 4. Voxelize the full object and (separately) only the wavy-half fibres, for the GT mask.
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

    # 6. Metrics: overall corr, and corr inside the wavy vs calm halves (the QC contrast).
    c_all = corr_with_gt(recon_obj, obj)
    c_bad = region_corr(recon_obj, obj, bad_mask)
    c_good = region_corr(recon_obj, obj, good_mask)
    res = resolution_along_across(register(recon_obj, obj), fibre_axis=0)
    print(
        f"  corr overall={c_all:.4f}  corr(wavy half)={c_bad}  corr(calm half)={c_good}"
    )

    # 7. Save recon + GT wavy-half mask + per-fibre waviness truth (QC map is built on the laptop).
    np.save(out_dir / "local_recon.npy", recon_obj.astype(np.float32))
    np.save(out_dir / "bad_mask.npy", bad_mask.astype(np.uint8))
    save_kw = dict(
        base_x=base_centers[0],
        base_y=base_centers[1],
        alpha_deg=alpha_deg,
        peak_deg=peak_deg,
        bad=bad,
        u=u,
        g=g,
    )
    if u_y is not None:
        save_kw["u_y"], save_kw["g_y"] = (
            u_y,
            g_y,
        )  # independent y-waviness profile (for GT replay)
    np.savez(out_dir / "wave_gt.npz", **save_kw)
    meta = wave_metrics_meta(
        args, n_fibres, n_bad, max_half_angle, transition, nz, ny, nx, pad_t, pad_d
    )
    result = {
        "corr_all": float(c_all),
        "corr_bad_region": c_bad,
        "corr_good_region": c_good,
        "psnr": float(psnr(recon_obj, obj)),
        "ssim": float(ssim(recon_obj, obj)),
        "resolution": res,
        "peak_deg_wavy_mean": float(np.mean(peak_deg[bad])) if n_bad else None,
        "peak_deg_calm_mean": float(np.mean(peak_deg[~bad]))
        if (n_fibres - n_bad)
        else None,
    }
    with open(out_dir / "local_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "result": result}, f, indent=2)
    print(
        f"Saved {out_dir / 'local_metrics.json'}  (recon + bad_mask + wave_gt alongside)"
    )


if __name__ == "__main__":
    main()
