"""CLI: point-spread sweep vs array half-angle. Longitudinal (z) resolution as a function of
the plenoptic array half-angle (follow-up to run_plenoptic_psf).

run_plenoptic_psf measured the point-spread once, at the current array (max ray half-angle 9.66 deg), and
found the z FWHM (24 um) far coarser than the transverse FWHM (5 um). This driver repeats that
measurement while widening the array: for a sweep of target half-angles it sets the detector-array
span so the most oblique ray makes that angle with the optical axis, images the same central
sphere, reconstructs, and records the z and transverse FWHM. The plenoptic depth channel is
parallax-limited, so the prediction is dz ~ dx / tan(theta): the anisotropy should fall as
cot(theta) and reach isotropy near 45 deg. This run tests that law directly and shows the
detector span each angle costs.

The half-angle is set by the SOURCE array as seen from the sample, theta = atan(source_range / SOD),
so widening the aperture means opening the source span. The detector span is then not free: a
source at lateral s casts the sample's shadow at -s * (SDD - SOD) / SOD, so the sub-aperture
detector tiles must sit at exactly that magnification or they image empty space.
    source_range = SOD * tan(theta)
    det_range    = SDD * tan(theta) - source_range   ( = source_range * (SDD - SOD) / SOD )
Both are half-spans (plenoptic_grid spans -range .. +range). The two expressions for det_range
agree only while source_range tracks theta. Holding the source span fixed and growing det_range
alone slides every sub-aperture off the shadow it is meant to catch, and the reconstruction
silently loses its views rather than failing: past a mismatch of one tile half-width only the
central sub-aperture still sees the sample, and since that one has src = det = 0 for every theta,
every angle then reconstructs the same single projection and returns identical metrics.

SOD and SDD are held fixed, so theta is the only variable. The phantom, voxel size, iteration
count and TV-along-z prior are identical to run_plenoptic_psf so the 9.66 deg point reproduces the
theme-5 anchor as a built-in control (SOD * tan(9.663 deg) = 22000 um, the current array).

Module-level code is stdlib + numpy only, so this module imports on a box without torch/astra.
`main` defers every torch/astra/CUDA import to its body.

Usage (cluster, CUDA interpreter):
    python -m phantom_sim.run_plenoptic_psf_sweep --smoke --out /tmp/psf_sweep_smoke
    python -m phantom_sim.run_plenoptic_psf_sweep --out phantom_sim_results/exp5b_psf_sweep
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from phantom_sim.run_plenoptic_psf import (
    make_sphere_phantom,
    axis_profile,
    fwhm,
    mtf,
    psf_entry,
)


def source_range_for_half_angle(theta_deg, sod_um):
    """Source-array half-span (um) whose outermost source subtends theta_deg at the sample.

    This is what actually sets the plenoptic aperture: the parallax baseline is the spread of
    source positions seen from the sample, theta = atan(source_range / SOD). The current array
    (22000 um at SOD 129.24 mm) is the 9.663 deg control point. Pure, no torch.
    """
    return sod_um * math.tan(math.radians(theta_deg))


def det_range_for_half_angle(theta_deg, source_range_um, sdd_um):
    """Detector-array half-span (um) that makes the most oblique plenoptic ray subtend theta_deg.

    Inverse of run_plenoptic_tilt.max_plenoptic_half_angle_deg: the most oblique ray spans
    (source_range + det_range) over the SDD, so
        det_range = SDD * tan(theta) - source_range.
    `source_range_um` MUST be the source half-span for this same theta (see
    source_range_for_half_angle), otherwise the returned span is geometrically inconsistent:
    the tiles land where the sample's shadow is not. With the matched source span this reduces
    to the magnification relation det_range = source_range * (SDD - SOD) / SOD.
    Raises ValueError if the requested angle is so small the detector span would be non-positive
    (theta below atan(source_range / SDD)).
    """
    det_range = sdd_um * math.tan(math.radians(theta_deg)) - source_range_um
    if det_range <= 0.0:
        raise ValueError(
            "half-angle %.3f deg needs det_range <= 0 with source_range=%.0f um, SDD=%.0f um; "
            "the source span alone already exceeds that angle"
            % (theta_deg, source_range_um, sdd_um)
        )
    return det_range


def sweep_entry(
    theta_deg,
    det_range_um,
    det_span_mm,
    axis_entries,
    transverse_um,
    ratio,
    corr,
    source_range_um=0.0,
):
    """One half-angle point of the sweep. Pure, no torch. `axis_entries` are psf_entry dicts.

    `det_range_um` and `source_range_um` are half-spans (the plenoptic_grid arguments);
    `det_span_mm` is the full detector width the angle costs, i.e. 2 * det_range_um.
    """
    by = {e["axis"]: e for e in axis_entries}
    return {
        "theta_deg": float(theta_deg),
        "source_range_um": float(source_range_um),
        "source_span_mm": float(2.0 * source_range_um / 1000.0),
        "det_range_um": float(det_range_um),
        "det_span_mm": float(det_span_mm),
        "fwhm_z_um": float(by["z"]["fwhm_um"]),
        "fwhm_x_um": float(by["x"]["fwhm_um"]),
        "fwhm_y_um": float(by["y"]["fwhm_um"]),
        "fwhm_transverse_um": float(transverse_um),
        "z_to_transverse_ratio": float(ratio),
        "corr_with_gt": float(corr),
        "psf": axis_entries,
    }


def assemble_psf_sweep_metrics(entries, meta):
    """Pure schema assembly for psf_sweep_metrics.json. No torch/astra."""
    return {"meta": meta, "sweep": entries}


def default_half_angles():
    """The requested sweep; the current 9.66 deg array point is injected separately as a control."""
    return [5.0, 7.0, 12.0, 15.0, 17.0, 21.0, 25.0, 30.0, 35.0, 40.0, 45.0]


def resolve_half_angles(
    requested, source_range_um, current_det_range_um, sdd_um, include_current=True
):
    """Sorted, de-duplicated half-angle list, optionally with the current array's half-angle.

    The current point (det_range = current_det_range_um) reproduces the theme-5 anchor, so it is
    a within-run control that the pipeline still gives ~24/5. Pure, no torch.
    """
    angles = list(requested)
    if include_current:
        cur = math.degrees(math.atan((source_range_um + current_det_range_um) / sdd_um))
        if not any(abs(cur - a) < 0.05 for a in angles):
            angles.append(round(cur, 2))
    return sorted(set(float(a) for a in angles))


def build_parser():
    p = argparse.ArgumentParser(
        description="Point-spread sweep vs array half-angle: longitudinal resolution vs aperture."
    )
    p.add_argument("--out", required=True)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny volume/grid/iters and 3 angles for a fast cluster smoke test.",
    )
    p.add_argument(
        "--half-angles",
        type=float,
        nargs="+",
        default=None,
        help="Target max ray half-angles in degrees (default: 5..45 requested set).",
    )
    p.add_argument(
        "--no-include-current",
        dest="include_current",
        action="store_false",
        help="Do not inject the current array half-angle (theme-5 control point).",
    )
    p.add_argument("--nz", type=int, default=256)
    p.add_argument("--ny", type=int, default=256)
    p.add_argument("--nx", type=int, default=256)
    p.add_argument(
        "--radius-vox", type=float, default=2.0, help="Sphere radius in voxels."
    )
    p.add_argument("--voxel-um", type=float, default=2.0)
    p.add_argument("--mu-fibre", type=float, default=1.0)
    p.add_argument("--mu-matrix", type=float, default=0.0)
    p.add_argument("--sod-mm", type=float, default=129.24)
    p.add_argument("--sdd-mm", type=float, default=1171.57)
    p.add_argument("--detector-pixel-um", type=float, default=55.0)
    p.add_argument(
        "--source-range-um",
        type=float,
        default=22000.0,
        help="Source half-span of the CURRENT array. Reference only: it fixes the "
        "injected control half-angle. Each swept angle sets its own source span "
        "from SOD * tan(theta).",
    )
    p.add_argument(
        "--current-det-range-um",
        type=float,
        default=177400.0,
        help="Detector half-span of the current array; sets the injected control "
        "half-angle. Must stay the magnification partner of --source-range-um.",
    )
    p.add_argument("--n-grid", type=int, default=21)
    p.add_argument("--super-sampling", type=int, default=2)
    p.add_argument("--pad-transverse-vox", type=int, default=32)
    p.add_argument("--pad-depth-vox", type=int, default=32)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument(
        "--tv-weight",
        type=float,
        default=0.5,
        help="TV weight along z (the fibre axis); matches run_plenoptic_psf.",
    )
    p.add_argument(
        "--save-recons",
        action="store_true",
        help="Also save each angle's cropped recon .npy (large; off by default).",
    )
    p.add_argument("--log-every", type=int, default=10)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke:
        nz, ny, nx = 32, 32, 32
        n_grid, n_iter, log_every = 3, 6, 1
        pad_t, pad_d = 4, 4
        radius = 2.0
        requested = [5.0, 25.0]
    else:
        nz, ny, nx = args.nz, args.ny, args.nx
        n_grid, n_iter, log_every = args.n_grid, args.n_iter, args.log_every
        pad_t, pad_d = args.pad_transverse_vox, args.pad_depth_vox
        radius = args.radius_vox
        requested = (
            args.half_angles if args.half_angles is not None else default_half_angles()
        )

    voxel_um = args.voxel_um
    sdd_um = args.sdd_mm * 1000.0
    sod_um = args.sod_mm * 1000.0
    thetas = resolve_half_angles(
        requested,
        args.source_range_um,
        args.current_det_range_um,
        sdd_um,
        include_current=args.include_current,
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from phantom_sim.geometry import (
        build_plenoptic_geometry,
        make_plenoptic_projector,
        plenoptic_grid,
        XrayOperator,
        delete_projector,
    )
    from phantom_sim.reconstruct import reconstruct
    from phantom_sim.metrics import corr_with_gt

    obj = make_sphere_phantom(nz, ny, nx, radius, args.mu_fibre, args.mu_matrix)
    vol = np.pad(obj, ((pad_d, pad_d), (pad_t, pad_t), (pad_t, pad_t)))
    padded_shape = vol.shape
    obj_sl = (
        slice(pad_d, pad_d + nz),
        slice(pad_t, pad_t + ny),
        slice(pad_t, pad_t + nx),
    )

    detector_z = sdd_um / voxel_um
    pixel_size = args.detector_pixel_um / voxel_um
    vol_z0 = sod_um / voxel_um
    num_imgs = n_grid * n_grid
    sinogram_size = [padded_shape[1], padded_shape[2], num_imgs]
    volt = torch.tensor(vol, device="cuda")

    def corr_metric(recon_padded_np):
        return corr_with_gt(np.ascontiguousarray(recon_padded_np[obj_sl]), obj)

    print(
        "half-angle sweep (SOD=%.2f mm, SDD=%.2f mm fixed; source and detector spans both "
        "scale with theta):" % (args.sod_mm, args.sdd_mm)
    )
    entries = []
    for theta in thetas:
        # The aperture is opened at the SOURCE array; the detector span then follows from
        # magnification. Scaling only the detector span walks the sub-apertures off the sample's
        # shadow and quietly reduces the run to its central view (see module docstring).
        source_range_um = source_range_for_half_angle(theta, sod_um)
        det_range_um = det_range_for_half_angle(theta, source_range_um, sdd_um)
        det_span_mm = (
            2.0 * det_range_um / 1000.0
        )  # full detector width; det_range is a half-span
        src, det = plenoptic_grid(
            source_range_um / voxel_um, det_range_um / voxel_um, n=n_grid
        )
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
        proj = A(volt).detach()
        torch.cuda.empty_cache()

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

        axis_entries = []
        for name, ax in (("z", 0), ("y", 1), ("x", 2)):
            prof = axis_profile(recon_obj, ax)
            w = fwhm(prof, spacing=voxel_um)
            freqs, m = mtf(prof, spacing=voxel_um)
            axis_entries.append(psf_entry(name, w, prof, freqs, m))
        by = {e["axis"]: e["fwhm_um"] for e in axis_entries}
        transverse = 0.5 * (by["x"] + by["y"])
        ratio = (by["z"] / transverse) if transverse else 0.0
        corr = float(corr_with_gt(recon_obj, obj))
        entries.append(
            sweep_entry(
                theta,
                det_range_um,
                det_span_mm,
                axis_entries,
                transverse,
                ratio,
                corr,
                source_range_um=source_range_um,
            )
        )
        if args.save_recons:
            np.save(
                out_dir / ("psf_recon_%05.2f.npy" % theta), recon_obj.astype(np.float32)
            )
        print(
            "  theta %5.2f deg (src span %7.1f mm, det span %8.1f mm): z-FWHM %6.2f um  "
            "transverse %5.2f um  ratio %5.2f  corr %.4f"
            % (
                theta,
                2.0 * source_range_um / 1000.0,
                det_span_mm,
                by["z"],
                transverse,
                ratio,
                corr,
            )
        )

    del volt
    torch.cuda.empty_cache()

    meta = {
        "smoke": bool(args.smoke),
        "voxel_um": voxel_um,
        "nz": nz,
        "ny": ny,
        "nx": nx,
        "radius_vox": radius,
        "mu_fibre": args.mu_fibre,
        "mu_matrix": args.mu_matrix,
        "sod_mm": args.sod_mm,
        "sdd_mm": args.sdd_mm,
        "detector_pixel_um": args.detector_pixel_um,
        "source_range_um": args.source_range_um,
        "current_det_range_um": args.current_det_range_um,
        "n_grid": n_grid,
        "n_iter": n_iter,
        "lr": args.lr,
        "tv_weight": args.tv_weight,
        "pad_transverse_vox": pad_t,
        "pad_depth_vox": pad_d,
        "half_angles_deg": [float(t) for t in thetas],
    }
    metrics = assemble_psf_sweep_metrics(entries, meta)
    with open(out_dir / "psf_sweep_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved %s" % (out_dir / "psf_sweep_metrics.json"))
    print(
        "  z-FWHM vs theta: "
        + ", ".join("%.0f:%.1f" % (e["theta_deg"], e["fwhm_z_um"]) for e in entries)
    )


if __name__ == "__main__":
    main()
