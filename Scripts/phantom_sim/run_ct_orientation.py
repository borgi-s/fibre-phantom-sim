"""CLI: Experiment 1, CT rotation-axis vs beam-axis fibre orientation.

Packs ONE fibre bundle, voxelises the SAME packing along axis=0 (fibres
parallel to the scanner's rotation axis) and axis=1 (fibres transverse to
the rotation axis, lying in the rotation plane), simulates an identical
full-360-degree CT scan for each orientation, reconstructs with a shared
iterative TV solver, registers to ground truth, and scores PSNR, SSIM, and
directional (along vs across fibre) resolution. Does not assert which
orientation wins; that is the experiment's result.

Array-axis convention: `voxelize_config`'s `axis` argument only moves where
the FIBRE geometry sits inside the voxel array; array dim 0 is ALWAYS the
scanner's fixed physical rotation axis (ASTRA Z), for both orientations. So
axis=0 volumes have fibres parallel to the rotation axis (shape (Z, ny, nx),
Z = n_slices), and axis=1 volumes have fibres transverse to it (shape
(ny, Z, nx)); either way, array dim 0 feeds the CT geometry's rotation axis
and array dim -1 (nx) is the other in-plane transverse direction.

This-box vs cluster split: `simulate_projections` and `reconstruct` require
a CUDA GPU with ASTRA's cuda3d projector, absent on this box. Every
astra/torch-cuda-touching import is deferred to inside `main`, so
`import phantom_sim.run_ct_orientation` succeeds on this box without astra
or a CUDA torch build; module-level imports are stdlib + numpy + the
import-safe `phantom_sim.phantom` functions (`pack_bundle`'s own torch
import is itself lazy, inside the function body).

Usage (cluster, using its CUDA-enabled interpreter):
    python -m phantom_sim.run_ct_orientation --out DataFiles/phantom_sim/exp1
    python -m phantom_sim.run_ct_orientation --smoke --out /tmp/exp1_smoke

Outputs under --out:
    ct_axis0.npy / ct_axis0.meta.json   axis=0 (rotation-aligned) recon + meta
    ct_axis1.npy / ct_axis1.meta.json   axis=1 (beam/transverse) recon + meta
    ct_metrics.json                     {"axis0": {...}, "axis1": {...}}
"""
import argparse
import json
from pathlib import Path

import numpy as np

from phantom_sim.phantom import pack_bundle, voxelize_config


def write_meta(path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def build_parser():
    p = argparse.ArgumentParser(
        description="Experiment 1: CT rotation-axis vs beam-axis fibre orientation.")
    p.add_argument("--out", required=True, help="Output directory (created if missing).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny volume, few angles, few packing/recon iters, for a fast "
                        "cluster smoke test. Overrides the study-scale defaults below.")

    # packing params (see phantom.pack_bundle), study-scale defaults match make_phantom.py
    p.add_argument("--domain-radius-um", type=float, default=100.0,
                   help="Packing domain (bundle cross-section) radius, um.")
    p.add_argument("--fvf", type=float, default=0.3,
                   help="Fibre volume fraction as a 0-1 fraction.")
    p.add_argument("--r-mean-um", type=float, default=3.5, help="Mean fibre radius, um.")
    p.add_argument("--r-sigma-um", type=float, default=0.0,
                   help="Fibre radius std dev, um (0 = uniform radii).")
    p.add_argument("--n-slices", type=int, default=64,
                   help="Number of packing slices along the fibre axis.")
    p.add_argument("--misalignment", default="none",
                   help="Vendored fibre-pack preset string: one of 'none', 'very low', "
                        "'moderate', 'high', 'very high'. Avoid 'low': it has an upstream "
                        "UnboundLocalError bug in the vendored code.")
    p.add_argument("--iters", type=int, default=200, help="Packing optimisation iterations.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for packing and noise.")

    # voxelisation params (see phantom.voxelize_config); voxel-um is the
    # global recon_pixel_size constraint (2.0 um throughout the study)
    p.add_argument("--ny", type=int, default=300, help="Transverse grid height, voxels.")
    p.add_argument("--nx", type=int, default=300, help="Transverse grid width, voxels.")
    p.add_argument("--voxel-um", type=float, default=2.0, help="Voxel size, um.")
    p.add_argument("--mu-fibre", type=float, default=6.2e-4,
                   help="Fibre (glass) per-voxel absorbance (mu times voxel size, ASTRA "
                        "sums voxel values over unit-length rays). Reference-scale value; "
                        "ASTRA integrates this directly, so it must be mu[1/length] times "
                        "voxel_size, NOT mu in 1/cm, or line integrals saturate the beam.")
    p.add_argument("--mu-matrix", type=float, default=3.5e-4,
                   help="Matrix (resin) per-voxel absorbance; see --mu-fibre. Reference "
                        "PlenoXFiber scale (matrix 3.5e-4, fibre 6.2e-4, ~75 percent "
                        "contrast); override with your calibrated 12 keV values rescaled "
                        "to per-voxel absorbance.")
    p.add_argument("--axial-um", type=float, default=None,
                   help="Physical spacing per packing slice along the fibre axis, um. "
                        "Defaults to --voxel-um (isotropic voxels).")

    # CT geometry params: study-scale defaults match the real 260616 rotation-CT
    # scan (DataFiles/260616_Glasfiber_Tomo): source-to-sample 131.5 mm,
    # source-to-detector 1148.0 mm, Timepix detector pixel 55 um, ~405
    # projections over a full 360 degree turn.
    p.add_argument("--n-angles", type=int, default=405,
                   help="Number of projection angles over a full 360 degree turn.")
    p.add_argument("--sod-mm", type=float, default=131.5,
                   help="Source-to-sample (volume centre) distance, mm.")
    p.add_argument("--sdd-mm", type=float, default=1148.0,
                   help="Source-to-detector distance, mm.")
    p.add_argument("--detector-pixel-um", type=float, default=55.0,
                   help="Physical detector pixel size, um.")
    p.add_argument("--super-sampling", type=int, default=2,
                   help="ASTRA voxel/detector supersampling factor.")

    # forward model / reconstruction, shared identically across both orientations
    p.add_argument("--i0", type=float, default=5e4, help="Incident photon flux (Poisson).")
    p.add_argument("--n-iter", type=int, default=200, help="TV solver iterations.")
    p.add_argument("--lr", type=float, default=1e-1, help="Adam learning rate.")
    p.add_argument("--tv-weight", type=float, default=0.5,
                   help="Squared-L2 gradient penalty weight applied along each "
                        "orientation's OWN fibre axis only (dim 0 for axis0, dim 1 for "
                        "axis1), transverse axes unpenalised, matching the reference "
                        "along-fibre smoothing. May need one GPU tuning pass.")
    p.add_argument("--log-every", type=int, default=10,
                   help="Convergence-history logging interval, iterations.")
    return p


def _ct_geometry(voxel_um, sod_mm, sdd_mm, detector_pixel_um, n_angles):
    """Full-360-degree CT angles and vol_z0_pix/detector_z/pixel_size in voxel units.

    build_ct_vectors' distance args share the volume's voxel-unit convention
    (astra.create_vol_geom(*vol.shape) sizes each voxel as 1 unit), so all
    physical distances (mm/um) are divided by voxel_um to land in that scale.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    vol_z0_pix = (sod_mm * 1000.0) / voxel_um
    detector_z = (sdd_mm * 1000.0) / voxel_um
    pixel_size = detector_pixel_um / voxel_um
    return angles, vol_z0_pix, detector_z, pixel_size


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.smoke:
        domain_radius_um = 40.0
        r_mean_um = 3.0
        n_slices = 24
        iters = 10
        ny = nx = 48
        n_angles = 40
        n_iter = 5
        log_every = 1
    else:
        domain_radius_um = args.domain_radius_um
        r_mean_um = args.r_mean_um
        n_slices = args.n_slices
        iters = args.iters
        ny = args.ny
        nx = args.nx
        n_angles = args.n_angles
        n_iter = args.n_iter
        log_every = args.log_every

    voxel_um = args.voxel_um
    axial_um = voxel_um if args.axial_um is None else args.axial_um

    configuration, radii = pack_bundle(
        domain_radius_um=domain_radius_um,
        fvf=args.fvf,
        r_mean_um=r_mean_um,
        r_sigma_um=args.r_sigma_um,
        n_slices=n_slices,
        misalignment=args.misalignment,
        iters=iters,
        seed=args.seed,
    )

    angles, vol_z0_pix, detector_z, pixel_size = _ct_geometry(
        voxel_um, args.sod_mm, args.sdd_mm, args.detector_pixel_um, n_angles)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deferred to here: everything below touches astra and/or CUDA torch.
    import torch
    from phantom_sim.geometry import build_ct_vectors, make_ct_projector, XrayOperator
    from phantom_sim.forward import simulate_projections
    from phantom_sim.reconstruct import reconstruct
    from phantom_sim.metrics import psnr, ssim, register, resolution_along_across

    vecs = build_ct_vectors(angles, vol_z0_pix, detector_z, pixel_size, pixel_size)

    results = {}
    for axis, name in [(0, "axis0"), (1, "axis1")]:
        vol = voxelize_config(
            configuration, radii, (ny, nx), voxel_um=voxel_um,
            axis=axis, mu_fibre=args.mu_fibre, mu_matrix=args.mu_matrix,
        )

        # Array dim 0 is always the fixed rotation axis regardless of axis
        # (see module docstring); array dim -1 (nx) is the other in-plane
        # transverse direction. Both orientations get the IDENTICAL vecs.
        sinogram_size = [vol.shape[0], vol.shape[2], n_angles]
        pid, _ = make_ct_projector(vol.shape, vecs, sinogram_size=sinogram_size,
                                   super_sampling=args.super_sampling)
        A = XrayOperator(pid)
        volt = torch.tensor(vol, device="cuda")
        proj = simulate_projections(A, volt, i0=args.i0, seed=args.seed)

        # Strong squared-L2 TV along THIS orientation's fibre axis only (the fibres
        # run along array dim `axis` after voxelize_config placed them there); the
        # transverse axes stay unpenalised so fibre cross-sections keep their edges.
        tv_weights = [0.0, 0.0, 0.0]
        tv_weights[axis] = args.tv_weight
        recon, history = reconstruct(
            A, proj, mask=None, n_iter=n_iter, lr=args.lr,
            tv_weights=tuple(tv_weights), x_init=None,
            log_every=log_every, gt=vol, target_psnr=None, lower_clamp=0.0,
        )

        p = psnr(recon, vol)
        s = ssim(recon, vol)
        recon_reg = register(recon, vol)
        res = resolution_along_across(recon_reg, fibre_axis=axis)
        metrics_entry = {"psnr": p, "ssim": s, "resolution": res}
        results[name] = metrics_entry

        npy_path = out_dir / f"ct_{name}.npy"
        meta_path = out_dir / f"ct_{name}.meta.json"
        np.save(npy_path, recon.astype(np.float32))

        meta = {
            "orientation_axis": axis,
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
                "r_sigma_um": args.r_sigma_um,
                "n_slices": n_slices,
                "misalignment": args.misalignment,
                "iters": iters,
                "seed": args.seed,
                "axial_um": axial_um,
            },
            "geometry": {
                "n_angles": n_angles,
                "angle_range_deg": 360.0,
                "sod_mm": args.sod_mm,
                "sdd_mm": args.sdd_mm,
                "detector_pixel_um": args.detector_pixel_um,
                "vol_z0_pix": vol_z0_pix,
                "detector_z_pix": detector_z,
                "pixel_size_pix": pixel_size,
                "super_sampling": args.super_sampling,
            },
            "reconstruction": {
                "i0": args.i0,
                "n_iter": n_iter,
                "lr": args.lr,
                "tv_weight": args.tv_weight,
                "log_every": log_every,
            },
            "metrics": metrics_entry,
            "history": history,
        }
        write_meta(meta_path, meta)
        print(f"Saved {npy_path}")
        print(f"Saved {meta_path}")

    metrics_path = out_dir / "ct_metrics.json"
    write_meta(metrics_path, results)
    print(f"Saved {metrics_path}")
    print(f"axis0: psnr={results['axis0']['psnr']:.2f} ssim={results['axis0']['ssim']:.4f}")
    print(f"axis1: psnr={results['axis1']['psnr']:.2f} ssim={results['axis1']['ssim']:.4f}")


if __name__ == "__main__":
    main()
