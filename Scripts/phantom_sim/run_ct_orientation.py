"""CLI: Experiment 1, CT rotation-axis vs transverse fibre orientation.

Builds ONE deep straight-fibre bundle (unidirectional composite, extruded along dim 0),
then reconstructs it in two orientations from an identical full-360-degree CT scan:
  - axis0: fibres PARALLEL to the scanner's rotation axis (array dim 0);
  - axis1: fibres TRANSVERSE to it, lying in the rotation plane (fibres along dim 1).
Scores whether rotation-axis alignment reconstructs better. Does not assert which wins;
that is the experiment's result.

Recipe (reproduces the reference vedrana CT reconstruction; see the notebooks
Generate_fibre_phantom_vedrana_script.ipynb and
plenoptic-NewAstra-SimData_210725-CT_Comparison.ipynb):
  - Frame-filling void phantom: the bundle cross-section fills the ny=nx transverse frame
    (`--domain-radius-um` 300 at ny=nx=300), packed at `--fvf` 0.4, voxelised as a
    fill-fraction in [0,1] against a ZERO background (`--mu-matrix 0`, `--mu-fibre 1`) so
    fibres sit in void, exactly as the reference phantom. Switch to the physical
    glass-in-resin study with `--mu-matrix 3.5e-4 --mu-fibre 6.2e-4`.
  - Deep volume: the fibre axis is `--depth` voxels (default 512) so line integrals
    accumulate strong along-fibre contrast (a shallow volume reconstructs nothing).
  - Void recon margin: the object is padded off the reconstruction boundary
    (`--pad-transverse-vox` 175, `--pad-depth-vox` 100; the reference pads 300->650 and
    600->800). Seamless because the background is void; scored on the object crop only.
  - Solver: squared-L2 TV along each orientation's OWN fibre axis only, non-negativity
    clamp, lr 0.05, 300 iters, ReduceLROnPlateau (the reference solver).
  - Forward: CLEAN line integrals by default (`--i0 0`); pass `--i0 >0` to add Poisson
    photon noise at that flux (the SNR dial), meaningful only with physical absorbances.
  - Headline metric: corr-with-GT (Pearson, scale-invariant), reliable where global
    PSNR/SSIM sit at the background-dominated floor. PSNR/SSIM/resolution still recorded.

Array-axis convention: array dim 0 is ALWAYS the scanner's fixed rotation axis (ASTRA Z)
for both orientations. axis0 volumes are (depth, ny, nx) with fibres along dim 0; axis1
volumes are (ny, depth, nx) with fibres along dim 1 (obtained by moving the fibre axis).

This-box vs cluster split: everything astra/CUDA-touching is deferred into `main`, so
`import phantom_sim.run_ct_orientation` works on this box; `make_straight_fibre_phantom`
is torch-CPU (its torch import is lazy inside pack_bundle).

Usage (cluster):
    python -m phantom_sim.run_ct_orientation --out phantom_sim_results/exp1
    python -m phantom_sim.run_ct_orientation --smoke --out /tmp/exp1_smoke

Outputs under --out:
    ct_axis0.npy / ct_axis0.meta.json   axis=0 (rotation-aligned) recon + meta
    ct_axis1.npy / ct_axis1.meta.json   axis=1 (transverse) recon + meta
    ct_metrics.json                     {"axis0": {...}, "axis1": {...}}
"""
import argparse
import json
from pathlib import Path

import numpy as np

from phantom_sim.phantom import make_straight_fibre_phantom


def write_meta(path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def build_parser():
    p = argparse.ArgumentParser(
        description="Experiment 1: CT rotation-axis vs transverse fibre orientation.")
    p.add_argument("--out", required=True, help="Output directory (created if missing).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny volume, few angles/iters, for a fast cluster smoke test.")

    # phantom (frame-filling void bundle, matching the reference vedrana phantom)
    p.add_argument("--domain-radius-um", type=float, default=300.0,
                   help="Bundle cross-section radius, um. Default 300 makes the bundle fill "
                        "the ny=nx=300 transverse frame (reference: domain_radius=300 at "
                        "voxelize xy=300). Keep it near ny/2 * voxel_um.")
    p.add_argument("--fvf", type=float, default=0.4,
                   help="Fibre volume fraction (0-1); reference vedrana phantom uses 0.4.")
    p.add_argument("--r-mean-um", type=float, default=6.0,
                   help="Mean fibre radius, um (glass, ~12um diameter = 6um radius).")
    p.add_argument("--iters", type=int, default=200, help="2D packing iterations.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for packing and noise.")
    p.add_argument("--ny", type=int, default=300, help="Transverse grid height, voxels.")
    p.add_argument("--nx", type=int, default=300, help="Transverse grid width, voxels.")
    p.add_argument("--depth", type=int, default=512,
                   help="Fibre-axis length in voxels (deep volume; the load-bearing "
                        "difference from the earlier shallow study).")
    p.add_argument("--voxel-um", type=float, default=2.0, help="Voxel size, um.")
    p.add_argument("--mu-fibre", type=float, default=1.0,
                   help="Fibre value. Default 1.0 with --mu-matrix 0 reproduces the "
                        "reference [0,1] fill-fraction phantom (fibres in void). For the "
                        "physical glass-in-resin study use 6.2e-4 (per-voxel absorbance).")
    p.add_argument("--mu-matrix", type=float, default=0.0,
                   help="Background (matrix) value. Default 0.0 = fibres in void (reference "
                        "recipe; a zero background also makes the recon padding seamless). "
                        "For glass-in-resin use 3.5e-4.")

    # CT geometry: study-scale defaults match the real 260616 rotation-CT scan
    p.add_argument("--n-angles", type=int, default=405,
                   help="Projection angles over a full 360 degree turn.")
    p.add_argument("--sod-mm", type=float, default=131.5, help="Source-to-sample distance, mm.")
    p.add_argument("--sdd-mm", type=float, default=1148.0, help="Source-to-detector distance, mm.")
    p.add_argument("--detector-pixel-um", type=float, default=55.0, help="Detector pixel, um.")
    p.add_argument("--super-sampling", type=int, default=2, help="ASTRA supersampling.")

    # forward / reconstruction, shared identically across both orientations
    p.add_argument("--i0", type=float, default=0.0,
                   help="0 = clean line integrals (no noise); >0 = Poisson photon noise at "
                        "that incident flux (the SNR dial).")
    p.add_argument("--n-iter", type=int, default=300, help="TV solver iterations (reference 300).")
    p.add_argument("--lr", type=float, default=5e-2, help="Adam learning rate (reference 0.05).")
    p.add_argument("--tv-weight", type=float, default=0.5,
                   help="Squared-L2 TV weight along each orientation's fibre axis only.")
    p.add_argument("--pad-transverse-vox", type=int, default=175,
                   help="Void margin (voxels per side) added to each transverse recon axis, "
                        "so the object sits off the reconstruction boundary (reference pads "
                        "the 300-wide object into a 650-wide frame). Seamless only with a "
                        "void background (--mu-matrix 0); metrics score the object crop only.")
    p.add_argument("--pad-depth-vox", type=int, default=100,
                   help="Void margin (voxels per side) added to the fibre-length recon axis "
                        "(reference pads 600 -> 800).")
    p.add_argument("--plateau-scheduler", dest="plateau_scheduler", action="store_true",
                   default=True, help="ReduceLROnPlateau (factor 0.5, patience 10), the "
                                      "reference solver (default on).")
    p.add_argument("--no-plateau-scheduler", dest="plateau_scheduler", action="store_false",
                   help="Disable the LR scheduler (fixed lr).")
    p.add_argument("--log-every", type=int, default=10, help="History logging interval.")
    return p


def _ct_geometry(voxel_um, sod_mm, sdd_mm, detector_pixel_um, n_angles):
    """Full-360-degree CT angles and vol_z0_pix/detector_z/pixel_size in voxel units."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    vol_z0_pix = (sod_mm * 1000.0) / voxel_um
    detector_z = (sdd_mm * 1000.0) / voxel_um
    pixel_size = detector_pixel_um / voxel_um
    return angles, vol_z0_pix, detector_z, pixel_size


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.smoke:
        domain_radius_um = 40.0
        r_mean_um = 6.0
        depth = 32
        iters = 20
        ny = nx = 64
        n_angles = 40
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
        n_angles = args.n_angles
        n_iter = args.n_iter
        log_every = args.log_every
        pad_transverse = args.pad_transverse_vox
        pad_depth = args.pad_depth_vox

    voxel_um = args.voxel_um
    angles, vol_z0_pix, detector_z, pixel_size = _ct_geometry(
        voxel_um, args.sod_mm, args.sdd_mm, args.detector_pixel_um, n_angles)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One deep straight bundle (axis0 orientation); axis1 is the same bundle with its
    # fibre axis moved from dim 0 to dim 1.
    print(f"Building deep straight bundle ({depth}, {ny}, {nx}) ...")
    vol0, radii = make_straight_fibre_phantom(
        nz=depth, ny=ny, nx=nx, voxel_um=voxel_um, domain_radius_um=domain_radius_um,
        fvf=args.fvf, r_mean_um=r_mean_um, mu_fibre=args.mu_fibre, mu_matrix=args.mu_matrix,
        iters=iters, seed=args.seed)
    print(f"  {radii.shape[0]} fibres, values in [{vol0.min():.2e}, {vol0.max():.2e}]")

    # Void margin around the object (reference pads 300->650 transverse, 600->800 axial),
    # so the object sits off the reconstruction boundary. Pad the CANONICAL volume (fibre
    # along dim 0); axis1 is this same padded volume with its fibre axis moved 0->1, so both
    # orientations get identical physical margin. Seamless only because the background is
    # void (mu_matrix 0); a nonzero pedestal would make the pad a fake void edge.
    pw = ((pad_depth, pad_depth), (pad_transverse, pad_transverse),
          (pad_transverse, pad_transverse))
    vol0_pad = np.pad(vol0, pw)
    # The object's slice inside the padded canonical volume (fibre along dim 0), used to
    # crop the recon back before scoring so the ~zero margin cannot inflate the metrics.
    obj_sl = (slice(pad_depth, pad_depth + depth),
              slice(pad_transverse, pad_transverse + ny),
              slice(pad_transverse, pad_transverse + nx))

    # Deferred: everything below touches astra and/or CUDA torch.
    import torch
    from phantom_sim.geometry import build_ct_vectors, make_ct_projector, XrayOperator
    from phantom_sim.forward import simulate_projections
    from phantom_sim.reconstruct import reconstruct
    from phantom_sim.metrics import psnr, ssim, register, resolution_along_across, corr_with_gt

    vecs = build_ct_vectors(angles, vol_z0_pix, detector_z, pixel_size, pixel_size)
    use_noise = bool(args.i0 and args.i0 > 0)
    forward_desc = f"poisson_i0_{args.i0:g}" if use_noise else "clean_line_integrals"

    results = {}
    for axis, name in [(0, "axis0"), (1, "axis1")]:
        vol = vol0_pad if axis == 0 else np.ascontiguousarray(np.moveaxis(vol0_pad, 0, 1))

        sinogram_size = [vol.shape[0], vol.shape[2], n_angles]
        pid, _ = make_ct_projector(vol.shape, vecs, sinogram_size=sinogram_size,
                                   super_sampling=args.super_sampling)
        A = XrayOperator(pid)
        volt = torch.tensor(vol, device="cuda")
        if use_noise:
            proj = simulate_projections(A, volt, i0=args.i0, seed=args.seed)
        else:
            proj = A(volt).detach()

        # Strong squared-L2 TV along THIS orientation's fibre axis only.
        tv_weights = [0.0, 0.0, 0.0]
        tv_weights[axis] = args.tv_weight
        recon, history = reconstruct(
            A, proj, mask=None, n_iter=n_iter, lr=args.lr, tv_weights=tuple(tv_weights),
            x_init=None, log_every=log_every, gt=vol, target_psnr=None, lower_clamp=0.0,
            plateau_scheduler=args.plateau_scheduler)

        # Undo the axis move so the crop is always in canonical coordinates, then compare
        # the object crop against the unpadded canonical GT (fibre axis 0 for both).
        recon_canon = recon if axis == 0 else np.moveaxis(recon, 1, 0)
        recon_obj = np.ascontiguousarray(recon_canon[obj_sl])

        corr = corr_with_gt(recon_obj, vol0)
        p = psnr(recon_obj, vol0)
        s = ssim(recon_obj, vol0)
        res = resolution_along_across(register(recon_obj, vol0), fibre_axis=0)
        metrics_entry = {"corr_with_gt": corr, "psnr": p, "ssim": s, "resolution": res}
        results[name] = metrics_entry

        npy_path = out_dir / f"ct_{name}.npy"
        np.save(npy_path, recon_obj.astype(np.float32))
        meta = {
            "orientation_axis": axis,
            "fibre_orientation": "parallel_to_rotation_axis" if axis == 0 else "transverse",
            "smoke": bool(args.smoke),
            "object_shape": list(vol0.shape),
            "recon_padded_shape": list(vol.shape),
            "saved_shape": list(recon_obj.shape),
            "pad_transverse_vox": pad_transverse,
            "pad_depth_vox": pad_depth,
            "dtype": "float32",
            "voxel_um": voxel_um,
            "mu_fibre": args.mu_fibre,
            "mu_matrix": args.mu_matrix,
            "n_fibres": int(radii.shape[0]),
            "packing": {"domain_radius_um": domain_radius_um, "fvf": args.fvf,
                        "r_mean_um": r_mean_um, "depth": depth, "iters": iters,
                        "seed": args.seed},
            "geometry": {"n_angles": n_angles, "angle_range_deg": 360.0, "sod_mm": args.sod_mm,
                         "sdd_mm": args.sdd_mm, "detector_pixel_um": args.detector_pixel_um,
                         "vol_z0_pix": vol_z0_pix, "detector_z_pix": detector_z,
                         "pixel_size_pix": pixel_size, "super_sampling": args.super_sampling},
            "reconstruction": {"forward": forward_desc, "i0": args.i0, "n_iter": n_iter,
                               "lr": args.lr, "tv_weight": args.tv_weight,
                               "tv_axis": axis, "plateau_scheduler": bool(args.plateau_scheduler),
                               "log_every": log_every},
            "metrics": metrics_entry,
            "history": history,
        }
        write_meta(out_dir / f"ct_{name}.meta.json", meta)
        print(f"Saved {npy_path}")
        print(f"  {name}: corr_with_gt={corr:.4f} psnr={p:.2f} ssim={s:.4f} "
              f"res_ratio={res['ratio']:.3f}")

    write_meta(out_dir / "ct_metrics.json", results)
    print(f"Saved {out_dir / 'ct_metrics.json'}  (forward: {forward_desc})")
    print("HEADLINE corr-with-GT:  "
          f"axis0 (rotation-aligned) = {results['axis0']['corr_with_gt']:.4f}   "
          f"axis1 (transverse) = {results['axis1']['corr_with_gt']:.4f}")


if __name__ == "__main__":
    main()
