"""CLI: pack a fibre bundle (vendored fibre-pack, torch), voxelise it into an
attenuation phantom volume, and save the volume plus ground-truth table and a
metadata sidecar.

This-box only (pulls in torch via `pack_bundle`; CPU is fine, no CUDA needed).

Usage (from Scripts/, using the 3d_recon env interpreter):
    & "C:\\Users\\borgi\\anaconda3\\envs\\3d_recon\\python.exe" phantom_sim/make_phantom.py ^
        --out ..\\DataFiles\\phantom_sim\\phantom_dev

Outputs under --out:
    phantom_mu.npy       float32 attenuation volume, shape (Z or ny/nx permuted per --axis)
    phantom_gt.npz        ground_truth_table dict arrays (centre_um, orientation, radius_um)
    phantom_mu.meta.json  all packing/voxelisation params + derived shape info (utf-8)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from phantom_sim.phantom import ground_truth_table, pack_bundle, voxelize_config


def build_parser():
    p = argparse.ArgumentParser(
        description="Pack a fibre bundle and voxelise it into an attenuation phantom.")
    p.add_argument("--out", required=True, help="Output directory (created if missing).")

    # packing params (see phantom.pack_bundle)
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
                   help="Vendored fibre-pack preset string ('none', 'very low', 'moderate', "
                        "'high', 'very high') or a dict of raw params. Avoid 'low': it has an "
                        "upstream UnboundLocalError bug in the vendored code.")
    p.add_argument("--iters", type=int, default=200, help="Packing optimisation iterations.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the packer.")

    # voxelisation params (see phantom.voxelize_config)
    p.add_argument("--ny", type=int, default=300, help="Transverse grid height, voxels.")
    p.add_argument("--nx", type=int, default=300, help="Transverse grid width, voxels.")
    p.add_argument("--voxel-um", type=float, default=1.0, help="Transverse voxel size, um.")
    p.add_argument("--axis", type=int, default=0, help="Array axis the fibre direction is moved to.")
    p.add_argument("--mu-fibre", type=float, default=0.30, help="Fibre linear attenuation coeff.")
    p.add_argument("--mu-matrix", type=float, default=0.28, help="Matrix (resin) attenuation coeff.")
    p.add_argument("--axial-um", type=float, default=None,
                   help="Physical spacing per packing slice along the fibre axis, um. "
                        "Defaults to --voxel-um (isotropic voxels).")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    axial_um = args.voxel_um if args.axial_um is None else args.axial_um

    configuration, radii = pack_bundle(
        domain_radius_um=args.domain_radius_um,
        fvf=args.fvf,
        r_mean_um=args.r_mean_um,
        r_sigma_um=args.r_sigma_um,
        n_slices=args.n_slices,
        misalignment=args.misalignment,
        iters=args.iters,
        seed=args.seed,
    )

    vol = voxelize_config(
        configuration, radii, (args.ny, args.nx), voxel_um=args.voxel_um,
        axis=args.axis, mu_fibre=args.mu_fibre, mu_matrix=args.mu_matrix,
    )

    gt = ground_truth_table(configuration, radii, axial_um=axial_um, axis=args.axis)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_path = out_dir / "phantom_mu.npy"
    gt_path = out_dir / "phantom_gt.npz"
    meta_path = out_dir / "phantom_mu.meta.json"

    np.save(mu_path, vol.astype(np.float32))
    np.savez(gt_path, **gt)

    n_fibres = int(radii.shape[0])
    meta = {
        "domain_radius_um": args.domain_radius_um,
        "fvf": args.fvf,
        "r_mean_um": args.r_mean_um,
        "r_sigma_um": args.r_sigma_um,
        "n_slices": args.n_slices,
        "misalignment": args.misalignment,
        "iters": args.iters,
        "seed": args.seed,
        "transverse_shape": [args.ny, args.nx],
        "voxel_um": args.voxel_um,
        "axis": args.axis,
        "mu_fibre": args.mu_fibre,
        "mu_matrix": args.mu_matrix,
        "axial_um": axial_um,
        "n_fibres": n_fibres,
        "volume_shape": list(vol.shape),
        "volume_dtype": str(vol.dtype),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {mu_path}")
    print(f"Saved {gt_path}")
    print(f"Saved {meta_path}")
    print(f"N fibres: {n_fibres}, volume shape: {vol.shape}")


if __name__ == "__main__":
    main()
