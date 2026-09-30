"""CLI: point and step-edge resolution probes for the plenoptic geometry.

Companion to run_plenoptic_psf.py (whose test object is a sphere of radius 2 voxels, not
point-like on the transverse scale). Two objects, each imaged in a single plenoptic capture
and reconstructed with the same solver and prior as the fibre studies:

  --object point      one voxel of value mu at the volume centre. Its reconstruction is the
                      point-spread function; profiles along z / y / x give the FWHM and MTF.
  --object fibre_end  a cylinder of radius --edge-radius-vox along z (the fibre axis) that
                      starts at the centre plane and runs --rod-length-vox beyond it: a step
                      edge along z, the geometry of a fibre end or break. The edge-spread
                      function (ESF) along the rod gives the 10-90% rise distance and, by
                      differentiation, the line-spread function (LSF), its FWHM and the MTF.

The rod radius sets whether the edge is a
single fibre end (radius 3 voxels = 6 um) or a broad face (e.g. 25 voxels).

Module level is stdlib + numpy only (imports on a box without torch/astra); main defers every
torch/astra import. Usage (cluster, CUDA interpreter):
    python -m phantom_sim.run_plenoptic_psf_edge --object point --smoke --out /tmp/pe_smoke
    python -m phantom_sim.run_plenoptic_psf_edge --object fibre_end --out phantom_sim_results/exp5c_fibre_end
"""

import argparse
import json
from pathlib import Path

import numpy as np

from phantom_sim.run_plenoptic_psf import axis_profile, fwhm, mtf


def make_point_phantom(nz, ny, nx, mu=1.0):
    """One voxel of value `mu` at the volume centre (nz//2, ny//2, nx//2), zero elsewhere."""
    v = np.zeros((nz, ny, nx), dtype=np.float32)
    v[nz // 2, ny // 2, nx // 2] = mu
    return v


def make_rod_phantom(nz, ny, nx, radius_vox, z_start, z_stop, mu=1.0, mu_matrix=0.0):
    """Cylinder of value `mu` along axis 0 (z) over slices [z_start, z_stop), radius
    `radius_vox` about the transverse centre, in a `mu_matrix` background."""
    yc, xc = ny // 2, nx // 2
    yy, xx = np.ogrid[:ny, :nx]
    disc = (yy - yc) ** 2 + (xx - xc) ** 2 <= radius_vox**2
    v = np.full((nz, ny, nx), mu_matrix, dtype=np.float32)
    v[z_start:z_stop][:, disc] = mu
    return v


def rod_core_mask(ny, nx, radius_vox, shrink_vox=1):
    """Transverse mask of the rod core (radius - shrink_vox), used to average the ESF over the
    cross-section away from the rod surface."""
    yc, xc = ny // 2, nx // 2
    yy, xx = np.ogrid[:ny, :nx]
    r = max(radius_vox - shrink_vox, 0.5)
    return (yy - yc) ** 2 + (xx - xc) ** 2 <= r**2


def _crossing(p, level, start, step):
    """Linearly interpolated index where p crosses `level`, searching from `start` in
    direction `step` (+1 or -1). None if never crossed."""
    i = start
    while 0 <= i + step < len(p):
        a, b = p[i], p[i + step]
        if (a - level) * (b - level) <= 0 and a != b:
            return i + step * (level - a) / (b - a)
        i += step
    return None


def edge_metrics(esf, edge_index, spacing=1.0, window=None):
    """Edge-spread analysis of a rising edge at about `edge_index` (object at higher index).

    The profile is normalised between its base (median of the samples before the edge) and
    plateau (median of the samples after it), each taken over `window` samples (default: up to
    40, clipped to the available length) starting 10 samples away from the edge. Returns the
    10-90% rise distance, the FWHM of the line-spread function (the derivative of the ESF,
    negative lobes clipped), and the MTF of the LSF, all in `spacing` units."""
    p = np.asarray(esf, dtype=np.float64)
    n = len(p)
    w = window or 40
    lo = p[max(edge_index - 10 - w, 0) : max(edge_index - 10, 1)]
    hi = p[min(edge_index + 10, n - 1) : min(edge_index + 10 + w, n)]
    base = float(np.median(lo))
    plateau = float(np.median(hi))
    q = (p - base) / (plateau - base) if plateau != base else np.zeros_like(p)

    # search from the start of the base window: a blurred edge can pass 50% before edge_index,
    # and a forward search from there would land on the rod's far (falling) end instead
    i50 = _crossing(q, 0.5, max(edge_index - 10 - w, 0), +1)
    if i50 is None:
        i50 = float(edge_index)
    k = int(np.floor(i50))
    i10 = _crossing(q, 0.1, k + 1, -1)
    i90 = _crossing(q, 0.9, k, +1)
    rise = (i90 - i10) * spacing if (i10 is not None and i90 is not None) else None

    # LSF on a +-60 sample window around the edge, forward difference placed on the half grid
    a, b = max(k - 60, 0), min(k + 61, n)
    lsf = np.clip(np.diff(q[a:b]), 0.0, None)
    lsf_w = fwhm(lsf, spacing=spacing)
    freqs, m = mtf(lsf, spacing=spacing)
    return {
        "base": base,
        "plateau": plateau,
        "edge_50_index": float(i50),
        "rise_10_90_um": None if rise is None else float(rise),
        "lsf_fwhm_um": float(lsf_w),
        "lsf": [float(x) for x in lsf],
        "freqs_inv_um": [float(x) for x in freqs],
        "mtf": [float(x) for x in m],
    }


def build_parser():
    p = argparse.ArgumentParser(description="Point / step-edge plenoptic resolution probes.")
    p.add_argument("--out", required=True)
    p.add_argument("--object", choices=("point", "fibre_end"), required=True)
    p.add_argument("--smoke", action="store_true", help="Tiny volume/grid/iters.")
    p.add_argument("--nz", type=int, default=256)
    p.add_argument("--ny", type=int, default=256)
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--edge-radius-vox", type=float, default=3.0, help="Rod radius (fibre_end).")
    p.add_argument(
        "--rod-length-vox", type=int, default=96, help="Rod length beyond the edge (fibre_end)."
    )
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
    p.add_argument("--pad-transverse-vox", type=int, default=32)
    p.add_argument("--pad-depth-vox", type=int, default=32)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--tv-weight", type=float, default=0.5, help="Penalty weight along z.")
    p.add_argument("--log-every", type=int, default=10)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke:
        nz, ny, nx = 48, 32, 32
        n_grid, n_iter, log_every = 3, 6, 1
        pad_t, pad_d = 4, 4
        rod_len = 12
    else:
        nz, ny, nx = args.nz, args.ny, args.nx
        n_grid, n_iter, log_every = args.n_grid, args.n_iter, args.log_every
        pad_t, pad_d = args.pad_transverse_vox, args.pad_depth_vox
        rod_len = args.rod_length_vox
    voxel_um = args.voxel_um
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    zc = nz // 2
    if args.object == "point":
        obj = make_point_phantom(nz, ny, nx, args.mu_fibre)
    else:
        obj = make_rod_phantom(
            nz, ny, nx, args.edge_radius_vox, zc, min(zc + rod_len, nz), args.mu_fibre, args.mu_matrix
        )

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

    vol = np.pad(obj, ((pad_d, pad_d), (pad_t, pad_t), (pad_t, pad_t)))
    padded_shape = vol.shape
    obj_sl = (slice(pad_d, pad_d + nz), slice(pad_t, pad_t + ny), slice(pad_t, pad_t + nx))

    detector_z = (args.sdd_mm * 1000.0) / voxel_um
    pixel_size = args.detector_pixel_um / voxel_um
    src, det = plenoptic_grid(args.source_range_um / voxel_um, args.det_range_um / voxel_um, n=n_grid)
    num_imgs = n_grid * n_grid
    vol_z0 = (args.sod_mm * 1000.0) / voxel_um
    sinogram_size = [padded_shape[1], padded_shape[2], num_imgs]
    geom = build_plenoptic_geometry(src, det, vol_z0, detector_z, pixel_size, pixel_size)
    pid, _ = make_plenoptic_projector(
        padded_shape, geom, sinogram_size=sinogram_size, super_sampling=args.super_sampling
    )
    A = XrayOperator(pid)
    volt = torch.tensor(vol, device="cuda")
    proj = A(volt).detach()
    del volt
    torch.cuda.empty_cache()

    def corr_metric(recon_padded_np):
        return corr_with_gt(np.ascontiguousarray(recon_padded_np[obj_sl]), obj)

    recon, history = reconstruct(
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
    np.save(out_dir / "recon.npy", recon_obj.astype(np.float32))

    meta = {
        "smoke": bool(args.smoke),
        "object": args.object,
        "voxel_um": voxel_um,
        "nz": nz,
        "ny": ny,
        "nx": nx,
        "mu_fibre": args.mu_fibre,
        "mu_matrix": args.mu_matrix,
        "sod_mm": args.sod_mm,
        "sdd_mm": args.sdd_mm,
        "source_range_um": args.source_range_um,
        "det_range_um": args.det_range_um,
        "n_grid": n_grid,
        "n_iter": n_iter,
        "lr": args.lr,
        "tv_weight": args.tv_weight,
        "pad_transverse_vox": pad_t,
        "pad_depth_vox": pad_d,
        "final_corr_with_gt": float(corr_with_gt(recon_obj, obj)),
        "recon_sum": float(recon_obj.sum()),
        "gt_sum": float(obj.sum()),
    }
    result = {}
    if args.object == "point":
        for name, ax in (("z", 0), ("y", 1), ("x", 2)):
            prof = axis_profile(recon_obj, ax)
            freqs, m = mtf(prof, spacing=voxel_um)
            result[name] = {
                "fwhm_um": float(fwhm(prof, spacing=voxel_um)),
                "gt_fwhm_um": float(fwhm(axis_profile(obj, ax), spacing=voxel_um)),
                "profile": [float(x) for x in prof],
                "freqs_inv_um": [float(x) for x in freqs],
                "mtf": [float(x) for x in m],
            }
            print(f"  point {name}: FWHM {result[name]['fwhm_um']:.2f} um")
    else:
        meta.update({"edge_radius_vox": args.edge_radius_vox, "rod_length_vox": rod_len, "edge_z_index": zc})
        core = rod_core_mask(ny, nx, args.edge_radius_vox, shrink_vox=1)
        for tag, prof_r, prof_g in (
            ("axis", axis_profile(recon_obj, 0), axis_profile(obj, 0)),
            ("core_mean", recon_obj[:, core].mean(axis=1), obj[:, core].mean(axis=1)),
        ):
            em = edge_metrics(prof_r, zc, spacing=voxel_um)
            eg = edge_metrics(prof_g, zc, spacing=voxel_um)
            em["esf"] = [float(x) for x in prof_r]
            em["gt_rise_10_90_um"] = eg["rise_10_90_um"]
            em["gt_lsf_fwhm_um"] = eg["lsf_fwhm_um"]
            result[tag] = em
            print(f"  fibre_end {tag}: 10-90 rise {em['rise_10_90_um']} um, LSF FWHM {em['lsf_fwhm_um']:.2f} um")
        # transverse profile through the rod, well inside it, for reference
        zi = min(zc + rod_len // 2, nz - 1)
        result["transverse_mid_rod"] = {
            "z_index": int(zi),
            "profile_y": [float(x) for x in recon_obj[zi, :, nx // 2]],
            "fwhm_um": float(fwhm(recon_obj[zi, :, nx // 2], spacing=voxel_um)),
            "gt_fwhm_um": float(fwhm(obj[zi, :, nx // 2], spacing=voxel_um)),
        }
    out = {"meta": meta, "result": result, "history_corr": history.get("corr", [])}
    with open(out_dir / "psf_edge_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_dir / 'psf_edge_metrics.json'}")


if __name__ == "__main__":
    main()
