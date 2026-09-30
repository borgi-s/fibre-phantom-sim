"""Shared helpers for the M1-M4 fibre-diameter methods. Torch-free (numpy + scipy only).

Array convention (same as the phantom_sim drivers): volumes are (nz, ny, nx) with the fibre axis
along dim 0 (z). Transverse um coordinates follow ``phantom.voxelize_config``:
    x_um = (col - (nx - 1) / 2) * voxel_um,   y_um = (row - (ny - 1) / 2) * voxel_um
"""

import json
from pathlib import Path

import numpy as np


class Truth:
    """Ground truth of a simulated bundle: per-slice fibre centres (um) and per-fibre radii (um).

    cfg: (nz, 2, N) centres, index 0 = x, 1 = y.  radii: (N,).  rope_radius_um: bundle radius.
    """

    def __init__(self, cfg, radii, rope_radius_um, voxel_um):
        self.cfg = np.asarray(cfg, dtype=np.float64)
        self.radii = np.asarray(radii, dtype=np.float64)
        self.rope_radius_um = float(rope_radius_um)
        self.voxel_um = float(voxel_um)

    @property
    def diam_um(self):
        return 2.0 * self.radii

    @property
    def mean_diam_um(self):
        return float(2.0 * self.radii.mean())

    def nearest_true_diam(self, z, rows, cols, shape_yx, max_dist_um=None):
        """True diameter (um) of the fibre nearest each detected centre (rows, cols) in slice z.

        Returns (diam_um, dist_um) arrays; entries farther than ``max_dist_um`` (default: that
        fibre's own true radius) are NaN, so unmatched detections do not get a fake truth.
        """
        ny, nx = shape_yx
        z = int(np.clip(round(z), 0, self.cfg.shape[0] - 1))
        x_um, y_um = pix_to_um(
            np.asarray(rows, float), np.asarray(cols, float), shape_yx, self.voxel_um
        )
        cx, cy = self.cfg[z, 0], self.cfg[z, 1]
        d = np.hypot(x_um[:, None] - cx[None, :], y_um[:, None] - cy[None, :])
        k = np.argmin(d, axis=1)
        dist = d[np.arange(len(k)), k]
        lim = (
            self.radii[k]
            if max_dist_um is None
            else np.full(len(k), float(max_dist_um))
        )
        diam = np.where(dist <= lim, 2.0 * self.radii[k], np.nan)
        return diam, dist


def pix_to_um(rows, cols, shape_yx, voxel_um):
    ny, nx = shape_yx
    return (np.asarray(cols) - (nx - 1) / 2.0) * voxel_um, (
        np.asarray(rows) - (ny - 1) / 2.0
    ) * voxel_um


def rope_interior_mask(shape_yx, voxel_um, rope_radius_um, margin_um=80.0):
    """Boolean (ny, nx) mask of the rope interior at least ``margin_um`` inside the bundle edge.

    Mirrors the real-data rule (interior >= 40 voxels = 80 um from the sample edge)."""
    ny, nx = shape_yx
    rr, cc = np.mgrid[0:ny, 0:nx]
    x_um, y_um = pix_to_um(rr, cc, shape_yx, voxel_um)
    return np.hypot(x_um, y_um) <= (rope_radius_um - margin_um)


def analysis_z_slices(nz, n=6, skip=40):
    """``n`` evenly spaced transverse slice indices, skipping ``skip`` slices at each end
    (the real-data rule skipped 60 of ~550; 40 of 256 here clears the ~24 um z-blur edges)."""
    lo, hi = skip, nz - 1 - skip
    if hi <= lo:
        lo, hi = 0, nz - 1
    return [int(round(v)) for v in np.linspace(lo, hi, n)]


def summary_stats(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "q1": None,
            "q3": None,
        }
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
    }


def load_run(run_dir):
    """Load one diameter-sweep rung written by ``run_plenoptic_diamsweep``.

    Returns dict(recon (nz,ny,nx) float32, gt (nz,ny,nx) float32 in [0,1], truth=Truth, meta=dict).
    """
    run_dir = Path(run_dir)
    recon = np.load(run_dir / "recon.npy").astype(np.float32)
    gt = np.load(run_dir / "gt_vol_u8.npy").astype(np.float32) / 255.0
    g = np.load(run_dir / "fibre_gt.npz")
    with open(run_dir / "metrics.json", encoding="utf-8") as f:
        meta = json.load(f)["meta"]
    truth = Truth(g["cfg"], g["radii"], meta["domain_radius_um"], meta["voxel_um"])
    return {"recon": recon, "gt": gt, "truth": truth, "meta": meta}


def synthetic_bundle(
    diam_um,
    shape_yx=(256, 256),
    nz=32,
    voxel_um=2.0,
    fvf=0.5,
    blur_sigma_um=2.5,
    noise_frac=0.02,
    rope_radius_um=None,
    seed=0,
):
    """Straight hex-ish packed synthetic bundle of known diameter for method self-checks.

    Returns (vol (nz,ny,nx) float32, Truth). Jittered hexagonal lattice at the pitch that gives
    ``fvf``, voxelised with anti-aliasing, Gaussian-blurred (sigma in um) plus Gaussian noise.
    """
    from scipy import ndimage as ndi
    from phantom_sim.phantom import voxelize_config

    rng = np.random.default_rng(seed)
    ny, nx = shape_yx
    if rope_radius_um is None:
        rope_radius_um = 0.45 * min(ny, nx) * voxel_um
    r = diam_um / 2.0
    pitch = np.sqrt(2.0 * np.pi * r * r / (np.sqrt(3.0) * fvf))
    pts = []
    n_side = int(np.ceil(rope_radius_um / pitch)) + 2
    for j in range(-n_side, n_side + 1):
        for i in range(-n_side, n_side + 1):
            x = (i + 0.5 * (j % 2)) * pitch
            y = j * pitch * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= rope_radius_um - r:
                pts.append((x, y))
    pts = np.asarray(pts).T
    jitter = max(0.0, 0.5 * (pitch - diam_um) - 0.05)
    pts = pts + rng.uniform(-jitter, jitter, pts.shape)
    cfg = np.repeat(pts[None], nz, axis=0)
    radii = np.full(pts.shape[1], r)
    vol = voxelize_config(
        cfg,
        radii,
        shape_yx,
        voxel_um=voxel_um,
        axis=0,
        mu_fibre=1.0,
        mu_matrix=0.0,
        supersample=4,
    )
    if blur_sigma_um > 0:
        vol = ndi.gaussian_filter(
            vol, sigma=(0.0, blur_sigma_um / voxel_um, blur_sigma_um / voxel_um)
        )
    vol = vol + noise_frac * rng.standard_normal(vol.shape).astype(np.float32)
    return vol.astype(np.float32), Truth(cfg, radii, rope_radius_um, voxel_um)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
