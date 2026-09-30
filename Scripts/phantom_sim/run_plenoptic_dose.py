"""Real-data plenoptic DOSE-REDUCTION study (exp8).

Reconstructs the 260703 glass-fibre SINGLE capture at position 0 only (the "Zpos1"
recon) at a ladder of dose fractions f. A lower f simulates a shorter exposure by
POISSON (BINOMIAL) THINNING of the raw photon counts: each detected photon is kept
with probability f, so N_f = rng.binomial(N_int, f). Both the sample frames and their
per-position flats are thinned by f (a shorter scan shortens both), which preserves the
mean transmission and grows only the noise (as 1 / sqrt(f)); thinning the sample alone
would add a spurious -log(f) global absorbance offset. f = 1 short-circuits to the raw
counts unchanged. The goal is the lowest f whose reconstruction still correlates well
with the f = 1 reconstruction.

This is a faithful port of the real-data reconstruction path from
    Scripts/plenoptic-NewAstra-Al12thr_Aug2026_cleaned.ipynb
(the pipeline that produced the delivered 260703 recons), with two owner-confirmed
simplifications relative to the notebook:
  * the multibang penalty is dropped (it only activates at it >= 500; we run 200);
  * oversampling and sub-pixel shift refinement are dropped (single position, single
    projector, out = Projection.apply(A, x) directly).
Only position 0 (the lowest xs plane, sorted ascending) is reconstructed.

Runs on the DTU HPC cluster (Linux, NVIDIA A100, ASTRA + torch + numpy; no OpenCV, so
it runs in pleno_v100_min or try_pleno). It is NOT runnable on the Windows dev laptop
(no data, no GPU, no ASTRA); the cuda
default-device set is deferred into main() so the module still imports / py_compiles
without cuda present.
"""

import os
import glob
import json
import time
import argparse
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import astra
import astra.experimental


# ============================================================================
# Dataset location and loader tuning constants (copied verbatim from the notebook)
# ============================================================================
DATA_ROOT = "/path/to/raw_data"
DATASET = "260703_plenoptics_glassfiber"
PATTERN = "Image_26_*_Event.txt"
IMG_DIR_OVERRIDE = None  # set to an absolute path to skip auto-detection

STD_THR = 1000.0  # flat/sample split on frame spatial std.
DEAD_LO_FRAC = 0.25  # per-pixel median < this * global median -> dead
DEAD_HI_FRAC = 4.0  # per-pixel median > this * global median -> hot
DEAD_MAX_FRAC = 0.05  # abort if the mask covers more than this fraction of the detector
FLAT_DONOR = None  # xs plane whose per-position grid is shared; None = lowest available
FLAT_DRIFT_RESCALE = True
FLAT_DRIFT_REF = "post"  # 'post' | 'pre' | 'mean' | 'openbeam'

DEFAULT_OUT_DIR = "/path/to/workspace/phantom_sim_results/exp8_dose"


# ============================================================================
# Dead-pixel inpaint (pure numpy; drop-in replacement for the notebook's cv2.inpaint)
# ============================================================================
def inpaint_image(image, mask, max_iter=100):
    """Fill masked (dead/hot) pixels by iterative 4-neighbour averaging (pure numpy).

    Drop-in replacement for the notebook's cv2.inpaint(TELEA): the OpenCV wheel in the
    cluster envs needs a newer libstdc++ (CXXABI_1.3.15) than the system provides, the same
    break that removed scipy earlier. Dead pixels are sparse and scattered, so a few
    neighbour-diffusion passes fill them; anything still empty after max_iter falls back to
    the frame median. The fill is cosmetic and identical across every dose level, so it does
    not bias the sweep.
    """
    img = np.array(image, dtype=np.float32, copy=True)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return img
    img[m] = np.nan
    for _ in range(max_iter):
        empty = np.isnan(img)
        if not empty.any():
            break
        acc = np.zeros_like(img)
        cnt = np.zeros_like(img)
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            s = np.roll(img, shift, axis=axis)
            good = ~np.isnan(s)
            take = good & empty
            acc[take] += s[take]
            cnt[take] += 1.0
        fillable = empty & (cnt > 0)
        img[fillable] = acc[fillable] / cnt[fillable]
    still = np.isnan(img)
    if still.any():
        img[still] = float(np.median(image[~m])) if (~m).any() else 0.0
    return img


# ============================================================================
# Loader helpers (verbatim from the notebook)
# ============================================================================
def _img_stem(fp):
    b = os.path.basename(fp)
    return (
        b[: -len("_Event.txt")] if b.endswith("_Event.txt") else os.path.splitext(b)[0]
    )


def _parse_poslog(path, num_header_lines=15):
    row = {}
    with open(path) as f:
        for _ in range(num_header_lines):
            next(f)
        for line in f:
            p = line.split()
            if not p:
                continue
            row[p[0]] = list(map(float, p[1:]))
            if line.startswith("s1vg"):
                break
    return row


def _read_frame(path, shape=(256, 256)):
    """Whitespace-delimited integer counts. Equivalent to np.loadtxt for this fixed
    format but roughly 40x faster, which matters at 2213 frames."""
    with open(path, "rb") as fh:
        a = np.array(fh.read().split(), dtype=np.float32)
    if a.size != shape[0] * shape[1]:
        raise ValueError(f"{path}: {a.size} values, expected {shape[0] * shape[1]}")
    return a.reshape(shape)


def _find_img_dir(root, dataset, pattern):
    if IMG_DIR_OVERRIDE:
        return IMG_DIR_OVERRIDE
    base = os.path.join(root, dataset)
    for c in (os.path.join(base, dataset), base):  # unzip often doubles the folder
        if glob.glob(os.path.join(c, pattern)):
            return c
    for dirpath, _, _ in os.walk(base):
        if glob.glob(os.path.join(dirpath, pattern)):
            return dirpath
    raise FileNotFoundError(
        f"no files matching {pattern} under {base}. "
        "Set IMG_DIR_OVERRIDE to the folder holding the images."
    )


def load_plenoptic_with_flats(
    img_dir,
    log_dir,
    img_pattern,
    std_thr=STD_THR,
    dead_lo=DEAD_LO_FRAC,
    dead_hi=DEAD_HI_FRAC,
    flat_donor=FLAT_DONOR,
    drift_rescale=FLAT_DRIFT_RESCALE,
    drift_ref=FLAT_DRIFT_REF,
):
    img_files = glob.glob(os.path.join(img_dir, img_pattern))
    log_by_stem = {
        os.path.basename(f)[: -len(".poslog")]: f
        for f in glob.glob(os.path.join(log_dir, "*.poslog"))
    }
    pairs = [
        (_img_stem(f), f, log_by_stem[_img_stem(f)])
        for f in img_files
        if _img_stem(f) in log_by_stem
    ]
    pairs.sort(key=lambda t: t[0])  # stem sorts chronologically = acq order
    print(
        f"images={len(img_files)}  poslogs={len(log_by_stem)}  paired={len(pairs)}  "
        f"images_without_log={len(img_files) - len(pairs)}  "
        f"logs_without_image={len(log_by_stem) - len(pairs)}"
    )
    assert pairs, "0 pairs, stems did not match; check image vs poslog filenames"

    n = len(pairs)
    imgs = np.zeros((n, 256, 256), np.float32)
    pdict = defaultdict(list)
    for i, (stem, imf, logf) in enumerate(pairs):
        imgs[i] = np.rot90(_read_frame(imf), 1, axes=(0, 1))
        for lab, val in _parse_poslog(logf).items():
            pdict[lab].append(val)
        if i % 250 == 0:
            print(f"  loaded {i}/{n}", flush=True)

    pos = {
        k: np.array([v[0] for v in pdict[k]], float)
        for k in ("xs", "xd", "ys", "zs", "yd", "zd")
        if k in pdict
    }
    xs_per = pos["xs"]
    fmean = imgs.mean(axis=(1, 2))
    fstd = imgs.std(axis=(1, 2))
    is_flat = fstd < std_thr  # smooth frame == open beam
    print(
        f"\nframe spatial std: flats {fstd[is_flat].min():.0f}..{fstd[is_flat].max():.0f}   "
        f"samples {fstd[~is_flat].min():.0f}..{fstd[~is_flat].max():.0f}   (STD_THR={std_thr:g})"
    )
    print(f"flats={int(is_flat.sum())}  samples={int((~is_flat).sum())}")

    # ---- dead / hot pixels, measured on the flats -----------------------------
    # The old rule ((imgs==0) | (imgs>9999)).any(0) marks 100% of the detector on
    # 260703, whose flats sit near 12.5k counts, so use a relative per-pixel test.
    F = imgs[is_flat]
    pmed = np.median(F, axis=0)
    gmed = float(np.median(pmed))
    dead = ((pmed < dead_lo * gmed) | (pmed > dead_hi * gmed) | (F == 0).any(0)).astype(
        np.uint8
    )
    print(
        f"flat median = {gmed:.0f} counts   dead/hot pixels = {int(dead.sum())} "
        f"({100 * dead.mean():.3f}%)"
    )
    if dead.mean() > DEAD_MAX_FRAC:
        raise ValueError(
            f"dead mask covers {100 * dead.mean():.1f}% of the detector. "
            "Check DEAD_LO_FRAC / DEAD_HI_FRAC before inpainting anything."
        )

    # ---- split each xs block into projections + per-position flat grid --------
    scans_by_xs, grid_by_xs, centre_by_xs = OrderedDict(), OrderedDict(), OrderedDict()
    for xs in sorted(set(xs_per)):
        s_idx = np.where((xs_per == xs) & (~is_flat))[
            0
        ]  # acquisition order = grid order
        f_idx = np.where((xs_per == xs) & (is_flat))[0]
        ns, nf = len(s_idx), len(f_idx)
        scans_by_xs[xs] = np.moveaxis(imgs[s_idx], 0, 2)  # RAW samples

        if nf == ns + 2:  # [pre, grid, post]
            g_idx = f_idx[1:-1]
        elif nf == 2:  # centre flats only
            g_idx = np.empty(0, int)
        else:
            raise ValueError(
                f"xs={xs:g}: {ns} projections but {nf} flats; expected "
                f"{ns + 2} (with grid) or 2 (centre only)."
            )

        centre_by_xs[xs] = dict(
            pre=float(imgs[f_idx[0]].mean()), post=float(imgs[f_idx[-1]].mean())
        )

        if g_idx.size:
            # HARD CHECK: each grid flat must sit at its projection's detector position.
            # This is exactly what the old off-by-one violated (it reached 380 mm).
            dyd = float(np.abs(pos["yd"][s_idx] - pos["yd"][g_idx]).max())
            dzd = float(np.abs(pos["zd"][s_idx] - pos["zd"][g_idx]).max())
            if max(dyd, dzd) > 1e-6:
                raise ValueError(
                    f"xs={xs:g}: flat/projection detector mismatch "
                    f"max|dyd|={dyd:.3f} max|dzd|={dzd:.3f} mm. Pairing is wrong."
                )
            dzs = float(np.median(pos["zs"][s_idx] - pos["zs"][g_idx]))
            grid_by_xs[xs] = np.moveaxis(
                np.stack([inpaint_image(imgs[i], dead) for i in g_idx]), 0, 2
            )
            print(
                f"  xs={xs:g}: {ns} projections, {g_idx.size} per-position flats, "
                f"(yd,zd) match exact, sample dropped {dzs:+.2f} mm in zs for the flats"
            )
        else:
            print(
                f"  xs={xs:g}: {ns} projections, no per-position grid ({nf} centre flats)"
            )

    # ---- share the grid across planes that do not have one --------------------
    if not grid_by_xs:
        raise ValueError(
            "no per-position flat grid anywhere in this dataset; "
            "per-projection flat-fielding is not possible."
        )
    if flat_donor is None:
        flat_donor = sorted(grid_by_xs)[0]
    elif flat_donor not in grid_by_xs:
        raise ValueError(
            f"FLAT_DONOR={flat_donor} has no grid; available: {sorted(grid_by_xs)}"
        )

    def _ref(x):
        c = centre_by_xs[x]
        key = "post" if drift_ref == "openbeam" else drift_ref
        return c[key] if key in ("pre", "post") else 0.5 * (c["pre"] + c["post"])

    def _openbeam_peak(scans, ff, step=21):
        """Where this plane's UNOBSTRUCTED pixels actually land in projection/flat.

        The mode of the ratio histogram around 1. With roughly 46% of the frame open
        beam on this sample the mode is well defined; the guard below skips any frame
        where too little of it survives, so a fully obstructed projection cannot drag
        the estimate. Returns the extra factor the flats need for open beam to sit at 1.
        """
        pk = []
        for k in range(0, scans.shape[2], step):
            r = inpaint_image(scans[:, :, k], dead) / np.maximum(ff[:, :, k], 1.0)
            sel = r[(r > 0.85) & (r < 1.15)]
            if sel.size < 0.05 * r.size:
                continue
            h, e = np.histogram(sel, bins=200, range=(0.85, 1.15))
            pk.append(0.5 * (e[h.argmax()] + e[h.argmax() + 1]))
        return float(np.median(pk)) if pk else 1.0

    flats_by_xs = OrderedDict()
    print(
        f"\nflat grid donor: xs={flat_donor:g}   drift reference: '{drift_ref}' centre flat"
        f"   rescale: {drift_rescale}"
    )
    openbeam_by_xs = OrderedDict()
    for xs in scans_by_xs:
        if xs in grid_by_xs:
            g, src, scale = grid_by_xs[xs], xs, 1.0
        else:
            g, src = grid_by_xs[flat_donor], flat_donor
            scale = _ref(xs) / _ref(flat_donor) if drift_rescale else 1.0
        # share the array when the scale is exactly 1 so we do not copy ~115 MB per plane
        ff = g if abs(scale - 1.0) < 1e-9 else (g * scale).astype(np.float32)
        ob = 1.0
        if drift_ref == "openbeam":
            ob = _openbeam_peak(scans_by_xs[xs], ff)
            if abs(ob - 1.0) > 1e-9:
                ff = (ff * ob).astype(np.float32)
        openbeam_by_xs[xs] = ob
        flats_by_xs[xs] = ff
        if flats_by_xs[xs].shape[2] != scans_by_xs[xs].shape[2]:
            raise ValueError(
                f"xs={xs:g}: {scans_by_xs[xs].shape[2]} projections vs "
                f"{flats_by_xs[xs].shape[2]} flats."
            )
        print(
            f"  xs={xs:g}: grid from xs={src:g}, drift scale={scale:.4f} "
            f"({100 * (scale - 1):+.2f}%)"
            + (
                f", open-beam refine={ob:.4f} ({100 * (ob - 1):+.2f}%)"
                if drift_ref == "openbeam"
                else ""
            )
            + f", centre pre/post = "
            f"{centre_by_xs[xs]['pre']:.0f}/{centre_by_xs[xs]['post']:.0f} counts"
        )

    # vestigial for the plenoptic path (it uses src/det positions, not rotation angles);
    # kept so the CT helper and commented cells that name `angles` still resolve.
    tzs = np.array([v[0] for v in pdict.get("tzs", [[0.0]] * n)], float)
    angles = (
        np.deg2rad(tzs[~is_flat]) if tzs.size == n else np.zeros(int((~is_flat).sum()))
    )

    diag = dict(
        frame_mean=fmean,
        frame_std=fstd,
        is_flat=is_flat,
        xs_per=xs_per,
        pos=pos,
        std_thr=std_thr,
        dead_frac=float(dead.mean()),
        flat_median=gmed,
        ex_flat=imgs[np.where(is_flat)[0][:4]].copy(),
        ex_sample=imgs[np.where(~is_flat)[0][:4]].copy(),
        flat_order_mean=fmean[is_flat],
        flat_order_xs=xs_per[is_flat],
        centre_by_xs=centre_by_xs,
        flat_donor=flat_donor,
        grid_planes=sorted(grid_by_xs),
        drift_ref=drift_ref,
        openbeam_by_xs=openbeam_by_xs,
    )
    return scans_by_xs, flats_by_xs, dead, sorted(scans_by_xs), angles, diag


# ============================================================================
# Geometry (verbatim constants + setup_astra_plenoptic)
# ============================================================================
# Geometry constants copied verbatim from the notebook geometry cell.
recon_pixel_size = 2.0  # (um!!)
GRID_N = 21  # 260703 is a full 21 x 21 = 441 grid per plane

GEOM_FROM_POSLOG = False
SOURCE_RANGE_MM_FIXED = 22.0
DET_RANGE_MM_FIXED = 177.4

# Calibration constants (260507_Geometry_Pin, run 2026-05-08); these are NOT read from
# the poslogs. In setup_astra_plenoptic, sample_source_distance = SOD, detector_distance
# = total SDD (despite the name). z_voxels_downscale / z_pos_offset are exactly the
# notebook values (found at execution_count 25); do_oversampling = 1 in the notebook but
# the oversampling PATH is dropped here per the owner-confirmed simplification.
z_voxels_downscale = 1.0
z_pos_offset = 0
do_oversampling = (
    1  # notebook value; the oversampling code path is intentionally NOT used
)

pixel_size_x = 55.0 / recon_pixel_size  # 55 um pixel size
pixel_size_y = 55.0 / recon_pixel_size

CT_vol_size = [650, 650, 800]
vol_z_offset_pix = (
    200  # shift volume bbox +200 voxels along beam axis so sample sits centered
)


def setup_astra_plenoptic(
    vol_recon_size,
    src_vu_pix,
    det_vu_pix,
    vol_z0_pix,
    detector_z,
    pixel_size_x,
    pixel_size_y,
    sinogram_size,
    super_sampling=2,
    vol_z_offset_pix=0,
):
    rows, cols, slices = vol_recon_size
    vol_geom = astra.create_vol_geom(
        rows,
        cols,
        slices,
        -cols / 2,
        cols / 2,
        -rows / 2,
        rows / 2,
        -slices / 2 + vol_z_offset_pix,
        slices / 2 + vol_z_offset_pix,
    )
    num_imgs = src_vu_pix.shape[-1]
    src_vu2xyz = np.empty((num_imgs, 3))
    src_vu2xyz[:, 0] = src_vu_pix[0]  # source x
    src_vu2xyz[:, 1] = src_vu_pix[1]  # source y
    src_vu2xyz[
        :, 2
    ] = -vol_z0_pix  # source z, defined as negative distance from 0 (sample center)
    det_xyz = np.zeros_like(src_vu2xyz)
    det_xyz[:, 0] = det_vu_pix[0]
    det_xyz[:, 1] = det_vu_pix[1]
    det_xyz[:, 2] = detector_z - vol_z0_pix

    dir_x = np.zeros_like(src_vu2xyz)
    dir_x[:, 0] = pixel_size_x
    dir_y = np.zeros_like(src_vu2xyz)
    dir_y[:, 1] = pixel_size_y
    det_geometry = np.hstack([src_vu2xyz, det_xyz, dir_x, dir_y])
    proj_geom = astra.create_proj_geom(
        "cone_vec", sinogram_size[0], sinogram_size[1], det_geometry
    )
    opts = {
        "VoxelSuperSampling": super_sampling,
        "DetectorSuperSampling": super_sampling,
    }
    projector_id = astra.create_projector("cuda3d", proj_geom, vol_geom, opts)
    return projector_id, vol_geom


def build_position0_geometry():
    """Build ONLY the position-0 plenoptic geometry (the first / lowest xs plane).

    Returns (proj_id, vol_geom, n_proj) where n_proj is the number of modelled rays
    (GRID_N * GRID_N = 441 for the full 260703 grid). Positions 1..3 are NOT built.
    """
    if GEOM_FROM_POSLOG:
        raise NotImplementedError(
            "This driver reproduces the delivered recon geometry (GEOM_FROM_POSLOG=False, "
            "hardcoded 22.0 / 177.4 mm). Poslog-measured travel is not wired here."
        )
    source_range = SOURCE_RANGE_MM_FIXED * 1e3 / recon_pixel_size
    det_range = DET_RANGE_MM_FIXED * 1e3 / recon_pixel_size
    print(
        f"source range {source_range * recon_pixel_size * 1e-3:.3f} mm   "
        f"det range {det_range * recon_pixel_size * 1e-3:.3f} mm"
    )

    sources_v = np.linspace(-source_range, source_range, GRID_N)
    sources_u = np.linspace(source_range, -source_range, GRID_N)
    det_v = np.linspace(det_range, -det_range, GRID_N)
    det_u = np.linspace(-det_range, det_range, GRID_N)
    src_vu_pix = np.meshgrid(sources_v, sources_u, indexing="ij")
    src_vu_pix = np.flip(np.stack([c.flatten() for c in src_vu_pix], axis=0), 0)
    det_vu_pix = np.meshgrid(det_v, det_u, indexing="ij")
    det_vu_pix = np.flip(np.stack([c.flatten() for c in det_vu_pix], axis=0), 0)

    sample_source_distance = (
        z_voxels_downscale
    ) * 12.924e4 / recon_pixel_size + z_pos_offset
    detector_distance = (
        z_voxels_downscale
    ) * 117.157e4 / recon_pixel_size - z_pos_offset

    sinogram_size = [256, 256, det_vu_pix.shape[-1]]
    n_proj = det_vu_pix.shape[-1]
    mag_factor = detector_distance / sample_source_distance  # SDD / SOD
    print(
        f"magnification factor {mag_factor:.4f}   modelled rays (position 0) = {n_proj}"
    )

    # Position 0 uses vol_z0_pix = sample_source_distance and detector_z = detector_distance
    # (proj_id_plenoptic / proj_id_plenoptic1 in the notebook; identical geometry).
    proj_id, vol_geom = setup_astra_plenoptic(
        vol_recon_size=CT_vol_size,
        src_vu_pix=src_vu_pix,
        det_vu_pix=det_vu_pix,
        vol_z0_pix=sample_source_distance,
        detector_z=detector_distance,
        pixel_size_x=pixel_size_x,
        pixel_size_y=pixel_size_y,
        sinogram_size=sinogram_size,
        super_sampling=2,
        vol_z_offset_pix=vol_z_offset_pix,
    )
    return proj_id, vol_geom, n_proj


# ============================================================================
# Torch FP/BP operators (verbatim)
# ============================================================================
class XrayOperator:
    def __init__(self, projector_id):
        self.projector_id = projector_id
        self.vol_geom = astra.projector3d.volume_geometry(projector_id)
        self.vol_shape = astra.geom_size(self.vol_geom)
        self.proj_geom = astra.projector3d.projection_geometry(projector_id)
        self.proj_shape = astra.geom_size(self.proj_geom)

    def __call__(self, vol_data):
        proj_data = torch.zeros(self.proj_shape, dtype=torch.float32, device="cuda")
        astra.experimental.direct_FP3D(
            self.projector_id, vol=vol_data.detach(), proj=proj_data
        )
        return proj_data

    def T(self, proj_data):
        vol_data = torch.zeros(self.vol_shape, dtype=torch.float32, device="cuda")
        astra.experimental.direct_BP3D(
            self.projector_id, vol=vol_data, proj=proj_data.detach()
        )
        return vol_data


class Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, vol):
        ctx.A = A
        return A(vol)

    @staticmethod
    def backward(ctx, grad_output):
        A = ctx.A
        return None, A.T(grad_output)


# ============================================================================
# Reconstruction hyper-parameters (verbatim from the notebook optimizer cell)
# ============================================================================
learning_rate = 2e-4
tvz_alpha = 5e3  # fibre direction (dim 0)
tvy_alpha = 0
tvx_alpha = 0


# ============================================================================
# Normalization / -log (verbatim from the notebook, execution_count 21)
# ============================================================================
COR = 0  # integer detector roll; sub-pixel shifts are learned in the optimizer (dropped here)


def normalize_to_sinogram(projections, ff, dead, cor=COR):
    """Per-projection flat-fielding + -log, then to the ASTRA sinogram tensor.

    `projections` is the RAW (256, 256, N_PROJ) sample-count block; `ff` the matched
    (256, 256, N_PROJ) flat grid. Verbatim from the projection-correction cell.
    """
    projections_corrected = np.zeros_like(projections, dtype=np.float32)
    for n1 in range(projections.shape[-1]):
        ratio = inpaint_image(projections[:, :, n1], dead) / np.maximum(
            ff[:, :, n1], 1.0
        )
        projections_corrected[:, :, n1] = -np.log(np.clip(ratio, 1e-3, None))
    pc = projections_corrected
    t = torch.tensor(pc.transpose(0, 2, 1), dtype=torch.float32)
    return torch.roll(t, cor, 2).contiguous()


# ============================================================================
# NEW: dose thinning + reconstruction + metrics
# ============================================================================
def thin_counts(counts, f, rng):
    """Poisson (binomial) thinning of integer photon counts.

    f >= 1.0 short-circuits to the counts unchanged (identity). Otherwise each detected
    photon is kept with probability f: N_f = rng.binomial(round(N), f).
    """
    if f >= 1.0:
        return counts
    return rng.binomial(np.rint(counts).astype(np.int64), f).astype(np.float32)


def reconstruct(A, sinogram, n_iter):
    """Faithful port of the notebook reconstruction loop, single position / projector.

    ADAM lr=2e-4 on x; ReduceLROnPlateau(factor=0.5, patience=5, threshold=1e-6,
    cooldown=5); tvz only (tvy_alpha=tvx_alpha=0); clamp x to [-1e-5, 100.0] every
    iteration; loss = mean(|out - proj|^2) + tvz. x is initialized from A.T(sinogram)
    exactly as the notebook does: x = Parameter(zeros_like(backproj)).

    Returns (x_detached, final_loss).
    """
    backproj = A.T(sinogram)
    x = torch.nn.Parameter(torch.zeros_like(backproj))

    optimizer = torch.optim.Adam(
        [{"params": [x], "lr": learning_rate, "weight_decay": 0}]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, threshold=1e-6, cooldown=5
    )

    final_loss = float("nan")
    for it in range(n_iter):
        start_time = time.time()
        optimizer.zero_grad()

        out = Projection.apply(A, x)

        tvz = torch.mean(torch.abs(torch.diff(x, dim=0)) ** 2) * tvz_alpha
        loss = torch.mean(torch.abs(out - sinogram) ** 2) + tvz

        loss.backward()
        optimizer.step()

        # non-negativity / clamp, every iteration
        with torch.no_grad():
            x.data = torch.clamp(x.data, -1e-5, 100.0)

        avg_loss = loss.item()
        scheduler.step(avg_loss)
        final_loss = avg_loss

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  iter: {it}/{n_iter}, loss: {avg_loss:.3e}, tvz: {tvz:.2e}, "
            f"LR: {current_lr:.2e}, time: {time.time() - start_time:.2f}s",
            flush=True,
        )

    return x.detach(), final_loss


def pearson(a, b):
    """Flattened Pearson correlation between two numpy arrays."""
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def parse_float_list(s):
    return [float(v) for v in str(s).split(",") if v.strip() != ""]


def parse_int_list(s):
    return [int(v) for v in str(s).split(",") if v.strip() != ""]


def build_argparser():
    p = argparse.ArgumentParser(
        description="Real-data plenoptic dose-reduction sweep (exp8)."
    )
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument("--pattern", default=PATTERN)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--doses", default="1.0,0.5,0.25,0.10,0.05,0.02")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument(
        "--no-thin-flats",
        action="store_true",
        help="thin ONLY the sample, not the per-position flat. NOT physical (adds a "
        "-log(f) global absorbance offset); for debugging only. Default thins both.",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="quick check: n_iter=5, doses=1.0,0.1, seeds=0",
    )
    return p


def main():
    args = build_argparser().parse_args()

    # Torch default device: replicate the notebook, but only when cuda is present so the
    # module still imports on a cuda-less box. On the cluster this selects the A100.
    if torch.cuda.is_available():
        torch.set_default_device("cuda:0")
        torch.cuda.set_device("cuda:0")
        print("cuda available: default device set to cuda:0")
    else:
        print(
            "WARNING: cuda not available. ASTRA reconstruction cannot run on this host."
        )

    n_iter = args.n_iter
    doses = parse_float_list(args.doses)
    seeds = parse_int_list(args.seeds)
    thin_both = not bool(args.no_thin_flats)
    if args.smoke:
        n_iter = 5
        doses = [1.0, 0.1]
        seeds = [0]
        print("SMOKE mode: n_iter=5, doses=[1.0, 0.1], seeds=[0]")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "dose_metrics.json")

    # ---- load the raw data ONCE ------------------------------------------------
    img_dir = _find_img_dir(args.data_root, args.dataset, args.pattern)
    log_dir = os.path.join(img_dir, "ADVACAM_LOG")
    if not os.path.isdir(log_dir):
        log_dir = img_dir
    print(f"IMG_DIR = {img_dir}")
    print(f"LOG_DIR = {log_dir}\n")

    all_scans_by_xs, flatfields_by_xs, dead_pixels, unique_xs, angles, diag = (
        load_plenoptic_with_flats(img_dir, log_dir, args.pattern)
    )

    # Position 0 = the lowest xs plane, sorted ascending = the "Zpos1" recon. It carries
    # its own full per-position flat grid (it is the donor), so no drift-share is used.
    xs0 = unique_xs[0]
    raw_sample_block = all_scans_by_xs[xs0]  # (256, 256, N_PROJ) RAW counts
    flat_block = flatfields_by_xs[xs0]  # (256, 256, N_PROJ) matched flats
    n_proj_data = raw_sample_block.shape[2]
    print(
        f"\nposition 0: xs={xs0:g}   projections={n_proj_data}   thin_both={thin_both}"
    )

    # ---- geometry + operator ONCE ---------------------------------------------
    proj_id, vol_geom, n_proj_geom = build_position0_geometry()
    if n_proj_geom != n_proj_data:
        raise ValueError(
            f"geometry builds {n_proj_geom} rays but position 0 has "
            f"{n_proj_data} projections; check GRID_N."
        )
    A = XrayOperator(proj_id)

    metrics = []

    def save_and_record(x, f, seed, final_loss, ref_np):
        """Save the recon .pt, compute correlations vs the f=1 reference, append + flush."""
        x_cpu = x.detach().cpu()
        if f >= 1.0:
            fname = "dose_f1_full.pt"
        else:
            fname = f"dose_f{f:g}_s{seed}.pt"
        out_path = os.path.join(out_dir, fname)
        metadata = {
            "dose_fraction": f,
            "seed": seed,
            "thin_both": thin_both,
            "n_iter": n_iter,
            "recon_pixel_size": recon_pixel_size,
            "CT_vol_size": CT_vol_size,
            "tvz_alpha": tvz_alpha,
            "lr": learning_rate,
            "position": 0,
            "dataset": args.dataset,
            "final_loss": final_loss,
        }
        torch.save({"tensor": x_cpu, "metadata": metadata}, out_path)

        x_np = x_cpu.numpy()
        corr_all = pearson(ref_np, x_np)
        fibre_mask = ref_np > 0.5 * np.percentile(ref_np, 99)
        corr_fibre = pearson(ref_np[fibre_mask], x_np[fibre_mask])
        metrics.append(
            {
                "f": f,
                "seed": seed,
                "corr_all": corr_all,
                "corr_fibre": corr_fibre,
                "final_loss": final_loss,
            }
        )
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        print(
            f"[RESULT] f={f:g} seed={seed} corr_all={corr_all:.4f} "
            f"corr_fibre={corr_fibre:.4f} final_loss={final_loss:.3e} -> {fname}",
            flush=True,
        )

    # ---- reference: f = 1, FIRST and exactly once (identity; seed ignored) ------
    print("\n=== f = 1 (reference, identity, seed ignored) ===", flush=True)
    sino_ref = normalize_to_sinogram(raw_sample_block, flat_block, dead_pixels)
    if torch.cuda.is_available():
        sino_ref = sino_ref.to("cuda")
    x_ref, ref_loss = reconstruct(A, sino_ref, n_iter)
    ref_np = x_ref.detach().cpu().numpy()
    save_and_record(x_ref, 1.0, None, ref_loss, ref_np)
    del sino_ref, x_ref

    # ---- sweep: each f < 1, each seed ------------------------------------------
    for f in doses:
        if f >= 1.0:
            continue  # f = 1 already run once as the reference; f > 1 is not meaningful
        for seed in seeds:
            print(f"\n=== f = {f:g}, seed = {seed} ===", flush=True)
            rng = np.random.default_rng(seed)
            thinned_sample = thin_counts(raw_sample_block, f, rng)
            # Physical shorter-exposure model: thin BOTH the sample and its per-position flat by f,
            # so the mean transmission is preserved (no -log(f) offset) and noise scales as 1/sqrt(f).
            flat_used = flat_block if not thin_both else thin_counts(flat_block, f, rng)
            sino = normalize_to_sinogram(thinned_sample, flat_used, dead_pixels)
            if torch.cuda.is_available():
                sino = sino.to("cuda")
            x, final_loss = reconstruct(A, sino, n_iter)
            save_and_record(x, f, seed, final_loss, ref_np)
            del sino, x

    print(f"\nDONE. {len(metrics)} recons. Metrics: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
