"""M4: fibre-diameter estimate by granulometry (no individual fibre segmentation). Torch-free.

Ported from the scratch M4_granulometry code (common.py / master.py / make_figs.py).

Grey-level and binary (Otsu) pattern spectra from morphological openings with disks of
increasing diameter (2x upsampled, 1 um SE grid at 2 um voxels), peak / median of the
spectrum, plus radial autocorrelation (first minimum) and power-spectrum (dominant ring)
cross-checks, and synthetic hex-disk calibration curves.

Synthetic calibration uses ``common.synthetic_bundle`` (single slice, matched blur), not the
original hex-disk generator.

Use::

    from phantom_sim.diam_methods import m4_granulometry
    res = m4_granulometry.run(vol, voxel_um, out_dir, "recon D=16", truth=truth)
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import disk
from skimage.transform import rescale

from phantom_sim.diam_methods import common

DIAMS_FULL_UM = np.arange(2, 38, 2.0)  # 2..36 um step 2 (18 values), as the original
DIAMS_FAST_UM = np.arange(2, 30, 2.0)  # 2..28 um step 2 (14 values)
OPEN_DIAMS_FULL = [8, 12, 16, 20, 24]
OPEN_DIAMS_FAST = [8, 16, 24]
CROP_CAP_PX = 300
MIN_CROP_UM = 150.0  # below this the truth-based crop is relaxed (smaller margin)


# ------------------------------------------------------------------ crop
def get_interior_crop(slice2d, voxel_um, erode_um=80.0, target=CROP_CAP_PX):
    """Original blurred-Otsu rule: (target x target) crop centred on the sample interior,
    at least ``erode_um`` inside the sample edge. Returns (crop, (y0, y1, x0, x1))."""
    erode_px = max(1, int(round(erode_um / voxel_um)))
    H, W = slice2d.shape
    target = int(min(target, H, W))
    blur = ndi.gaussian_filter(slice2d, sigma=15 * 2.0 / voxel_um)
    mask = blur > threshold_otsu(blur)
    mask = ndi.binary_closing(mask, structure=np.ones((7, 7)))
    mask = ndi.binary_fill_holes(mask)
    lbl, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, lbl, range(1, n + 1))
        mask = lbl == (np.argmax(sizes) + 1)
    dist = ndi.distance_transform_edt(mask)
    ys, xs = np.where(dist >= erode_px)
    if len(ys) == 0:
        for e in range(erode_px - 5, 0, -5):
            ys, xs = np.where(dist >= e)
            if len(ys) > 0:
                break
    if len(ys) == 0:
        ys, xs = np.array([H // 2]), np.array([W // 2])
    cy, cx = int(np.mean(ys)), int(np.mean(xs))
    half = target // 2
    y0 = min(max(cy - half, 0), H - target)
    x0 = min(max(cx - half, 0), W - target)
    return slice2d[y0 : y0 + target, x0 : x0 + target], (
        y0,
        y0 + target,
        x0,
        x0 + target,
    )


def truth_crop_bbox(
    shape_yx, voxel_um, rope_radius_um, margin_um=80.0, cap=CROP_CAP_PX
):
    """Largest centred square fully inside ``common.rope_interior_mask``. Returns
    (bbox, margin_used_um). The margin is relaxed (40, 20, 0 um) only when the square would be
    narrower than MIN_CROP_UM (small test bundles)."""
    ny, nx = shape_yx
    best = None
    for m in (margin_um, 40.0, 20.0, 0.0):
        if m > margin_um:
            continue
        mask = common.rope_interior_mask(
            shape_yx, voxel_um, rope_radius_um, margin_um=m
        )
        if not mask.any():
            continue
        # grow a centred square until a corner leaves the mask
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        side = 2
        maxside = int(min(cap, ny, nx))
        while side + 2 <= maxside:
            h = (side + 2) / 2.0
            y0, y1 = int(np.ceil(cy - h + 0.5)), int(np.floor(cy + h - 0.5))
            x0, x1 = int(np.ceil(cx - h + 0.5)), int(np.floor(cx + h - 0.5))
            if y0 < 0 or x0 < 0 or y1 >= ny or x1 >= nx:
                break
            if not (mask[y0, x0] and mask[y0, x1] and mask[y1, x0] and mask[y1, x1]):
                break
            side += 2
        y0 = int(round(cy - side / 2.0 + 0.5))
        x0 = int(round(cx - side / 2.0 + 0.5))
        best = ((y0, y0 + side, x0, x0 + side), m)
        if side * voxel_um >= MIN_CROP_UM:
            break
    if best is None:
        side = int(min(cap, ny, nx, 32))
        y0, x0 = (ny - side) // 2, (nx - side) // 2
        best = ((y0, y0 + side, x0, x0 + side), float("nan"))
    return best


# ------------------------------------------------------------------ granulometry
def _edges_mids(diams_um):
    edges = np.concatenate([[0.0], diams_um])
    return edges, 0.5 * (edges[:-1] + edges[1:])


def _spectrum_from_G(G, diams_um):
    loss = np.clip(-np.diff(np.asarray(G, dtype=np.float64)), 0, None)
    total = loss.sum()
    return loss / total if total > 0 else loss


def grey_granulometry(img, diams_um, voxel_um, upsample=2, smooth_sigma_vox=0.7):
    smooth = ndi.gaussian_filter(img.astype(np.float64), sigma=smooth_sigma_vox)
    up = rescale(smooth, upsample, order=1, anti_aliasing=False, preserve_range=True)
    up_voxel_um = voxel_um / upsample
    G = [up.sum()]
    for d_um in diams_um:
        r_px = max(1, int(round((d_um / 2.0) / up_voxel_um)))
        G.append(ndi.grey_opening(up, footprint=disk(r_px)).sum())
    _, mids = _edges_mids(diams_um)
    return mids, _spectrum_from_G(G, diams_um)


def binary_granulometry(img, diams_um, voxel_um, upsample=2, smooth_sigma_vox=0.7):
    smooth = ndi.gaussian_filter(img.astype(np.float64), sigma=smooth_sigma_vox)
    binimg = smooth > threshold_otsu(smooth)
    up = (
        rescale(
            binimg.astype(np.float32),
            upsample,
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        )
        > 0.5
    )
    up_voxel_um = voxel_um / upsample
    G = [up.sum()]
    for d_um in diams_um:
        r_px = max(1, int(round((d_um / 2.0) / up_voxel_um)))
        G.append(ndi.binary_opening(up, structure=disk(r_px)).sum())
    _, mids = _edges_mids(diams_um)
    return mids, _spectrum_from_G(G, diams_um)


def peak_and_median(mids, ps):
    """Bin peak, bin median, parabolic sub-bin peak, linearly interpolated median (um)."""
    ps = np.asarray(ps, dtype=np.float64)
    if ps.sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan
    i = int(np.argmax(ps))
    peak = float(mids[i])
    cum = np.cumsum(ps) / ps.sum()
    mi = min(int(np.searchsorted(cum, 0.5)), len(mids) - 1)
    median = float(mids[mi])
    # parabolic interpolation around the max
    peak_i = peak
    if 0 < i < len(ps) - 1:
        ym, y0, yp = ps[i - 1], ps[i], ps[i + 1]
        den = ym - 2 * y0 + yp
        if den < 0:
            delta = float(np.clip(0.5 * (ym - yp) / den, -1, 1))
            peak_i = peak + delta * (mids[1] - mids[0])
    # median interpolated on the bin edges (cum reaches cum[k] at edge k+1)
    step = mids[1] - mids[0]
    edges = np.concatenate([[mids[0] - step / 2.0], mids + step / 2.0])
    cum0 = np.concatenate([[0.0], cum])
    med_i = float(np.interp(0.5, cum0, edges))
    return peak, median, float(peak_i), med_i


def radial_profile(img2d, voxel_um):
    img = img2d.astype(np.float64)
    img = img - img.mean()
    imgw = img * np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(imgw))) ** 2
    N0, N1 = img.shape
    cy, cx = N0 // 2, N1 // 2
    y, x = np.indices((N0, N1))
    r_int = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    nmax = min(cy, cx)
    idx = np.arange(0, nmax)
    P_radial = ndi.mean(P, r_int, index=idx)
    k = idx * (1.0 / (N0 * voxel_um))
    acf2d = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.ifftshift(P))))
    acf2d = acf2d / acf2d[cy, cx]
    C_radial = ndi.mean(acf2d, r_int, index=idx)
    return {"k": k, "P": P_radial, "r": idx * voxel_um, "C": C_radial}


def first_min_after_zero(r, C, min_r_um=3.0):
    mask = r >= min_r_um
    rr, CC = r[mask], C[mask]
    for i in range(1, len(CC) - 1):
        if CC[i] < CC[i - 1] and CC[i] < CC[i + 1]:
            return float(rr[i])
    below = np.where(CC < 0)[0]
    if len(below):
        return float(rr[below[0]])
    return float("nan")


def first_ring_peak(k, P, min_k=0.006):
    from scipy.signal import find_peaks

    mask = k >= min_k
    kk, PP = k[mask], P[mask]
    if len(PP) < 5:
        return float("nan")
    logP_s = ndi.uniform_filter1d(np.log10(PP + 1e-12), size=3)
    peaks, props = find_peaks(logP_s, prominence=0.05)
    if len(peaks) == 0:
        return float("nan")
    return float(kk[peaks[np.argmax(props["prominences"])]])


def analyse_crop(crop, voxel_um, diams_um):
    """All M4 quantities for one 2D crop."""
    mids, ps_g = grey_granulometry(crop, diams_um, voxel_um)
    _, ps_b = binary_granulometry(crop, diams_um, voxel_um)
    prof = radial_profile(crop, voxel_um)
    fmin = first_min_after_zero(prof["r"], prof["C"])
    kpeak = first_ring_peak(prof["k"], prof["P"])
    period = 1.0 / kpeak if (kpeak == kpeak and kpeak > 0) else float("nan")
    return {
        "mids": mids,
        "ps_g": ps_g,
        "ps_b": ps_b,
        "prof": prof,
        "autocorr_first_min_um": fmin,
        "ring_period_um": period,
    }


def _slice_job(args):
    crop, voxel_um, diams_um = args
    return analyse_crop(crop, voxel_um, diams_um)


def _map(jobs, n_workers):
    if n_workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(n_workers, len(jobs))) as ex:
            return list(ex.map(_slice_job, jobs))
    return [_slice_job(j) for j in jobs]


def _aggregate(results, mids):
    ps_g = np.mean([r["ps_g"] for r in results], axis=0)
    ps_b = np.mean([r["ps_b"] for r in results], axis=0)
    g = peak_and_median(mids, ps_g)
    b = peak_and_median(mids, ps_b)
    fm = [r["autocorr_first_min_um"] for r in results]
    rp = [r["ring_period_um"] for r in results]
    out = {
        "ps_g": ps_g,
        "ps_b": ps_b,
        "grey": g,
        "binary": b,
        "fmin": float(np.nanmean(fm)) if np.isfinite(fm).any() else float("nan"),
        "period": float(np.nanmean(rp)) if np.isfinite(rp).any() else float("nan"),
    }
    return out


def _interior(results):
    """Slices used for the median autocorrelation / power curves (drop first and last as the
    original figure did, when there are enough)."""
    return results[1:-1] if len(results) > 3 else results


def _median_profile(results):
    inter = _interior(results)
    r = inter[0]["prof"]["r"]
    k = inter[0]["prof"]["k"]
    C = np.median([x["prof"]["C"] for x in inter], axis=0)
    P = np.median([x["prof"]["P"] for x in inter], axis=0)
    return r, C, k, P


# ------------------------------------------------------------------ synthetic calibration
def synthetic_crop(diam_um, side_px, voxel_um, blur_sigma_um, fvf, seed):
    """Central (side_px x side_px) crop of a one-slice ``common.synthetic_bundle``."""
    s = (
        int(np.ceil(side_px / 0.636)) + 4
    )  # rope radius 0.45*s -> inscribed square 0.636*s
    vol, _ = common.synthetic_bundle(
        diam_um,
        shape_yx=(s, s),
        nz=1,
        voxel_um=voxel_um,
        fvf=fvf,
        blur_sigma_um=blur_sigma_um,
        seed=seed,
    )
    y0 = (s - side_px) // 2
    return vol[0, y0 : y0 + side_px, y0 : y0 + side_px]


# ------------------------------------------------------------------ figures
def _grey_opened(img, diam_um, voxel_um, upsample=2, smooth_sigma_vox=0.7):
    smooth = ndi.gaussian_filter(img.astype(np.float64), sigma=smooth_sigma_vox)
    up = rescale(smooth, upsample, order=1, anti_aliasing=False, preserve_range=True)
    r_px = max(1, int(round((diam_um / 2.0) / (voxel_um / upsample))))
    opened = ndi.grey_opening(up, footprint=disk(r_px))
    return rescale(
        opened, 1.0 / upsample, order=1, anti_aliasing=True, preserve_range=True
    )


def _fig_openings(path, label, rows, open_diams, voxel_um):
    import matplotlib.pyplot as plt

    ncol = len(open_diams) + 1
    fig, axes = plt.subplots(
        len(rows), ncol, figsize=(3.0 * ncol, 3.2 * len(rows) + 0.6), squeeze=False
    )
    for ri, (name, img) in enumerate(rows):
        vmax = float(np.percentile(img, 99.5))
        vmin = float(min(0.0, np.percentile(img, 0.5)))
        ax = axes[ri, 0]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}  original", fontsize=9)
        ax.axis("off")
        common_bar(ax, 50, voxel_um)
        for ci, d in enumerate(open_diams, start=1):
            ax = axes[ri, ci]
            ax.imshow(_grey_opened(img, d, voxel_um), cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"opened {d} um", fontsize=9)
            ax.axis("off")
    fig.suptitle(
        f"{label}: morphological opening of the interior crop (grey-level, disk SE)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def common_bar(ax, length_um, voxel_um):
    import matplotlib.patches as mpatches

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    W, H = abs(xlim[1] - xlim[0]), abs(ylim[0] - ylim[1])
    length_px = length_um / voxel_um
    if length_px > 0.7 * W:
        return
    box_h, box_w = 0.10 * H, length_px + 0.06 * W
    bx, by = min(xlim) + 0.03 * W, max(ylim) - 0.03 * H - box_h
    ax.add_patch(
        mpatches.Rectangle(
            (bx, by),
            box_w,
            box_h,
            facecolor="black",
            edgecolor="none",
            alpha=0.6,
            zorder=5,
        )
    )
    bar_y = by + box_h * 0.68
    ax.plot(
        [bx + 0.03 * W, bx + 0.03 * W + length_px],
        [bar_y, bar_y],
        color="white",
        lw=3,
        zorder=6,
        solid_capstyle="butt",
    )
    ax.text(
        bx + 0.03 * W + length_px / 2.0,
        by + box_h * 0.28,
        f"{int(length_um)} um",
        color="white",
        ha="center",
        va="center",
        fontsize=8,
        zorder=6,
    )


def _fig_spectrum(path, label, mids, curves, truth_d):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, kind, pk in [(axes[0], "ps_g", "grey"), (axes[1], "ps_b", "binary")]:
        for name, color, agg in curves:
            ax.plot(mids, agg[kind], color=color, lw=2, label=name)
            peak = agg[pk][2]  # sub-bin peak
            if np.isfinite(peak):
                val = np.interp(peak, mids, agg[kind])
                ax.annotate(
                    f"{peak:.1f}",
                    xy=(peak, val),
                    xytext=(peak, val * 1.04),
                    color=color,
                    fontsize=9,
                    ha="center",
                )
        if truth_d is not None:
            ax.axvline(
                truth_d,
                color="k",
                ls=":",
                lw=1.2,
                label=f"true mean D = {truth_d:.1f} um",
            )
        else:
            for d in (12, 20):
                ax.axvline(d, color="k", ls=":", lw=1)
        ax.set_xlabel("opening diameter (um)")
        ax.set_xlim(0, float(mids[-1]) + 1)
    axes[0].set_ylabel("normalised pattern spectrum (fraction of total loss)")
    axes[0].set_title("Grey-level granulometry")
    axes[1].set_title("Binary granulometry (Otsu-thresholded)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{label}: granulometric pattern spectra, sub-bin peaks labelled")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fig_autocorr(path, label, curves):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, color, r, C, k, P, fmin, period in curves:
        axes[0].plot(r, C, color=color, label=name)
        if fmin == fmin:
            axes[0].axvline(fmin, color=color, ls=":", lw=1)
            axes[0].annotate(
                f"{fmin:.0f}", xy=(fmin, -0.05), color=color, fontsize=8, ha="center"
            )
        axes[1].plot(k, P, color=color, label=name)
        if period == period:
            axes[1].axvline(1.0 / period, color=color, ls=":", lw=1)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlim(0, 60)
    axes[0].set_xlabel("r (um)")
    axes[0].set_ylabel("radial autocorrelation C(r)")
    axes[0].set_title("Autocorrelation (dotted = first minimum)")
    axes[0].legend(fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_xlim(0, 0.3)
    axes[1].set_xlabel("spatial frequency k (1/um)")
    axes[1].set_ylabel("radial power spectrum P(k)")
    axes[1].set_title("Power spectrum (dotted = dominant ring)")
    axes[1].legend(fontsize=8)
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ driver
def _clean(o):
    """JSON-safe copy: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def run(
    vol,
    voxel_um,
    out_dir,
    label,
    truth=None,
    fast=False,
    n_workers=None,
    blur_sigma_um=2.5,
    syn_fvf=0.4,
):
    """Run M4 on ``vol`` (nz, ny, nx float32, fibre axis = dim 0). Writes M4_openings.png,
    M4_spectrum.png, M4_autocorr.png, M4_summary.json into ``out_dir`` and returns the summary.

    ``blur_sigma_um`` / ``syn_fvf`` set the matched blur and fibre volume fraction of the
    synthetic calibration bundles.
    """
    import matplotlib

    matplotlib.use("Agg")

    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vol = np.asarray(vol)
    nz, ny, nx = vol.shape
    voxel_um = float(voxel_um)
    if n_workers is None:
        n_workers = max(1, os.cpu_count() or 1)
    diams = DIAMS_FAST_UM if fast else DIAMS_FULL_UM
    open_diams = OPEN_DIAMS_FAST if fast else OPEN_DIAMS_FULL

    # slices
    zs = common.analysis_z_slices(nz, n=6, skip=40)
    zs = sorted(set(zs))
    if fast:
        zs = (
            sorted({zs[len(zs) // 2], zs[max(0, len(zs) // 2 - 1)]})
            if len(zs) > 1
            else zs
        )
        zs = zs[:2]

    # crops
    crops, bboxes = [], []
    margin_used = None
    if truth is not None:
        bbox, margin_used = truth_crop_bbox(
            (ny, nx), voxel_um, truth.rope_radius_um, 80.0
        )
        y0, y1, x0, x1 = bbox
    for z in zs:
        if truth is not None:
            crop, bb = vol[z, y0:y1, x0:x1], bbox
        else:
            crop, bb = get_interior_crop(vol[z], voxel_um)
        crops.append(np.ascontiguousarray(crop))
        bboxes.append([int(v) for v in bb])
    side_px = crops[0].shape[0]

    res = _map([(c, voxel_um, diams) for c in crops], n_workers)
    mids = res[0]["mids"]
    agg = _aggregate(res, mids)

    per_slice = []
    for z, bb, r in zip(zs, bboxes, res):
        pg = peak_and_median(mids, r["ps_g"])
        pb = peak_and_median(mids, r["ps_b"])
        per_slice.append(
            {
                "z": int(z),
                "bbox": bb,
                "grey_peak_um": pg[0],
                "grey_median_um": pg[1],
                "grey_peak_interp_um": pg[2],
                "binary_peak_um": pb[0],
                "binary_median_um": pb[1],
                "binary_peak_interp_um": pb[2],
                "autocorr_first_min_um": r["autocorr_first_min_um"],
                "ring_period_um": r["ring_period_um"],
                "ps_g": r["ps_g"],
                "ps_b": r["ps_b"],
            }
        )

    # synthetic calibration
    truth_d = float(truth.mean_diam_um) if truth is not None else None
    syn_ds = []
    if truth_d is not None:
        syn_ds.append(float(round(truth_d)))
    if not fast or truth_d is None:
        for d in (12.0, 20.0):
            if d not in syn_ds:
                syn_ds.append(d)
    syn = {}
    syn_agg, syn_res, syn_crops = {}, {}, {}
    seeds = (1,) if fast else (1, 2)
    for d in syn_ds:
        scrops = [
            synthetic_crop(d, side_px, voxel_um, blur_sigma_um, syn_fvf, sd)
            for sd in seeds
        ]
        sres = _map([(c, voxel_um, diams) for c in scrops], n_workers)
        sagg = _aggregate(sres, mids)
        syn_agg[d], syn_res[d], syn_crops[d] = sagg, sres, scrops[0]
        g, b = sagg["grey"], sagg["binary"]
        syn[f"{int(d)}"] = {
            "true_diam_um": d,
            "grey_peak_um": g[0],
            "grey_median_um": g[1],
            "grey_peak_interp_um": g[2],
            "grey_median_interp_um": g[3],
            "binary_peak_um": b[0],
            "binary_median_um": b[1],
            "binary_peak_interp_um": b[2],
            "binary_median_interp_um": b[3],
            "autocorr_first_min_um": sagg["fmin"],
            "ring_period_um": sagg["period"],
        }

    # figures
    _fig_openings_rows = [(f"z={zs[len(zs) // 2]}", crops[len(zs) // 2])]
    if syn_ds:
        d_ref = syn_ds[0]
        _fig_openings_rows.append((f"synthetic D={int(d_ref)} um", syn_crops[d_ref]))
    _fig_openings(
        out_dir / "M4_openings.png", label, _fig_openings_rows, open_diams, voxel_um
    )

    palette = ["tab:green", "tab:red", "tab:purple", "tab:brown"]
    curves = [(f"{label} (avg {len(zs)} slices)", "tab:blue", agg)]
    for i, d in enumerate(syn_ds):
        curves.append(
            (f"synthetic D={int(d)} um (truth)", palette[i % len(palette)], syn_agg[d])
        )
    _fig_spectrum(out_dir / "M4_spectrum.png", label, mids, curves, truth_d)

    r_, C_, k_, P_ = _median_profile(res)
    ac_curves = [
        (
            f"{label} [{len(_interior(res))} interior z]",
            "tab:blue",
            r_,
            C_,
            k_,
            P_,
            first_min_after_zero(r_, C_),
            (lambda kp: 1.0 / kp if kp == kp and kp > 0 else float("nan"))(
                first_ring_peak(k_, P_)
            ),
        )
    ]
    for i, d in enumerate(syn_ds):
        sr = syn_res[d][0]["prof"]
        kp = first_ring_peak(sr["k"], sr["P"])
        ac_curves.append(
            (
                f"synthetic D={int(d)} um (truth)",
                palette[i % len(palette)],
                sr["r"],
                sr["C"],
                sr["k"],
                sr["P"],
                first_min_after_zero(sr["r"], sr["C"]),
                1.0 / kp if kp == kp and kp > 0 else float("nan"),
            )
        )
    _fig_autocorr(out_dir / "M4_autocorr.png", label, ac_curves)

    g, b = agg["grey"], agg["binary"]
    bias = lambda v: float(v - truth_d) if (truth_d is not None and v == v) else None
    summary = {
        "method": "M4",
        "label": label,
        "grey_peak_um": g[0],
        "grey_median_um": g[1],
        "grey_peak_interp_um": g[2],
        "grey_median_interp_um": g[3],
        "binary_peak_um": b[0],
        "binary_median_um": b[1],
        "binary_peak_interp_um": b[2],
        "binary_median_interp_um": b[3],
        "autocorr_first_min_um": agg["fmin"],
        "ring_period_um": agg["period"],
        "per_slice": per_slice,
        "mids_um": mids,
        "ps_g_mean": agg["ps_g"],
        "ps_b_mean": agg["ps_b"],
        "synthetic": syn,
        "truth_mean_diam_um": truth_d,
        "bias_grey_peak_um": bias(g[0]),
        "bias_grey_peak_interp_um": bias(g[2]),
        "bias_grey_median_um": bias(g[1]),
        "bias_binary_peak_um": bias(b[0]),
        "bias_binary_peak_interp_um": bias(b[2]),
        "bias_binary_median_um": bias(b[1]),
        "voxel_um": voxel_um,
        "crop_px": int(side_px),
        "crop_margin_um": margin_used,
        "z_slices": [int(z) for z in zs],
        "fast": bool(fast),
        "runtime_s": time.time() - t0,
    }
    summary = _clean(summary)
    with open(out_dir / "M4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary
