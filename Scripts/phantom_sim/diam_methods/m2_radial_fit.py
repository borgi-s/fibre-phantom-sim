"""Method M2: blurred-disk radial-profile fibre-diameter fit (torch-free port of the scratch code).

Fibre centres are local maxima of the lightly smoothed transverse slice (maxima closer than 8 um
are merged to handle bright double rims, then centroid-refined). Around every centre the
azimuthally averaged radial profile is fitted with a Gaussian-blurred solid disk (erf-edge model,
exact via the non-central chi-square CDF): a free-sigma fit and a shared-sigma fit (sigma = median of
the good free fits of the volume), plus a half-level crossing diameter. "Isolated" fibres have a
nearest neighbour farther than 1.3 D. A synthetic sanity check recovers known D from disks blurred
and noised to match the volume.

Volumes are (nz, ny, nx) float32 with the fibre axis along dim 0. Entry point: ``run``.
"""

import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt
from scipy.optimize import curve_fit
from scipy.special import chndtr
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402

from phantom_sim.diam_methods import common  # noqa: E402

ISO_FACTOR = 1.3


# ----------------------------------------------------------------------------- core
def sample_interior_mask(vol, z, voxel_um, half_avg=5, erode_um=40.0, blur_um=16.0):
    """Original real-data rule: blurred z-average -> Otsu -> largest blob -> >= 40 um from edge."""
    z0 = max(0, z - half_avg)
    z1 = min(vol.shape[0], z + half_avg + 1)
    img = vol[z0:z1].mean(axis=0)
    blurred = gaussian_filter(img, sigma=blur_um / voxel_um)
    mask = blurred > threshold_otsu(blurred)
    mask = ndi.binary_closing(mask, structure=np.ones((7, 7)))
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_opening(mask, structure=np.ones((7, 7)))
    lbl, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, lbl, range(1, n + 1))
        mask = lbl == (np.argmax(sizes) + 1)
    return mask & (distance_transform_edt(mask) * voxel_um >= erode_um)


def merge_close_peaks(coords, weight_img, voxel_um, merge_um=8.0):
    """Single-linkage merge of maxima closer than merge_um (KD-tree, memory-frugal)."""
    n = len(coords)
    if n == 0:
        return np.zeros((0, 2)), np.zeros((0,), dtype=int)
    if n == 1:
        return coords.astype(float), np.array([1])
    pts_um = coords * voxel_um
    pairs = cKDTree(pts_um).query_pairs(merge_um, output_type="ndarray")
    if len(pairs):
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, labels = connected_components(g, directed=False)
    else:
        labels = np.arange(n)
    merged, counts = [], []
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        pts = coords[idx].astype(float)
        w = weight_img[coords[idx, 0], coords[idx, 1]].astype(float)
        w = w - w.min() + 1e-9
        merged.append((pts * w[:, None]).sum(axis=0) / w.sum())
        counts.append(len(idx))
    return np.array(merged), np.array(counts)


def refine_center(img, center_rc, voxel_um, window_um=9.0, n_iter=3):
    c = np.array(center_rc, dtype=float)
    win_px = window_um / voxel_um
    for _ in range(n_iter):
        r0, c0 = c
        rmin = max(int(np.floor(r0 - win_px)), 0)
        rmax = min(int(np.ceil(r0 + win_px)) + 1, img.shape[0])
        cmin = max(int(np.floor(c0 - win_px)), 0)
        cmax = min(int(np.ceil(c0 + win_px)) + 1, img.shape[1])
        if rmax <= rmin or cmax <= cmin:
            break
        patch = img[rmin:rmax, cmin:cmax]
        rr, cc = np.mgrid[rmin:rmax, cmin:cmax]
        dist = np.sqrt((rr - r0) ** 2 + (cc - c0) ** 2) * voxel_um
        wmask = dist <= window_um
        vals = patch[wmask]
        if vals.size == 0:
            break
        w = np.clip(vals - np.percentile(vals, 10), 0, None)
        if w.sum() <= 0:
            break
        newc = np.array(
            [
                (rr[wmask].astype(float) * w).sum() / w.sum(),
                (cc[wmask].astype(float) * w).sum() / w.sum(),
            ]
        )
        done = np.linalg.norm(newc - c) < 0.02
        c = newc
        if done:
            break
    return c


def detect_centers(
    img,
    interior_mask,
    voxel_um,
    smooth_um=2.0,
    min_distance_um=4.0,
    merge_um=8.0,
    refine_window_um=9.0,
):
    smooth = gaussian_filter(img, sigma=smooth_um / voxel_um)
    vals = smooth[interior_mask]
    if vals.size < 10:
        return np.zeros((0, 2)), 0, 0
    thr = threshold_otsu(vals)
    coords = peak_local_max(
        smooth,
        min_distance=max(1, int(round(min_distance_um / voxel_um))),
        threshold_abs=thr,
        labels=interior_mask.astype(int),
    )
    raw_n = len(coords)
    merged, counts = merge_close_peaks(coords, smooth, voxel_um, merge_um)
    n_events = int(np.sum(counts > 1))
    if len(merged) == 0:
        return np.zeros((0, 2)), raw_n, n_events
    refined = np.array(
        [refine_center(img, c, voxel_um, refine_window_um) for c in merged]
    )
    ri = np.rint(refined[:, 0]).astype(int)
    ci = np.rint(refined[:, 1]).astype(int)
    ok = (
        (ri >= 0)
        & (ri < interior_mask.shape[0])
        & (ci >= 0)
        & (ci < interior_mask.shape[1])
    )
    keep = np.zeros(len(refined), dtype=bool)
    keep[ok] = interior_mask[ri[ok], ci[ok]]
    return refined[keep], raw_n, n_events


def radial_profiles(
    img, centers_rc, voxel_um, r_max_um=20.0, dr_um=0.5, n_theta=72, chunk=400
):
    """Azimuthally averaged profiles for all centres: (radii (nr,), I (N, nr))."""
    radii = np.arange(dr_um / 2, r_max_um, dr_um)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    RR, TT = np.meshgrid(radii, th, indexing="ij")
    dr_px = (RR / voxel_um) * np.sin(TT)
    dc_px = (RR / voxel_um) * np.cos(TT)
    out = np.zeros((len(centers_rc), len(radii)))
    for a in range(0, len(centers_rc), chunk):
        c = np.asarray(centers_rc[a : a + chunk], dtype=float)
        rows = c[:, 0, None, None] + dr_px[None]
        cols = c[:, 1, None, None] + dc_px[None]
        s = map_coordinates(img, [rows, cols], order=1, mode="nearest")
        out[a : a + chunk] = s.mean(axis=2)
    return radii, out


def disk_model(r, A, B, R, s):
    """Azimuthal average of a solid disk (radius R, levels A inside / B outside) blurred by sigma s."""
    x = (R**2) / (s**2)
    nc = (np.asarray(r, dtype=float) / s) ** 2
    return B + (A - B) * chndtr(x, 2.0, nc)


def half_level_crossing(r, I, A=None, B=None):
    if A is None:
        A = I[:2].mean()
    if B is None:
        B = I[-4:].mean()
    half = (A + B) / 2.0
    below = I < half if A >= B else I > half
    idx = np.where(below)[0]
    if len(idx) == 0:
        return np.nan
    i1 = idx[0]
    if i1 == 0:
        return r[0]
    i0 = i1 - 1
    if I[i1] == I[i0]:
        return r[i1]
    frac = (half - I[i0]) / (I[i1] - I[i0])
    return r[i0] + frac * (r[i1] - r[i0])


def fit_profile(r, I, s_fixed=None, s0=1.5):
    A0 = float(I[:2].mean())
    B0 = float(I[-4:].mean())
    R0 = half_level_crossing(r, I, A0, B0)
    if not np.isfinite(R0) or R0 <= 0.3:
        R0 = 6.0
    R0 = float(np.clip(R0, 1.0, 14.0))
    span = max(abs(A0 - B0), 1e-9)
    lo = min(I.min(), A0, B0) - 3 * span - 1e-9
    hi = max(I.max(), A0, B0) + 3 * span + 1e-9
    try:
        if s_fixed is None:
            popt, _ = curve_fit(
                disk_model,
                r,
                I,
                p0=[A0, B0, R0, s0],
                bounds=([lo, lo, 0.5, 0.2], [hi, hi, 15.0, 6.0]),
                maxfev=8000,
            )
            A, B, R, s = popt
            pred = disk_model(r, *popt)
        else:

            def f(rr, A, B, R):
                return disk_model(rr, A, B, R, s_fixed)

            popt, _ = curve_fit(
                f,
                r,
                I,
                p0=[A0, B0, R0],
                bounds=([lo, lo, 0.5], [hi, hi, 15.0]),
                maxfev=8000,
            )
            A, B, R = popt
            s = s_fixed
            pred = f(r, *popt)
    except Exception:
        return None
    ss_res = float(np.sum((I - pred) ** 2))
    ss_tot = float(np.sum((I - I.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return dict(
        A=float(A), B=float(B), R=float(R), s=float(s), D=float(2 * R), r2=float(r2)
    )


def is_bad_fit(fit, r2_min=0.85, s_lo=0.2, s_hi=5.5, R_lo=0.6, R_hi=14.5):
    if fit is None:
        return True
    if not np.isfinite(fit["r2"]) or fit["r2"] < r2_min:
        return True
    if not (s_lo < fit["s"] < s_hi):
        return True
    if not (R_lo < fit["R"] < R_hi):
        return True
    return False


def _fit_chunk(args):
    r, Iarr, s_fixed = args
    return [fit_profile(r, I, s_fixed=s_fixed) for I in Iarr]


def fit_all(r, Iarr, s_fixed, pool, chunk=64):
    """Fit every profile row (in a process pool if given); returns list of dict|None in order."""
    tasks = [(r, Iarr[a : a + chunk], s_fixed) for a in range(0, len(Iarr), chunk)]
    if pool is None:
        res = map(_fit_chunk, tasks)
    else:
        res = pool.map(_fit_chunk, tasks)
    out = []
    for part in res:
        out.extend(part)
    return out


def nearest_neighbor_um(centers_rc, voxel_um):
    if len(centers_rc) < 2:
        return np.full(len(centers_rc), np.nan)
    pts = np.asarray(centers_rc) * voxel_um
    d, _ = cKDTree(pts).query(pts, k=2)
    return d[:, 1]


# ----------------------------------------------------------------------------- synthetic
def make_synthetic(
    D_true,
    s_true,
    A,
    B,
    noise_std,
    voxel_um,
    rng,
    box_um=400.0,
    min_sep_factor=1.6,
    n_target=30,
):
    box_px = int(round(box_um / voxel_um))
    R_true = D_true / 2.0
    min_sep = min_sep_factor * D_true
    margin = D_true * 1.2
    centers = []
    attempts = 0
    while len(centers) < n_target and attempts < 40000:
        attempts += 1
        y = rng.uniform(margin, box_um - margin)
        x = rng.uniform(margin, box_um - margin)
        if all(np.hypot(y - cy, x - cx) >= min_sep for cy, cx in centers):
            centers.append((y, x))
    yy, xx = np.mgrid[0:box_px, 0:box_px]
    yy_um, xx_um = yy * voxel_um, xx * voxel_um
    img = np.full((box_px, box_px), B, dtype=np.float64)
    for cy, cx in centers:
        img[np.hypot(yy_um - cy, xx_um - cx) <= R_true] = A
    img = gaussian_filter(img, sigma=s_true / voxel_um)
    img = img + rng.normal(0, noise_std, size=img.shape)
    return img.astype(np.float32)


def measure_contrast_noise(img, interior, voxel_um):
    smooth = gaussian_filter(img, sigma=2.0 / voxel_um)
    vals = smooth[interior]
    thr = threshold_otsu(vals)
    fibre = img[interior & (smooth > thr)]
    matrix = img[interior & (smooth <= thr)]
    A = float(np.percentile(fibre, 75))
    B = float(np.percentile(matrix, 25))
    hp = img - gaussian_filter(img, sigma=1.0)
    noise = float(np.std(hp[interior & (smooth <= thr)]))
    return A, B, noise


def synthetic_check(D_true, s_true, A, B, noise, voxel_um, rng, pool):
    img = make_synthetic(D_true, s_true, A, B, noise, voxel_um, rng)
    h, w = img.shape
    m = int(round(20.0 / voxel_um))
    interior = np.zeros(img.shape, dtype=bool)
    interior[m : h - m, m : w - m] = True
    centers, _, _ = detect_centers(img, interior, voxel_um)
    if len(centers) == 0:
        Ds = ss = np.array([])
        n_tot = 0
    else:
        r, P = radial_profiles(img, centers, voxel_um)
        fits = fit_all(r, P, None, pool)
        good = [f for f in fits if not is_bad_fit(f)]
        Ds = np.array([f["D"] for f in good])
        ss = np.array([f["s"] for f in good])
        n_tot = len(fits)
    res = dict(
        D_true=float(D_true),
        n_detected=int(len(centers)),
        n_good=int(len(Ds)),
        n_total=int(n_tot),
        D_mean=float(Ds.mean()) if len(Ds) else None,
        D_std=float(Ds.std()) if len(Ds) else None,
        D_median=float(np.median(Ds)) if len(Ds) else None,
        s_mean=float(ss.mean()) if len(ss) else None,
        s_std=float(ss.std()) if len(ss) else None,
        A=A,
        B=B,
        noise_std=noise,
        s_true=float(s_true),
    )
    return res, Ds


# ----------------------------------------------------------------------------- figures
def _fmt(s):
    if s["n"] == 0:
        return "n=0"
    return (
        f"n={s['n']}  mean={s['mean']:.2f}+/-{s['std']:.2f}um  "
        f"median={s['median']:.2f} [{s['q1']:.2f},{s['q3']:.2f}]um"
    )


def _ref_lines(truth):
    """[(D_um, colour, label)] reference diameters: true mean D, or the original 12/20 um."""
    if truth is not None:
        return [
            (truth.mean_diam_um, "tab:red", f"true mean D={truth.mean_diam_um:.1f}um")
        ]
    return [(12.0, "tab:cyan", "D=12um ref"), (20.0, "tab:orange", "D=20um ref")]


def fig_examples(path, label, exs, voxel_um, truth, raw_txt):
    n = len(exs)
    fig, axs = plt.subplots(n, 2, figsize=(9.5, 3.05 * n), squeeze=False)
    refs = _ref_lines(truth)
    for i, ex in enumerate(exs):
        ax_im, ax_pr = axs[i]
        crop = ex["crop"]
        h, w = crop.shape
        extent = [
            -w / 2 * voxel_um,
            w / 2 * voxel_um,
            h / 2 * voxel_um,
            -h / 2 * voxel_um,
        ]
        vmin, vmax = np.percentile(crop, [1, 99])
        ax_im.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax, extent=extent)
        for D, col, _ in refs:
            ax_im.add_patch(
                mpatches.Circle(
                    (0, 0), D / 2, fill=False, ec=col, lw=1.0, ls=(0, (2, 2))
                )
            )
        ax_im.add_patch(
            mpatches.Circle((0, 0), ex["free"]["R"], fill=False, ec="lime", lw=1.6)
        )
        ax_im.add_patch(
            mpatches.Circle(
                (0, 0), ex["shared"]["R"], fill=False, ec="magenta", lw=1.2, ls="--"
            )
        )
        ttl = f"z={ex['z']}  NN={ex['nn']:.1f}um"
        if np.isfinite(ex["d_true"]):
            ttl += f"  true D={ex['d_true']:.1f}um"
        ax_im.set_title(f"{label} {ttl}", fontsize=9)
        ax_im.set_xlabel("um")
        ax_im.set_ylabel("um")
        r, I = ex["r"], ex["I"]
        ax_pr.plot(r, I, "o", ms=3, color="k", label="data (azimuthal avg)")
        rr = np.linspace(0, r.max(), 200)
        f, s = ex["free"], ex["shared"]
        ax_pr.plot(
            rr,
            disk_model(rr, f["A"], f["B"], f["R"], f["s"]),
            "-",
            color="lime",
            lw=1.6,
            label=f"free-s fit D={f['D']:.2f}um s={f['s']:.2f}um R2={f['r2']:.3f}",
        )
        ax_pr.plot(
            rr,
            disk_model(rr, s["A"], s["B"], s["R"], s["s"]),
            "--",
            color="magenta",
            lw=1.4,
            label=f"shared-s fit D={s['D']:.2f}um s={s['s']:.2f}um R2={s['r2']:.3f}",
        )
        rh = half_level_crossing(r, I, float(I[:2].mean()), float(I[-4:].mean()))
        if np.isfinite(rh):
            ax_pr.axvline(
                rh,
                color="tab:blue",
                lw=1.2,
                ls=":",
                label=f"half-level D={2 * rh:.2f}um",
            )
        for D, col, lbl in refs:
            ax_pr.axvline(D / 2, color=col, lw=1.0, ls=(0, (2, 2)), label=lbl)
        ax_pr.legend(fontsize=6.5, loc="best")
        ax_pr.set_xlabel("r (um)")
        ax_pr.set_ylabel("intensity")
        ax_pr.set_title(f"radial profile ({label} #{i})", fontsize=9)
    ref_txt = "true mean D" if truth is not None else "12/20um"
    fig.suptitle(
        f"{label}: M2 examples: crop with fitted circle (green=free-s, magenta dashed=shared-s,\n"
        f"dashed = {ref_txt} reference) and radial profile with fit + half-level crossing\n"
        + raw_txt,
        fontsize=7.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_overlay(path, label, ov, voxel_um):
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.5))
    crop = ov["crop"]
    h, w = crop.shape
    vmin, vmax = np.percentile(crop, [1, 99])
    ax.imshow(
        crop,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        extent=[0, w * voxel_um, h * voxel_um, 0],
    )
    n_acc = n_rej = 0
    for yy, xx, D, bad in ov["rows"]:
        if not np.isfinite(D) or D <= 0:
            continue
        n_rej += bool(bad)
        n_acc += not bad
        ax.add_patch(
            mpatches.Circle(
                (xx * voxel_um, yy * voxel_um),
                D / 2,
                fill=False,
                ec="red" if bad else "lime",
                lw=0.9,
                alpha=0.85,
            )
        )
    ax.set_title(
        f"{label}: z={ov['z']}  {w * voxel_um:.0f}x{h * voxel_um:.0f}um crop\n"
        f"accepted(green)={n_acc}  rejected(red)={n_rej}",
        fontsize=10,
    )
    ax.set_xlabel("um")
    ax.set_ylabel("um")
    fig.suptitle(
        f"{label}: M2 overlay: all fitted circles (free-s blurred-disk fit), colour = accept/reject",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fig_distribution(
    path,
    label,
    D_shared_all,
    D_shared_iso,
    D_half_all,
    D_half_iso,
    s_shared,
    syn,
    syn_Ds,
    truth,
):
    refs = _ref_lines(truth)
    hi = max(30.0, 2.2 * refs[0][0])
    bins = np.linspace(4, hi, 40)
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9])
    for j, (a, b, col, ttl) in enumerate(
        [
            (
                D_shared_all,
                D_shared_iso,
                "tab:blue",
                f"blurred-disk fit D (shared-s={s_shared:.2f}um)",
            ),
            (D_half_all, D_half_iso, "tab:green", "half-level crossing D"),
        ]
    ):
        ax = fig.add_subplot(gs[0, j])
        ax.hist(
            a,
            bins=bins,
            color="tab:gray",
            alpha=0.55,
            label=f"all accepted (n={len(a)})",
        )
        ax.hist(b, bins=bins, color=col, alpha=0.75, label=f"isolated (n={len(b)})")
        for D, c, lbl in refs:
            ax.axvline(
                D,
                color="k" if truth is None else c,
                lw=1,
                ls="--",
                label=lbl if truth else None,
            )
        ax.set_title(
            f"{label}: {ttl}\nall: {_fmt(common.summary_stats(a))}\n"
            f"iso: {_fmt(common.summary_stats(b))}",
            fontsize=8.3,
        )
        ax.set_xlabel("D (um)")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
    ax3 = fig.add_subplot(gs[1, :])
    keys = list(syn.keys())
    for i, k in enumerate(keys):
        Ds = syn_Ds[k]
        Dt = syn[k]["D_true"]
        if len(Ds):
            ax3.boxplot(
                [Ds],
                positions=[i],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor="tab:orange", alpha=0.6),
            )
        ax3.plot([i - 0.3, i + 0.3], [Dt, Dt], color="k", lw=2, ls="--")
        top = np.nanmax(Ds) if len(Ds) else Dt
        m = syn[k]["D_mean"]
        txt = f"true={Dt:.0f}um"
        if m is not None:
            txt += f"\nrecov={m:.2f}+/-{syn[k]['D_std']:.2f}um\n(n={syn[k]['n_good']}/{syn[k]['n_total']} good)"
        ax3.text(i, top + 0.04 * Dt, txt, ha="center", va="bottom", fontsize=7)
    ax3.set_xticks(range(len(keys)))
    ax3.set_xticklabels(keys)
    ax3.set_xlim(-0.6, len(keys) - 0.4)
    ax3.margins(y=0.35)
    ax3.set_ylabel("recovered D (um)")
    ax3.set_title(
        f"{label}: synthetic sanity check: known-D disks, blur sigma={s_shared:.2f}um + noise "
        "matched to this volume, recovered via the same pipeline",
        fontsize=9,
    )
    fig.suptitle(
        f"{label}: M2 diameter distributions: blurred-disk fit vs half-level crossing, "
        "all vs isolated fibres",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=125)
    plt.close(fig)


def fig_truth_scatter(path, label, d_true, iso, methods, truth):
    """methods: list of (name, values, accepted_mask). Rows: all accepted, isolated only."""
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
    rng = np.random.default_rng(0)
    lim_lo = 0.6 * np.nanmin(d_true) if np.isfinite(d_true).any() else 4
    lim_hi = 1.6 * np.nanmax(d_true) if np.isfinite(d_true).any() else 30
    stats = {}
    for ri, (rname, sel_iso) in enumerate(
        [("all accepted", False), ("isolated", True)]
    ):
        for ci, (name, vals, acc) in enumerate(methods):
            ax = axs[ri, ci]
            m = acc & np.isfinite(d_true) & np.isfinite(vals)
            if sel_iso:
                m = m & iso
            x, y = d_true[m], vals[m]
            xj = x + rng.uniform(-0.15, 0.15, x.shape)
            ax.scatter(
                xj, y, s=6, alpha=0.3, color="tab:blue" if ri == 0 else "tab:green"
            )
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="identity")
            if len(x):
                bias = float(np.mean(y - x))
                rmse = float(np.sqrt(np.mean((y - x) ** 2)))
                stats[f"{name}_{'iso' if sel_iso else 'all'}"] = dict(
                    n=int(len(x)), bias_um=bias, rmse_um=rmse
                )
                ax.text(
                    0.04,
                    0.95,
                    f"n={len(x)}\nbias={bias:+.2f}um\nRMSE={rmse:.2f}um",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=9,
                    bbox=dict(fc="w", alpha=0.8, ec="0.7"),
                )
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
            ax.set_xlabel("true D (um, per fibre; x jittered +/-0.15um)")
            ax.set_ylabel("measured D (um)")
            ax.set_title(f"{name}: {rname}", fontsize=9)
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(
        f"{label}: M2 measured vs true per-fibre diameter (true mean D={truth.mean_diam_um:.2f}um)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return stats


# ----------------------------------------------------------------------------- driver
def _default_workers():
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(n, 16))


def run(vol, voxel_um, out_dir, label, truth=None, fast=False, n_workers=None):
    """Run M2 on ``vol`` (nz, ny, nx float32, fibre axis = dim 0). See module docstring."""
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    vol = np.asarray(vol, dtype=np.float32)
    nz, ny, nx = vol.shape
    voxel_um = float(voxel_um)
    n_workers = _default_workers() if n_workers is None else max(1, int(n_workers))
    rng = np.random.default_rng(20260928)
    zs = common.analysis_z_slices(nz, n=2 if fast else 6, skip=40)
    max_fibres_per_slice = 40 if fast else None

    pool = ProcessPoolExecutor(max_workers=n_workers) if n_workers > 1 else None
    try:
        # ---- detection + profiles per slice
        slices = []  # dict(z, img, interior, centers, nn, P, d_true)
        raw_counts = []
        radii = None
        for z in zs:
            img = vol[z]
            if truth is not None:
                interior = common.rope_interior_mask(
                    (ny, nx), voxel_um, truth.rope_radius_um, margin_um=80.0
                )
            else:
                interior = sample_interior_mask(vol, z, voxel_um)
            centers, raw_n, n_ev = detect_centers(img, interior, voxel_um)
            nn = nearest_neighbor_um(centers, voxel_um)
            if max_fibres_per_slice and len(centers) > max_fibres_per_slice:
                sel = np.sort(
                    rng.choice(len(centers), max_fibres_per_slice, replace=False)
                )
                centers, nn = centers[sel], nn[sel]
            if len(centers):
                radii, P = radial_profiles(img, centers, voxel_um)
            else:
                P = np.zeros((0, 1))
            if truth is not None and len(centers):
                d_true, _ = truth.nearest_true_diam(
                    z, centers[:, 0], centers[:, 1], (ny, nx)
                )
            else:
                d_true = np.full(len(centers), np.nan)
            raw_counts.append((int(z), int(raw_n), int(len(centers)), int(n_ev)))
            slices.append(
                dict(
                    z=int(z),
                    img=img,
                    interior=interior,
                    centers=centers,
                    nn=nn,
                    P=P,
                    d_true=d_true,
                )
            )
        n_det = sum(len(s["centers"]) for s in slices)

        # ---- pass 1: free-s fits; shared sigma
        for s in slices:
            s["free"] = fit_all(radii, s["P"], None, pool) if len(s["centers"]) else []
        good_s = [f["s"] for s in slices for f in s["free"] if not is_bad_fit(f)]
        shared_s = float(np.median(good_s)) if good_s else 1.5
        # ---- pass 2: shared-s fits
        for s in slices:
            s["shared"] = (
                fit_all(radii, s["P"], shared_s, pool) if len(s["centers"]) else []
            )

        # ---- per-fibre arrays
        cols = {
            k: []
            for k in (
                "z",
                "idx",
                "y_um",
                "x_um",
                "nn_um",
                "D_free",
                "s_free",
                "r2_free",
                "bad_free",
                "D_shared",
                "r2_shared",
                "bad_shared",
                "D_half",
                "D_true",
            )
        }
        nan = np.nan
        for s in slices:
            for i, c in enumerate(s["centers"]):
                ff, fs = s["free"][i], s["shared"][i]
                I = s["P"][i]
                rh = half_level_crossing(
                    radii, I, float(I[:2].mean()), float(I[-4:].mean())
                )
                cols["z"].append(s["z"])
                cols["idx"].append(i)
                cols["y_um"].append(float(c[0] * voxel_um))
                cols["x_um"].append(float(c[1] * voxel_um))
                cols["nn_um"].append(
                    float(s["nn"][i]) if np.isfinite(s["nn"][i]) else nan
                )
                cols["D_free"].append(ff["D"] if ff else nan)
                cols["s_free"].append(ff["s"] if ff else nan)
                cols["r2_free"].append(ff["r2"] if ff else nan)
                cols["bad_free"].append(is_bad_fit(ff))
                cols["D_shared"].append(fs["D"] if fs else nan)
                cols["r2_shared"].append(fs["r2"] if fs else nan)
                cols["bad_shared"].append(
                    is_bad_fit(
                        fs, r2_min=0.80, s_lo=shared_s * 0.5, s_hi=shared_s * 2.0
                    )
                )
                cols["D_half"].append(2 * rh if np.isfinite(rh) else nan)
                cols["D_true"].append(float(s["d_true"][i]))
        A_ = {k: np.asarray(v) for k, v in cols.items()}
        acc_sh = (~A_["bad_shared"]) & np.isfinite(A_["D_shared"])
        acc_fr = (~A_["bad_free"]) & np.isfinite(A_["D_free"])
        acc_hf = np.isfinite(A_["D_half"]) & (A_["D_half"] > 4) & (A_["D_half"] < 30)
        iso = (
            np.isfinite(A_["nn_um"])
            & (A_["nn_um"] > ISO_FACTOR * A_["D_shared"])
            & acc_sh
        )

        stats = dict(
            fit_shared_all=common.summary_stats(A_["D_shared"][acc_sh]),
            fit_shared_iso=common.summary_stats(A_["D_shared"][acc_sh & iso]),
            fit_free_all=common.summary_stats(A_["D_free"][acc_fr]),
            half_level_all=common.summary_stats(A_["D_half"][acc_hf]),
            half_level_iso=common.summary_stats(A_["D_half"][acc_hf & iso]),
        )

        # ---- synthetic sanity check (contrast/noise/blur from this volume)
        zc = zs[len(zs) // 2]
        interior_c = slices[len(slices) // 2]["interior"]
        try:
            A_c, B_c, noise_c = measure_contrast_noise(vol[zc], interior_c, voxel_um)
        except Exception:
            A_c, B_c, noise_c = (
                float(np.percentile(vol[zc], 90)),
                float(np.percentile(vol[zc], 10)),
                float(np.std(vol[zc] - gaussian_filter(vol[zc], 1.0))),
            )
        if truth is not None:
            d_list = [float(round(truth.mean_diam_um))] + ([] if fast else [12.0, 20.0])
        else:
            d_list = [12.0] if fast else [12.0, 20.0]
        seen = []
        d_list = [d for d in d_list if not (d in seen or seen.append(d))]
        syn, syn_Ds = {}, {}
        for D in d_list:
            res, Ds = synthetic_check(
                D, shared_s, A_c, B_c, noise_c, voxel_um, rng, pool
            )
            key = f"D{int(D)}"
            syn[key], syn_Ds[key] = res, Ds
    finally:
        if pool is not None:
            pool.shutdown()

    # ---- figures
    raw_txt = (
        f"raw peaks {sum(x[1] for x in raw_counts)} -> fitted centers {sum(x[2] for x in raw_counts)} "
        f"({sum(x[3] for x in raw_counts)} merge events, close maxima consolidated at 8um)"
    )
    # examples: 3 well-fit isolated fibres (free-s), low/mid/high D
    pool_ex = []
    for s in slices:
        for i in range(len(s["centers"])):
            f = s["free"][i]
            if (
                f is not None
                and not is_bad_fit(f)
                and np.isfinite(s["nn"][i])
                and f["D"] > 0
                and s["nn"][i] > ISO_FACTOR * f["D"]
            ):
                pool_ex.append((f["D"], s, i))
    pool_ex.sort(key=lambda t: t[0])
    if len(pool_ex) >= 3:
        picks = [
            pool_ex[int(len(pool_ex) * 0.15)],
            pool_ex[len(pool_ex) // 2],
            pool_ex[int(len(pool_ex) * 0.85)],
        ]
    elif pool_ex:
        picks = pool_ex
    else:
        picks = [(0, s, i) for s in slices for i in range(len(s["centers"]))][:3]
    exs = []
    half_px = int(round(14.0 / voxel_um)) + 2
    for _, s, i in picks:
        c = s["centers"][i]
        r0, c0 = int(round(c[0])), int(round(c[1]))
        crop = s["img"][
            max(0, r0 - half_px) : r0 + half_px, max(0, c0 - half_px) : c0 + half_px
        ]
        if s["free"][i] is None or s["shared"][i] is None:
            continue
        exs.append(
            dict(
                z=s["z"],
                crop=crop,
                r=radii,
                I=s["P"][i],
                free=s["free"][i],
                shared=s["shared"][i],
                nn=float(s["nn"][i]) if np.isfinite(s["nn"][i]) else np.nan,
                d_true=float(s["d_true"][i]),
            )
        )
    if exs:
        fig_examples(
            os.path.join(out_dir, "M2_examples.png"),
            label,
            exs,
            voxel_um,
            truth,
            raw_txt,
        )

    mid = slices[len(slices) // 2]
    half_ov = max(4, min(int(round(100.0 / voxel_um)), ny // 2, nx // 2))
    r0o, r1o = ny // 2 - half_ov, ny // 2 + half_ov
    c0o, c1o = nx // 2 - half_ov, nx // 2 + half_ov
    ov_rows = []
    for i, c in enumerate(mid["centers"]):
        yy, xx = c[0] - r0o, c[1] - c0o
        if 0 <= yy < 2 * half_ov and 0 <= xx < 2 * half_ov:
            f = mid["free"][i]
            ov_rows.append((yy, xx, f["D"] if f else np.nan, is_bad_fit(f)))
    fig_overlay(
        os.path.join(out_dir, "M2_overlay.png"),
        label,
        dict(z=mid["z"], crop=mid["img"][r0o:r1o, c0o:c1o], rows=ov_rows),
        voxel_um,
    )

    fig_distribution(
        os.path.join(out_dir, "M2_distribution.png"),
        label,
        A_["D_shared"][acc_sh],
        A_["D_shared"][acc_sh & iso],
        A_["D_half"][acc_hf],
        A_["D_half"][acc_hf & iso],
        shared_s,
        syn,
        syn_Ds,
        truth,
    )

    bias = rmse = None
    truth_stats = None
    if truth is not None:
        fig_stats = fig_truth_scatter(
            os.path.join(out_dir, "M2_truth_scatter.png"),
            label,
            A_["D_true"],
            iso,
            [
                ("fit D shared-s", A_["D_shared"], acc_sh),
                ("fit D free-s", A_["D_free"], acc_fr),
                ("half-level D", A_["D_half"], acc_hf),
            ],
            truth,
        )
        m = acc_sh & np.isfinite(A_["D_true"])
        if m.any():
            dd = A_["D_shared"][m] - A_["D_true"][m]
            bias, rmse = float(dd.mean()), float(np.sqrt(np.mean(dd**2)))
        truth_stats = fig_stats

    # ---- CSV
    with open(
        os.path.join(out_dir, "M2_fibres.csv"), "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        names = list(cols.keys()) + ["isolated", "shared_s_volume"]
        w.writerow(names)
        for k in range(len(A_["z"])):
            w.writerow(
                [
                    A_[c][k].item() if hasattr(A_[c][k], "item") else A_[c][k]
                    for c in cols
                ]
                + [bool(iso[k]), shared_s]
            )

    out = dict(
        method="M2",
        label=label,
        **stats,
        sigma_shared_um=shared_s,
        n_detected=int(n_det),
        n_raw_peaks=int(sum(x[1] for x in raw_counts)),
        slices=[int(z) for z in zs],
        voxel_um=voxel_um,
        synthetic=syn,
        truth_mean_diam_um=float(truth.mean_diam_um) if truth is not None else None,
        bias_fit_um=bias,
        rmse_fit_um=rmse,
        truth_scatter_stats=truth_stats,
        runtime_s=float(time.time() - t_start),
        fast=bool(fast),
    )
    common.save_json(out, os.path.join(out_dir, "M2_summary.json"))
    return out
