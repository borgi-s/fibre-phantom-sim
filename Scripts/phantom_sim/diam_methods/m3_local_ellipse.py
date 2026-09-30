# -*- coding: utf-8 -*-
"""M3 local-threshold ellipse fibre-diameter method (torch-free, voxel-size aware).

Ported from the scratch M3 code (m3_core.py + m3_full_run.py). Per xy slice:
  1. interior mask: analytic rope disc (when ground truth is given) or, for real data, Otsu on a
     heavily smoothed silhouette (largest component, filled, eroded 80 um).
  2. seeds = local maxima of a lightly smoothed slice (sigma 3 um), seeds closer than 8 um merged.
  3. per-fibre local threshold: boundary = matrix + frac * (peak - matrix), peak = 90th pct in a
     4 um radius disk at the seed, matrix = 15th pct in a 10-20 um annulus.
  4. marker watershed on the inverted (sigma 2 um) intensity; each basin trimmed to its own
     boundary level, holes filled; regionprops ellipse fit; quality gates.
All um constants are converted to pixels with ``voxel_um``.

Entry point: ``run(vol, voxel_um, out_dir, label, truth=None, fast=False, n_workers=None)``.
"""

import csv
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import find_contours, regionprops
from skimage.segmentation import watershed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse, FancyBboxPatch  # noqa: E402

from phantom_sim.diam_methods import common  # noqa: E402

# ---- parameters in um (converted to px inside Params) ----
SIGMA_SEED_UM = 3.0
SIGMA_SEG_UM = 2.0
MIN_SEED_SEP_UM = 8.0
ANNULUS_IN_UM = 10.0
ANNULUS_OUT_UM = 20.0
PEAK_R_UM = 4.0
INTERIOR_ERODE_UM = 80.0
SIGMA_BOUNDARY_UM = 20.0
MATRIX_PCTL = 15
PEAK_PCTL = 90
ELONGATION_MAX = 1.8
SOLIDITY_MIN = 0.85
MIN_AREA_PX = 4
FRACS = (0.4, 0.5, 0.6)

GATE_COLORS = {
    "accepted": "#22c55e",
    "border": "#ef4444",
    "low_contrast": "#f59e0b",
    "elongated": "#a855f7",
    "low_solidity": "#7c2d12",
    "degenerate": "#eab308",
}


class Params:
    def __init__(self, voxel_um):
        v = float(voxel_um)
        self.voxel_um = v
        self.sigma_seed = SIGMA_SEED_UM / v
        self.sigma_seg = SIGMA_SEG_UM / v
        self.merge_radius_px = MIN_SEED_SEP_UM / v
        self.min_distance_px = max(1, int(round(MIN_SEED_SEP_UM / v)))
        self.sigma_boundary = SIGMA_BOUNDARY_UM / v
        self.erode_px = max(1, int(round(INTERIOR_ERODE_UM / v)))
        r_in = max(1, int(round(ANNULUS_IN_UM / v)))
        r_out = max(r_in + 1, int(round(ANNULUS_OUT_UM / v)))
        r_pk = max(1, int(round(PEAK_R_UM / v)))
        self.ann_y, self.ann_x = _ring_offsets(r_in, r_out)
        self.pk_y, self.pk_x = _disk_offsets(r_pk)


def _ring_offsets(r_in, r_out):
    ys, xs = np.mgrid[-r_out : r_out + 1, -r_out : r_out + 1]
    rr = np.sqrt(ys**2 + xs**2)
    ring = (rr >= r_in) & (rr <= r_out)
    return ys[ring], xs[ring]


def _disk_offsets(r):
    ys, xs = np.mgrid[-r : r + 1, -r : r + 1]
    disk = np.sqrt(ys**2 + xs**2) <= r
    return ys[disk], xs[disk]


# --------------------------------------------------------------------------
# core segmentation
# --------------------------------------------------------------------------
def otsu_interior_mask(slice_img, prm):
    """Real-data rule: Otsu of a heavily smoothed slice, largest component, filled, eroded."""
    sm_big = gaussian(slice_img, sigma=prm.sigma_boundary, preserve_range=True)
    fg = sm_big > threshold_otsu(sm_big)
    lab, n = ndi.label(fg)
    if n == 0:
        return np.zeros(fg.shape, dtype=bool)
    sizes = ndi.sum(fg, lab, index=np.arange(1, n + 1))
    comp = ndi.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))
    return ndi.binary_erosion(comp, structure=np.ones((3, 3)), iterations=prm.erode_px)


def find_seeds(slice_img, interior, prm):
    sm = gaussian(slice_img, sigma=prm.sigma_seed, preserve_range=True)
    if not interior.any():
        return np.zeros((0, 2), dtype=int), sm
    floor = np.percentile(sm[interior], 60)
    coords = peak_local_max(
        sm,
        min_distance=prm.min_distance_px,
        threshold_abs=floor,
        labels=interior.astype(int),
    )
    if len(coords) == 0:
        return coords, sm
    vals = sm[coords[:, 0], coords[:, 1]]
    coords = coords[np.argsort(-vals)]  # brightest first
    # greedy merge (identical result to the incremental-KD-tree version, O(n log n))
    tree = cKDTree(coords.astype(float))
    neigh = tree.query_ball_point(coords.astype(float), r=prm.merge_radius_px - 1e-9)
    removed = np.zeros(len(coords), dtype=bool)
    keep = []
    for i in range(len(coords)):
        if removed[i]:
            continue
        keep.append(i)
        for j in neigh[i]:
            if j != i:
                removed[j] = True
    return coords[keep], sm


def local_levels(sm, seeds, prm):
    H, W = sm.shape
    n = len(seeds)
    peak = np.zeros(n)
    matrix = np.zeros(n)
    noise = np.zeros(n)
    for i, (r, c) in enumerate(seeds):
        py = np.clip(r + prm.pk_y, 0, H - 1)
        px = np.clip(c + prm.pk_x, 0, W - 1)
        peak[i] = np.percentile(sm[py, px], PEAK_PCTL)
        ay = np.clip(r + prm.ann_y, 0, H - 1)
        ax = np.clip(c + prm.ann_x, 0, W - 1)
        ann = sm[ay, ax]
        matrix[i] = np.percentile(ann, MATRIX_PCTL)
        lowv = ann[ann <= np.percentile(ann, 40)]
        noise[i] = np.std(lowv)
    return peak, matrix, noise


def segment_slice(slice_img, seeds, interior, sm_seg, frac, prm):
    n = len(seeds)
    if n == 0:
        return np.zeros(slice_img.shape, dtype=int), {}
    markers = np.zeros(slice_img.shape, dtype=int)
    for i, (r, c) in enumerate(seeds):
        markers[r, c] = i + 1
    floor = (
        np.percentile(sm_seg[interior], 20)
        if interior.any()
        else np.percentile(sm_seg, 20)
    )
    broad = interior & (sm_seg > floor)
    broad[markers > 0] = True
    labels_ws = watershed(-sm_seg, markers=markers, mask=broad)

    peak, matrix, noise = local_levels(sm_seg, seeds, prm)
    boundary = matrix + frac * (peak - matrix)
    contrast = peak - matrix

    # bounding boxes of every basin in one pass
    slices = ndi.find_objects(labels_ws, max_label=n)
    final = np.zeros(slice_img.shape, dtype=int)
    for i in range(n):
        sl = slices[i]
        if sl is None:
            continue
        sub_basin = labels_ws[sl] == (i + 1)
        region = sub_basin & (sm_seg[sl] >= boundary[i])
        region = ndi.binary_fill_holes(region)
        final[sl][region] = i + 1
    return final, dict(
        peak=peak, matrix=matrix, boundary=boundary, contrast=contrast, noise=noise
    )


def measure_and_gate(final_labels, interior, per_seed, seeds, voxel_um):
    outside_dil = ndi.binary_dilation(~interior, iterations=1)
    props_by_label = {p.label: p for p in regionprops(final_labels)}

    contrast_vals = per_seed["contrast"] if len(per_seed) else np.array([])
    global_floor = 0.0
    if len(contrast_vals) and np.any(contrast_vals > 0):
        global_floor = 0.2 * float(np.median(contrast_vals[contrast_vals > 0]))

    records = []
    for i in range(len(seeds)):
        lbl = i + 1
        rec = dict(
            seed_row=int(seeds[i][0]),
            seed_col=int(seeds[i][1]),
            label=lbl,
            peak=float(per_seed["peak"][i]),
            matrix=float(per_seed["matrix"][i]),
            boundary=float(per_seed["boundary"][i]),
            contrast=float(per_seed["contrast"][i]),
            noise=float(per_seed["noise"][i]),
        )
        p = props_by_label.get(lbl)
        if p is None or p.area < MIN_AREA_PX:
            rec.update(
                accepted=False,
                reason="degenerate",
                area_px=0 if p is None else int(p.area),
                minor_axis_um=np.nan,
                major_axis_um=np.nan,
                equiv_diam_um=np.nan,
                solidity=np.nan,
                tilt_deg=np.nan,
                elongation=np.nan,
                centroid=None,
                orientation=np.nan,
            )
            records.append(rec)
            continue
        r0, c0, r1, c1 = p.bbox
        touches_border = bool(np.any(p.image & outside_dil[r0:r1, c0:c1]))
        fails_contrast = rec["contrast"] < max(global_floor, 3.0 * rec["noise"])
        major, minor = (
            _ax(p, "axis_major_length", "major_axis_length"),
            _ax(p, "axis_minor_length", "minor_axis_length"),
        )
        elong = (major / minor) if minor > 1e-9 else np.inf
        if touches_border:
            reason = "border"
        elif fails_contrast:
            reason = "low_contrast"
        elif elong > ELONGATION_MAX:
            reason = "elongated"
        elif p.solidity < SOLIDITY_MIN:
            reason = "low_solidity"
        else:
            reason = "accepted"
        tilt = (
            np.degrees(np.arccos(np.clip(minor / major, 0, 1)))
            if major > 1e-9
            else np.nan
        )
        rec.update(
            accepted=(reason == "accepted"),
            reason=reason,
            area_px=int(p.area),
            minor_axis_um=minor * voxel_um,
            major_axis_um=major * voxel_um,
            equiv_diam_um=_ax(p, "equivalent_diameter_area", "equivalent_diameter")
            * voxel_um,
            solidity=float(p.solidity),
            tilt_deg=float(tilt),
            elongation=float(elong),
            centroid=tuple(p.centroid),
            orientation=float(p.orientation),
        )
        records.append(rec)
    return records


def _ax(p, new, old):
    return getattr(p, new) if hasattr(p, new) else getattr(p, old)


def _attach_truth(records, truth, z, shape_yx):
    for r in records:
        r["true_diam_um"] = np.nan
    if truth is None:
        return
    acc = [r for r in records if r["accepted"] and r.get("centroid") is not None]
    for s in range(0, len(acc), 400):
        chunk = acc[s : s + 400]
        rows = np.array([r["centroid"][0] for r in chunk])
        cols = np.array([r["centroid"][1] for r in chunk])
        d, _ = truth.nearest_true_diam(z, rows, cols, shape_yx)
        for r, dv in zip(chunk, d):
            r["true_diam_um"] = float(dv)


def process_slice(vol, zi, prm, truth, fracs, keep_maps):
    sl = np.asarray(vol[zi], dtype=np.float32)
    if truth is not None:
        interior = common.rope_interior_mask(
            sl.shape, prm.voxel_um, truth.rope_radius_um, margin_um=80.0
        )
    else:
        interior = otsu_interior_mask(sl, prm)
    seeds, _ = find_seeds(sl, interior, prm)
    sm_seg = gaussian(sl, sigma=prm.sigma_seg, preserve_range=True)
    out = dict(z=zi, n_seeds=len(seeds), fracs={})
    for frac in fracs:
        final, per_seed = segment_slice(sl, seeds, interior, sm_seg, frac, prm)
        recs = measure_and_gate(final, interior, per_seed, seeds, prm.voxel_um)
        for r in recs:
            r["z"] = zi
            r["frac"] = frac
        _attach_truth(recs, truth, zi, sl.shape)
        out["fracs"][frac] = dict(records=recs, final=final if keep_maps else None)
    if keep_maps:
        out.update(sl=sl, sm_seg=sm_seg, interior=interior)
    return out


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------
def _acc(records):
    return [r for r in records if r["accepted"]]


def _rmse_bias(est, true):
    est = np.asarray(est, float)
    true = np.asarray(true, float)
    ok = np.isfinite(est) & np.isfinite(true)
    if not ok.any():
        return None, None
    e = est[ok] - true[ok]
    return float(e.mean()), float(np.sqrt(np.mean(e**2)))


def summarize_frac(records):
    acc = _acc(records)
    return dict(
        minor=common.summary_stats([r["minor_axis_um"] for r in acc]),
        equiv=common.summary_stats([r["equiv_diam_um"] for r in acc]),
        tilt=common.summary_stats([r["tilt_deg"] for r in acc]),
        reasons={k: int(v) for k, v in Counter(r["reason"] for r in records).items()},
        n_total=len(records),
        n_accept=len(acc),
    )


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _ellipse_patch(rec, voxel_um, centroid=None):
    y0, x0 = centroid if centroid is not None else rec["centroid"]
    o = rec["orientation"]
    major = rec["major_axis_um"] / voxel_um
    minor = rec["minor_axis_um"] / voxel_um
    angle = math.degrees(math.atan2(-math.cos(o), -math.sin(o)))
    return Ellipse(
        (x0, y0),
        width=major,
        height=minor,
        angle=angle,
        fill=False,
        edgecolor="yellow",
        lw=1.0,
        linestyle=(0, (3, 2)),
    )


def _scale_bar(ax, x0, y0, length_px, text):
    ax.add_patch(
        FancyBboxPatch(
            (x0 - 1, y0 - 5.5),
            length_px + 2,
            7,
            boxstyle="round,pad=0.3",
            linewidth=0,
            facecolor="black",
            alpha=0.55,
            zorder=5,
        )
    )
    ax.plot([x0, x0 + length_px], [y0, y0], color="white", lw=2.0, zorder=6)
    ax.text(
        x0 + length_px / 2,
        y0 - 1.5,
        text,
        color="white",
        ha="center",
        va="bottom",
        fontsize=7,
        zorder=6,
    )


def _save(fig, path, dpi):
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def fig_distribution(path, label, by_frac_recs, fracs, truth, dpi=120):
    recs05 = by_frac_recs[0.5]
    acc = _acc(recs05)
    minors = np.array([r["minor_axis_um"] for r in acc])
    equivs = np.array([r["equiv_diam_um"] for r in acc])
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 0.9])
    )
    for ax, x, name, color in [
        (axes[0], minors, "minor-axis diameter", "#3b82f6"),
        (axes[1], equivs, "equivalent-area diameter", "#8b5cf6"),
    ]:
        if len(x):
            ax.hist(
                x, bins=30, color=color, alpha=0.8, edgecolor="white", linewidth=0.3
            )
            q1, q3 = np.percentile(x, [25, 75])
            ax.set_title(
                f"{label}: {name} (n={len(x)})\nmean={x.mean():.1f}+/-{x.std():.1f} um, "
                f"median={np.median(x):.1f} [{q1:.1f},{q3:.1f}] um",
                fontsize=9,
            )
        else:
            ax.set_title(f"{label}: {name} (n=0)", fontsize=9)
        if truth is not None:
            ax.axvline(
                truth.mean_diam_um,
                color="green",
                ls="--",
                lw=1.4,
                label=f"true mean {truth.mean_diam_um:.1f} um",
            )
        else:
            ax.axvline(12, color="green", ls="--", lw=1.2, label="12 um (spec)")
            ax.axvline(
                20, color="darkorange", ls="--", lw=1.2, label="20 um (spacing ref)"
            )
        ax.set_xlabel(f"{name} (um)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("count")

    ax = axes[2]
    for metric, key, marker, ls in [
        ("minor", "minor_axis_um", "o", "-"),
        ("equiv", "equiv_diam_um", "s", "--"),
    ]:
        means, stds = [], []
        for fr in fracs:
            v = np.array([r[key] for r in _acc(by_frac_recs[fr])], float)
            means.append(v.mean() if len(v) else np.nan)
            stds.append(v.std() if len(v) else np.nan)
        ax.errorbar(
            list(fracs),
            means,
            yerr=stds,
            marker=marker,
            ls=ls,
            capsize=3,
            lw=1.5,
            ms=5,
            color="#2563eb",
            label=f"{metric}",
        )
    if truth is not None:
        tm = []
        for fr in fracs:
            t = np.array([r["true_diam_um"] for r in _acc(by_frac_recs[fr])], float)
            tm.append(np.nanmean(t) if np.isfinite(t).any() else np.nan)
        ax.plot(
            list(fracs),
            tm,
            color="green",
            marker="^",
            ls=":",
            lw=1.4,
            label="true D (matched fibres)",
        )
        ax.axhline(
            truth.mean_diam_um,
            color="green",
            ls="-",
            lw=0.8,
            alpha=0.6,
            label=f"true mean {truth.mean_diam_um:.1f} um",
        )
    else:
        ax.axhline(12, color="green", ls=":", lw=1)
    ax.set_xticks(list(fracs))
    ax.set_xlabel("threshold fraction (boundary = matrix + frac*(peak-matrix))")
    ax.set_ylabel("diameter (um), mean +/- std")
    ax.set_title(f"{label}: threshold sensitivity", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    _save(fig, path, dpi)


def _pick_crop_centers(interior, half):
    H, W = interior.shape
    ys, xs = np.where(interior)
    if len(ys) == 0:
        return [(H // 2, W // 2)]
    pts = [
        (int(np.median(ys)), int(np.median(xs))),
        (int(np.percentile(ys, 30)), int(np.percentile(xs, 70))),
    ]
    m = half + 1
    out = []
    for cy, cx in pts:
        out.append(
            (int(np.clip(cy, m, max(m, H - m))), int(np.clip(cx, m, max(m, W - m))))
        )
    return out


def fig_overlay(path, label, mid, voxel_um, dpi=120):
    sl, interior = mid["sl"], mid["interior"]
    final = mid["fracs"][0.5]["final"] if 0.5 in mid["fracs"] else None
    recs = {r["label"]: r for r in mid["fracs"][0.5]["records"]}
    H, W = sl.shape
    half = int(min(round(100.0 / voxel_um / 2), H // 2 - 1, W // 2 - 1))
    vmax = np.percentile(sl, 99.9)
    centers = _pick_crop_centers(interior, half)
    fig, axes = plt.subplots(
        len(centers), 2, figsize=(9, 4.5 * len(centers)), squeeze=False
    )
    bar_px = 20.0 / voxel_um
    for k, (cy, cx) in enumerate(centers):
        r0, c0 = max(0, cy - half), max(0, cx - half)
        r1, c1 = min(H, r0 + 2 * half), min(W, c0 + 2 * half)
        crop = sl[r0:r1, c0:c1]
        lbl_crop = final[r0:r1, c0:c1]
        ax = axes[k, 0]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(f"{label} z={mid['z']} raw crop", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        _scale_bar(ax, 4, crop.shape[0] - 4, bar_px, "20 um")
        ax = axes[k, 1]
        ax.imshow(crop, cmap="gray", vmin=0, vmax=vmax)
        for lb in np.unique(lbl_crop):
            if lb == 0 or lb not in recs:
                continue
            rec = recs[lb]
            m = lbl_crop == lb
            if m.sum() < 2:
                continue
            for cont in find_contours(np.pad(m.astype(float), 1), 0.5):
                ax.plot(
                    cont[:, 1] - 1,
                    cont[:, 0] - 1,
                    color=GATE_COLORS.get(rec["reason"], "red"),
                    lw=0.9,
                )
            if rec["accepted"]:
                cy0, cx0 = rec["centroid"]
                if r0 <= cy0 < r1 and c0 <= cx0 < c1:
                    ax.add_patch(
                        _ellipse_patch(rec, voxel_um, centroid=(cy0 - r0, cx0 - c0))
                    )
        ax.set_xlim(-0.5, crop.shape[1] - 0.5)
        ax.set_ylim(crop.shape[0] - 0.5, -0.5)
        ax.set_title(f"{label} z={mid['z']} contours+ellipses", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        _scale_bar(ax, 4, crop.shape[0] - 4, bar_px, "20 um")
    handles = [
        plt.Line2D([0], [0], color=c, lw=2, label=kk) for kk, c in GATE_COLORS.items()
    ]
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color="yellow",
            lw=1.5,
            ls=(0, (3, 2)),
            label="fitted ellipse (accepted only)",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, path, dpi)


def _radial_profile(sm, ry, rx, voxel_um, max_r_um=36.0, n_ang=16):
    rr = np.arange(0, max_r_um / voxel_um, 0.5)
    profs = []
    for a in range(n_ang):
        th = 2 * np.pi * a / n_ang
        yy = np.clip(ry + rr * np.sin(th), 0, sm.shape[0] - 1)
        xx = np.clip(rx + rr * np.cos(th), 0, sm.shape[1] - 1)
        profs.append(map_coordinates(sm, [yy, xx], order=1))
    return rr * voxel_um, np.array(profs)


def fig_examples(path, label, mid, voxel_um, dpi=110):
    recs = mid["fracs"][0.5]["records"]
    acc = sorted(
        [r for r in recs if r["accepted"] and r["area_px"] > 10],
        key=lambda r: r["equiv_diam_um"],
    )
    picks = []
    if acc:
        n = len(acc)
        for q in (0.35, 0.65):
            r = acc[min(n - 1, int(n * q))]
            if r not in picks:
                picks.append(r)
    rej = [r for r in recs if not r["accepted"] and r["area_px"] > 10]
    if rej:
        top = Counter(r["reason"] for r in rej).most_common(1)[0][0]
        cand = sorted(
            [r for r in rej if r["reason"] == top], key=lambda r: -r["area_px"]
        )
        picks.append(cand[len(cand) // 3])
    if not picks:
        fig = plt.figure(figsize=(4, 3))
        fig.text(0.5, 0.5, f"{label}: no fibres to show", ha="center")
        _save(fig, path, dpi)
        return
    sl, sm, final = mid["sl"], mid["sm_seg"], mid["fracs"][0.5]["final"]
    H, W = sl.shape
    half = int(min(round(40.0 / voxel_um), H // 2 - 1, W // 2 - 1))
    vmax = np.percentile(sl, 99.9)
    fig, axes = plt.subplots(
        2, len(picks), figsize=(4.2 * len(picks), 8), squeeze=False
    )
    for k, rec in enumerate(picks):
        ry, rx = rec["seed_row"], rec["seed_col"]
        r0 = int(np.clip(ry - half, 0, H - 2 * half))
        c0 = int(np.clip(rx - half, 0, W - 2 * half))
        patch = sl[r0 : r0 + 2 * half, c0 : c0 + 2 * half]
        m = final[r0 : r0 + 2 * half, c0 : c0 + 2 * half] == rec["label"]
        ax = axes[0, k]
        ax.imshow(patch, cmap="gray", vmin=0, vmax=vmax)
        color = GATE_COLORS.get(rec["reason"], "red")
        if m.any():
            for cont in find_contours(np.pad(m.astype(float), 1), 0.5):
                ax.plot(cont[:, 1] - 1, cont[:, 0] - 1, color=color, lw=1.3)
        ax.scatter([rx - c0], [ry - r0], c="red", marker="+", s=30)
        status = "ACCEPTED" if rec["accepted"] else f"REJECTED ({rec['reason']})"
        ax.set_title(
            f"{label} {status}\nminor={rec['minor_axis_um']:.1f} equiv={rec['equiv_diam_um']:.1f} um",
            fontsize=8.5,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, patch.shape[1] - 0.5)
        ax.set_ylim(patch.shape[0] - 0.5, -0.5)
        ax = axes[1, k]
        rr_um, profs = _radial_profile(sm, ry, rx, voxel_um)
        for p in profs:
            ax.plot(rr_um, p, color="gray", lw=0.4, alpha=0.5)
        ax.plot(rr_um, profs.mean(axis=0), color="blue", lw=2, label="mean profile")
        ax.axhline(rec["peak"], color="green", ls="--", lw=1, label="peak level")
        ax.axhline(rec["matrix"], color="brown", ls="--", lw=1, label="matrix level")
        ax.axhline(rec["boundary"], color="red", ls="-", lw=1.4, label="boundary")
        ax.set_xlabel("radius (um)", fontsize=8)
        if k == 0:
            ax.set_ylabel("intensity")
        ax.legend(fontsize=6.5, loc="upper right")
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    _save(fig, path, dpi)


def fig_truth_scatter(path, label, recs05, truth, dpi=120):
    acc = [r for r in _acc(recs05) if np.isfinite(r.get("true_diam_um", np.nan))]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    t = np.array([r["true_diam_um"] for r in acc], float)
    for ax, key, name, color in [
        (axes[0], "minor_axis_um", "minor-axis", "#3b82f6"),
        (axes[1], "equiv_diam_um", "equivalent-area", "#8b5cf6"),
    ]:
        e = np.array([r[key] for r in acc], float)
        ax.scatter(t, e, s=8, alpha=0.5, color=color)
        lo = np.nanmin([t.min(), e.min()]) - 1 if len(t) else 0
        hi = np.nanmax([t.max(), e.max()]) + 1 if len(t) else 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
        b, rm = _rmse_bias(e, t)
        txt = (
            f"n={len(t)}\nbias={b:+.2f} um\nRMSE={rm:.2f} um"
            if b is not None
            else "no matched fibres"
        )
        ax.text(
            0.04,
            0.96,
            txt,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        )
        ax.set_xlabel("true per-fibre diameter (um)")
        ax.set_ylabel(f"{name} diameter (um)")
        ax.set_title(f"{label}: {name} vs truth (frac 0.5)", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, path, dpi)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
_CSV_FIELDS = [
    "frac",
    "z",
    "label",
    "seed_row",
    "seed_col",
    "accepted",
    "reason",
    "area_px",
    "centroid_row",
    "centroid_col",
    "minor_axis_um",
    "major_axis_um",
    "equiv_diam_um",
    "solidity",
    "tilt_deg",
    "elongation",
    "peak",
    "matrix",
    "boundary",
    "contrast",
    "noise",
    "true_diam_um",
]


def _write_csv(path, by_frac_recs):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(_CSV_FIELDS)
        for fr in sorted(by_frac_recs):
            for r in by_frac_recs[fr]:
                c = r.get("centroid")
                row = dict(
                    r,
                    centroid_row=None if c is None else c[0],
                    centroid_col=None if c is None else c[1],
                )
                out = []
                for k in _CSV_FIELDS:
                    v = row.get(k)
                    if v is None or (isinstance(v, float) and not np.isfinite(v)):
                        out.append("")
                    elif isinstance(v, (bool, np.bool_)):
                        out.append(int(v))
                    elif isinstance(v, (float, np.floating)):
                        out.append(f"{float(v):.6g}")
                    else:
                        out.append(v)
                w.writerow(out)


def run(vol, voxel_um, out_dir, label, truth=None, fast=False, n_workers=None):
    """Run M3 on a (nz, ny, nx) float32 volume (fibre axis = dim 0); write figures/CSV/JSON to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    vol = np.asarray(vol)
    nz = vol.shape[0]
    prm = Params(voxel_um)
    z_list = common.analysis_z_slices(nz, n=6, skip=40)
    if fast:
        mid_i = len(z_list) // 2
        z_list = sorted(set([z_list[mid_i - 1], z_list[mid_i]]))
    fracs = (0.5,) if fast else FRACS
    mid_z = z_list[len(z_list) // 2] if not fast else z_list[-1]
    if not fast:
        mid_z = z_list[2] if len(z_list) > 2 else z_list[-1]

    def work(zi):
        return process_slice(vol, zi, prm, truth, fracs, keep_maps=(zi == mid_z))

    nw = (
        1
        if fast and n_workers is None
        else (n_workers or min(len(z_list), os.cpu_count() or 1))
    )
    if nw <= 1:
        results = [work(z) for z in z_list]
    else:
        with ThreadPoolExecutor(max_workers=nw) as ex:
            results = list(ex.map(work, z_list))
    by_z = {r["z"]: r for r in results}

    by_frac_recs = {
        fr: [rec for z in z_list for rec in by_z[z]["fracs"][fr]["records"]]
        for fr in fracs
    }
    mid = by_z[mid_z]

    _write_csv(os.path.join(out_dir, "M3_fibres.csv"), by_frac_recs)
    fig_distribution(
        os.path.join(out_dir, "M3_distribution.png"), label, by_frac_recs, fracs, truth
    )
    fig_overlay(os.path.join(out_dir, "M3_overlay.png"), label, mid, prm.voxel_um)
    fig_examples(os.path.join(out_dir, "M3_examples.png"), label, mid, prm.voxel_um)
    if truth is not None:
        fig_truth_scatter(
            os.path.join(out_dir, "M3_truth_scatter.png"),
            label,
            by_frac_recs[0.5],
            truth,
        )

    by_frac = {str(fr): summarize_frac(by_frac_recs[fr]) for fr in fracs}
    acc05 = _acc(by_frac_recs[0.5])
    bias_m = rmse_m = bias_e = rmse_e = None
    if truth is not None:
        t = [r["true_diam_um"] for r in acc05]
        bias_m, rmse_m = _rmse_bias([r["minor_axis_um"] for r in acc05], t)
        bias_e, rmse_e = _rmse_bias([r["equiv_diam_um"] for r in acc05], t)
    summary = dict(
        method="M3",
        label=label,
        voxel_um=float(voxel_um),
        fast=bool(fast),
        z_slices=[int(z) for z in z_list],
        by_frac=by_frac,
        truth_mean_diam_um=None if truth is None else float(truth.mean_diam_um),
        bias_minor_um=bias_m,
        rmse_minor_um=rmse_m,
        bias_equiv_um=bias_e,
        rmse_equiv_um=rmse_e,
    )
    with open(os.path.join(out_dir, "M3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary
