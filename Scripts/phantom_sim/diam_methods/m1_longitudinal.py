"""M1 fibre diameter: longitudinal (fibre-parallel) slices, widest point + blurred chord-law fit.

Port of the 2026-09-28 real-data scratch study (lib_m1 / run_m1 / figs_m1 / chord_model / esf) into a
reusable, torch-free module. Volumes are (nz, ny, nx) float32 with the fibre axis along dim 0 (z).
All internal geometry is in voxels; every length constant is defined in um and converted with
``voxel_um`` (at 2 um voxels the constants equal the original ones exactly).

Per candidate fibre (local maximum in an xy slice at a candidate z-level):
  1. track the centre through xy slices (+-64 um), fit a line -> local axis a and centre c0.
  2. tilt-compensated longitudinal slices (planes containing a; 'yz' ~ fixed x, 'xz' ~ fixed y)
     at offsets +-30 um (1 um step), trilinear sampling; transverse profiles averaged along the
     fibre over a segment of L = 80 um (primary).
  3. per offset plane: band peak, walk to the first dip on each side, FWHM edges (local-dip or global
     matrix baseline), single-peak and shoulder checks, dip criterion (>= 50% toward matrix).
  4. width-vs-offset curve -> max width (widest plane) and blurred chord-law fit W(dx; R, sigma) -> 2R.
Rejections: track, tilt, no_band, merged, multi_peak, unbracketed, elongated, off_centre, fit.

Public entry point: ``run(vol, voxel_um, out_dir, label, truth=None, fast=False, n_workers=None)``.
"""

import csv
import os
import tempfile
import time

import numpy as np

REASONS = [
    "ok",
    "track",
    "tilt",
    "no_band",
    "merged",
    "multi_peak",
    "unbracketed",
    "elongated",
    "off_centre",
    "fit",
]
ORIENTS_ALL = ("yz", "xz")
# key: (rowset, FWHM baseline mode, dip criterion, sigma multiplier relative to the estimated sigma).
# The sigma multipliers reproduce the original 1.48 / 2.00 vox variants around the 1.69 vox ESF sigma.
SETTINGS_ALL = {
    "P": ("L80", "loc", 0.5, 1.0),
    "G": ("L80", "glb", 0.5, 1.0),
    "D3": ("L80", "loc", 0.3, 1.0),
    "D7": ("L80", "loc", 0.7, 1.0),
    "L40": ("L40", "loc", 0.5, 1.0),
    "L120": ("L120", "loc", 0.5, 1.0),
    "LIT": ("LIT", "loc", 0.5, 1.0),
    "Slo": ("L80", "loc", 0.5, 1.48 / 1.69),
    "Shi": ("L80", "loc", 0.5, 2.00 / 1.69),
}
TABLE_VERSION = 1
_G = {}


# ------------------------------------------------------------------------------------------------ params
def _params(voxel_um, nz, fast=False):
    v = float(voxel_um)

    def vx(um):
        return um / v

    def ivx(um):
        return int(round(um / v))

    zpad = ivx(16.0)  # z padding for the (4 um sigma) tracking smoothing
    track_half = min(ivx(64.0), max(6, (nz - 1) // 2 - zpad - 1))
    n_side = min(ivx(24.0), max(3, track_half // 2))
    n_min = min(ivx(60.0), int(1.4 * track_half))
    t_half = ivx(60.0)
    p = dict(
        v=v,
        S_GRID=np.arange(-vx(28.0), vx(28.0) + 1e-9, vx(0.5)),
        DX_GRID=np.arange(-vx(30.0), vx(30.0) + 1e-9, vx(1.0)),
        T_HALF=t_half,
        T_GRID=np.arange(-t_half, t_half + 1, dtype=float),
        SEG_HALF={"L40": ivx(20.0), "L80": ivx(40.0), "L120": ivx(60.0)},
        LIT_K=np.arange(-ivx(30.0), ivx(30.0) + 1),
        LIT_THALF=ivx(20.0),
        TRACK_HALF=track_half,
        ZPAD=zpad,
        BOX_XY=ivx(40.0),
        TRACK_SM=(vx(4.0), vx(2.0), vx(2.0)),
        PEAK_RAD=max(2, ivx(4.0)),
        TRACK_N_SIDE=n_side,
        TRACK_N_MIN=n_min,
        TRACK_RMS=vx(2.0),
        TRACK_OUTLIER=vx(4.0),
        TILT_MAX=25.0,
        SEARCH0=vx(6.0),
        FOLLOW=vx(2.5),
        SHOULDER=vx(3.0),
        MIN_SIDE=vx(3.0),
        MAX_OFF=vx(12.0),
        FIT_RMS=vx(1.0),
        FIT_MIN_PLANES=6,
        FIT_DX=vx(6.0),
        R_LO=vx(3.0),
        R_HI=vx(20.0),
        SIG_LO=vx(0.6),
        SIG_HI=vx(7.0),
        SIG_STARTS=(vx(2.0), vx(3.4), vx(5.0)),
        SHARP_LO=vx(2.0),
        SHARP_HI=vx(24.0),
        CAND_SMOOTH=vx(2.0),
        CAND_MIN_DIST=max(2, ivx(8.0)),
        Z_STEP=ivx(80.0),
        SKIP=track_half + zpad,
        LINK_TOL=vx(6.0),
        fast=bool(fast),
    )
    return p


def candidate_levels(nz, prm):
    """Candidate z-levels: original 80 um spacing, centred, skipping the tracking margin at each end
    (40 slices for 2 um voxels once nz is large enough)."""
    lo, hi = prm["SKIP"], nz - 1 - prm["SKIP"]
    if hi < lo:
        return [nz // 2]
    step = prm["Z_STEP"]
    n = int((hi - lo) // step) + 1
    start = lo + ((hi - lo) - (n - 1) * step) / 2.0
    return [int(round(start + i * step)) for i in range(n)]


# ------------------------------------------------------------------------------------------------ chord model
def _build_chord_table(prm, r_step=0.01):
    from scipy.stats import ncx2

    R_GRID = np.round(np.arange(prm["R_LO"], prm["R_HI"] + 1e-4, 0.1), 4)
    SIG_GRID = np.round(np.arange(prm["SIG_LO"], prm["SIG_HI"] + 1e-4, 0.1), 4)
    DX_GRID = np.round(np.arange(0.0, 2.0 * prm["R_HI"] + 1e-4, 0.1), 4)
    r = np.arange(0.0, max(35.0, prm["R_HI"] + 7.0 * prm["SIG_HI"]), r_step)
    W = np.full((len(R_GRID), len(SIG_GRID), len(DX_GRID)), np.nan, np.float32)
    for i, R in enumerate(R_GRID):
        for j, s in enumerate(SIG_GRID):
            rr = r[r <= R + 10 * s]
            f = np.clip(ncx2.cdf((R / s) ** 2, 2, (rr / s) ** 2), 0, 1)
            fdx = np.interp(DX_GRID, rr, f)
            rh = np.interp(fdx / 2, f[::-1], rr[::-1])
            w = 2 * np.sqrt(np.clip(rh**2 - DX_GRID**2, 0, None))
            w[fdx < 1e-4] = np.nan
            W[i, j] = w
    return dict(W=W, R=R_GRID, S=SIG_GRID, DX=DX_GRID)


def chord_table(prm):
    """Blurred chord-law table W(R, sigma, dx) in voxels; cached in the system temp dir."""
    name = f"phantom_sim_m1_chord_v{prm['v']:.6g}_t{TABLE_VERSION}.npz"
    path = os.path.join(tempfile.gettempdir(), name)
    if os.path.exists(path):
        try:
            d = np.load(path)
            return {k: d[k] for k in ("W", "R", "S", "DX")}
        except Exception:
            pass
    t = _build_chord_table(prm)
    tmp = path + f".{os.getpid()}.tmp.npz"
    try:
        np.savez(tmp, **t)
        os.replace(tmp, path)
    except OSError:
        pass
    return t


class _ChordModel:
    def __init__(self, table):
        from scipy.interpolate import RegularGridInterpolator

        W = np.array(table["W"], dtype=float)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                w = W[i, j]
                bad = ~np.isfinite(w)
                if bad.any():
                    k = np.where(~bad)[0].max()
                    w[k + 1 :] = w[k]
        self.R = table["R"]
        self.S = table["S"]
        self.DX = table["DX"]
        self.itp = RegularGridInterpolator(
            (self.R, self.S, self.DX), W, bounds_error=False, fill_value=None
        )

    def w(self, dx, R, s):
        dx = np.clip(np.abs(np.asarray(dx, float)), 0, self.DX[-1])
        pts = np.stack([np.full_like(dx, R), np.full_like(dx, s), dx], -1)
        return self.itp(pts)


# ------------------------------------------------------------------------------------------------ tracking
def _refine(img, iy, ix):
    c = img[iy, ix]
    dy = dx = 0.0
    a, b = img[iy - 1, ix], img[iy + 1, ix]
    den = a - 2 * c + b
    if den < 0:
        dy = 0.5 * (a - b) / den
    a, b = img[iy, ix - 1], img[iy, ix + 1]
    den = a - 2 * c + b
    if den < 0:
        dx = 0.5 * (a - b) / den
    return iy + float(np.clip(dy, -0.5, 0.5)), ix + float(np.clip(dx, -0.5, 0.5))


def _local_peak(img, py, px, rad=2):
    if not (np.isfinite(py) and np.isfinite(px)):
        return None
    iy0, ix0 = int(round(py)), int(round(px))
    if (
        iy0 - rad < 1
        or ix0 - rad < 1
        or iy0 + rad >= img.shape[0] - 1
        or ix0 + rad >= img.shape[1] - 1
    ):
        return None
    win = img[iy0 - rad : iy0 + rad + 1, ix0 - rad : ix0 + rad + 1]
    k = int(np.argmax(win))
    wy, wx = divmod(k, win.shape[1])
    if wy in (0, 2 * rad) or wx in (0, 2 * rad):
        return None
    return _refine(img, iy0 - rad + wy, ix0 - rad + wx)


def track(vol, z0, y0, x0, prm):
    """Track the fibre centre through xy slices around z0. Returns dict (zs, ys, xs, line fit, rms) or None."""
    from scipy import ndimage as ndi

    nz, ny, nx = vol.shape
    half, B, pad, rad = prm["TRACK_HALF"], prm["BOX_XY"], prm["ZPAD"], prm["PEAK_RAD"]
    zlo, zhi = max(0, z0 - half - pad), min(nz, z0 + half + pad + 1)
    nb = 2 * B + 1
    ylo = int(np.clip(int(y0) - B, 0, max(0, ny - nb)))
    xlo = int(np.clip(int(x0) - B, 0, max(0, nx - nb)))
    box = np.asarray(vol[zlo:zhi, ylo : ylo + nb, xlo : xlo + nb], dtype=np.float32)
    S = ndi.gaussian_filter(box, prm["TRACK_SM"])
    k0 = z0 - zlo
    p = _local_peak(S[k0], y0 - ylo, x0 - xlo, rad)
    if p is None:
        p = _local_peak(S[k0], y0 - ylo, x0 - xlo, rad + 1)
    if p is None:
        return None
    zs, ys, xs = [z0], [p[0]], [p[1]]
    for sg in (1, -1):
        py, px = p
        vy = vx = 0.0
        hist = [(0, p[0], p[1])]
        miss = 0
        for k in range(1, half + 1):
            kk = k0 + sg * k
            if kk < 0 or kk >= S.shape[0]:
                break
            q = _local_peak(S[kk], py + vy, px + vx, rad)
            if q is None:
                miss += 1
                if miss > 3:
                    break
                py, px = py + vy, px + vx
                continue
            miss = 0
            py, px = q
            hist.append((k, py, px))
            zs.append(z0 + sg * k)
            ys.append(py)
            xs.append(px)
            h = np.array(hist[-8:])
            if len(h) >= 3:
                vy = np.polyfit(h[:, 0], h[:, 1], 1)[0]
                vx = np.polyfit(h[:, 0], h[:, 2], 1)[0]
    zs = np.array(zs, float)
    ys = np.array(ys) + ylo
    xs = np.array(xs) + xlo
    o = np.argsort(zs)
    zs, ys, xs = zs[o], ys[o], xs[o]
    res = dict(
        zs=zs,
        ys=ys,
        xs=xs,
        n=len(zs),
        n_up=int((zs > z0).sum()),
        n_dn=int((zs < z0).sum()),
    )
    if len(zs) < 10:
        res.update(ok=False)
        return res
    dz = zs - z0
    A = np.stack([np.ones_like(dz), dz], 1)
    keep = np.ones(len(dz), bool)
    for _ in range(2):
        cy = np.linalg.lstsq(A[keep], ys[keep], rcond=None)[0]
        cx = np.linalg.lstsq(A[keep], xs[keep], rcond=None)[0]
        r = np.hypot(ys - A @ cy, xs - A @ cx)
        keep = r < prm["TRACK_OUTLIER"]
        if keep.sum() < 10:
            break
    rms = float(np.sqrt(np.mean(r[keep] ** 2))) if keep.sum() else np.inf
    res.update(y0=cy[0], sy=cy[1], x0=cx[0], sx=cx[1], rms=rms, n_keep=int(keep.sum()))
    return res


# ------------------------------------------------------------------------------------------------ sampling
def basis(sy, sx, orient):
    a = np.array([1.0, sy, sx])
    a /= np.linalg.norm(a)
    e = np.array([0.0, 0.0, 1.0]) if orient == "yz" else np.array([0.0, 1.0, 0.0])
    u = e - (e @ a) * a
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    v /= np.linalg.norm(v)
    comp = 1 if orient == "yz" else 2
    if v[comp] < 0:
        v = -v
    return a, u, v


def sample_tc(vol, c0, a, u, v, dxs, ss, ts):
    """Tilt-compensated sampling -> array (n_dx, n_s, n_t)."""
    from scipy import ndimage as ndi

    pts = (
        c0[:, None, None, None]
        + u[:, None, None, None] * dxs[None, :, None, None]
        + v[:, None, None, None] * ss[None, None, :, None]
        + a[:, None, None, None] * ts[None, None, None, :]
    )
    vals = ndi.map_coordinates(vol, pts.reshape(3, -1), order=1, mode="nearest")
    return vals.reshape(len(dxs), len(ss), len(ts)).astype(np.float32)


def sample_literal(vol, tr, z0, orient, ss, k, thalf):
    """Axis-aligned slices at integer x (yz) or integer y (xz). Returns P (n_k, n_s), dx offsets, scale."""
    from scipy import ndimage as ndi

    zs = np.arange(z0 - thalf, z0 + thalf + 1, dtype=float)
    yc = tr["y0"] + tr["sy"] * (zs - z0)
    xc = tr["x0"] + tr["sx"] * (zs - z0)
    if orient == "yz":
        planes = np.round(tr["x0"]) + k
        Z = np.broadcast_to(zs[None, None, :], (len(k), len(ss), len(zs)))
        Y = yc[None, None, :] + ss[None, :, None] + 0 * planes[:, None, None]
        X = np.broadcast_to(planes[:, None, None], Z.shape)
        dx = planes - tr["x0"]
        scale = np.cos(np.arctan(tr["sy"]))
    else:
        planes = np.round(tr["y0"]) + k
        Z = np.broadcast_to(zs[None, None, :], (len(k), len(ss), len(zs)))
        X = xc[None, None, :] + ss[None, :, None] + 0 * planes[:, None, None]
        Y = np.broadcast_to(planes[:, None, None], Z.shape)
        dx = planes - tr["y0"]
        scale = np.cos(np.arctan(tr["sx"]))
    pts = np.stack([Z, Y, X]).reshape(3, -1)
    vals = ndi.map_coordinates(vol, pts, order=1, mode="nearest").reshape(Z.shape)
    return vals.mean(-1), dx.astype(float), float(scale)


# ------------------------------------------------------------------------------------------------ per-row analysis
def row_measure(p, s_center, halfwin, M, C, ss, shoulder_tol):
    from scipy.signal import find_peaks

    ds = ss[1] - ss[0]
    n = len(p)
    ic = int(round((s_center - ss[0]) / ds))
    hw = int(round(halfwin / ds))
    lo, hi = max(ic - hw, 1), min(ic + hw, n - 2)
    if hi <= lo:
        return None
    k = lo + int(np.argmax(p[lo : hi + 1]))
    if p[k] < p[k - 1] or p[k] < p[k + 1]:
        return None
    if (k == lo and p[lo - 1] > p[k]) or (k == hi and p[hi + 1] > p[k]):
        return None
    Pk = float(p[k])
    if Pk - M < 0.2 * C:
        return None
    tol = 0.04 * C
    i = k
    mn = p[k]
    imn = k
    while i > 0:
        i -= 1
        if p[i] < mn:
            mn = p[i]
            imn = i
        elif p[i] > mn + tol:
            break
    mL, iL = float(mn), imn
    i = k
    mn = p[k]
    imn = k
    while i < n - 1:
        i += 1
        if p[i] < mn:
            mn = p[i]
            imn = i
        elif p[i] > mn + tol:
            break
    mR, iR = float(mn), imn

    def eL(h):
        for j in range(k - 1, iL - 1, -1):
            if p[j] < h:
                return ss[j] + (h - p[j]) / (p[j + 1] - p[j]) * ds
        return np.nan

    def eR(h):
        for j in range(k + 1, iR + 1):
            if p[j] < h:
                return ss[j] - (h - p[j]) / (p[j - 1] - p[j]) * ds
        return np.nan

    r = dict(
        k=k,
        s_pk=float(ss[k]),
        Pk=Pk,
        mL=mL,
        mR=mR,
        iL=iL,
        iR=iR,
        sL_dip=float(ss[iL]),
        sR_dip=float(ss[iR]),
    )
    r["eL_loc"] = eL(0.5 * (Pk + mL))
    r["eR_loc"] = eR(0.5 * (Pk + mR))
    r["eL_glb"] = eL(0.5 * (Pk + M))
    r["eR_glb"] = eR(0.5 * (Pk + M))
    a_ = r["eL_loc"]
    b_ = r["eR_loc"]
    multi = False
    if np.isfinite(a_) and np.isfinite(b_):
        ja = int(np.ceil((a_ - ss[0]) / ds))
        jb = int(np.floor((b_ - ss[0]) / ds))
        seg = p[max(ja - 1, 0) : jb + 2]
        pk_, _ = find_peaks(seg, prominence=0.08 * max(Pk - M, 1e-12))
        multi = len(pk_) > 1
    r["multi"] = multi
    r["contrast"] = Pk - M
    sh = False
    for side in ("L", "R"):
        if side == "L":
            seg = p[iL : k + 1]
            pos = ss[iL:k] + ds / 2
            e = r["eL_loc"]
        else:
            seg = p[k : iR + 1][::-1]
            pos = ss[k + 1 : iR + 1][::-1] - ds / 2
            e = r["eR_loc"]
        if len(seg) < 4 or not np.isfinite(e):
            continue
        g = np.diff(seg)
        gm = g.max()
        if gm <= 0:
            continue
        pk_, _ = find_peaks(np.concatenate([[0], g, [0]]), prominence=0.3 * gm)
        s_g = pos[int(np.argmax(g))]
        if len(pk_) > 1 or abs(e - s_g) > shoulder_tol:
            sh = True
    r["shoulder"] = sh
    return r


def width_rows(P, i0, M, C, prm, ss):
    """Follow the band from row i0 outward in both directions (peak searched within +-FOLLOW of the
    start-row peak). Returns list of row dicts / None."""
    n = P.shape[0]
    rows = [None] * n
    r0 = row_measure(P[i0], 0.0, prm["SEARCH0"], M, C, ss, prm["SHOULDER"])
    if r0 is None:
        return rows
    rows[i0] = r0
    anchor = r0["s_pk"]
    for sg in (1, -1):
        prev = r0
        runmax = r0["contrast"]
        i = i0 + sg
        while 0 <= i < n:
            r = row_measure(P[i], anchor, prm["FOLLOW"], M, C, ss, prm["SHOULDER"])
            if r is None:
                break
            if r["contrast"] < 0.3 * runmax:
                break
            if (
                r["contrast"] < 0.8 * runmax
                and r["contrast"] > prev["contrast"] + 0.03 * C
            ):
                break
            rows[i] = r
            runmax = max(runmax, r["contrast"])
            prev = r
            i += sg
    return rows


def evaluate(rows, dxs, M, prm, mode="loc", dip_frac=0.5, scale=1.0):
    """Pre-fit evaluation of a width-vs-offset curve (everything except the chord fit, which depends
    on sigma and is done afterwards). reason == 'ok' means 'ready for the fit'."""
    n = len(rows)
    exists = np.array([r is not None for r in rows])
    con = np.array([r["contrast"] if r is not None else np.nan for r in rows])
    w = np.full(n, np.nan)
    valid = np.zeros(n, bool)
    dipfail = np.zeros(n, bool)
    multif = np.zeros(n, bool)
    for i, r in enumerate(rows):
        if r is None:
            continue
        eL, eR = r["eL_" + mode], r["eR_" + mode]
        dl = (r["Pk"] - r["mL"]) >= dip_frac * (r["Pk"] - M)
        dr = (r["Pk"] - r["mR"]) >= dip_frac * (r["Pk"] - M)
        dipfail[i] = (not (dl and dr)) or r["shoulder"]
        multif[i] = r["multi"]
        if np.isfinite(eL) and np.isfinite(eR):
            w[i] = (eR - eL) * scale
        valid[i] = (not dipfail[i]) and (not multif[i]) and np.isfinite(w[i])
    out = dict(w=w, valid=valid, exists=exists, reason="ok", scale=float(scale))
    if not exists.any():
        out["reason"] = "no_band"
        return out
    if valid.sum() < 3:
        out["reason"] = (
            "merged" if dipfail[exists].sum() >= multif[exists].sum() else "multi_peak"
        )
        return out
    wv = np.where(valid, w, -np.inf)
    ib = int(np.argmax(wv))
    a = ib
    while a - 1 >= 0 and valid[a - 1]:
        a -= 1
    b = ib
    while b + 1 < n and valid[b + 1]:
        b += 1
    out.update(ib=ib, run=(a, b), w_max=float(w[ib]), dx_best=float(dxs[ib]))
    step = abs(dxs[1] - dxs[0])
    min_side = prm["MIN_SIDE"]
    if (ib - a) * step < min_side - 1e-9 or (b - ib) * step < min_side - 1e-9:
        out["reason"] = "unbracketed"
        edge = a - 1 if (ib - a) * step < min_side - 1e-9 else b + 1
        if 0 <= edge < n and rows[edge] is not None:
            out["why_end"] = (
                "merged" if dipfail[edge] else ("multi" if multif[edge] else "nan")
            )
        else:
            out["why_end"] = "band_end"
        return out
    cmax = np.nanmax(con)
    lo = ib
    while lo - 1 >= 0 and exists[lo - 1] and con[lo - 1] >= 0.5 * cmax:
        lo -= 1
    hi = ib
    while hi + 1 < n and exists[hi + 1] and con[hi + 1] >= 0.5 * cmax:
        hi += 1
    out["extent"] = float((hi - lo) * step)
    if out["extent"] > 1.6 * w[ib] / scale:
        out["reason"] = "elongated"
        return out
    if abs(dxs[ib]) > prm["MAX_OFF"]:
        out["reason"] = "off_centre"
        return out
    out["fit_in"] = (
        np.asarray(dxs[a : b + 1], np.float64),
        np.asarray(w[a : b + 1] / scale, np.float64),
        float(dxs[ib]),
        float(w[ib] / scale),
    )
    return out


# ------------------------------------------------------------------------------------------------ chord fits
def fit_fixed(x, y, model, dx_start, wmax, sig, prm):
    """Primary blurred chord fit: sigma fixed, free R and dx0 (voxels, unscaled)."""
    from scipy.optimize import least_squares

    out = dict(fit_ok=False, D_fit=np.nan, dx0=np.nan, rms=np.nan)
    rlo, rhi, fdx = prm["R_LO"], prm["R_HI"], prm["FIT_DX"]
    sig = float(np.clip(sig, model.S[0], model.S[-1]))

    def res1(p):
        return model.w(x - p[1], p[0], sig) - y

    try:
        r = least_squares(
            res1,
            [np.clip(wmax / 2 + 0.3, rlo + 0.1, rhi - 0.1), dx_start],
            bounds=([rlo, dx_start - fdx], [rhi, dx_start + fdx]),
            diff_step=1e-3,
        )
        R, dx0 = r.x
        rms = float(np.sqrt(np.mean(r.fun**2)))
        out.update(
            D_fit=2 * float(R),
            dx0=float(dx0),
            rms=rms,
            fit_ok=bool(
                rms < prm["FIT_RMS"]
                and rlo + 0.05 < R < rhi - 0.05
                and len(x) >= prm["FIT_MIN_PLANES"]
            ),
        )
    except Exception:
        pass
    return out


def fit_free(x, y, model, dx_start, wmax, prm):
    """Free-sigma chord fit (3 parameters) and sharp (unblurred) chord on the top 25% of the curve."""
    from scipy.optimize import least_squares

    out = dict(D_free=np.nan, sig_free=np.nan, rms_free=np.nan, D_sharp=np.nan)
    rlo, rhi, fdx = prm["R_LO"], prm["R_HI"], prm["FIT_DX"]
    slo, shi = float(model.S[0]), float(model.S[-1])

    def res3(p):
        return model.w(x - p[2], p[0], p[1]) - y

    best = None
    for s0 in prm["SIG_STARTS"]:
        try:
            r = least_squares(
                res3,
                [
                    np.clip(wmax / 2 + 0.3, rlo + 0.1, rhi - 0.1),
                    np.clip(s0, slo, shi),
                    dx_start,
                ],
                bounds=([rlo, slo, dx_start - fdx], [rhi, shi, dx_start + fdx]),
                diff_step=1e-3,
                x_scale=[1, 0.5, 1],
            )
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is not None:
        out.update(
            D_free=2 * float(best.x[0]),
            sig_free=float(best.x[1]),
            rms_free=float(np.sqrt(np.mean(best.fun**2))),
        )
    sel = y >= 0.75 * np.nanmax(y)
    if sel.sum() >= 4:
        xs_, ys_ = x[sel], y[sel]

        def r2(p):
            return 2 * np.sqrt(np.clip(p[0] ** 2 - (xs_ - p[1]) ** 2, 0, None)) - ys_

        try:
            q = least_squares(
                r2,
                [
                    np.clip(
                        wmax / 2 + 0.2, prm["SHARP_LO"] + 0.1, prm["SHARP_HI"] - 0.1
                    ),
                    dx_start,
                ],
                bounds=(
                    [prm["SHARP_LO"], dx_start - fdx],
                    [prm["SHARP_HI"], dx_start + fdx],
                ),
            )
            out["D_sharp"] = 2 * float(q.x[0])
        except Exception:
            pass
    return out


# ------------------------------------------------------------------------------------------------ workers
def _init_worker(vol_src, levels, prm, table, orients, settings):
    if isinstance(vol_src, str):
        vol = np.load(vol_src, mmap_mode="r")
    else:
        vol = vol_src
    _G.update(
        vol=vol,
        M=float(levels[0]),
        C=float(levels[1]),
        prm=prm,
        model=_ChordModel(table),
        orients=tuple(orients),
        settings=dict(settings),
    )


def _track_check(tr, prm):
    if tr is None:
        return "nopeak"
    if "sy" not in tr:
        return "short"
    if (
        tr["n_up"] < prm["TRACK_N_SIDE"]
        or tr["n_dn"] < prm["TRACK_N_SIDE"]
        or tr["n"] < prm["TRACK_N_MIN"]
    ):
        return "short"
    if tr["rms"] > prm["TRACK_RMS"]:
        return "rms"
    return None


def _rowsets(vol, tr, z0, orient, M, C, prm, needed):
    c0 = np.array([z0, tr["y0"], tr["x0"]], float)
    a, u, v = basis(tr["sy"], tr["sx"], orient)
    ss, dxg, th = prm["S_GRID"], prm["DX_GRID"], prm["T_HALF"]
    V = sample_tc(vol, c0, a, u, v, dxg, ss, prm["T_GRID"])
    i0 = int(np.argmin(np.abs(dxg)))
    out, Ps = {}, {}
    for name in needed:
        if name == "LIT":
            Pl, dxl, scl = sample_literal(
                vol, tr, z0, orient, ss, prm["LIT_K"], prm["LIT_THALF"]
            )
            il = int(np.argmin(np.abs(dxl)))
            out["LIT"] = (width_rows(Pl, il, M, C, prm, ss), dxl, scl)
            Ps["LIT"] = Pl
        else:
            hl = prm["SEG_HALF"][name]
            P = V[:, :, th - hl : th + hl + 1].mean(-1)
            out[name] = (width_rows(P, i0, M, C, prm, ss), dxg, 1.0)
            Ps[name] = P
    return out, Ps, (c0, a, u, v)


def measure(cand):
    """Worker: track + sample + per-plane widths + pre-fit evaluation for every setting/orientation."""
    vol, M, C, prm = _G["vol"], _G["M"], _G["C"], _G["prm"]
    orients, settings = _G["orients"], _G["settings"]
    cid, z0, y0, x0 = cand
    rec = dict(id=int(cid), z0=int(z0), y0=int(y0), x0=int(x0))
    tr = track(vol, z0, y0, x0, prm)
    why = _track_check(tr, prm)
    if why is not None:
        rec["track_ok"] = False
        rec["track_why"] = why
        rec["res"] = {o: {k: dict(reason="track") for k in settings} for o in orients}
        return rec
    tilt = float(np.degrees(np.arctan(np.hypot(tr["sy"], tr["sx"]))))
    rec.update(
        track_ok=True,
        yc=float(tr["y0"]),
        xc=float(tr["x0"]),
        sy=float(tr["sy"]),
        sx=float(tr["sx"]),
        trms=float(tr["rms"]),
        tilt=tilt,
    )
    if tilt > prm["TILT_MAX"]:
        rec["res"] = {o: {k: dict(reason="tilt") for k in settings} for o in orients}
        return rec
    needed = sorted(set(s[0] for s in settings.values()))
    rec["res"] = {}
    for orient in orients:
        rowsets, _, (c0, a, u, v) = _rowsets(vol, tr, z0, orient, M, C, prm, needed)
        rec["basis_" + orient] = (a.tolist(), u.tolist(), v.tolist())
        rr = {}
        for key, (Lname, mode, dip, _sm) in settings.items():
            rows, dxs, scl_ = rowsets[Lname]
            ev = evaluate(rows, dxs, M, prm, mode=mode, dip_frac=dip, scale=scl_)
            keep = {
                k: ev[k]
                for k in (
                    "reason",
                    "w_max",
                    "dx_best",
                    "why_end",
                    "extent",
                    "scale",
                    "fit_in",
                )
                if k in ev
            }
            if key == "P":
                keep["w"] = ev["w"].astype(np.float32)
                keep["valid"] = ev["valid"]
                keep["exists"] = ev["exists"]
                keep["run"] = ev.get("run")
                keep["eL"] = np.array(
                    [r["eL_loc"] if r is not None else np.nan for r in rows], np.float32
                )
                keep["eR"] = np.array(
                    [r["eR_loc"] if r is not None else np.nan for r in rows], np.float32
                )
            rr[key] = keep
        rec["res"][orient] = rr
    return rec


def _fit_job(job):
    kind, x, y, dx_start, wmax, sig = job
    model, prm = _G["model"], _G["prm"]
    if kind == "fixed":
        return fit_fixed(x, y, model, dx_start, wmax, sig, prm)
    return fit_free(x, y, model, dx_start, wmax, prm)


# ------------------------------------------------------------------------------------------------ levels / masks
def _hist_levels(vals):
    """Matrix (M) and fibre (F) levels.

    Bimodal interior histogram (the real-data case): the two modes of the smoothed histogram below /
    above the Otsu split. If the histogram is not clearly bimodal (dense, blurred bundles where the
    matrix level is never reached), fall back to the 1st / 99th percentiles.
    Returns (M, F, source)."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu

    vals = np.asarray(vals, np.float64)
    vals = vals[np.isfinite(vals)]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    t = float(threshold_otsu(np.clip(vals, lo, hi)))
    h, e = np.histogram(vals, 400, range=(lo, hi))
    hs = ndi.gaussian_filter1d(h.astype(float), 3)
    c = 0.5 * (e[1:] + e[:-1])
    it = int(np.searchsorted(c, t))
    if 3 < it < len(c) - 3:
        iM = int(np.argmax(hs[:it]))
        iF = it + int(np.argmax(hs[it:]))
        valley = float(hs[iM : iF + 1].min()) if iF > iM else np.inf
        bimodal = it - iM >= 4 and iF - it >= 4 and valley < 0.7 * min(hs[iM], hs[iF])
        if bimodal:
            return float(c[iM]), float(c[iF]), "histogram_modes"
    M, F = np.percentile(vals, [1, 99])
    return float(M), float(F), "percentile_1_99"


def _sample_mask(sl, prm):
    """Original real-data sample-edge rule: heavy Gaussian blur, threshold (Otsu here, a fixed
    level originally), close/open, largest component, fill holes."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu

    s = ndi.gaussian_filter(np.asarray(sl, np.float32), 8.0 / prm["v"])
    m = s > threshold_otsu(s)
    m = ndi.binary_closing(m, iterations=6)
    m = ndi.binary_opening(m, iterations=4)
    lab, n = ndi.label(m)
    if n == 0:
        return np.zeros_like(m)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    return ndi.binary_fill_holes(lab == (int(np.argmax(sizes)) + 1))


def _erf_model(x, a, b, x0, s):
    from scipy.special import erf

    return a + b * 0.5 * (1 + erf((x - x0) / (np.sqrt(2) * s)))


def _fit_esf_profiles(profiles, M, C):
    """Profiles run from outside (air) to inside (sample), edge near index 12 (original layout)."""
    import warnings

    from scipy.optimize import curve_fit

    sig = []
    for seg in profiles:
        x = np.arange(len(seg), dtype=float)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, _ = curve_fit(
                    _erf_model,
                    x,
                    seg,
                    p0=[seg[:4].mean(), seg[-4:].mean() - seg[:4].mean(), 12, 1.5],
                    maxfev=4000,
                )
        except Exception:
            continue
        fitres = np.sqrt(np.mean((_erf_model(x, *p) - seg) ** 2))
        inside = p[0] + p[1]
        # clean edge: matrix-like inside level, clear step above the outside level, good erf fit.
        # The outside (air) level must lie clearly below the matrix level, otherwise the "edge" is a
        # fibre boundary touching the sample edge (e.g. matrix == air in the synthetic bundles).
        if (
            0.3 < abs(p[3]) < 6
            and fitres < 0.08 * abs(p[1])
            and abs(inside - M) < 0.4 * C
            and p[1] > 0.5 * C
            and p[0] < M - 0.5 * C
            and 8 <= p[2] <= 16
        ):
            sig.append(abs(p[3]))
    return np.array(sig)


def esf_sigma(vol, zlevels, M, C, prm, truth=None, masks=None):
    """PSF sigma (vox) from the sample-edge ESF. Circular rope edge (radial profiles) when truth is
    given, else the four faces of the blurred-Otsu sample mask. Returns (sigma or None, n_edges)."""
    from scipy import ndimage as ndi

    nz, ny, nx = vol.shape
    hz = min(5, max(0, min(zlevels) - 0), nz - 1 - max(zlevels))
    profiles = []
    for z in zlevels:
        sl = np.asarray(vol[max(0, z - hz) : z + hz + 1], np.float32).mean(0)
        if truth is not None:
            Rp = truth.rope_radius_um / prm["v"]
            cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
            ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)
            rr = Rp + 12 - np.arange(22, dtype=float)
            Y = cy + np.sin(ang)[:, None] * rr[None, :]
            X = cx + np.cos(ang)[:, None] * rr[None, :]
            ok = (
                (Y.min(1) >= 0)
                & (Y.max(1) <= ny - 1)
                & (X.min(1) >= 0)
                & (X.max(1) <= nx - 1)
            )
            if not ok.any():
                continue
            vals = ndi.map_coordinates(
                sl, np.stack([Y[ok].ravel(), X[ok].ravel()]), order=1
            ).reshape(-1, 22)
            profiles += list(vals)
        else:
            m = masks.get(z) if masks is not None else None
            if m is None or not m.any():
                continue
            rows_ = np.where(m.any(1))[0]
            cols_ = np.where(m.any(0))[0]
            r_lo, r_hi = np.percentile(rows_, [25, 75]).astype(int)
            c_lo, c_hi = np.percentile(cols_, [25, 75]).astype(int)
            for line in range(r_lo, r_hi, 4):
                idx = np.where(m[line])[0]
                if len(idx) == 0:
                    continue
                e = idx[0]
                if e - 12 >= 0:
                    profiles.append(sl[line, e - 12 : e + 10])
                e = idx[-1]
                if e + 13 <= nx:
                    profiles.append(sl[line, e - 9 : e + 13][::-1])
            for line in range(c_lo, c_hi, 4):
                idx = np.where(m[:, line])[0]
                if len(idx) == 0:
                    continue
                e = idx[0]
                if e - 12 >= 0:
                    profiles.append(sl[e - 12 : e + 10, line])
                e = idx[-1]
                if e + 13 <= ny:
                    profiles.append(sl[e - 9 : e + 13, line][::-1])
    profiles = [np.asarray(p, float) for p in profiles if len(p) == 22]
    sig = _fit_esf_profiles(profiles, M, C)
    if len(sig) < 20:
        return None, int(len(sig))
    return float(np.median(sig)), int(len(sig))


# ------------------------------------------------------------------------------------------------ helpers
def _stats(values):
    from phantom_sim.diam_methods.common import summary_stats

    return summary_stats(values)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _link_unique(acc, tol):
    """Number of distinct fibres among accepted segments (segments at different z-levels linked when
    each one's axis predicts the other's centre within tol voxels)."""
    n = len(acc)
    if n == 0:
        return 0
    z = np.array([r["z0"] for r in acc], float)
    yc = np.array([r["yc"] for r in acc])
    xc = np.array([r["xc"] for r in acc])
    sy = np.array([r["sy"] for r in acc])
    sx = np.array([r["sx"] for r in acc])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        dz = z - z[i]
        ya, xa = yc[i] + sy[i] * dz, xc[i] + sx[i] * dz
        yb, xb = yc - sy * dz, xc - sx * dz
        hit = (
            (dz != 0)
            & (np.hypot(ya - yc, xa - xc) < tol)
            & (np.hypot(yb - yc[i], xb - xc[i]) < tol)
        )
        for j in np.where(hit)[0]:
            if j > i:
                parent[find(i)] = find(int(j))
    return len(set(find(i) for i in range(n)))


# ------------------------------------------------------------------------------------------------ main entry
def run(
    vol, voxel_um, out_dir, label, truth=None, fast=False, n_workers=None, levels=None
):
    """Run M1 on ``vol`` (nz, ny, nx; fibre axis = z) and write figures, CSV and summary into out_dir.

    levels: optional (M, F) matrix / fibre intensity levels; estimated from the interior histogram
    when omitted. Returns a JSON-serialisable summary dict.
    """
    from concurrent.futures import ProcessPoolExecutor
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max

    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    vol = np.ascontiguousarray(np.asarray(vol, dtype=np.float32))
    nz, ny, nx = vol.shape
    prm = _params(voxel_um, nz, fast=fast)
    v = prm["v"]
    orients = ("yz",) if fast else ORIENTS_ALL
    settings = {"P": SETTINGS_ALL["P"]} if fast else dict(SETTINGS_ALL)
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, int(n_workers))

    # --- candidate levels, interior, intensity levels
    ZL = candidate_levels(nz, prm)
    masks, inner = {}, {}
    if truth is not None:
        from phantom_sim.diam_methods.common import rope_interior_mask

        im = rope_interior_mask((ny, nx), v, truth.rope_radius_um, margin_um=80.0)
        for z in ZL:
            inner[z] = im
        interior_rule = "rope_interior_mask(margin 80 um)"
    else:
        for z in ZL:
            masks[z] = _sample_mask(vol[z], prm)
            inner[z] = (
                ndi.distance_transform_edt(masks[z]) >= 80.0 / v
                if masks[z].any()
                else masks[z]
            )
        interior_rule = "blurred-Otsu sample mask, >= 80 um from its edge"
    hz = int(round(4 * prm["CAND_SMOOTH"]))
    smooth = {}
    for z in ZL:
        blk = np.asarray(vol[max(0, z - hz) : z + hz + 1], np.float32)
        smooth[z] = ndi.gaussian_filter(blk, prm["CAND_SMOOTH"])[z - max(0, z - hz)]
    if levels is None:
        vals = np.concatenate(
            [smooth[z][inner[z]] for z in ZL if inner[z].any()]
            or [smooth[ZL[0]].ravel()]
        )
        M, F, level_source = _hist_levels(vals)
    else:
        M, F = float(levels[0]), float(levels[1])
        level_source = "given"
    C = F - M

    # --- candidates
    cands, all_peaks = [], {}
    for z in ZL:
        lab = inner[z].astype(int)
        if not lab.any():
            all_peaks[z] = np.zeros((0, 2), int)
            continue
        pk = peak_local_max(
            smooth[z],
            min_distance=prm["CAND_MIN_DIST"],
            threshold_abs=M + 0.4 * C,
            labels=lab,
            exclude_border=False,
        )
        all_peaks[z] = pk
        for i in range(len(pk)):
            cands.append((len(cands), z, int(pk[i, 0]), int(pk[i, 1])))
    n_all = len(cands)
    if fast and len(cands) > 24:
        rng = np.random.default_rng(1)
        cands = [cands[i] for i in sorted(rng.choice(len(cands), 24, replace=False))]

    # --- sigma from the ESF (before the heavy pass)
    sig_esf, n_esf = esf_sigma(vol, ZL, M, C, prm, truth=truth, masks=masks)

    table = chord_table(prm)
    tmp_vol = None
    pool = None
    try:
        if n_workers > 1 and len(cands) > 1:
            tmp_vol = os.path.join(out_dir, f"_m1_vol_tmp_{os.getpid()}.npy")
            np.save(tmp_vol, vol)
            pool = ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(tmp_vol, (M, C), prm, table, orients, settings),
            )
            pmap = lambda f, it, cs: list(pool.map(f, it, chunksize=cs))  # noqa: E731
        else:
            _init_worker(vol, (M, C), prm, table, orients, settings)
            pmap = lambda f, it, cs: [f(x) for x in it]  # noqa: E731
        cs = max(1, min(8, len(cands) // (4 * n_workers) + 1))
        recs = pmap(measure, cands, cs)
        t_meas = time.time() - t_start
        t_fit0 = time.time()

        # --- free-sigma + sharp fits for every pre-fit-ok measurement (Slo/Shi share P's inputs)
        free_keys = [k for k in settings if k not in ("Slo", "Shi")]
        jobs, where = [], []
        for ir, r in enumerate(recs):
            for o in orients:
                for k in free_keys:
                    q = r["res"][o][k]
                    if "fit_in" in q:
                        x, y, dx0, wm = q["fit_in"]
                        jobs.append(("free", x, y, dx0, wm, 0.0))
                        where.append((ir, o, k))
        fcs = max(1, min(16, len(jobs) // (4 * n_workers) + 1))
        for (ir, o, k), fr in zip(where, pmap(_fit_job, jobs, fcs)):
            recs[ir]["res"][o][k].update(fr)
        for r in recs:
            for o in orients:
                for k in ("Slo", "Shi"):
                    if k in r["res"][o] and "fit_in" in r["res"][o][k]:
                        p_ = r["res"][o]["P"]
                        r["res"][o][k].update(
                            {
                                kk: p_.get(kk, np.nan)
                                for kk in ("D_free", "sig_free", "rms_free", "D_sharp")
                            }
                        )
        # --- sigma choice
        sf = np.array(
            [
                r["res"][o]["P"].get("sig_free", np.nan)
                for r in recs
                for o in orients
                if "fit_in" in r["res"][o]["P"]
                and r["res"][o]["P"].get("rms_free", np.inf) < prm["FIT_RMS"]
            ]
        )
        sf = sf[np.isfinite(sf)]
        sig_free_med = float(np.median(sf)) if len(sf) else None
        if sig_esf is not None:
            sigma, sigma_source = sig_esf, "esf"
        elif sig_free_med is not None:
            sigma, sigma_source = sig_free_med, "free_fit"
        else:
            sigma, sigma_source = 1.69 * 2.0 / v, "default"

        # --- fixed-sigma fits for all settings
        jobs, where = [], []
        for ir, r in enumerate(recs):
            for o in orients:
                for k, (_L, _m, _d, smul) in settings.items():
                    q = r["res"][o][k]
                    if "fit_in" in q:
                        x, y, dx0, wm = q["fit_in"]
                        jobs.append(("fixed", x, y, dx0, wm, sigma * smul))
                        where.append((ir, o, k))
        for (ir, o, k), fr in zip(where, pmap(_fit_job, jobs, fcs)):
            recs[ir]["res"][o][k].update(fr)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        if tmp_vol is not None:
            try:
                os.remove(tmp_vol)
            except OSError:
                pass
    t_fit = time.time() - t_fit0
    # finalise reasons, scale fit results
    for r in recs:
        for o in orients:
            for k in settings:
                q = r["res"][o][k]
                if "fit_in" in q:
                    sc = q.get("scale", 1.0)
                    for kk in ("D_fit", "D_free", "D_sharp"):
                        if kk in q:
                            q[kk] = q[kk] * sc
                    if not q.get("fit_ok", False):
                        q["reason"] = "fit"
                    del q["fit_in"]

    # --- truth lookup
    acc = [r for r in recs if any(r["res"][o]["P"]["reason"] == "ok" for o in orients)]
    if truth is not None and acc:
        for z in sorted(set(r["z0"] for r in acc)):
            grp = [r for r in acc if r["z0"] == z]
            d, dist = truth.nearest_true_diam(
                z, [r["yc"] for r in grp], [r["xc"] for r in grp], (ny, nx)
            )
            for r, dd, di in zip(grp, d, dist):
                r["true_diam_um"] = float(dd)
                r["true_dist_um"] = float(di)

    ctx = dict(
        vol=vol,
        recs=recs,
        prm=prm,
        M=M,
        F=F,
        C=C,
        sigma=sigma,
        sigma_source=sigma_source,
        orients=orients,
        settings=settings,
        ZL=ZL,
        inner=inner,
        truth=truth,
        label=label,
        out_dir=out_dir,
        table=table,
    )
    summary = _summarise(ctx)
    summary.update(
        n_candidates_all=n_all,
        levels=dict(M=M, F=F, source=level_source),
        interior_rule=interior_rule,
        sigma_esf_vox=sig_esf,
        n_esf_edges=n_esf,
        sigma_free_median_vox=sig_free_med,
        z_levels=ZL,
        voxel_um=v,
        fast=bool(fast),
        n_workers=n_workers,
        runtime_measure_s=round(t_meas, 1),
        runtime_fits_s=round(t_fit, 1),
    )
    ctx["summary"] = summary
    _write_csv(ctx)
    t_fig0 = time.time()
    _make_figures(ctx)
    summary["runtime_figures_s"] = round(time.time() - t_fig0, 1)
    summary["runtime_s"] = round(time.time() - t_start, 1)
    out = _jsonable(summary)
    from phantom_sim.diam_methods.common import save_json

    save_json(out, os.path.join(out_dir, "M1_summary.json"))
    return out


def _summarise(ctx):
    recs, orients, settings, v, truth = (
        ctx["recs"],
        ctx["orients"],
        ctx["settings"],
        ctx["prm"]["v"],
        ctx["truth"],
    )
    per_setting = {}
    for key in settings:
        wm, df, dfree, sfree, per_o = [], [], [], [], {}
        for o in orients:
            ok = [r["res"][o][key] for r in recs if r["res"][o][key]["reason"] == "ok"]
            a = [x["w_max"] * v for x in ok]
            b = [x["D_fit"] * v for x in ok]
            per_o[o] = dict(wmax=_stats(a), dfit=_stats(b))
            wm += a
            df += b
            dfree += [x["D_free"] * v for x in ok]
            sfree += [x["sig_free"] for x in ok]
        per_setting[key] = dict(
            wmax=_stats(wm),
            dfit=_stats(df),
            dfree=_stats(dfree),
            sig_free=_stats(sfree),
            per_orient=per_o,
        )
    reasons = {o: {} for o in orients}
    for r in recs:
        for o in orients:
            k = r["res"][o]["P"]["reason"]
            reasons[o][k] = reasons[o].get(k, 0) + 1
    tot = {}
    for o in orients:
        for k, n in reasons[o].items():
            tot[k] = tot.get(k, 0) + n
    acc = [r for r in recs if any(r["res"][o]["P"]["reason"] == "ok" for o in orients)]
    both = (
        [r for r in recs if all(r["res"][o]["P"]["reason"] == "ok" for o in orients)]
        if len(orients) > 1
        else []
    )
    tw = {}
    for r in recs:
        if not r.get("track_ok"):
            tw[r.get("track_why")] = tw.get(r.get("track_why"), 0) + 1
    dxb = [
        r["res"][o]["P"]["dx_best"] * v
        for r in recs
        for o in orients
        if r["res"][o]["P"]["reason"] == "ok"
    ]
    counts = dict(
        candidates=len(recs),
        accepted_segments=len(acc),
        unique_fibres=_link_unique(acc, ctx["prm"]["LINK_TOL"]),
        accepted_measurements=int(sum(reasons[o].get("ok", 0) for o in orients)),
        both_orient=len(both),
        reasons=reasons,
        track_failures=tw,
        dx_best_um=_stats(dxb),
        tilt_accepted=_stats([r["tilt"] for r in acc]),
    )
    wm = [
        r["res"][o]["P"]["w_max"] * v
        for r in recs
        for o in orients
        if r["res"][o]["P"]["reason"] == "ok"
    ]
    df = [
        r["res"][o]["P"]["D_fit"] * v
        for r in recs
        for o in orients
        if r["res"][o]["P"]["reason"] == "ok"
    ]
    out = {
        "method": "M1",
        "label": ctx["label"],
        "max_width": _stats(wm),
        "chord_fit": _stats(df),
        "n_accepted": int(counts["accepted_measurements"]),
        "n_candidates": len(recs),
        "reasons": tot,
        "sigma_vox": float(ctx["sigma"]),
        "sigma_source": ctx["sigma_source"],
        "truth_mean_diam_um": None,
        "bias_chord_um": None,
        "rmse_chord_um": None,
        "bias_maxw_um": None,
        "rmse_maxw_um": None,
        "counts": counts,
        "settings": per_setting,
    }
    if truth is not None:
        out["truth_mean_diam_um"] = truth.mean_diam_um
        tt, mw, mf = [], [], []
        for r in recs:
            for o in orients:
                q = r["res"][o]["P"]
                if q["reason"] == "ok" and np.isfinite(r.get("true_diam_um", np.nan)):
                    tt.append(r["true_diam_um"])
                    mw.append(q["w_max"] * v)
                    mf.append(q["D_fit"] * v)
        if tt:
            tt, mw, mf = map(np.asarray, (tt, mw, mf))
            out.update(
                bias_chord_um=float(np.mean(mf - tt)),
                rmse_chord_um=float(np.sqrt(np.mean((mf - tt) ** 2))),
                bias_maxw_um=float(np.mean(mw - tt)),
                rmse_maxw_um=float(np.sqrt(np.mean((mw - tt) ** 2))),
                n_truth_matched=int(len(tt)),
            )
        else:
            out["n_truth_matched"] = 0
    return out


def _write_csv(ctx):
    recs, orients, v = ctx["recs"], ctx["orients"], ctx["prm"]["v"]
    path = os.path.join(ctx["out_dir"], "M1_measurements.csv")

    def g(q, k, s=v):
        x = q.get(k, np.nan)
        return round(float(x) * s, 4) if x is not None and np.isfinite(x) else ""

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cand_id",
                "z0_vox",
                "y0_vox",
                "x0_vox",
                "yc_vox",
                "xc_vox",
                "tilt_deg",
                "orient",
                "w_max_um",
                "D_chordfit_um",
                "D_freesigma_um",
                "sigma_free_vox",
                "D_sharp_um",
                "dx_best_um",
                "fit_rms_um",
                "true_diam_um",
                "true_dist_um",
            ]
        )
        for r in recs:
            for o in orients:
                q = r["res"][o]["P"]
                if q["reason"] != "ok":
                    continue
                td = r.get("true_diam_um", np.nan)
                tdist = r.get("true_dist_um", np.nan)
                w.writerow(
                    [
                        r["id"],
                        r["z0"],
                        r["y0"],
                        r["x0"],
                        round(r["yc"], 3),
                        round(r["xc"], 3),
                        round(r["tilt"], 2),
                        o,
                        g(q, "w_max"),
                        g(q, "D_fit"),
                        g(q, "D_free"),
                        g(q, "sig_free", 1.0),
                        g(q, "D_sharp"),
                        g(q, "dx_best"),
                        g(q, "rms"),
                        round(td, 4) if np.isfinite(td) else "",
                        round(tdist, 4) if np.isfinite(tdist) else "",
                    ]
                )


# ------------------------------------------------------------------------------------------------ figures
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GOOD, CRIT = "#0ca30c", "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
REASON_LABEL = {
    "track": "xy tracking failed",
    "tilt": "tilt > 25 deg",
    "no_band": "no band peak at start",
    "merged": "neighbour merges (no dip / shoulder)",
    "multi_peak": "band not single-peaked",
    "unbracketed": "widest plane not reached (merges first)",
    "elongated": "band too long across offsets",
    "off_centre": "widest plane > 12 um off track",
    "fit": "chord fit poor (rms > 1 um)",
}
EXPLAIN = {
    "merged": "A neighbour touches the band: one flank has only a\nshoulder, no dip. The local half-max level then sits\n"
    "too high and the FWHM would read too narrow.",
    "unbracketed": "The width is still rising where the planes turn\ninvalid (a neighbour merges on one side), so the\n"
    "widest plane is never observed. No value reported.",
    "elongated": "The band persists over > 1.6 x its width in slice\noffset: two fibres stacked along the viewing\n"
    "direction. The chord law does not apply.",
    "fit": "The width-vs-offset curve does not follow the\nblurred chord law (rms > 1 um).",
}


def _refs(ctx):
    """Reference diameter lines: the true mean diameter when truth is known, else 12 um (spec) / 20 um."""
    if ctx["truth"] is not None:
        d = ctx["truth"].mean_diam_um
        return [(d, f"true mean {d:.1f} um", "--", AQUA)]
    return [(12.0, "12 um (spec)", "--", VIOLET), (20.0, "20 um", "-.", AQUA)]


def _save(fig, path, dpi):
    tmp = path[:-4] + "_tmp.png"
    fig.savefig(tmp, dpi=min(dpi, 150))
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.5)


def _make_figures(ctx):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rc = {
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.titlesize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#fcfcfb",
    }
    with plt.rc_context(rc):
        ctx["model"] = _ChordModel(ctx["table"])
        _fig_examples(ctx, plt)
        _fig_failures(ctx, plt)
        _fig_overview(ctx, plt)
        _fig_distribution(ctx, plt)
        if ctx["truth"] is not None:
            _fig_truth(ctx, plt)
    plt.close("all")


def _recompute(ctx, r, o):
    vol, prm, M, C = ctx["vol"], ctx["prm"], ctx["M"], ctx["C"]
    tr = track(vol, r["z0"], r["y0"], r["x0"], prm)
    rowsets, Ps, (c0, a, u, v) = _rowsets(vol, tr, r["z0"], o, M, C, prm, ["L80"])
    rows, dxs, _ = rowsets["L80"]
    ev = evaluate(rows, dxs, M, prm)
    if "fit_in" in ev:
        x, y, dx0, wm = ev.pop("fit_in")
        ev.update(fit_fixed(x, y, ctx["model"], dx0, wm, ctx["sigma"], prm))
        if not ev.get("fit_ok", False):
            ev["reason"] = "fit"
    return dict(tr=tr, c0=c0, a=a, u=u, v=v, P=Ps["L80"], rows=rows, ev=ev)


def _yscale(ctx):
    F = abs(ctx["F"]) if ctx["F"] != 0 else 1.0
    if F < 0.05:
        e = int(np.floor(np.log10(F)))
        return 10.0 ** (-e), f"(x1e{e}, arb.)"
    return 1.0, "(arb.)"


def _plot_example(ctx, axrow, o, d, title_extra=""):
    from scipy import ndimage as ndi

    prm, vol, M, model, sigma = (
        ctx["prm"],
        ctx["vol"],
        ctx["M"],
        ctx["model"],
        ctx["sigma"],
    )
    V = prm["v"]
    S_GRID, DX_GRID = prm["S_GRID"], prm["DX_GRID"]
    ev, rows, P = d["ev"], d["rows"], d["P"]
    c0, a, u, v = d["c0"], d["a"], d["u"], d["v"]
    seg_um = 2 * prm["SEG_HALF"]["L80"] * V / 2.0
    ib = ev.get("ib", None)
    if ib is None:
        ex = [i for i, rr in enumerate(rows) if rr is not None]
        ib = ex[len(ex) // 2] if ex else int(np.argmin(np.abs(DX_GRID)))
    dxb = DX_GRID[ib]
    # (a) longitudinal slice at the widest offset, +-80 um along the fibre
    ts = np.arange(-80.0 / V, 80.0 / V + 1e-9, 0.5)
    ss = S_GRID
    pts = (
        c0[:, None, None]
        + u[:, None, None] * dxb
        + v[:, None, None] * ss[None, None, :]
        + a[:, None, None] * ts[None, :, None]
    )
    img = ndi.map_coordinates(vol, pts.reshape(3, -1), order=1, mode="nearest").reshape(
        len(ts), len(ss)
    )
    ax = axrow[0]
    lo, hi = np.percentile(img, [1, 99.5])
    ax.imshow(
        img,
        cmap="gray",
        vmin=lo,
        vmax=hi,
        extent=(ss[0] * V, ss[-1] * V, ts[-1] * V, ts[0] * V),
        aspect="equal",
        interpolation="bilinear",
    )
    rr = rows[ib]
    if rr is not None:
        for e in (rr["eL_loc"], rr["eR_loc"]):
            if np.isfinite(e):
                ax.plot([e * V, e * V], [-seg_um, seg_um], color=ORANGE, lw=1.4)
    ax.plot([ss[0] * V + 1, ss[0] * V + 1], [-seg_um, seg_um], color=AQUA, lw=3)
    ax.set_title(
        f"{o} slice (fibre-parallel), offset {dxb * V:+.1f} um\n{title_extra}\n"
        f"aqua: averaged {2 * seg_um:.0f} um; orange: edges",
        fontsize=8,
    )
    ax.set_xlabel("across band (um)")
    ax.set_ylabel("along fibre (um)")
    # (b) mean cross-section with the scanned planes
    ax = axrow[1]
    ext = (S_GRID[0] * V, S_GRID[-1] * V, DX_GRID[-1] * V, DX_GRID[0] * V)
    lo, hi = np.percentile(P, [1, 99.5])
    ax.imshow(P, cmap="gray", vmin=lo, vmax=hi, extent=ext, aspect="equal")
    for i, q in enumerate(rows):
        if q is None:
            continue
        good = ev["valid"][i]
        for e in (q["eL_loc"], q["eR_loc"]):
            if np.isfinite(e):
                ax.plot(
                    e * V, DX_GRID[i] * V, ".", ms=2.2, color=GOOD if good else CRIT
                )
    ax.axhline(dxb * V, color=ORANGE, lw=1.0, ls="--")
    ax.set_xlim(-24, 24)
    ax.set_ylim(24, -24)
    ax.set_title(
        "mean cross-section over segment;\nFWHM edges per plane (green valid, red rejected)",
        fontsize=8.5,
    )
    ax.set_xlabel("across band (um)")
    ax.set_ylabel("slice offset (um)")
    # (c) width vs offset
    ax = axrow[2]
    w = ev["w"] * V
    ex = ev["exists"]
    ax.plot(
        DX_GRID[ex] * V,
        w[ex],
        "o",
        mfc="none",
        mec=MUTED,
        ms=4,
        label="measured, rejected plane",
    )
    ax.plot(
        DX_GRID[ev["valid"]] * V,
        w[ev["valid"]],
        "o",
        color=BLUE,
        ms=4,
        label="measured, valid plane",
    )
    if np.isfinite(ev.get("D_fit", np.nan)):
        exi = np.where(ex)[0]
        xx = np.linspace(DX_GRID[exi.min()] - 1.5, DX_GRID[exi.max()] + 1.5, 300)
        ax.plot(
            xx * V,
            model.w(xx - ev["dx0"], ev["D_fit"] / 2, sigma) * V,
            color=ORANGE,
            lw=1.8,
            label=f"blurred chord fit: 2R = {ev['D_fit'] * V:.1f} um",
        )
        ax.axhline(ev["D_fit"] * V, color=ORANGE, lw=0.8, ls=":")
    if "w_max" in ev and ev["reason"] == "ok":
        ax.plot(
            dxb * V,
            ev["w_max"] * V,
            "*",
            color=INK,
            ms=10,
            label=f"max width = {ev['w_max'] * V:.1f} um",
        )
    refs = _refs(ctx)
    ytop = max(
        30.0, 1.4 * max(r_[0] for r_ in refs), np.nanmax(np.append(w[ex], 0)) * 1.15
    )
    for val, lab, ls, _col in refs:
        ax.axhline(val, color=MUTED, lw=0.8, ls=ls)
        ax.text(
            29,
            val + 0.3,
            lab.split(" ")[0] if ctx["truth"] is None else lab,
            color=MUTED,
            fontsize=7,
            ha="right",
        )
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, ytop)
    ax.set_xlabel("slice offset (um)")
    ax.set_ylabel("band FWHM (um)")
    ax.set_title(f"width vs offset   [{ev['reason']}]", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="lower center", frameon=False)
    # (d) profile at widest plane
    ax = axrow[3]
    ysc, ylab = _yscale(ctx)
    p = P[ib]
    ax.plot(S_GRID * V, p * ysc, color=INK, lw=1.4)
    ax.axhline(M * ysc, color=MUTED, lw=0.8, ls=":")
    ax.text(-27, M * ysc, "matrix level", color=MUTED, fontsize=7, va="bottom")
    if rr is not None:
        hL = 0.5 * (rr["Pk"] + rr["mL"])
        hR = 0.5 * (rr["Pk"] + rr["mR"])
        ax.plot(
            [S_GRID[0] * V, rr["s_pk"] * V],
            [hL * ysc] * 2,
            color=ORANGE,
            lw=0.9,
            ls="--",
        )
        ax.plot(
            [rr["s_pk"] * V, S_GRID[-1] * V],
            [hR * ysc] * 2,
            color=ORANGE,
            lw=0.9,
            ls="--",
        )
        rng_ = (p.max() - min(M, p.min())) * ysc
        eL, eR = rr["eL_loc"], rr["eR_loc"]
        if np.isfinite(eL) and np.isfinite(eR):
            yv = 0.5 * (hL + hR) * ysc
            ax.annotate(
                "",
                xy=(eL * V, yv),
                xytext=(eR * V, yv),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.4),
            )
            ax.text(
                0.5 * (eL + eR) * V,
                yv + 0.02 * rng_,
                f"FWHM {abs(eR - eL) * V:.1f} um",
                color=INK,
                ha="center",
                fontsize=8,
            )
        ax.plot(rr["sL_dip"] * V, rr["mL"] * ysc, "v", color=MUTED, ms=5)
        ax.plot(rr["sR_dip"] * V, rr["mR"] * ysc, "v", color=MUTED, ms=5)
        yl = ax.get_ylim()
        ymin = min(M * ysc, p.min() * ysc) - 0.12 * rng_
        sc = rr["s_pk"] * V
        for k_, (val, lab, _ls, col) in enumerate(refs):
            yy = ymin + 0.06 * rng_ * (len(refs) - 1 - k_)
            ax.plot(
                [sc - val / 2, sc + val / 2],
                [yy, yy],
                color=col,
                lw=3,
                solid_capstyle="butt",
            )
            ax.text(
                sc + val / 2 + 1,
                yy,
                f"{val:.0f} um" if ctx["truth"] is None else f"{val:.1f} um (true)",
                fontsize=7,
                va="center",
                color=INK2,
            )
        ax.set_ylim(ymin - 0.04 * rng_, max(yl[1], p.max() * ysc + 0.05 * rng_))
    ax.set_xlim(-28, 28)
    ax.set_xlabel("across band (um)")
    ax.set_ylabel(f"intensity {ylab}")
    ax.set_title(
        "transverse profile at widest plane\n(half-max per side, dips marked)",
        fontsize=8.5,
    )


def _empty_fig(plt, path, text, dpi=100):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(
        0.5, 0.5, text, ha="center", va="center", fontsize=11, transform=ax.transAxes
    )
    _save(fig, path, dpi)
    plt.close(fig)


def _fig_examples(ctx, plt):
    recs, orients, v = ctx["recs"], ctx["orients"], ctx["prm"]["v"]
    path = os.path.join(ctx["out_dir"], "M1_examples.png")
    ok_meas = [
        (r, o) for r in recs for o in orients if r["res"][o]["P"]["reason"] == "ok"
    ]
    good = [(r, o) for r, o in ok_meas if r["res"][o]["P"]["rms"] < 0.2 * 2.0 / v]
    good.sort(key=lambda t: t[0]["res"][t[1]]["P"]["D_fit"])
    if len(good) >= 4:
        picks = [good[int(q * (len(good) - 1))] for q in (0.15, 0.4, 0.65, 0.9)]
    else:
        picks = sorted(ok_meas, key=lambda t: t[0]["res"][t[1]]["P"]["D_fit"])[:4]
    if not picks:
        _empty_fig(plt, path, f"{ctx['label']}: M1 found no accepted measurement")
        return
    fig, axs = plt.subplots(
        len(picks),
        4,
        figsize=(17, 4.3 * len(picks)),
        gridspec_kw=dict(width_ratios=[0.95, 1, 1.25, 1.25]),
        squeeze=False,
    )
    for axrow, (r, o) in zip(axs, picks):
        d = _recompute(ctx, r, o)
        _plot_example(
            ctx,
            axrow,
            o,
            d,
            title_extra=f"#{r['id']}, z = {r['z0'] * v:.0f} um, tilt {r['tilt']:.0f} deg",
        )
    fig.suptitle(
        f"{ctx['label']}, M1: accepted examples (15th, 40th, 65th, 90th percentile of chord-fit diameter)",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _save(fig, path, 110 if len(picks) == 4 else 120)
    plt.close(fig)


def _fig_failures(ctx, plt):
    recs, orients = ctx["recs"], ctx["orients"]
    path = os.path.join(ctx["out_dir"], "M1_failures.png")
    fail_picks = []
    for reason in ("merged", "unbracketed", "elongated", "fit"):
        cand = [
            (r, o)
            for r in recs
            for o in orients
            if r["res"][o]["P"]["reason"] == reason
        ]
        if cand:
            fail_picks.append(cand[len(cand) // 3])
        if len(fail_picks) == 3:
            break
    if not fail_picks:
        _empty_fig(
            plt,
            path,
            f"{ctx['label']}: no merged / unbracketed / elongated / fit rejections",
        )
        return
    fig, axs = plt.subplots(
        len(fail_picks),
        4,
        figsize=(17, 4.3 * len(fail_picks)),
        gridspec_kw=dict(width_ratios=[0.8, 1, 1.25, 1.25]),
        squeeze=False,
    )
    for axrow, (r, o) in zip(axs, fail_picks):
        d = _recompute(ctx, r, o)
        reason = r["res"][o]["P"]["reason"]
        _plot_example(ctx, axrow, o, d, title_extra=f"REJECTED ({reason})")
        axrow[2].text(
            0.02,
            0.97,
            EXPLAIN.get(reason, ""),
            transform=axrow[2].transAxes,
            fontsize=7.5,
            va="top",
            color=INK,
            bbox=dict(fc="white", ec=MUTED, lw=0.5, alpha=0.9),
        )
    fig.suptitle(
        f"{ctx['label']}, M1: rejected examples (what goes wrong)",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    _save(fig, path, 115)
    plt.close(fig)


def _fig_overview(ctx, plt):
    from matplotlib.patches import Rectangle

    recs, prm, vol, ZL, inner = (
        ctx["recs"],
        ctx["prm"],
        ctx["vol"],
        ctx["ZL"],
        ctx["inner"],
    )
    V = prm["v"]
    DX_GRID = prm["DX_GRID"]
    nz, ny, nx = vol.shape
    orients = ctx["orients"]
    TOL = 7.0 / V
    xs_best = []
    for r in recs:
        q = r["res"]["yz"]["P"]
        if q["reason"] == "ok":
            u = np.array(r["basis_yz"][1])
            xs_best.append(r["xc"] + q["dx_best"] * u[2])
    xs_best = np.array(xs_best)
    zmid = ZL[len(ZL) // 2]
    in_mid = np.asarray(inner[zmid])
    cols = np.where(in_mid.any(0))[0]
    grid = np.arange(cols.min(), cols.max() + 1) if len(cols) else np.arange(nx)
    if len(xs_best):
        cnt = [(np.abs(xs_best - g) <= TOL).sum() for g in grid]
        X_OV = int(grid[int(np.argmax(cnt))])
    else:
        X_OV = int(grid[len(grid) // 2])
    col = np.zeros(ny, bool)
    for z in ZL:
        col |= np.asarray(inner[z][:, X_OV])
    yy = np.where(col)[0]
    pad = int(round(80.0 / V))
    y_lo, y_hi = (
        (max(yy.min() - pad, 0), min(yy.max() + pad, ny)) if len(yy) else (0, ny)
    )
    z_lo, z_hi = 0, nz
    sl = np.asarray(vol[z_lo:z_hi, y_lo:y_hi, X_OV])
    lo_, hi_ = np.percentile(sl, [1, 99.7])
    abbrev = {
        "merged": "m",
        "unbracketed": "u",
        "elongated": "e",
        "multi_peak": "p",
        "no_band": "n",
        "fit": "f",
        "off_centre": "o",
        "track": "t",
        "tilt": "T",
    }
    ann_acc, ann_rej = [], []
    for r in recs:
        q = r["res"]["yz"]["P"]
        if r.get("track_ok") and "basis_yz" in r:
            a, u, v = (np.array(t) for t in r["basis_yz"])
            dxb = (
                q.get("dx_best", 0.0)
                if q["reason"]
                not in ("track", "tilt", "no_band", "merged", "multi_peak")
                else 0.0
            )
            x_pl = r["xc"] + dxb * u[2]
        else:
            x_pl = r["x0"]
            u = v = None
        if abs(x_pl - X_OV) > TOL:
            continue
        z0 = r["z0"]
        if q["reason"] == "ok":
            ib = int(np.argmin(np.abs(DX_GRID - q["dx_best"])))
            eL, eR = q["eL"][ib], q["eR"][ib]
            yb = r["yc"] + q["dx_best"] * u[1]
            ann_acc.append(
                (z0, yb + eL * v[1], yb + eR * v[1], q["w_max"] * V, q["D_fit"] * V)
            )
        else:
            yb = r["yc"] if r.get("track_ok") else r["y0"]
            ann_rej.append((z0, yb, abbrev.get(q["reason"], "?")))
    zt = 4.0 * 2.0 / V

    def draw_annotations(ax, big=False):
        seen = {}
        for z0, y1, y2, wmx, dft in sorted(
            ann_acc, key=lambda t: (t[0], min(t[1], t[2]))
        ):
            k_ = seen.get(z0, 0)
            seen[z0] = k_ + 1
            dz_lab = -(4 if big else 3) if k_ % 2 == 0 else (16 if big else 14)
            ax.plot(
                [y1 * V, y2 * V],
                [z0 * V, z0 * V],
                color=ORANGE,
                lw=2.6 if big else 2.0,
                solid_capstyle="butt",
            )
            for yv_ in (y1, y2):
                ax.plot(
                    [yv_ * V] * 2,
                    [(z0 - zt) * V, (z0 + zt) * V],
                    color=ORANGE,
                    lw=1.4 if big else 1.1,
                )
            ax.text(
                max(y1, y2) * V + 2,
                z0 * V + dz_lab,
                f"{wmx:.1f} / {dft:.1f}" if big else f"{wmx:.1f}",
                color="#ffd9c7",
                fontsize=8.5 if big else 7,
                fontweight="bold",
                clip_on=True,
            )
        for z0, yb, ab in ann_rej:
            ax.plot(yb * V, z0 * V, "x", color="#ff8080", ms=6 if big else 4.5, mew=1.0)
            ax.text(
                yb * V + 2,
                z0 * V + (9 if big else 7),
                ab,
                color="#ffb3b3",
                fontsize=8 if big else 6.5,
                clip_on=True,
            )

    fig = plt.figure(figsize=(17, 11))
    ax = fig.add_axes([0.04, 0.05, 0.50, 0.86])
    ax.imshow(
        sl,
        cmap="gray",
        vmin=lo_,
        vmax=hi_,
        extent=(y_lo * V, y_hi * V, z_hi * V, z_lo * V),
        aspect="equal",
    )
    draw_annotations(ax)
    for z in ZL:
        ax.axhline(z * V, color=AQUA, lw=0.4, alpha=0.6)
    ax.set_xlabel("y (um)")
    ax.set_ylabel("z (um), fibre direction")
    ax.set_title(
        f"{ctx['label']}: yz slice at x = {X_OV * V:.0f} um. Orange bars: accepted fibres whose widest plane "
        f"lies within {TOL * V:.0f} um of this slice\n(bar = FWHM edges at the widest plane, label = max width "
        f"in um; n={len(ann_acc)}). Red x: rejected candidates near this plane (n={len(ann_rej)});\nm merged, "
        f"u unbracketed, e elongated, f fit, t track, p multi-peak, o off-centre. Thin aqua lines: candidate "
        f"z-levels.",
        fontsize=8.5,
        loc="left",
    )
    hz = 60.0 * 2.0 / V
    if ann_acc:
        A = np.array([(z0, 0.5 * (y1 + y2)) for z0, y1, y2, _, _ in ann_acc])
        best, bc = None, -1
        for zc_, yc_ in A:
            n_ = (
                (np.abs(A[:, 0] - zc_) < 55 * 2 / V)
                & (np.abs(A[:, 1] - yc_) < 55 * 2 / V)
            ).sum()
            if n_ > bc:
                bc, best = n_, (zc_, yc_)
        zc_, yc_ = best
        zc_ = np.clip(zc_, z_lo + hz, max(z_lo + hz, z_hi - hz))
        yc_ = np.clip(yc_, y_lo + hz, max(y_lo + hz, y_hi - hz))
        axz = fig.add_axes([0.58, 0.05, 0.40, 0.40], anchor="W")
        axz.imshow(
            sl,
            cmap="gray",
            vmin=lo_,
            vmax=hi_,
            extent=(y_lo * V, y_hi * V, z_hi * V, z_lo * V),
            aspect="equal",
            interpolation="bilinear",
        )
        draw_annotations(axz, big=True)
        axz.set_xlim((yc_ - hz) * V, (yc_ + hz) * V)
        axz.set_ylim((zc_ + hz) * V, (zc_ - hz) * V)
        axz.set_title(
            "zoom of the same yz slice (label: max width / chord-fit 2R, um)",
            fontsize=8.5,
            loc="left",
        )
        axz.set_xlabel("y (um)")
        axz.set_ylabel("z (um)")
        ax.add_patch(
            Rectangle(
                ((yc_ - hz) * V, (zc_ - hz) * V),
                2 * hz * V,
                2 * hz * V,
                fill=False,
                ec="#7fd4ff",
                lw=1.0,
            )
        )
    zs = zmid
    ax2 = fig.add_axes([0.58, 0.52, 0.40, 0.40], anchor="W")
    xy = np.asarray(vol[zs])
    im = np.asarray(inner[zs])
    lo2, hi2 = np.percentile(xy[im] if im.any() else xy, [1, 99.7])
    ax2.imshow(xy, cmap="gray", vmin=lo2, vmax=hi2, extent=(0, nx * V, ny * V, 0))
    if im.any():
        ax2.contour(
            np.arange(nx) * V,
            np.arange(ny) * V,
            im.astype(float),
            [0.5],
            colors=[AQUA],
            linewidths=0.8,
        )
    cat = {"both": [], "one": [], "rej": [], "trk": []}
    for r in recs:
        if r["z0"] != zs:
            continue
        st = [r["res"][o]["P"]["reason"] == "ok" for o in orients]
        key = (
            "both"
            if all(st) and len(orients) > 1
            else (
                "one"
                if any(st)
                else ("trk" if r["res"]["yz"]["P"]["reason"] == "track" else "rej")
            )
        )
        cat[key].append((r["x0"], r["y0"]))
    sty = {
        "rej": ("x", "#ff8080", 3, "rejected in all orientations"),
        "trk": ("+", "#ffd24d", 3.5, "xy tracking failed"),
        "one": ("o", "#7fd4ff", 4.5, "accepted in one orientation"),
        "both": ("o", GOOD, 5.5, "accepted in yz and xz"),
    }
    for k in ("rej", "trk", "one", "both"):
        if cat[k]:
            arr = np.array(cat[k]) * V
            mk, colr, ms, lab = sty[k]
            ax2.plot(
                arr[:, 0],
                arr[:, 1],
                mk,
                color=colr,
                ms=ms,
                mfc="none" if mk == "o" else colr,
                mew=1.2,
                label=f"{lab} ({len(cat[k])})",
                ls="none",
            )
    ax2.axvline(
        X_OV * V, color=ORANGE, lw=1.0, ls="--", label="position of the yz slice (left)"
    )
    ax2.set_xlim(0, nx * V)
    ax2.set_ylim(ny * V, 0)
    ax2.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
    ax2.set_title(
        f"xy slice at z = {zs * V:.0f} um: every candidate fibre (local maximum) at this level;\n"
        f"aqua contour = interior used (>= 80 um from the sample edge)",
        fontsize=8.5,
        loc="left",
    )
    ax2.set_xlabel("x (um)")
    ax2.set_ylabel("y (um)")
    _save(fig, os.path.join(ctx["out_dir"], "M1_overview.png"), 110)
    plt.close(fig)


def _fig_distribution(ctx, plt):
    recs, orients, V, S = ctx["recs"], ctx["orients"], ctx["prm"]["v"], ctx["summary"]
    settings, sig = ctx["settings"], ctx["sigma"]
    refs = _refs(ctx)
    per = S["settings"]
    cc = S["counts"]
    reasons = cc["reasons"]
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.32, wspace=0.62)
    ax = fig.add_subplot(gs[0, 0])
    s = per["P"]
    wm = (
        np.array(
            [
                r["res"][o]["P"]["w_max"]
                for r in recs
                for o in orients
                if r["res"][o]["P"]["reason"] == "ok"
            ]
        )
        * V
    )
    df = (
        np.array(
            [
                r["res"][o]["P"]["D_fit"]
                for r in recs
                for o in orients
                if r["res"][o]["P"]["reason"] == "ok"
            ]
        )
        * V
    )
    allv = np.concatenate([wm, df, [r_[0] for r_ in refs]])
    b_lo = max(0.0, min(6.0, np.percentile(allv, 1) - 3))
    b_hi = max(32.0, np.percentile(allv, 99) + 4)
    bins = np.arange(b_lo, b_hi + 0.01, 0.75)

    def lab(name, st):
        if st["n"] == 0:
            return f"{name}\n(none)"
        return (
            f"{name}\n{st['mean']:.1f} +/- {st['std']:.1f} um\n"
            f"median {st['median']:.1f} [IQR {st['q1']:.1f}, {st['q3']:.1f}]"
        )

    if len(wm):
        ax.hist(
            wm,
            bins,
            color=BLUE,
            alpha=0.55,
            edgecolor="white",
            linewidth=0.6,
            label=lab("max width (FWHM at widest plane)", s["wmax"]),
        )
        ax.hist(
            df,
            bins,
            histtype="step",
            color=ORANGE,
            lw=2.0,
            label=lab(
                f"blurred chord-law fit, 2R (sigma {sig:.2f} vox, {ctx['sigma_source']})",
                s["dfit"],
            ),
        )
    yt = max(ax.get_ylim()[1], 1.0)
    for val, lbl, ls, _c in refs:
        ax.axvline(val, color=INK2, lw=1.0, ls=ls)
        ax.text(val + 0.2, yt * 0.97, lbl, fontsize=8, color=INK2, va="top")
    ax.set_xlabel("diameter (um)")
    ax.set_ylabel("measurements (" + " + ".join(orients) + ")")
    ax.set_title(
        f"{ctx['label']}: {cc['accepted_measurements']} accepted measurements "
        f"({cc['accepted_segments']} fibre segments, about {cc['unique_fibres']} distinct fibres) "
        f"out of {cc['candidates']} candidates x {len(orients)} orientation(s)",
        loc="left",
        fontsize=9,
    )
    ax.set_xlim(b_lo - 2, b_hi + 4)
    if len(wm):
        right = np.median(df if len(df) else wm) < 0.5 * (b_lo + b_hi)
        ax.legend(
            fontsize=8,
            loc="upper right" if right else "upper left",
            bbox_to_anchor=(1.0, 0.9) if right else (0.0, 0.9),
            frameon=False,
            labelspacing=1.2,
        )
    # rejection reasons
    ax = fig.add_subplot(gs[0, 1])
    order = [
        "ok",
        "merged",
        "unbracketed",
        "track",
        "elongated",
        "fit",
        "multi_peak",
        "off_centre",
        "no_band",
        "tilt",
    ]
    order = [k for k in order if any(reasons[o].get(k, 0) for o in orients)]
    yv = np.arange(len(order))
    mx = max([reasons[o].get(k, 0) for o in orients for k in order] + [1])
    for j, (o, colr) in enumerate(zip(orients, ("#52514e", "#b5b3ab"))):
        off = (j - 0.5) * 0.38 if len(orients) > 1 else 0.0
        vals = [reasons[o].get(k, 0) for k in order]
        ax.barh(yv + off, vals, height=0.36, color=colr, label=f"{o} slices")
        for yy_, vv in zip(yv, vals):
            ax.text(
                vv + 0.005 * mx, yy_ + off, str(vv), va="center", fontsize=7, color=INK2
            )
    ax.set_yticks(yv)
    ax.set_yticklabels(
        ["accepted" if k == "ok" else REASON_LABEL.get(k, k) for k in order], fontsize=8
    )
    ax.invert_yaxis()
    ax.set_xlabel("count")
    ax.set_title("outcome per candidate and orientation (primary settings)", loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    # sensitivity
    ax = fig.add_subplot(gs[1, 0])
    seg = {k: 2 * ctx["prm"]["SEG_HALF"][k] * V for k in ("L40", "L80", "L120")}
    set_label = {
        "P": f"primary: L={seg['L80']:.0f} um, local-dip FWHM, dip>=50%, sigma {sig:.2f}",
        "G": "global matrix baseline FWHM",
        "D3": "dip criterion >= 30% (looser)",
        "D7": "dip criterion >= 70% (stricter)",
        "L40": f"segment L = {seg['L40']:.0f} um",
        "L120": f"segment L = {seg['L120']:.0f} um",
        "LIT": f"literal axis-aligned slices, L = {2 * ctx['prm']['LIT_THALF'] * V:.0f} um",
        "Slo": f"chord fit with sigma = {sig * SETTINGS_ALL['Slo'][3]:.2f} vox",
        "Shi": f"chord fit with sigma = {sig * SETTINGS_ALL['Shi'][3]:.2f} vox",
    }
    rows_ = [
        k
        for k in ("P", "G", "D3", "D7", "L40", "L120", "LIT", "Slo", "Shi")
        if k in settings
    ]
    x_hi = max(32.0, b_hi)
    x_lo = max(0.0, min(8.0, b_lo))
    for i, key in enumerate(rows_):
        for est, colr, off in (("wmax", BLUE, -0.15), ("dfit", ORANGE, 0.15)):
            st = per[key][est]
            if st["n"] == 0:
                continue
            ax.errorbar(
                st["median"],
                i + off,
                xerr=[[st["median"] - st["q1"]], [st["q3"] - st["median"]]],
                fmt="o",
                color=colr,
                ms=5,
                capsize=2,
                lw=1.2,
            )
        ax.text(
            x_hi - 0.5,
            i,
            f"n={per[key]['wmax']['n']}",
            fontsize=7.5,
            va="center",
            color=INK2,
            ha="right",
        )
    i = len(rows_)
    st = per["P"]["dfree"]
    if st["n"]:
        ax.errorbar(
            st["median"],
            i + 0.15,
            xerr=[[st["median"] - st["q1"]], [st["q3"] - st["median"]]],
            fmt="o",
            color=ORANGE,
            ms=5,
            capsize=2,
            lw=1.2,
        )
        ax.text(
            x_hi - 0.5,
            i,
            f"sigma med {per['P']['sig_free']['median']:.2f} vox",
            fontsize=7.5,
            va="center",
            color=INK2,
            ha="right",
        )
    ax.set_yticks(range(len(rows_) + 1))
    ax.set_yticklabels(
        [set_label[k] for k in rows_] + ["chord fit with free sigma (3 parameters)"],
        fontsize=8,
    )
    ax.invert_yaxis()
    for val, lbl, ls, _c in refs:
        ax.axvline(val, color=INK2, lw=1.0, ls=ls)
    ax.set_xlim(x_lo, x_hi)
    ax.plot([], [], "o", color=BLUE, label="max width")
    ax.plot([], [], "o", color=ORANGE, label="chord fit 2R")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.set_xlabel("diameter (um): median and IQR")
    ttl = "sensitivity to method choices (each row changes one setting vs. primary)"
    if len(rows_) == 1:
        ttl += "\n(fast mode: primary only)"
    ax.set_title(ttl, loc="left")
    # yz vs xz
    ax = fig.add_subplot(gs[1, 1])
    lim = (x_lo, max(30.0, b_hi))
    both = [
        r
        for r in recs
        if len(orients) > 1 and all(r["res"][o]["P"]["reason"] == "ok" for o in orients)
    ]
    if both:
        a_ = np.array([r["res"]["yz"]["P"]["D_fit"] for r in both]) * V
        b_ = np.array([r["res"]["xz"]["P"]["D_fit"] for r in both]) * V
        a2 = np.array([r["res"]["yz"]["P"]["w_max"] for r in both]) * V
        b2 = np.array([r["res"]["xz"]["P"]["w_max"] for r in both]) * V
        ax.plot(
            a2,
            b2,
            "o",
            color=BLUE,
            ms=4.5,
            alpha=0.7,
            mec="white",
            mew=0.5,
            label="max width",
        )
        ax.plot(
            a_,
            b_,
            "o",
            color=ORANGE,
            ms=4.5,
            alpha=0.7,
            mec="white",
            mew=0.5,
            label="chord fit 2R",
        )
        r_ = (
            np.corrcoef(a_, b_)[0, 1]
            if len(a_) > 2 and np.std(a_) > 0 and np.std(b_) > 0
            else np.nan
        )
        ax.set_title(
            f"same fibre segment measured in yz and xz slices (n={len(both)}); "
            f"chord-fit r = {r_:.2f}\nmedian |yz - xz| = {np.median(np.abs(a_ - b_)):.1f} um (chord), "
            f"{np.median(np.abs(a2 - b2)):.1f} um (max width)",
            loc="left",
            fontsize=8.5,
        )
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.set_title(
            "yz vs xz: no segment accepted in both orientations"
            + (" (fast mode: yz only)" if len(orients) == 1 else ""),
            loc="left",
            fontsize=8.5,
        )
    ax.plot(lim, lim, color=MUTED, lw=0.8)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("diameter from yz slices (um)")
    ax.set_ylabel("diameter from xz slices (um)")
    path = os.path.join(ctx["out_dir"], "M1_distribution.png")
    tmp = path[:-4] + "_tmp.png"
    fig.savefig(tmp, dpi=110, bbox_inches="tight")
    os.replace(tmp, path)
    plt.close(fig)


def _fig_truth(ctx, plt):
    recs, orients, V, S, truth = (
        ctx["recs"],
        ctx["orients"],
        ctx["prm"]["v"],
        ctx["summary"],
        ctx["truth"],
    )
    tt, mw, mf = [], [], []
    for r in recs:
        for o in orients:
            q = r["res"][o]["P"]
            if q["reason"] == "ok" and np.isfinite(r.get("true_diam_um", np.nan)):
                tt.append(r["true_diam_um"])
                mw.append(q["w_max"] * V)
                mf.append(q["D_fit"] * V)
    path = os.path.join(ctx["out_dir"], "M1_truth_scatter.png")
    if not tt:
        _empty_fig(
            plt,
            path,
            f"{ctx['label']}: no accepted measurement matched to a true fibre",
        )
        return
    tt, mw, mf = map(np.asarray, (tt, mw, mf))
    fig, axs = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw=dict(width_ratios=[1, 1.1])
    )
    ax = axs[0]
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.15, 0.15, len(tt)) if np.ptp(tt) < 1e-6 else 0.0
    ax.plot(
        tt + jit,
        mw,
        "o",
        color=BLUE,
        ms=4,
        alpha=0.6,
        mec="white",
        mew=0.4,
        label=f"max width: bias {S['bias_maxw_um']:+.2f} um, RMSE {S['rmse_maxw_um']:.2f} um",
    )
    ax.plot(
        tt + jit,
        mf,
        "o",
        color=ORANGE,
        ms=4,
        alpha=0.6,
        mec="white",
        mew=0.4,
        label=f"chord fit 2R: bias {S['bias_chord_um']:+.2f} um, RMSE {S['rmse_chord_um']:.2f} um",
    )
    allv = np.concatenate([tt, mw, mf])
    lo, hi = (
        np.floor(np.percentile(allv, 0.5) - 2),
        np.ceil(np.percentile(allv, 99.5) + 2),
    )
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, label="identity")
    ax.axhline(truth.mean_diam_um, color=AQUA, lw=0.8, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("true diameter of the matched fibre (um)")
    ax.set_ylabel("measured (um)")
    ax.set_title(
        f"{ctx['label']}, M1: measured vs true per-fibre diameter (n={len(tt)} measurements)\n"
        f"true mean D = {truth.mean_diam_um:.2f} um",
        loc="left",
        fontsize=9,
    )
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax = axs[1]
    e1, e2 = mw - tt, mf - tt
    m = max(3.0, np.ceil(np.percentile(np.abs(np.concatenate([e1, e2])), 99)))
    bins = np.arange(-m, m + 0.01, 0.5)
    ax.hist(
        e1, bins, color=BLUE, alpha=0.55, edgecolor="white", label="max width - true"
    )
    ax.hist(
        e2, bins, histtype="step", color=ORANGE, lw=2.0, label="chord fit 2R - true"
    )
    ax.axvline(0, color=INK2, lw=1.0)
    ax.set_xlabel("error (um)")
    ax.set_ylabel("measurements")
    ax.set_title(
        f"error distribution; median error: max width {np.median(e1):+.2f} um, chord {np.median(e2):+.2f} um",
        loc="left",
        fontsize=9,
    )
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, path, 120)
    plt.close(fig)
