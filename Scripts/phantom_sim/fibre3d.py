"""Torch-free interactive 3D fibre-tube illustrations for the phantom study (this box).

Renders the GT fibre configurations (packer output, (Z, 2, N) centres in um) as 3D tubes
in self-contained Plotly HTML: crisp fibres, per-fibre transparency, light enough to spin
on the laptop. Three scenes, one per experiment:

  - exp1_orientation_html : fibres along the rotation axis vs transverse (+ rotation axis,
                            X-ray source, detector panel, and beam line schematic).
  - exp2_zpositions_html  : the beam-axis bundle drawn at N positions along the optical axis,
                            faded so the sweep reads as motion (+ optical axis, source,
                            detector).
  - exp3_chunks_html      : the wandering bundle once, with a chosen set of depth-chunks
                            highlighted (adjacent A vs every-second B), the rest faded. The
                            real wander is sub-fibre-diameter, so it is exaggerated (labelled)
                            for visibility.

Fibres are drawn as Scatter3d polylines (one NaN-separated trace) rather than voxel
isosurfaces, so 1000 packed fibres stay individually visible after subsampling. Scenes
auto-range with a cubic aspect so nothing (source/detector included) is clipped. Distances in
the source/detector schematic are illustrative, not to scale.
"""
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from phantom_sim.phantom import upsample_config


# ----------------------------------------------------------------------------------
# Fibre-line primitives
# ----------------------------------------------------------------------------------

def _subsample(cfg, radii, n_show, seed=0):
    """Keep at most n_show fibres, evenly spread across the pack for a representative subset."""
    cfg = np.asarray(cfg, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = cfg.shape[2]
    if n_show is None or n_show >= n:
        return cfg, radii
    idx = np.unique(np.linspace(0, n - 1, int(n_show)).round().astype(int))
    return cfg[:, :, idx], radii[idx]


def _exaggerate_wander(cfg, factor):
    """Amplify each fibre's transverse deviation from its own mean by `factor`.

    Leaves every fibre's mean position put (the bundle footprint is unchanged) and scales only
    the along-length wiggle, so a sub-fibre-diameter wander becomes visible without exploding
    the bundle. factor=1.0 is faithful.
    """
    cfg = np.asarray(cfg, dtype=np.float64)
    if factor == 1.0:
        return cfg
    mean = cfg.mean(axis=0, keepdims=True)
    return mean + factor * (cfg - mean)


def _lines_xyz(cfg, along_um, orient, mask=None):
    """NaN-separated polyline coordinates for every fibre in `cfg`.

    cfg: (Z, 2, N) centres in um (index 0 = x, 1 = y). along_um: (Z,) the along-fibre
    coordinate per slice. orient maps (along, x, y) to world axes: "z" puts the fibre axis
    on world Z (fibres run up), "x" puts it on world X (fibres run sideways). `mask`: optional
    (Z,) bool selecting which slices to draw; a break is inserted where it turns off so
    non-contiguous runs do not connect.
    """
    cfg = np.asarray(cfg, dtype=np.float64)
    along_um = np.asarray(along_um, dtype=np.float64)
    Z, _, N = cfg.shape
    X, Y, Zc = [], [], []
    for k in range(N):
        tx, ty = cfg[:, 0, k], cfg[:, 1, k]
        if orient == "z":
            xs, ys, zs = tx, ty, along_um
        else:  # "x"
            xs, ys, zs = along_um, tx, ty
        if mask is None:
            xs_k, ys_k, zs_k = list(xs), list(ys), list(zs)
        else:
            xs_k, ys_k, zs_k = [], [], []
            for zi in range(Z):
                if mask[zi]:
                    xs_k.append(xs[zi])
                    ys_k.append(ys[zi])
                    zs_k.append(zs[zi])
                elif xs_k and not np.isnan(xs_k[-1]):
                    xs_k.append(np.nan)
                    ys_k.append(np.nan)
                    zs_k.append(np.nan)
        X.extend(xs_k)
        X.append(np.nan)
        Y.extend(ys_k)
        Y.append(np.nan)
        Zc.extend(zs_k)
        Zc.append(np.nan)
    return X, Y, Zc


def _fibre_trace(cfg, along_um, orient, color, opacity, width, name, mask=None,
                 showlegend=True):
    X, Y, Z = _lines_xyz(cfg, along_um, orient, mask=mask)
    return go.Scatter3d(x=X, y=Y, z=Z, mode="lines",
                        line=dict(color=color, width=width),
                        opacity=opacity, name=name, showlegend=showlegend,
                        hoverinfo="skip", connectgaps=False)


def _arrow(p0, p1, color, name, width=7, sizeref=40.0, showlegend=True):
    """A world-space arrow: a thick line plus a cone head, as two traces."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    line = go.Scatter3d(x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                        mode="lines", line=dict(color=color, width=width),
                        name=name, hoverinfo="skip", showlegend=showlegend)
    head = go.Cone(x=[p1[0]], y=[p1[1]], z=[p1[2]], u=[d[0]], v=[d[1]], w=[d[2]],
                   sizemode="absolute", sizeref=sizeref, anchor="tip",
                   colorscale=[[0, color], [1, color]], showscale=False,
                   hoverinfo="skip", name=name, showlegend=False)
    return line, head


def _rect(const_axis, const, a_range, b_range, color, opacity, name, showlegend=True):
    """A rectangular panel perpendicular to one world axis (the detector), as a Mesh3d.

    const_axis in {"x","y","z"} is held at `const`; the panel spans a_range x b_range in the
    other two axes (in their natural order x<y<z).
    """
    a = [a_range[0], a_range[1], a_range[1], a_range[0]]
    b = [b_range[0], b_range[0], b_range[1], b_range[1]]
    c = [const] * 4
    if const_axis == "x":
        x, y, z = c, a, b
    elif const_axis == "y":
        x, y, z = a, c, b
    else:
        x, y, z = a, b, c
    return go.Mesh3d(x=x, y=y, z=z, i=[0, 0], j=[1, 2], k=[2, 3],
                     color=color, opacity=opacity, name=name, showlegend=showlegend,
                     hoverinfo="skip")


def _label(p, text, color):
    return go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="text",
                        text=[text], textfont=dict(color=color, size=12),
                        hoverinfo="skip", showlegend=False)


def _scene(eye=(1.5, 1.5, 0.9), aspectmode="cube", aspect=None):
    """Hidden-axis 3D scene auto-ranging to include EVERY trace (source/detector never
    clipped). aspectmode "cube" (equal box), "data" (true proportions), or pass `aspect`
    (x, y, z) for a manual ratio, e.g. to elongate a long bundle so it is not squashed."""
    ax = dict(showbackground=False, showticklabels=False, showgrid=False,
              zeroline=False, visible=False, title="")
    s = dict(xaxis=dict(**ax), yaxis=dict(**ax), zaxis=dict(**ax),
             camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2])))
    if aspect is not None:
        s["aspectmode"] = "manual"
        s["aspectratio"] = dict(x=aspect[0], y=aspect[1], z=aspect[2])
    else:
        s["aspectmode"] = aspectmode
    return s


# ----------------------------------------------------------------------------------
# Experiment 1: orientation vs the rotation axis
# ----------------------------------------------------------------------------------

def exp1_orientation_html(cfg, radii, out_html, depth_um=1024.0, n_show=150, seed=0):
    """Two 3D scenes: fibres along the rotation axis vs transverse to it.

    Rotation axis = world Z (bold red arrow). Beam runs along world Y from an X-ray source
    point, through the sample, to a detector panel (dotted beam line; schematic, not to scale).
    Fibres semi-transparent so the axis shows through. `cfg` is a straight-bundle config.
    """
    cfg, radii = _subsample(cfg, radii, n_show, seed)
    Z = cfg.shape[0]
    if Z == 1:
        cfg = np.repeat(cfg, 2, axis=0)
        Z = 2
    along = np.linspace(-depth_um / 2.0, depth_um / 2.0, Z)
    b = max(float(np.abs(cfg[:, :2, :]).max()), depth_um / 2.0)
    src_y, det_y, axis_z = -1.4 * b, 1.4 * b, 1.15 * b

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.02,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Fibres ALONG the rotation axis", "Fibres TRANSVERSE to it"))

    for col, orient in ((1, "z"), (2, "x")):
        first = col == 1
        fig.add_trace(_fibre_trace(cfg, along, orient, "#3b6ea5", 0.55, 2.5, "fibres",
                                   showlegend=first), row=1, col=col)
        ln, hd = _arrow((0, 0, -axis_z), (0, 0, axis_z), "#d62728", "rotation axis",
                        sizeref=b * 0.14, showlegend=first)
        fig.add_trace(ln, row=1, col=col)
        fig.add_trace(hd, row=1, col=col)
        fig.add_trace(_label((0, 0, axis_z), "  rotation axis", "#d62728"), row=1, col=col)
        fig.add_trace(go.Scatter3d(x=[0], y=[src_y], z=[0], mode="markers",
                                   marker=dict(size=9, color="#f0a000", symbol="diamond"),
                                   name="X-ray source", showlegend=first, hoverinfo="skip"),
                      row=1, col=col)
        fig.add_trace(_label((0, src_y, 0), "  source", "#c07800"), row=1, col=col)
        fig.add_trace(_rect("y", det_y, (-b, b), (-b, b), "#7788aa", 0.35, "detector",
                            showlegend=first), row=1, col=col)
        fig.add_trace(_label((0, det_y, -0.7 * b), "  detector", "#556699"), row=1, col=col)
        fig.add_trace(go.Scatter3d(x=[0, 0], y=[src_y, det_y], z=[0, 0], mode="lines",
                                   line=dict(color="#f0a000", width=3, dash="dot"),
                                   name="X-ray beam", showlegend=first, hoverinfo="skip"),
                      row=1, col=col)

    fig.update_layout(
        title="Experiment 1: fibre orientation vs the CT rotation axis (full 360 scan)",
        scene=_scene(), scene2=_scene(),
        margin=dict(l=0, r=0, t=60, b=0), showlegend=True,
        legend=dict(orientation="h", y=0))
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)
    return str(out_html)


# ----------------------------------------------------------------------------------
# Experiment 2: the z-position sweep along the optical axis
# ----------------------------------------------------------------------------------

def exp2_zpositions_html(cfg, radii, out_html, depth_um=1024.0, n_positions=4, n_show=150,
                         seed=0, position_gap_frac=0.35):
    """One scene: the bundle drawn at `n_positions` offsets ALONG world X (the optical axis,
    left to right), coloured on a gradient and fading with N so the sweep reads as movement.
    X-ray source at left, detector panel at right (schematic). The gap between positions is
    exaggerated to `position_gap_frac` of the bundle length so they read as distinct.
    """
    cfg, radii = _subsample(cfg, radii, n_show, seed)
    Z = cfg.shape[0]
    if Z == 1:
        cfg = np.repeat(cfg, 2, axis=0)
        Z = 2
    length = depth_um
    along0 = np.linspace(-length / 2.0, length / 2.0, Z)
    step = position_gap_frac * length
    max_off = ((n_positions - 1) / 2.0) * step
    bx = length / 2.0 + max_off
    byz = float(np.abs(cfg[:, :2, :]).max())
    pad = 0.35 * length
    axis_x, src_x, det_x = bx + 0.6 * pad, -(bx + pad), bx + pad

    colours = ["#2c7fb8", "#41b6c4", "#7fcdbb", "#c7e9b4", "#fec44f", "#f03b20"]
    fig = go.Figure()
    for n in range(n_positions):
        off = (n - (n_positions - 1) / 2.0) * step
        opacity = max(0.82 - 0.13 * n, 0.22)
        fig.add_trace(_fibre_trace(cfg, along0 + off, "x", colours[n % len(colours)],
                                   opacity, 2.6, "N=%d" % n))
    ln, hd = _arrow((-axis_x, 0, 0), (axis_x, 0, 0), "#d62728", "optical axis",
                    sizeref=byz * 0.6)
    fig.add_trace(ln)
    fig.add_trace(hd)
    fig.add_trace(_label((axis_x, 0, 0), "  optical axis", "#d62728"))
    fig.add_trace(go.Scatter3d(x=[src_x], y=[0], z=[0], mode="markers",
                               marker=dict(size=9, color="#f0a000", symbol="diamond"),
                               name="X-ray source", hoverinfo="skip"))
    fig.add_trace(_label((src_x, 0, 0), "  source", "#c07800"))
    fig.add_trace(_rect("x", det_x, (-byz, byz), (-byz, byz), "#7788aa", 0.35, "detector"))
    fig.add_trace(_label((det_x, 0, -0.7 * byz), "  detector", "#556699"))

    fig.update_layout(
        title="Experiment 2: same bundle swept N=0..%d along the optical axis "
              "(warm-start reuses the previous solve)" % (n_positions - 1),
        scene=_scene(eye=(0.4, -2.0, 0.8), aspectmode="data"),
        margin=dict(l=0, r=0, t=60, b=0), legend=dict(orientation="h", y=0))
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)
    return str(out_html)


# ----------------------------------------------------------------------------------
# Experiment 3: the wandering bundle, chunks highlighted
# ----------------------------------------------------------------------------------

def exp3_chunks_html(cfg, radii, out_html, n_chunks=8, chunk_render_slices=14,
                     highlight=(0, 1, 2, 3), n_show=120, seed=0, length_um=8192.0,
                     wander_exaggeration=15.0, long_aspect=4.5):
    """One scene: the full wandering bundle (faint) with the `highlight` depth-chunks drawn
    bright and opaque, the rest faded. The bundle axis runs left to right along world X; the
    scene uses a manual `long_aspect` so the long bundle is drawn elongated, not squashed to a
    stub. The real transverse wander is sub-fibre-diameter, so it is exaggerated by
    `wander_exaggeration` (labelled). Ordering A highlights adjacent chunks (0,1,2,3), B
    every-second (0,2,4,6).
    """
    cfg, radii = _subsample(cfg, radii, n_show, seed)
    render_slices = int(n_chunks * chunk_render_slices)
    full = upsample_config(cfg, render_slices)
    full = _exaggerate_wander(full, wander_exaggeration)
    along = np.linspace(-length_um / 2.0, length_um / 2.0, render_slices)
    byz = float(np.abs(full[:, :2, :]).max())
    axis_x = 1.06 * (length_um / 2.0)

    chunk_of = np.arange(render_slices) // chunk_render_slices

    fig = go.Figure()
    fig.add_trace(_fibre_trace(full, along, "x", "#b0b0b0", 0.16, 1.5, "full bundle (faded)"))
    palette = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b", "#e377c2",
               "#17becf"]
    for c in highlight:
        m = chunk_of == c
        fig.add_trace(_fibre_trace(full, along, "x", palette[c % len(palette)], 0.85, 3.0,
                                   "chunk %d" % c, mask=m))
    fig.add_trace(_arrow((-axis_x, 0, 0), (axis_x, 0, 0), "#333333", "bundle axis",
                         sizeref=byz * 0.6)[0])

    order = ("adjacent (0,1,2,3)" if list(highlight) == [0, 1, 2, 3] else
             "every-second (0,2,4,6)" if list(highlight) == [0, 2, 4, 6] else str(list(highlight)))
    note = (" - transverse wander exaggerated %gx for visibility" % wander_exaggeration
            if wander_exaggeration != 1.0 else "")
    fig.update_layout(
        title="Experiment 3: wandering bundle, warm chain over chunks %s%s" % (order, note),
        scene=_scene(eye=(0.3, -2.0, 0.9), aspect=(long_aspect, 1.0, 1.0)),
        margin=dict(l=0, r=0, t=60, b=0), legend=dict(orientation="h", y=0))
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)
    return str(out_html)
