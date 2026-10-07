"""Time voxelize_config, Python vs Rust, on the same machine and inputs.

Run from Scripts/:  PYTHONPATH=. python phantom_sim/rust_voxelize/bench.py
Single-thread Rust: RAYON_NUM_THREADS=1 PYTHONPATH=. python phantom_sim/rust_voxelize/bench.py
Fibre centres are random (not packed), at the packed bundles' density, so timing is
representative; the bit-identity check lives in tests/test_voxelize_rust.py.
"""
import os
import time

import numpy as np

from phantom_sim.phantom import voxelize_config

CASES = {  # name: (slices, domain radius um, grid, voxel um)
    "make_phantom default": (64, 100.0, 300, 2.0),
    "large bundle": (256, 300.0, 600, 2.0),
}


def best_of(fn, n):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return min(times), out


rng = np.random.default_rng(0)
threads = os.environ.get("RAYON_NUM_THREADS", f"all ({os.cpu_count()})")
print(f"rust threads: {threads}")
for name, (Z, dom, g, vox) in CASES.items():
    n_fibres = int(0.3 * dom**2 / 6.0**2)          # fvf 0.3, 6 um radius
    ang = rng.uniform(0, 2 * np.pi, n_fibres)
    rad = dom * np.sqrt(rng.uniform(0, 1, n_fibres))
    xy = np.stack([rad * np.cos(ang), rad * np.sin(ang)])           # (2, N)
    cfg = np.repeat(xy[None], Z, axis=0) + rng.normal(0, 1.0, (Z, 2, n_fibres))
    radii = np.full(n_fibres, 6.0)
    kw = dict(transverse_shape=(g, g), voxel_um=vox, supersample=4)
    t_py, v_py = best_of(lambda: voxelize_config(cfg, radii, backend="python", **kw), 3)
    t_rs, v_rs = best_of(lambda: voxelize_config(cfg, radii, backend="rust", **kw), 7)
    assert np.array_equal(v_py, v_rs)
    print(f"{name:22s} Z={Z} N={n_fibres} grid={g}x{g}: "
          f"python {t_py:.3f} s, rust {t_rs*1e3:.1f} ms, {t_py / t_rs:.0f}x")
