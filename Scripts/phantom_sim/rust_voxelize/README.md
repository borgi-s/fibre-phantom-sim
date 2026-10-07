# phantom_rs: Rust port of `voxelize_config`

The per-slice disc rasteriser from `phantom.voxelize_config`, ported to Rust with PyO3. Same
arithmetic in the same order as the NumPy version, so the output is bit-identical
(`tests/test_voxelize_rust.py`). Slices along the fibre axis are independent and run in parallel
with rayon.

```
pip install maturin
cd Scripts/phantom_sim/rust_voxelize && maturin develop --release   # or: pip install .
```

`voxelize_config(..., backend="auto")` uses it when installed and falls back to Python otherwise;
`backend="python"` / `"rust"` force one.

## Measured (2026-10-07, AMD Ryzen 5 5600X, 6 cores / 12 threads, `bench.py`)

| Case | Python | Rust, 12 threads | Rust, 1 thread |
|---|---|---|---|
| make_phantom default: 64 slices, 83 fibres, 300x300 | 0.645 s | 2.8 ms (231x) | 10.0 ms (65x) |
| large bundle: 256 slices, 750 fibres, 600x600 | 23.8 s | 199 ms (119x) | 497 ms (47x) |

Best of 3 (Python) and 7 (Rust) runs, supersample 4, random fibre centres at the packed density.
The large case's thread scaling is capped by writing the 370 MB output volume.
