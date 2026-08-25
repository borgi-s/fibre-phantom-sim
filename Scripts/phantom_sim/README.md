# phantom_sim

A torch-free-where-possible fibre phantom forward-projection and reconstruction
study: simulated glass-fibre-in-resin phantoms, forward projection, and
reconstruction, built to compare against the plenoptic and CT recon pipelines
elsewhere in this repo.

## This-box vs cluster split

Work in this module runs in two places, and code should be written so it is
obvious at a glance which side a piece of functionality belongs to:

- **This box (Windows 11, no CUDA GPU)**: fibre-pack configuration generation,
  voxelisation, geometry setup, and any numpy-only test that does not need a
  GPU. Vendored `fibre_pack` code itself imports `torch`, but only in CPU
  mode; that is fine on this box since the `3d_recon` conda env has a CPU
  build of torch installed. Tests that exercise this side of the module are
  plain pytest tests (no marker) and are expected to pass here.
- **Cluster (CUDA GPU, torch + ASTRA cuda3d)**: the actual forward projector
  and iterative reconstruction, which require ASTRA's `cuda3d` projector and
  a CUDA-capable torch build. Neither is installed on this box (`astra` is
  absent here entirely). Tests that exercise this side are marked
  `@pytest.mark.cuda` and are only expected to run on the cluster.

## Interpreters

- **This box**: `C:\Users\borgi\anaconda3\envs\3d_recon\python.exe`
  (bare `python` on this machine is Python 2.7; always invoke by this
  absolute path).
- **Cluster**: a separate CUDA-enabled environment with torch and ASTRA's
  `cuda3d` projector, set up on the cluster side; not available on this box.

## Running tests

Register the `cuda` marker via `pytest.ini` (already in this directory).
From `Scripts/`, run the this-box test suite with:

```
& "C:\Users\borgi\anaconda3\envs\3d_recon\python.exe" -m pytest phantom_sim/tests -v -m "not cuda"
```

The `-m "not cuda"` convention excludes any test marked `cuda` (GPU-only,
cluster-side) so the this-box suite runs cleanly without ASTRA or a CUDA
torch build. On the cluster, drop the `-m "not cuda"` filter (or select
`-m cuda`) to run the GPU-dependent tests as well.

## Layout

- `third_party/fibre_pack/fibre_packer.py`: vendored from
  `vedranaa/fibre-pack` (see provenance comment at the top of the file for
  URL and vendor date). Provides `from_fvf`, `from_n`, and `FibrePacker` for
  generating fibre configurations (`configuration`, shape `(Z, 2, N)`, and
  `radii`, shape `(N,)`).
- `tests/fixtures/make_synth_config.py`: small, deterministic, torch-free
  synthetic fibre configurations (`straight_bundle`, `tilted_single`) used
  by this-box tests so they do not depend on the vendored (torch-based)
  packer at test time.
- `tests/conftest.py`: exposes the `synth_config` pytest fixture built on
  `straight_bundle`.
