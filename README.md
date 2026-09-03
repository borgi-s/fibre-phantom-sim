# fibre-phantom-sim

A controlled simulation study of glass-fibre-bundle phantoms for X-ray tomography. Known
ground-truth fibre configurations (from a fibre-packing model) are voxelised into attenuation
volumes, imaged through simulated cone-beam CT and plenoptic acquisition geometries
(Beer-Lambert attenuation with Poisson noise), and reconstructed with a total-variation
regularised iterative solver. Because the phantom is synthetic, every reconstruction is scored
against the exact truth. The study asks two questions: does fibre orientation relative to the
scan geometry change reconstruction quality, and does warm-starting each solve from a previous
reconstruction save compute.

## Context

Part of ongoing research on plenoptic X-ray tomography. The reconstruction recipe follows the
[PlenoXFiber](https://github.com/lsbesley/PlenoXFiber) pipeline, and the fibre packing uses
[fibre-pack](https://github.com/vedranaa/fibre-pack) (vendored under
`Scripts/phantom_sim/third_party/fibre_pack`). Both are GPL-3.0.

## The experiments

- **Experiment 1, CT orientation** (`run_ct_orientation.py`): the same fibre bundle reconstructed
  with its fibres along the rotation axis versus transverse to it, from a full 360-degree scan.
  Tests whether orientation drives reconstruction quality.
- **Experiment 2, plenoptic warm start** (`run_plenoptic_zpos.py`): the same bundle imaged at
  several z-positions, reconstructed either from scratch each time or warm-started from the
  previous position. Measures the compute needed to reach a fidelity threshold.
- **Experiment 3, warm start versus misorientation** (`run_plenoptic_chunkseq.py`): one long
  bundle whose fibres wander along its length, cut into depth chunks reconstructed as a
  warm-started sequence. Measures how the warm-start saving changes as the reused chunk becomes
  more misoriented. Batch scripts in `lsf/` sweep the bundle misalignment (very low, moderate,
  high).

## Requirements

| Component | Used for |
| --- | --- |
| Python 3.9+ | all code |
| `numpy` | phantom voxelisation, geometry, metrics |
| `torch` | fibre packing and the iterative reconstruction |
| `astra-toolbox` | CUDA cone-beam forward and back projection (GPU only) |
| `scikit-image` | registration and voxel-fidelity metrics |
| `matplotlib` | study figures |
| `plotly` | interactive 3D fibre illustrations |
| `tqdm` | progress reporting |

Packing, voxelisation, metrics, and all figures run CPU-only. The forward projector and the
reconstruction loops require a CUDA GPU with ASTRA's `cuda3d` backend; those code paths and
their tests are marked `cuda`.

## Contents

Core library (`Scripts/phantom_sim/`):

- `phantom.py` - torch-free fibre voxeliser, bundle orientation, packing wrapper, and the
  wandering-bundle helpers (`pack_bundle`, `make_straight_fibre_phantom`, `voxelize_config`,
  `upsample_config`, `chunk_gt`, `pair_misorientation`).
- `geometry.py` - ASTRA cone-beam geometry vector builders and operator.
- `forward.py` - Beer-Lambert plus Poisson forward model with a numpy reference.
- `reconstruct.py` - total-variation reconstructor with warm start, plateau scheduler, and
  early stopping.
- `metrics.py` - correlation-with-truth, PSNR, SSIM, and directional resolution.
- `analyze.py` - torch-free study figures (orientation bars, convergence sawtooths, cost tables).
- `fibre3d.py` - interactive 3D Plotly tube illustrations of the ground-truth bundles.
- `make_phantom.py`, `repro_plenoxfiber.py` - phantom CLI and a faithful PlenoXFiber baseline.

Experiment drivers: `run_ct_orientation.py`, `run_plenoptic_zpos.py`,
`run_plenoptic_chunkseq.py`. Batch scripts: `lsf/`. Vendored packing: `third_party/fibre_pack/`.
Tests: `tests/` (run with `pytest`; the `cuda`-marked tests need a GPU and ASTRA).

## Usage

```bash
# install the CPU-side dependencies
pip install numpy torch scikit-image matplotlib plotly tqdm
# astra-toolbox (GPU) is needed only for the forward model and reconstruction

# run the non-GPU test suite
pytest -m "not cuda" Scripts/phantom_sim/tests

# an experiment driver (reconstruction steps require a CUDA GPU with ASTRA)
python -m phantom_sim.run_plenoptic_chunkseq --out results/exp3_chunkseq --misalignment "very low"

# build the study figures from a results directory (torch-free)
python -m phantom_sim.analyze --results results/exp3_chunkseq
```

Run from the `Scripts/` directory (or add it to `PYTHONPATH`) so the `phantom_sim` package
imports.

## Acknowledgements

- [fibre-pack](https://github.com/vedranaa/fibre-pack) by Vedrana Andersen Dahl, vendored under
  `Scripts/phantom_sim/third_party/fibre_pack` (GPL-3.0).
- [PlenoXFiber](https://github.com/lsbesley/PlenoXFiber), the reference plenoptic reconstruction
  pipeline this study builds on.

## License

GPL-3.0, consistent with the vendored fibre-pack and with PlenoXFiber. See `LICENSE`.
