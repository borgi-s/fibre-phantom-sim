# fibre-phantom-sim

A controlled simulation study of glass-fibre-bundle phantoms for X-ray tomography. Known
ground-truth fibre configurations (from a fibre-packing model) are voxelised into attenuation
volumes, imaged through simulated cone-beam CT and plenoptic acquisition geometries
(Beer-Lambert attenuation with Poisson noise), and reconstructed with a total-variation
regularised iterative solver. Because the phantom is synthetic, every reconstruction is scored
against the exact truth. The study started from two questions: does fibre orientation relative to
the scan geometry change reconstruction quality, and does warm-starting each solve from a previous
reconstruction save compute. Later experiments characterise the plenoptic geometry itself: its
tolerance to fibre misalignment and how that depends on the array setup, its point-spread
function, its dose headroom on measured data, and how well fibre diameters can be recovered from
its reconstructions.

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
- **Experiment 4, uniform tilt** (`run_plenoptic_tilt.py`): a straight bundle sheared so that all
  fibres make the same angle with the optical axis, swept from 0 to 60 degrees and reconstructed
  from a single plenoptic capture. Measures how fidelity falls off once the tilt exceeds the
  largest ray angle the array samples.
- **Experiment 5, point-spread function** (`run_plenoptic_psf.py`, `run_plenoptic_psf_sweep.py`,
  `run_plenoptic_psf_edge.py`): a small sphere or a single voxel in an empty volume, reconstructed
  from one capture, gives the FWHM and MTF along z, y and x, with and without the penalty along z.
  The sweep repeats it for wider arrays; the edge driver adds a step edge along z.
- **Experiment 6, wider arrays** (`run_plenoptic_tilt.py` with scaled source and detector grids):
  the tilt sweep of experiment 4 repeated for arrays with larger maximum ray half-angles.
- **Experiment 7, local misalignment** (`run_plenoptic_local.py`, `run_plenoptic_waypoint.py`): a
  bundle whose fibres are straight in one half of the cross-section and wavy in the other, at
  several waviness levels, scored separately in each half. The waypoint model gives each fibre its
  own smooth random path while keeping the packing overlap-free.
- **Experiment 8, dose reduction on measured data** (`run_plenoptic_dose.py`, `analyze_dose.py`):
  Poisson-thins the photon-counting projections of a measured plenoptic scan to a fraction of the
  dose and compares each reconstruction with the full-dose one. Needs the raw scan (not included).
- **Experiment 9, fibre-diameter recovery** (`run_plenoptic_diamsweep.py`, `diam_analysis.py`,
  `diam_sweep_summary.py`): polydisperse bundles at a series of mean diameters, reconstructed with
  the plenoptic geometry, measured with four diameter estimators on both the reconstruction and
  the ground truth.

## Requirements

| Component | Used for |
| --- | --- |
| Python 3.9+ | all code |
| `numpy` | phantom voxelisation, geometry, metrics |
| `scipy` | diameter estimators, filtering, fits |
| `structure-tensor` | orientation read-out in the local-misalignment analysis |
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
- `analyze.py` - torch-free study figures (orientation bars, convergence sawtooths, cost tables,
  tilt and PSF curves, local-misalignment maps).
- `diam_methods/` - four torch-free fibre-diameter estimators: M1 widest point in fibre-parallel
  slices (blurred chord fit), M2 blurred-disk fit of the radial profile, M3 local-threshold
  segmentation with ellipse fit, M4 granulometry.
- `fibre3d.py` - interactive 3D Plotly tube illustrations of the ground-truth bundles.
- `make_phantom.py`, `repro_plenoxfiber.py` - phantom CLI and a faithful PlenoXFiber baseline.

Experiment drivers: `run_ct_orientation.py`, `run_plenoptic_zpos.py`,
`run_plenoptic_chunkseq.py`, `run_plenoptic_tilt.py`, `run_plenoptic_psf.py`,
`run_plenoptic_psf_sweep.py`, `run_plenoptic_psf_edge.py`, `run_plenoptic_local.py`,
`run_plenoptic_waypoint.py`, `run_plenoptic_dose.py`, `run_plenoptic_diamsweep.py`; analysis:
`analyze.py`, `analyze_dose.py`, `diam_analysis.py`, `diam_sweep_summary.py`.

Batch scripts: `lsf/`, one per experiment, for an LSF cluster. Set `WORKDIR`, the conda
environment and the email address (`<your-email>`) before submitting; `run_plenoptic_dose.py` and
the experiment 8 scripts also need `DATA_ROOT` pointed at the raw scan. Vendored packing:
`third_party/fibre_pack/`. Tests: `tests/` (run with `pytest`; the `cuda`-marked tests need a GPU
and ASTRA).

## Usage

```bash
# install the CPU-side dependencies
pip install numpy scipy torch scikit-image matplotlib plotly tqdm structure-tensor
# astra-toolbox (GPU) is needed only for the forward model and reconstruction

# run the non-GPU test suite
pytest -m "not cuda" Scripts/phantom_sim/tests

# an experiment driver (reconstruction steps require a CUDA GPU with ASTRA)
python -m phantom_sim.run_plenoptic_chunkseq --out results/exp3_chunkseq --misalignment "very low"

# build the study figures from a results directory (torch-free)
python -m phantom_sim.analyze --results results/exp3_chunkseq

# fibre-diameter estimators on one diameter-sweep rung, reconstruction and ground truth (torch-free)
python -m phantom_sim.diam_analysis --run results/exp9_diamsweep/D12

# every driver has a --smoke mode for a quick end-to-end check and lists its options with --help
python -m phantom_sim.run_plenoptic_tilt --help
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
