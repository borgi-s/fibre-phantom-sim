"""Fibre-diameter measurement methods M1-M4 (ported from the 2026-09-28 real-data study).

Each method module exposes ``run(vol, voxel_um, out_dir, label, truth=None, fast=False) -> dict``
and writes its figure set into ``out_dir``. Torch-free: numpy, scipy, scikit-image, matplotlib.
"""
