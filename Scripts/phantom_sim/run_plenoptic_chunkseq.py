"""CLI: chunk-sequence plenoptic study. Warm-start compute saving vs fibre misorientation.

Packs one long wandering bundle (fibres wander along dim 0), cuts it into n_chunks equal
chunks, and reconstructs a selected sequence of chunks from a single plenoptic capture
each, warm-starting each chunk from the previous chunk's result. Two orderings:
  A (adjacent, small step):     chunks 0,1,2,3
  B (every-second, large step): chunks 0,2,4,6
Anchor chunk 0 runs the full n_iter and defines C0 = its corr-with-GT; every other solve
(cold from scratch, and warm chained) early-stops es_extra iters after corr reaches C0.
The metric iters_to_threshold is the first iteration corr crosses C0 - es_tol.

Module-level code is stdlib + numpy + import-safe helpers only, so this module imports on a
box without torch/astra. `main` defers every torch/astra/CUDA import to its body.

See docs/superpowers/specs/2026-09-02-plenoptic-warmstart-misorientation-design.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from phantom_sim.reconstruct import first_crossing

ORDERINGS = {"A": [0, 1, 2, 3], "B": [0, 2, 4, 6]}


def cold_chunks(orderings, anchor=0):
    """Sorted unique non-anchor chunks needed by the cold arm across all orderings."""
    s = set()
    for seq in orderings.values():
        s.update(seq)
    s.discard(anchor)
    return sorted(s)


def chunkseq_entry(chunk, corr_val, psnr_val, ssim_val, resolution, history, bar):
    """One (arm, chunk) metrics entry. iters/time_to_threshold are the first crossing of
    `bar` (= C0 - es_tol) in the corr trace; null if never crossed. Pure, no torch."""
    iters_thr, time_thr = first_crossing(history, bar, key="corr")
    return {
        "chunk": int(chunk),
        "corr_with_gt": float(corr_val),
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "resolution": resolution,
        "iters_to_threshold": iters_thr,
        "time_to_threshold": time_thr,
        "history": history,
    }


def assemble_chunkseq_metrics(anchor_entry, cold_entries, runs, C0, meta):
    """Pure schema assembly for chunkseq_metrics.json. `runs` maps ordering name ->
    {warm:[entry...], pair_misorientation:[float...], cumulative_iters:int,
    cumulative_time:float}. No torch/astra."""
    return {
        "meta": {**meta, "C0": float(C0)},
        "anchor": anchor_entry,
        "cold": cold_entries,
        "runs": runs,
    }


def main(argv=None):
    raise NotImplementedError("main is implemented in Task 5")


if __name__ == "__main__":
    main()
