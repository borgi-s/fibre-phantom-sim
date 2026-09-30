import json
import time

import numpy as np

from phantom_sim.diam_methods import common
from phantom_sim.diam_methods.m1_longitudinal import run


def test_m1_recovers_known_diameter_fast(tmp_path):
    vol, truth = common.synthetic_bundle(
        16.0, shape_yx=(128, 128), nz=64, blur_sigma_um=2.5, fvf=0.4
    )
    t0 = time.time()
    res = run(
        vol, 2.0, str(tmp_path), "synthetic D16", truth=truth, fast=True, n_workers=1
    )
    elapsed = time.time() - t0
    assert (tmp_path / "M1_summary.json").exists()
    assert (tmp_path / "M1_distribution.png").exists()
    assert (tmp_path / "M1_measurements.csv").exists()
    with open(tmp_path / "M1_summary.json", encoding="utf-8") as f:
        summ = json.load(f)
    assert summ["method"] == "M1"
    json.dumps(res)  # JSON-serialisable
    assert res["n_accepted"] > 0
    med = res["chord_fit"]["median"]
    assert med is not None and abs(med - 16.0) < 2.5, med
    assert res["sigma_source"] in ("esf", "free_fit")
    assert res["truth_mean_diam_um"] == np.float64(16.0)
    assert elapsed < 60.0, elapsed
