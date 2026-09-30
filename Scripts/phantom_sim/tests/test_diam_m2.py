"""Fast smoke test of the torch-free M2 blurred-disk diameter method on a synthetic bundle."""

import json
import os

from phantom_sim.diam_methods import common, m2_radial_fit


def test_m2_fast_recovers_diameter(tmp_path):
    vol, truth = common.synthetic_bundle(
        16.0, shape_yx=(128, 128), nz=64, blur_sigma_um=2.5, fvf=0.4
    )
    out = tmp_path / "m2"
    res = m2_radial_fit.run(
        vol, truth.voxel_um, str(out), "test", truth=truth, fast=True, n_workers=1
    )
    for name in (
        "M2_examples.png",
        "M2_overlay.png",
        "M2_distribution.png",
        "M2_truth_scatter.png",
        "M2_fibres.csv",
        "M2_summary.json",
    ):
        assert os.path.getsize(out / name) > 0, name
    assert res["n_detected"] > 0
    assert res["method"] == "M2"
    med = res["fit_shared_all"]["median"]
    assert med is not None and abs(med - 16.0) < 2.0, med
    with open(out / "M2_summary.json", encoding="utf-8") as f:
        assert json.load(f)["truth_mean_diam_um"] == 16.0
