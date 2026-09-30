import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phantom_sim.diam_methods import common, m3_local_ellipse  # noqa: E402


def test_m3_fast_synthetic(tmp_path):
    vol, truth = common.synthetic_bundle(
        16.0, shape_yx=(128, 128), nz=64, blur_sigma_um=2.5, fvf=0.4
    )
    res = m3_local_ellipse.run(
        vol, 2.0, str(tmp_path), "synth16", truth=truth, fast=True, n_workers=1
    )
    for name in [
        "M3_overlay.png",
        "M3_distribution.png",
        "M3_examples.png",
        "M3_truth_scatter.png",
        "M3_fibres.csv",
        "M3_summary.json",
    ]:
        assert os.path.exists(tmp_path / name), name
    s = res["by_frac"]["0.5"]
    assert s["n_accept"] > 0
    assert abs(s["minor"]["median"] - 16.0) < 2.5
    assert res["truth_mean_diam_um"] == 16.0
    with open(tmp_path / "M3_summary.json", encoding="utf-8") as f:
        assert json.load(f)["method"] == "M3"
