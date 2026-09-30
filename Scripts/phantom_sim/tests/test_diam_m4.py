import json

from phantom_sim.diam_methods import common, m4_granulometry


def test_m4_fast_synthetic_16um(tmp_path):
    vol, truth = common.synthetic_bundle(
        16.0, shape_yx=(128, 128), nz=64, blur_sigma_um=2.5, fvf=0.4
    )
    res = m4_granulometry.run(
        vol, 2.0, tmp_path, "test D=16", truth=truth, fast=True, n_workers=1
    )
    for name in (
        "M4_openings.png",
        "M4_spectrum.png",
        "M4_autocorr.png",
        "M4_summary.json",
    ):
        assert (tmp_path / name).exists(), name
    with open(tmp_path / "M4_summary.json", encoding="utf-8") as f:
        js = json.load(f)
    assert js["method"] == "M4"
    assert res["truth_mean_diam_um"] == 16.0
    assert abs(res["grey_peak_interp_um"] - 16.0) < 4.0
