import json
import pytest

pytest.importorskip("astra")
pytest.importorskip("torch")


@pytest.mark.cuda
def test_ct_orientation_smoke(tmp_path):
    from phantom_sim.run_ct_orientation import main
    main(["--smoke", "--out", str(tmp_path)])
    for name in ["ct_axis0.npy", "ct_axis1.npy", "ct_metrics.json"]:
        assert (tmp_path / name).exists()
    m = json.loads((tmp_path / "ct_metrics.json").read_text(encoding="utf-8"))
    assert "axis0" in m and "psnr" in m["axis0"]
    assert "axis1" in m and "psnr" in m["axis1"]
