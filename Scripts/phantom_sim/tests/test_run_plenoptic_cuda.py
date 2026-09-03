import json
import pytest

pytest.importorskip("astra")
pytest.importorskip("torch")


@pytest.mark.cuda
def test_plenoptic_zpos_smoke(tmp_path):
    from phantom_sim.run_plenoptic_zpos import main
    main(["--smoke", "--out", str(tmp_path)])
    m = json.loads((tmp_path / "pleno_metrics.json").read_text(encoding="utf-8"))
    for strat in ["cold_joint", "warm_sequential"]:
        assert strat in m
        assert len(m[strat]["by_N"]) == 4
        assert "psnr" in m[strat]["by_N"][0]
        assert "iters_to_threshold" in m[strat]["by_N"][0]
        # the speed threshold is scored on the per-iteration corr-with-GT trace
        hist = m[strat]["by_N"][0]["history"]
        assert "corr" in hist and len(hist["corr"]) == len(hist["iter"])
        # both arms report total compute for the warm-vs-cold speed comparison
        assert "cumulative_iters" in m[strat] and "cumulative_time" in m[strat]
    assert m["meta"]["reconstruction"]["target_corr"] == 0.90
    assert m["meta"]["reconstruction"]["early_stop"] is True
