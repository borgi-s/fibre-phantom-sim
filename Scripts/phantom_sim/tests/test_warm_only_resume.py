"""Pure (torch-free) tests for the --warm-only resume helper.

A completed cold arm leaves pleno_cold_joint_z{1..4}.meta.json, each carrying its
per-stage "metrics" entry. load_cold_entries reads them back so the warm arm can be
run on its own and merged into the final pleno_metrics.json, instead of recomputing
the identical cold solves. No astra/torch here, so this runs on any box.
"""
import json

import pytest

from phantom_sim.run_plenoptic_zpos import build_parser, load_cold_entries


def test_parser_has_target_corr_default():
    args = build_parser().parse_args(["--out", "x"])
    assert args.target_corr == 0.90  # Experiment 2 speed threshold, corr-with-GT
    args2 = build_parser().parse_args(["--out", "x", "--target-corr", "0.85"])
    assert args2.target_corr == 0.85


def test_parser_early_stop_defaults_and_toggle():
    args = build_parser().parse_args(["--out", "x"])
    assert args.early_stop is True          # early-stop is the default protocol
    assert args.es_tol == 1e-3
    assert args.es_extra == 15
    off = build_parser().parse_args(["--out", "x", "--no-early-stop", "--es-tol", "0.002",
                                     "--es-extra", "20"])
    assert off.early_stop is False
    assert off.es_tol == 0.002
    assert off.es_extra == 20


def _fake_entry(n):
    return {
        "N": n,
        "corr_with_gt": 0.90 + 0.01 * n,
        "psnr": 14.0 + n,
        "ssim": 0.6,
        "resolution": {"along": 0.01, "across_mean": 0.1, "ratio": 0.1},
        "iters_to_threshold": None,
        "time_to_threshold": None,
        "history": {"iter": [0], "metric": [10.0]},
    }


def _write_cold_metas(out_dir, ns=(1, 2, 3, 4)):
    for n in ns:
        meta = {"strategy": "cold_joint", "N": n, "metrics": _fake_entry(n)}
        (out_dir / f"pleno_cold_joint_z{n}.meta.json").write_text(
            json.dumps(meta), encoding="utf-8")


def test_load_cold_entries_returns_four_in_order(tmp_path):
    _write_cold_metas(tmp_path)
    entries = load_cold_entries(tmp_path)
    assert [e["N"] for e in entries] == [1, 2, 3, 4]
    assert entries[3]["corr_with_gt"] == pytest.approx(0.94)
    assert "history" in entries[0]
    assert entries[0]["resolution"]["ratio"] == pytest.approx(0.1)


def test_load_cold_entries_missing_raises(tmp_path):
    _write_cold_metas(tmp_path, ns=(1, 2, 3))  # z4 never landed
    with pytest.raises(FileNotFoundError):
        load_cold_entries(tmp_path)
