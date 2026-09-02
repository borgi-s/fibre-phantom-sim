from phantom_sim.run_plenoptic_chunkseq import (
    ORDERINGS,
    cold_chunks,
    chunkseq_entry,
    assemble_chunkseq_metrics,
)


def test_orderings_are_adjacent_and_every_second():
    assert ORDERINGS["A"] == [0, 1, 2, 3]
    assert ORDERINGS["B"] == [0, 2, 4, 6]


def test_cold_chunks_is_union_minus_anchor():
    assert cold_chunks(ORDERINGS, anchor=0) == [1, 2, 3, 4, 6]


def test_chunkseq_entry_records_crossing_of_bar():
    history = {"corr": [0.40, 0.88, 0.91], "iter": [0, 10, 20], "time": [0.0, 1.0, 2.0]}
    e = chunkseq_entry(2, 0.912, 19.0, 0.94, {"ratio": 0.03}, history, bar=0.90)
    assert e["chunk"] == 2
    assert e["iters_to_threshold"] == 20 and e["time_to_threshold"] == 2.0
    assert e["corr_with_gt"] == 0.912


def test_assemble_chunkseq_metrics_schema():
    anchor = {"chunk": 0}
    cold = [{"chunk": 1}, {"chunk": 2}]
    runs = {
        "A": {
            "warm": [{"chunk": 0}],
            "pair_misorientation": [],
            "cumulative_iters": 0,
            "cumulative_time": 0.0,
        }
    }
    out = assemble_chunkseq_metrics(anchor, cold, runs, C0=0.96, meta={"seed": 0})
    assert out["meta"]["C0"] == 0.96 and out["meta"]["seed"] == 0
    assert out["anchor"]["chunk"] == 0
    assert out["cold"][1]["chunk"] == 2
    assert out["runs"]["A"]["cumulative_iters"] == 0
