from phantom_sim.reconstruct import bar_reached, first_crossing


def test_bar_reached_true_only_when_last_value_at_or_above_bar():
    assert bar_reached([0.5, 0.8, 0.92], 0.90) is True
    assert bar_reached([0.5, 0.8, 0.88], 0.90) is False
    assert bar_reached([], 0.90) is False


def test_first_crossing_reports_iteration_of_C0_bar():
    history = {
        "corr": [0.40, 0.70, 0.89, 0.91, 0.93],
        "iter": [0, 10, 20, 30, 40],
        "time": [0.0, 1.0, 2.0, 3.0, 4.0],
    }
    # C0 bar 0.90: first crossing is the 0.91 point at iter 30
    it, t = first_crossing(history, 0.90, key="corr")
    assert it == 30 and t == 3.0


def test_first_crossing_none_when_bar_never_reached():
    history = {"corr": [0.40, 0.70, 0.85], "iter": [0, 10, 20], "time": [0.0, 1.0, 2.0]}
    it, t = first_crossing(history, 0.90, key="corr")
    assert it is None and t is None
