from phantom_sim.reconstruct import (
    new_history, update_history, first_crossing, plateau_reached)


def test_plateau_reached_flat_above_floor():
    # last 4 points within 1e-3 and last >= floor 0.917 -> stop allowed
    assert plateau_reached([0.9181, 0.9182, 0.9184, 0.9185], tol=1e-3, patience=3, floor=0.917)


def test_plateau_reached_blocked_below_floor():
    # flat, but sitting at 0.905 (a warm restart transient) below floor 0.917 -> do NOT stop
    assert not plateau_reached([0.9048, 0.9049, 0.9050, 0.9051],
                               tol=1e-3, patience=3, floor=0.917)


def test_plateau_reached_still_rising():
    # spread 0.07 over the window >> tol -> not plateaued
    assert not plateau_reached([0.85, 0.88, 0.90, 0.92], tol=1e-3, patience=3)


def test_plateau_reached_needs_enough_history():
    assert not plateau_reached([0.95, 0.95], tol=1e-3, patience=3)


def test_plateau_reached_no_floor_flat():
    assert plateau_reached([0.9400, 0.9401, 0.9402, 0.9403], tol=1e-3, patience=3, floor=None)


def test_history_logs_corr_and_first_crossing_on_corr():
    h = new_history()
    assert "corr" in h  # corr-with-GT trace, for the Experiment 2 speed threshold
    for it, (t, ps, cr) in enumerate(zip([0.5, 1.0, 1.5, 2.0],
                                         [10.0, 12.0, 14.0, 15.0],
                                         [0.50, 0.70, 0.85, 0.92])):
        update_history(h, it, t, data_loss=1.0, tv_loss=0.1, metric=ps, corr=cr)
    assert h["corr"] == [0.50, 0.70, 0.85, 0.92]
    # thresholding on the corr trace, not PSNR
    it_cross, t_cross = first_crossing(h, target=0.80, key="corr")
    assert it_cross == 2
    assert t_cross == 1.5


def test_history_corr_optional_stays_empty():
    h = new_history()
    update_history(h, 0, 0.5, 1.0, 0.1, metric=12.0)  # no corr passed
    assert h["corr"] == []
    assert first_crossing(h, target=0.8, key="corr") == (None, None)


def test_history_records_and_finds_first_crossing():
    h = new_history()
    for it, (t, psnr) in enumerate(zip([0.5, 1.0, 1.5, 2.0],
                                       [10.0, 18.0, 22.0, 25.0])):
        update_history(h, it, t, data_loss=1.0 / (it + 1), tv_loss=0.1, metric=psnr)
    assert h["iter"] == [0, 1, 2, 3]
    it_cross, t_cross = first_crossing(h, target=20.0, key="metric")
    assert it_cross == 2
    assert t_cross == 1.5


def test_first_crossing_returns_none_when_never_reached():
    h = new_history()
    update_history(h, 0, 0.5, 1.0, 0.1, metric=5.0)
    assert first_crossing(h, target=50.0) == (None, None)
