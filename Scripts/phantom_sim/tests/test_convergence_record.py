from phantom_sim.reconstruct import new_history, update_history, first_crossing


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
