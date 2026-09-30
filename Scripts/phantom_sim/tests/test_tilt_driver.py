"""Torch-free tests for the tilt-sweep driver's pure helpers.

Only the config shear, the array half-angle, and the metrics schema are exercised here; the
forward projection and reconstruction in main() need the cluster and are not imported.
"""

import numpy as np

from phantom_sim.phantom import bundle_orientations
from phantom_sim.run_plenoptic_tilt import (
    tilt_config,
    max_plenoptic_half_angle_deg,
    tilt_entry,
    assemble_tilt_metrics,
)


def test_tilt_config_recovers_the_tilt_angle():
    # Base cross-section: row 0 = x, row 1 = y, three fibres.
    base = np.array([[0.0, 10.0, -5.0], [0.0, 3.0, 7.0]])
    voxel_um = 2.0
    for deg in (0.0, 5.0, 10.0, 20.0):
        cfg = tilt_config(base, depth=64, theta_rad=np.deg2rad(deg), voxel_um=voxel_um)
        assert cfg.shape == (64, 2, 3)
        ori = bundle_orientations(
            cfg, axial_um=voxel_um, axis=0
        )  # (N, 3) order (along, y, x)
        along = ori[:, 0]
        trans = np.sqrt(ori[:, 1] ** 2 + ori[:, 2] ** 2)
        recovered = np.rad2deg(np.arctan2(trans, along))
        assert np.allclose(recovered, deg, atol=1e-6)


def test_tilt_config_zero_angle_is_straight():
    base = np.array([[0.0, 10.0], [0.0, 3.0]])
    cfg = tilt_config(base, depth=16, theta_rad=0.0, voxel_um=2.0)
    for zi in range(16):
        assert np.allclose(cfg[zi], base)


def test_max_half_angle_matches_geometry():
    # 22 mm source range + 177.4 mm detector range over 1171.57 mm SDD -> ~9.66 deg.
    a = max_plenoptic_half_angle_deg(22000.0, 177400.0, 1171570.0)
    assert abs(a - 9.66) < 0.05


def test_assemble_tilt_metrics_schema():
    e = tilt_entry(
        8.0, 0.94, 17.0, 0.9, {"along": 0.01, "across_mean": 0.1, "ratio": 0.1}
    )
    m = assemble_tilt_metrics([e], {"max_half_angle_deg": 9.66})
    assert m["meta"]["max_half_angle_deg"] == 9.66
    assert m["sweep"][0]["theta_deg"] == 8.0
    assert m["sweep"][0]["corr_with_gt"] == 0.94
