"""Torch-free tests for the PSF-sweep driver's pure helpers.

The forward projection and reconstruction in main() need the cluster; here we pin the half-angle
<-> detector-span inverse (against the tilt driver's forward formula), the angle resolver, and the
metrics schema.
"""


import numpy as np
import pytest

from phantom_sim.run_plenoptic_tilt import max_plenoptic_half_angle_deg
from phantom_sim.run_plenoptic_psf import psf_entry
from phantom_sim.run_plenoptic_psf_sweep import (
    det_range_for_half_angle,
    source_range_for_half_angle,
    resolve_half_angles,
    sweep_entry,
    assemble_psf_sweep_metrics,
    default_half_angles,
)

SOURCE_RANGE = 22000.0
SOD_UM = 129.24 * 1000.0
SDD_UM = 1171.57 * 1000.0
CUR_DET = 177400.0
MAG = (SDD_UM - SOD_UM) / SOD_UM  # object plane -> detector plane, 8.0651


def test_det_range_inverts_the_half_angle_formula():
    # For each target angle, the det_range we compute must reproduce that angle through the
    # tilt driver's forward formula (round-trip).
    for theta in (5.0, 9.66, 15.0, 25.0, 45.0):
        det = det_range_for_half_angle(theta, SOURCE_RANGE, SDD_UM)
        back = max_plenoptic_half_angle_deg(SOURCE_RANGE, det, SDD_UM)
        assert abs(back - theta) < 1e-6


def test_current_array_det_range_is_the_9p66_control():
    # The current detector span (177.4 mm) must map to the ~9.66 deg anchor.
    theta = max_plenoptic_half_angle_deg(SOURCE_RANGE, CUR_DET, SDD_UM)
    assert abs(theta - 9.66) < 0.05
    det = det_range_for_half_angle(theta, SOURCE_RANGE, SDD_UM)
    assert abs(det - CUR_DET) < 1.0


def test_source_range_reproduces_the_current_array():
    # theta = atan(source_range / SOD): the 22 mm array at SOD 129.24 mm IS the 9.66 deg control.
    assert abs(source_range_for_half_angle(9.661, SOD_UM) - SOURCE_RANGE) < 50.0


def test_source_and_detector_spans_stay_magnification_matched():
    # REGRESSION (exp5b, 2026-09-21). A source at lateral s casts the sample's shadow at
    # -s * (SDD - SOD) / SOD, so the sub-aperture tiles only stay on target when the detector
    # half-span is that magnification of the source half-span. Opening the aperture at the
    # source keeps the pair consistent at every angle.
    for theta in (5.0, 7.0, 9.661, 12.0, 25.0, 45.0):
        src = source_range_for_half_angle(theta, SOD_UM)
        det = det_range_for_half_angle(theta, src, SDD_UM)
        assert abs(det - src * MAG) <= 1e-9 * det


def test_pinned_source_span_only_matches_at_the_control_angle():
    # Guards the bug this replaced: with the source span pinned at 22 mm, the detector span is
    # magnification-consistent ONLY at the 9.66 deg control. At 45 deg the tiles sit ~1 m away
    # from the shadows they should catch, so every sub-aperture but the central one goes blind
    # and the run silently collapses to a single projection.
    assert (
        abs(det_range_for_half_angle(9.661, SOURCE_RANGE, SDD_UM) - SOURCE_RANGE * MAG)
        < 100.0
    )
    off = abs(det_range_for_half_angle(45.0, SOURCE_RANGE, SDD_UM) - SOURCE_RANGE * MAG)
    assert off > 900000.0  # um


def test_det_range_rejects_too_small_an_angle():
    # Below atan(source_range / SDD) ~ 1.08 deg the detector span would be non-positive.
    with pytest.raises(ValueError):
        det_range_for_half_angle(0.5, SOURCE_RANGE, SDD_UM)


def test_resolve_half_angles_injects_and_sorts_current():
    angles = resolve_half_angles(
        [25.0, 5.0, 12.0], SOURCE_RANGE, CUR_DET, SDD_UM, include_current=True
    )
    assert angles == sorted(angles)
    assert any(abs(a - 9.66) < 0.05 for a in angles)  # control injected
    assert 5.0 in angles and 25.0 in angles


def test_resolve_half_angles_can_skip_current():
    angles = resolve_half_angles(
        [25.0, 5.0], SOURCE_RANGE, CUR_DET, SDD_UM, include_current=False
    )
    assert angles == [5.0, 25.0]
    assert not any(abs(a - 9.66) < 0.05 for a in angles)


def test_default_half_angles_are_the_requested_set():
    assert default_half_angles() == [
        5.0,
        7.0,
        12.0,
        15.0,
        17.0,
        21.0,
        25.0,
        30.0,
        35.0,
        40.0,
        45.0,
    ]


def test_sweep_entry_and_schema():
    axes = [
        psf_entry(
            "z",
            24.0,
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.1]),
            np.array([1.0, 0.5]),
        ),
        psf_entry(
            "y",
            5.0,
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.1]),
            np.array([1.0, 0.5]),
        ),
        psf_entry(
            "x",
            5.0,
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.1]),
            np.array([1.0, 0.5]),
        ),
    ]
    e = sweep_entry(
        9.66,
        CUR_DET,
        2.0 * CUR_DET / 1000.0,
        axes,
        transverse_um=5.0,
        ratio=4.8,
        corr=0.95,
        source_range_um=SOURCE_RANGE,
    )
    assert e["theta_deg"] == 9.66
    assert e["fwhm_z_um"] == 24.0
    assert e["fwhm_transverse_um"] == 5.0
    assert (
        abs(e["det_range_um"] - 177400.0) < 1e-6
    )  # half-span, the plenoptic_grid argument
    assert abs(e["det_span_mm"] - 354.8) < 1e-6  # full detector width the angle costs
    assert abs(e["source_range_um"] - 22000.0) < 1e-6
    assert abs(e["source_span_mm"] - 44.0) < 1e-6
    assert e["z_to_transverse_ratio"] == 4.8
    m = assemble_psf_sweep_metrics([e], {"voxel_um": 2.0, "sdd_mm": 1171.57})
    assert m["meta"]["voxel_um"] == 2.0
    assert m["sweep"][0]["theta_deg"] == 9.66
