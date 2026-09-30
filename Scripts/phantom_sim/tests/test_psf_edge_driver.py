"""Torch-free tests for the point / step-edge resolution driver.

main() needs the cluster; here we pin the point and rod phantoms, the edge-spread analysis
(10-90% rise, line-spread FWHM) and the metrics schema.
"""

import math

import numpy as np

from phantom_sim.run_plenoptic_psf_edge import (
    make_point_phantom,
    make_rod_phantom,
    edge_metrics,
    rod_core_mask,
)


def test_point_phantom_is_one_centre_voxel():
    v = make_point_phantom(9, 11, 13, mu=2.0)
    assert v.shape == (9, 11, 13)
    assert v.sum() == 2.0
    assert v[4, 5, 6] == 2.0


def test_rod_phantom_spans_z_range_and_radius():
    v = make_rod_phantom(40, 21, 21, radius_vox=3, z_start=20, z_stop=35, mu=1.0)
    assert v[19, 10, 10] == 0.0  # just before the end face
    assert v[20, 10, 10] == 1.0  # first rod slice
    assert v[34, 10, 10] == 1.0
    assert v[35, 10, 10] == 0.0
    assert v[25, 10, 13] == 1.0  # on the radius
    assert v[25, 10, 14] == 0.0  # outside the radius
    # cross-section area close to pi r^2
    assert abs(v[25].sum() - math.pi * 9) < 6


def test_rod_core_mask_is_inside_rod():
    core = rod_core_mask(21, 21, radius_vox=3, shrink_vox=1)
    rod = make_rod_phantom(4, 21, 21, radius_vox=3, z_start=0, z_stop=4)[0] > 0
    assert core.sum() > 0
    assert np.all(rod[core])


def test_edge_metrics_on_ideal_step():
    esf = np.zeros(100)
    esf[50:] = 1.0
    m = edge_metrics(esf, edge_index=50, spacing=2.0)
    # a sharp step rises over one sample: 10-90 = 0.8 sample, LSF FWHM = 1 sample
    assert abs(m["rise_10_90_um"] - 0.8 * 2.0) < 1e-6
    assert abs(m["lsf_fwhm_um"] - 1.0 * 2.0) < 1e-6
    assert abs(m["plateau"] - 1.0) < 1e-9 and abs(m["base"]) < 1e-9


def test_edge_metrics_on_gaussian_blurred_step():
    z = np.arange(400, dtype=float)
    sigma = 8.0
    esf = 3.0 * 0.5 * (1.0 + np.vectorize(math.erf)((z - 200.0) / (sigma * math.sqrt(2.0))))
    m = edge_metrics(esf, edge_index=200, spacing=1.0)
    # erf edge: 10-90 rise = 2.563 sigma; LSF is the Gaussian, FWHM = 2.3548 sigma
    assert abs(m["rise_10_90_um"] - 2.563 * sigma) < 0.3
    assert abs(m["lsf_fwhm_um"] - 2.3548 * sigma) < 0.3
    assert abs(m["plateau"] - 3.0) < 1e-3
    assert abs(m["mtf"][0] - 1.0) < 1e-9
    assert m["freqs_inv_um"][0] == 0.0


def test_edge_metrics_finds_edge_already_past_half_at_nominal_index():
    # blurred rod whose 50% point sits BEFORE the nominal edge, with a far end after it:
    # the 50% search must not skip ahead to the falling far end
    z = np.arange(256, dtype=float)
    erf = np.vectorize(math.erf)
    rise = 0.5 * (1.0 + erf((z - 120.0) / (6.0 * math.sqrt(2.0))))
    fall = 0.5 * (1.0 - erf((z - 224.0) / (6.0 * math.sqrt(2.0))))
    m = edge_metrics(rise * fall, edge_index=128, spacing=1.0)
    assert abs(m["edge_50_index"] - 120.0) < 0.5
    assert m["rise_10_90_um"] is not None
    assert abs(m["rise_10_90_um"] - 2.563 * 6.0) < 0.3
