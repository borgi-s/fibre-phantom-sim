import numpy as np
import pytest
from phantom_sim.phantom import voxelize_config, bundle_orientations
from phantom_sim.tests.fixtures.make_synth_config import straight_bundle, tilted_single


def test_supersampling_gives_soft_circular_edges():
    # A small fibre (r = 3 vox) rasterised hard is a blocky square/plus; anti-aliased
    # voxelisation must give edge voxels PARTIAL coverage (mu strictly between matrix and
    # fibre) so the disc reads as a circle, and its area estimate is at least as accurate.
    cfg, radii = straight_bundle(Z=1, n_side=1, radius_um=6.0)  # 1 fibre, r=3 vox at voxel 2
    hard = voxelize_config(cfg, radii, (41, 41), 2.0, axis=0, mu_fibre=1.0, mu_matrix=0.0,
                           supersample=1)
    aa = voxelize_config(cfg, radii, (41, 41), 2.0, axis=0, mu_fibre=1.0, mu_matrix=0.0,
                         supersample=4)
    # hard mask is binary; anti-aliased has intermediate coverage voxels
    assert set(np.unique(np.round(hard, 6)).tolist()).issubset({0.0, 1.0})
    partial = aa[(aa > 1e-4) & (aa < 1.0 - 1e-4)]
    assert partial.size > 8                       # a ring of soft edge voxels
    # coverage-summed area is closer to the true disc area than the hard count
    true_area = np.pi * 3.0 ** 2
    assert abs(float(aa.sum()) - true_area) <= abs(float(hard.sum()) - true_area) + 1e-6
    # interior stays full fibre, exterior stays matrix
    assert aa.max() == np.float32(1.0)
    assert aa.min() == np.float32(0.0)


def test_single_fibre_disc_area():
    cfg, radii = straight_bundle(Z=6, n_side=1, radius_um=6.0)  # 1 fibre at origin
    voxel_um = 2.0
    vol = voxelize_config(cfg, radii, transverse_shape=(41, 41), voxel_um=voxel_um,
                          axis=0, mu_fibre=1.0, mu_matrix=0.0)
    assert vol.dtype == np.float32
    assert vol.shape == (6, 41, 41)
    r_vox = 6.0 / voxel_um
    expected = np.pi * r_vox ** 2
    per_slice = (vol[0] > 0.5).sum()
    assert abs(per_slice - expected) / expected < 0.15
    # extruded: every slice identical
    assert np.allclose(vol[0], vol[-1])


def test_axis_placement_moves_fibre_axis():
    cfg, radii = straight_bundle(Z=6, n_side=1, radius_um=6.0)
    vol0 = voxelize_config(cfg, radii, (41, 41), 2.0, axis=0)
    vol1 = voxelize_config(cfg, radii, (41, 41), 2.0, axis=1)
    assert vol0.shape == (6, 41, 41)
    assert vol1.shape == (41, 6, 41)


def test_straight_orientation_is_axis_unit_vector():
    cfg, radii = straight_bundle(Z=8, n_side=2)
    n = bundle_orientations(cfg, axial_um=2.0, axis=0)
    assert n.shape == (4, 3)
    assert np.allclose(np.abs(n), np.array([1.0, 0.0, 0.0]), atol=1e-6)


def test_tilted_orientation_has_correct_angle():
    cfg, radii = tilted_single(Z=16, axial_um=2.0, tilt_x_um_per_slice=2.0)
    n = bundle_orientations(cfg, axial_um=2.0, axis=0)
    # tilt of 2 um per 2 um slice = 45 deg between along-axis and x
    ang = np.degrees(np.arccos(abs(n[0, 0])))
    assert abs(ang - 45.0) < 1.0


def test_bundle_orientations_axis_permutation():
    cfg, radii = straight_bundle(Z=8, n_side=2)
    # Test axis=1: along-fibre component should be at index 1
    n1 = bundle_orientations(cfg, axial_um=2.0, axis=1)
    assert n1.shape == (4, 3)
    assert np.allclose(np.abs(n1[0]), np.array([0.0, 1.0, 0.0]), atol=1e-6)
    # Test axis=2: along-fibre component should be at index 2
    n2 = bundle_orientations(cfg, axial_um=2.0, axis=2)
    assert n2.shape == (4, 3)
    assert np.allclose(np.abs(n2[0]), np.array([0.0, 0.0, 1.0]), atol=1e-6)
