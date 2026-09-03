"""ASTRA geometry for plenoptic grid and rotation CT, lifted from the notebook source."""
import numpy as np


def build_ct_vectors(angles, vol_z0_pix, detector_z, pixel_size_x, pixel_size_y):
    angles = np.asarray(angles, dtype=np.float64)
    n = len(angles)
    v = np.empty((n, 12))
    v[:, 0] = np.sin(angles) * vol_z0_pix
    v[:, 1] = -np.cos(angles) * vol_z0_pix
    v[:, 2] = 0.0
    v[:, 3] = -np.sin(angles) * detector_z
    v[:, 4] = np.cos(angles) * detector_z
    v[:, 5] = 0.0
    v[:, 6] = np.cos(angles) * pixel_size_x
    v[:, 7] = np.sin(angles) * pixel_size_x
    v[:, 8] = 0.0
    v[:, 9] = 0.0
    v[:, 10] = 0.0
    v[:, 11] = pixel_size_y
    return v


def plenoptic_grid(source_range, det_range, n=21):
    sources_v = np.linspace(-source_range, source_range, n)
    sources_u = np.linspace(source_range, -source_range, n)
    det_v = np.linspace(det_range, -det_range, n)
    det_u = np.linspace(-det_range, det_range, n)
    src = np.meshgrid(sources_v, sources_u, indexing="ij")
    src = np.flip(np.stack([c.flatten() for c in src], axis=0), 0)
    det = np.meshgrid(det_v, det_u, indexing="ij")
    det = np.flip(np.stack([c.flatten() for c in det], axis=0), 0)
    return src, det


def build_plenoptic_geometry(src_vu_pix, det_vu_pix, vol_z0_pix, detector_z,
                             pixel_size_x, pixel_size_y):
    num = src_vu_pix.shape[-1]
    src_xyz = np.empty((num, 3))
    src_xyz[:, 0] = src_vu_pix[0]
    src_xyz[:, 1] = src_vu_pix[1]
    src_xyz[:, 2] = -vol_z0_pix
    det_xyz = np.zeros_like(src_xyz)
    det_xyz[:, 0] = det_vu_pix[0]
    det_xyz[:, 1] = det_vu_pix[1]
    det_xyz[:, 2] = detector_z - vol_z0_pix
    dir_x = np.zeros_like(src_xyz); dir_x[:, 0] = pixel_size_x
    dir_y = np.zeros_like(src_xyz); dir_y[:, 1] = pixel_size_y
    return np.hstack([src_xyz, det_xyz, dir_x, dir_y])


def _vol_geom_args(vol_shape):
    """Map an array shape (nz, ny, nx) to astra.create_vol_geom's argument order.

    In this module array dim 0 is the special axis (CT rotation axis / plenoptic beam
    axis), which must land on ASTRA's Z (slices) so that astra.geom_size(vol_geom)
    equals the array shape and the volume tensor links without transposition. ASTRA's
    create_vol_geom takes (GridRowCount=Y, GridColCount=X, GridSliceCount=Z) and its
    data order is (Z, Y, X), so (nz, ny, nx) -> (ny, nx, nz).
    """
    nz, ny, nx = vol_shape
    return ny, nx, nz


def _vol_geom_args_windowed(vol_shape, vol_z_offset_pix=0):
    """Windowed create_vol_geom args for a (nz, ny, nx) array, voxel size 1 unit.

    Returns (Y, X, Z, minX, maxX, minY, maxY, minZ, maxZ); the vol_z_offset_pix shift
    is applied on the beam axis only (ASTRA Z = array dim 0 = nz).
    """
    nz, ny, nx = vol_shape
    return (ny, nx, nz,
            -nx / 2.0, nx / 2.0,
            -ny / 2.0, ny / 2.0,
            -nz / 2.0 + vol_z_offset_pix, nz / 2.0 + vol_z_offset_pix)


def make_plenoptic_projector(vol_recon_size, geometry, sinogram_size,
                             super_sampling=2, vol_z_offset_pix=0):
    import astra
    vol_geom = astra.create_vol_geom(
        *_vol_geom_args_windowed(vol_recon_size, vol_z_offset_pix))
    proj_geom = astra.create_proj_geom("cone_vec", sinogram_size[0], sinogram_size[1], geometry)
    opts = {"VoxelSuperSampling": super_sampling, "DetectorSuperSampling": super_sampling}
    return astra.create_projector("cuda3d", proj_geom, vol_geom, opts), vol_geom


def make_ct_projector(vol_recon_size, vectors, sinogram_size, super_sampling=2):
    import astra
    vol_geom = astra.create_vol_geom(*_vol_geom_args(vol_recon_size))
    proj_geom = astra.create_proj_geom("cone_vec", sinogram_size[0], sinogram_size[1], vectors)
    opts = {"VoxelSuperSampling": super_sampling, "DetectorSuperSampling": super_sampling}
    return astra.create_projector("cuda3d", proj_geom, vol_geom, opts), vol_geom


def delete_projector(projector_id):
    """Free an ASTRA cuda3d projector so the GPU can reclaim its workspace.

    Experiment 2 runs 8 solves and builds a fresh projector per stage; without freeing the
    old ones (and the concatenated projection tensors) the run accumulates dead allocations
    and OOMs on a 32 GB V100. Call between stages, paired with torch.cuda.empty_cache().
    """
    import astra
    astra.projector3d.delete(projector_id)


class XrayOperator:
    def __init__(self, projector_id):
        import astra
        self.projector_id = projector_id
        self.vol_geom = astra.projector3d.volume_geometry(projector_id)
        self.vol_shape = astra.geom_size(self.vol_geom)
        self.proj_geom = astra.projector3d.projection_geometry(projector_id)
        self.proj_shape = astra.geom_size(self.proj_geom)

    def __call__(self, vol_data):
        import torch, astra.experimental
        proj = torch.zeros(self.proj_shape, dtype=torch.float32, device="cuda")
        astra.experimental.direct_FP3D(self.projector_id, vol=vol_data.detach(), proj=proj)
        return proj

    def T(self, proj_data):
        import torch, astra.experimental
        vol = torch.zeros(self.vol_shape, dtype=torch.float32, device="cuda")
        astra.experimental.direct_BP3D(self.projector_id, vol=vol, proj=proj_data.detach())
        return vol


def make_projection_fn():
    import torch

    class Projection(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A, vol):
            ctx.A = A
            return A(vol)

        @staticmethod
        def backward(ctx, grad_output):
            return None, ctx.A.T(grad_output)

    return Projection
