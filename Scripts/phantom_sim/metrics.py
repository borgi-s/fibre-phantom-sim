"""Voxel-fidelity metrics: registration, PSNR, SSIM, directional resolution."""
import numpy as np


def register(recon, gt):
    a = np.asarray(recon, dtype=np.float64).ravel()
    b = np.asarray(gt, dtype=np.float64).ravel()
    M = np.vstack([a, np.ones_like(a)]).T
    gain, offset = np.linalg.lstsq(M, b, rcond=None)[0]
    return (gain * np.asarray(recon, dtype=np.float64) + offset).astype(np.float32)


def psnr(recon, gt, data_range=None):
    recon = register(recon, gt)
    gt = np.asarray(gt, dtype=np.float64)
    mse = np.mean((recon - gt) ** 2)
    if mse == 0:
        return 100.0
    if data_range is None:
        data_range = gt.max() - gt.min()
        if data_range == 0:
            data_range = 1.0
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def ssim(recon, gt):
    from skimage.metrics import structural_similarity
    recon = register(recon, gt)
    gt = np.asarray(gt, dtype=np.float64)
    dr = float(gt.max() - gt.min()) or 1.0
    return float(structural_similarity(gt, recon, data_range=dr))


def directional_sharpness(vol, axis):
    vol = np.asarray(vol, dtype=np.float64)
    d = np.diff(vol, axis=axis)
    return float(np.sqrt(np.mean(d ** 2)))


def resolution_along_across(vol, fibre_axis):
    axes = [0, 1, 2]
    along = directional_sharpness(vol, fibre_axis)
    across = [directional_sharpness(vol, a) for a in axes if a != fibre_axis]
    across_mean = float(np.mean(across))
    return {"along": along, "across_mean": across_mean,
            "ratio": float(along / across_mean) if across_mean else 0.0}


def corr_with_gt(recon, gt):
    """Pearson correlation of recon with gt (scale and offset invariant).

    A robust fidelity score for low-contrast fibre phantoms, where global PSNR/SSIM sit at
    the background-dominated floor: it measures whether the recon reproduces the spatial
    fibre pattern regardless of absolute scale. Valid only when gt is the EXACT ground
    truth from the same run (not a separately regenerated phantom with different packing).
    """
    a = np.asarray(recon, dtype=np.float64).ravel()
    b = np.asarray(gt, dtype=np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
