"""Iterative TV reconstruction with optional warm start and convergence logging."""
import time
import numpy as np


def new_history():
    return {"iter": [], "time": [], "data_loss": [], "tv_loss": [], "metric": []}


def update_history(history, it, t_elapsed, data_loss, tv_loss, metric=None):
    history["iter"].append(int(it))
    history["time"].append(float(t_elapsed))
    history["data_loss"].append(float(data_loss))
    history["tv_loss"].append(float(tv_loss))
    if metric is not None:
        history["metric"].append(float(metric))


def first_crossing(history, target, key="metric"):
    vals = history.get(key, [])
    for idx, m in enumerate(vals):
        if m >= target:
            return history["iter"][idx], history["time"][idx]
    return None, None


def _tv_isotropic_aniso(vol, weights):
    import torch
    wz, wy, wx = weights
    tv = 0.0
    if wz:
        tv = tv + wz * torch.mean(torch.abs(vol[1:, :, :] - vol[:-1, :, :]))
    if wy:
        tv = tv + wy * torch.mean(torch.abs(vol[:, 1:, :] - vol[:, :-1, :]))
    if wx:
        tv = tv + wx * torch.mean(torch.abs(vol[:, :, 1:] - vol[:, :, :-1]))
    return tv


def reconstruct(A, projections, mask=None, n_iter=200, lr=1e-2,
                tv_weights=(0.0, 0.0, 0.0), x_init=None, log_every=10,
                gt=None, target_psnr=None, lower_clamp=None):
    """Gradient-descent TV recon. Returns (recon numpy float32, history)."""
    import torch
    from phantom_sim.geometry import make_projection_fn
    from phantom_sim.metrics import psnr as psnr_fn
    Projection = make_projection_fn()

    if x_init is None:
        x = torch.zeros(A.vol_shape, dtype=torch.float32, device="cuda", requires_grad=True)
    else:
        x = torch.tensor(np.asarray(x_init, dtype=np.float32), device="cuda",
                         requires_grad=True)
    proj = projections if torch.is_tensor(projections) else torch.tensor(
        projections, dtype=torch.float32, device="cuda")
    m = mask if (mask is None or torch.is_tensor(mask)) else torch.tensor(
        mask, dtype=torch.float32, device="cuda")
    opt = torch.optim.Adam([x], lr=lr)
    gt_np = None if gt is None else np.asarray(gt, dtype=np.float32)

    history = new_history()
    t0 = time.time()
    for it in range(n_iter):
        opt.zero_grad()
        fp = Projection.apply(A, x)
        resid = fp - proj
        if m is not None:
            resid = resid * m
        data_loss = torch.mean(resid ** 2)
        tv_loss = _tv_isotropic_aniso(x, tv_weights)
        loss = data_loss + tv_loss
        loss.backward()
        opt.step()
        if lower_clamp is not None:
            with torch.no_grad():
                x.clamp_(min=lower_clamp)
        if it % log_every == 0 or it == n_iter - 1:
            metric = None
            if gt_np is not None:
                metric = psnr_fn(x.detach().cpu().numpy(), gt_np)
            update_history(history, it, time.time() - t0,
                           float(data_loss.detach()), float(tv_loss if isinstance(tv_loss, float)
                           else tv_loss.detach()), metric)
            if target_psnr is not None and metric is not None and metric >= target_psnr:
                # record the crossing but keep iterating to full budget for fairness
                pass
    return x.detach().cpu().numpy().astype(np.float32), history
