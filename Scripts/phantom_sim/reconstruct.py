"""Iterative TV reconstruction with optional warm start and convergence logging."""
import time
import numpy as np


def new_history():
    return {"iter": [], "time": [], "data_loss": [], "tv_loss": [], "metric": [], "corr": []}


def update_history(history, it, t_elapsed, data_loss, tv_loss, metric=None, corr=None):
    history["iter"].append(int(it))
    history["time"].append(float(t_elapsed))
    history["data_loss"].append(float(data_loss))
    history["tv_loss"].append(float(tv_loss))
    if metric is not None:
        history["metric"].append(float(metric))
    if corr is not None:
        history["corr"].append(float(corr))


def first_crossing(history, target, key="metric"):
    vals = history.get(key, [])
    for idx, m in enumerate(vals):
        if m >= target:
            return history["iter"][idx], history["time"][idx]
    return None, None


def plateau_reached(corr_hist, tol, patience, floor=None):
    """True once the corr trace has (a) risen to >= floor, if a floor is given, AND
    (b) flattened, its spread over the last `patience`+1 logged points being below tol.

    The Experiment 2 early-stopping rule: a warm stage inherits the previous position's
    volume, so its corr can sit on a transient flat spot BELOW the inherited fidelity
    before the new position's data pulls it back up. The floor (= prev_final_corr - tol)
    forbids stopping until corr has recovered to essentially the previous level, so only a
    genuine plateau AT OR ABOVE it ends the stage. Cold stages pass floor=None and stop at
    their own plateau. Pure: reads the logged corr list, no torch/astra.
    """
    if not corr_hist:
        return False
    if floor is not None and corr_hist[-1] < floor:
        return False
    if len(corr_hist) < patience + 1:
        return False
    window = corr_hist[-(patience + 1):]
    return (max(window) - min(window)) < tol


def bar_reached(corr_hist, bar):
    """True once the latest logged corr has reached a fixed bar.

    The chunk-sequence study's early-stop: every non-anchor solve stops es_extra
    iterations after its corr first reaches C0 (the anchor chunk's converged corr), so
    the recorded cost is 'iterations to match the anchor fidelity'. Unlike
    plateau_reached, this is an absolute target, not a flatness test. Pure: reads the
    logged corr list, no torch/astra.
    """
    return bool(corr_hist) and corr_hist[-1] >= bar


def _tv_isotropic_aniso(vol, weights):
    """Squared-L2 gradient penalty, independent per-axis weights (wz, wy, wx).

    Matches the reference PlenoXFiber solver: mean(diff**2) per axis, so a strong
    weight on the fibre axis (dim 0) smooths along coherent fibres without the
    edge-preserving-but-noisy behaviour of an L1 total variation, and the transverse
    axes can be left unpenalised so fibre cross-sections stay sharp.
    """
    import torch
    wz, wy, wx = weights
    tv = 0.0
    if wz:
        tv = tv + wz * torch.mean((vol[1:, :, :] - vol[:-1, :, :]) ** 2)
    if wy:
        tv = tv + wy * torch.mean((vol[:, 1:, :] - vol[:, :-1, :]) ** 2)
    if wx:
        tv = tv + wx * torch.mean((vol[:, :, 1:] - vol[:, :, :-1]) ** 2)
    return tv


def reconstruct(A, projections, mask=None, n_iter=200, lr=1e-2,
                tv_weights=(0.0, 0.0, 0.0), x_init=None, log_every=10,
                gt=None, target_psnr=None, lower_clamp=None, upper_clamp=None,
                plateau_scheduler=False, metric_fn=None,
                early_stop=False, es_tol=1e-3, es_patience=3, es_extra_iters=15,
                es_floor=None, es_bar=None):
    """Gradient-descent TV recon. Returns (recon numpy float32, history).

    lower_clamp/upper_clamp apply a per-iteration box constraint on x (the reference
    solver uses [0, 10]); non-negativity (lower_clamp=0.0) is the main stabiliser
    against the missing-wedge null-space blowing up.

    plateau_scheduler=True attaches a ReduceLROnPlateau (mode='min', factor 0.5,
    patience 10, threshold 1e-6) stepped on the total loss each iteration, matching the
    reference vedrana CT solver: it lets the run start at a large lr (0.05) and back off
    once the data term stops improving, which is most of why the reference converges
    smoothly over its 300 iterations.

    metric_fn, if given, is a callable taking the current recon as a numpy array (the full
    volume, in the solver's own padded frame) and returning one float logged per log step
    into history["corr"]. Experiment 2 passes a corr-with-GT-on-the-object-crop closure, so
    the speed threshold (iterations/wall-clock to a fidelity bar) can be scored on
    corr-with-GT rather than the padded-frame PSNR (which the void margin pins low). The
    numpy view of x is materialised once per log step and shared with the PSNR metric, so
    the extra cost is one cheap correlation, not a second GPU->CPU copy.

    early_stop=True ends the solve once the corr trace plateaus (see plateau_reached: spread
    < es_tol over the last es_patience+1 logged points) AND, for a warm stage, has recovered
    to es_floor (= previous position's final corr - es_tol); then it runs es_extra_iters more
    iterations and stops. n_iter is the hard cap. Requires metric_fn (the corr trace). This is
    the "reach the fidelity bar in fewer iterations" mechanism the Experiment 2 warm chain
    exploits; both arms early-stop so the iteration counts compare fairly.

    es_bar (chunk-sequence study): when set, the stage stops es_extra_iters after corr first
    reaches this absolute bar (C0, the anchor chunk's converged corr), instead of the plateau
    rule. es_bar=None keeps the plateau behaviour unchanged.
    """
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
    sched = None
    if plateau_scheduler:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=10, threshold=1e-6)
    gt_np = None if gt is None else np.asarray(gt, dtype=np.float32)

    history = new_history()
    t0 = time.time()
    stop_iter = None  # set to it+es_extra_iters once the plateau is first detected
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
        if sched is not None:
            sched.step(loss.item())
        if lower_clamp is not None or upper_clamp is not None:
            with torch.no_grad():
                x.clamp_(min=lower_clamp, max=upper_clamp)
        if it % log_every == 0 or it == n_iter - 1:
            metric = None
            corr = None
            if gt_np is not None or metric_fn is not None:
                x_np = x.detach().cpu().numpy()  # one GPU->CPU copy, shared by both metrics
                if gt_np is not None:
                    metric = psnr_fn(x_np, gt_np)
                if metric_fn is not None:
                    corr = float(metric_fn(x_np))
            update_history(history, it, time.time() - t0,
                           float(data_loss.detach()), float(tv_loss if isinstance(tv_loss, float)
                           else tv_loss.detach()), metric, corr)
            if early_stop and stop_iter is None and history["corr"]:
                if es_bar is not None:
                    triggered = bar_reached(history["corr"], es_bar)
                else:
                    triggered = plateau_reached(history["corr"], es_tol, es_patience, es_floor)
                if triggered:
                    stop_iter = it + es_extra_iters
        if stop_iter is not None and it >= stop_iter:
            break
    return x.detach().cpu().numpy().astype(np.float32), history
