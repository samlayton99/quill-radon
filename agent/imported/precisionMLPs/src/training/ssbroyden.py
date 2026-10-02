"""Self-scaling Broyden (SSB-II) quasi-Newton with a strong-Wolfe line search.

PyTorch fp64 port of the JAX/optax implementation in
`jaxpi/jaxpi/ssbroyden_optax.py` (scale_by_ssbroyden_wolfe_oracle). The update
keeps a dense inverse-Hessian H and needs the step size alpha *inside* the
update, so the line search is integrated rather than being a separate transform.

Interface is deliberately minimal and flat-vector based:

    x_star, info = ssbroyden_minimize(x0, fg, max_steps=..., ...)

where `fg(x) -> (f: float, g: Tensor)` is an oracle returning the objective and
its gradient at a flat float64 parameter vector.

Dense H is O(n^2); this is intended for the reduced (VarPro) problems where n is
a few thousand at most.

Defaults follow the tuned values in jaxpi/SSBROYDEN_FIX_SUMMARY.md
(lr=0.5, c1=1e-3, c2=0.3, max_ls=80).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

EPS_CURV = 1e-32      # |y.s| below this -> skip the H update
EPS_RHO = 1e-16       # |rho_k^-| below this -> skip the H update


@dataclass
class SSBroydenInfo:
    steps: int = 0
    f: float = math.inf
    grad_norm: float = math.inf
    converged: bool = False          # True only when the gradient norm reaches tol
    stalled: bool = False            # line search could not make progress
    restarts: int = 0
    history: list = field(default_factory=list)


def _strong_wolfe(x, fk, gk, pk, gk_dot_pk, fg, lr, c1, c2, max_ls,
                  alpha_min=1e-32, alpha_max=1e6, shrink=0.5, grow=1.1):
    """Line search mirroring the JAX `_wolfe_oracle` loop: Armijo + curvature,
    with a monotonicity guard, shrink/grow steps, and best-so-far fallback."""
    alpha = float(lr)
    prev_f, have_prev = math.inf, False
    best_a, best_f, best_g = 0.0, fk, gk
    accepted = False
    alpha_last, f_last, g_last = alpha, fk, gk

    for _ in range(int(max_ls)):
        alpha = min(max(alpha, alpha_min), alpha_max)
        f_new, g_new = fg(x + alpha * pk)
        if not math.isfinite(f_new):
            f_new = math.inf
        gpk = float(torch.dot(g_new, pk))

        armijo_basic = f_new <= fk + c1 * alpha * gk_dot_pk
        prev_violate = have_prev and (f_new >= prev_f)
        armijo_ok = armijo_basic and not prev_violate
        curvature_ok = abs(gpk) <= c2 * abs(gk_dot_pk)
        accept = armijo_ok and curvature_ok

        if math.isfinite(f_new) and f_new < best_f:
            best_a, best_f, best_g = alpha, f_new, g_new
        alpha_last, f_last, g_last = alpha, f_new, g_new

        if accept:
            accepted = True
            break

        shrink_now = (not armijo_ok) or prev_violate
        if shrink_now:
            prev_f, have_prev = f_new, True
        alpha = alpha * shrink if (shrink_now or gpk >= 0) else alpha * grow

    if accepted:
        return alpha_last, f_last, g_last
    return best_a, best_f, best_g


def ssbroyden_minimize(x0, fg, max_steps=200, lr=0.5, c1=1e-3, c2=0.3, max_ls=80,
                       init_scale=True, tol=1e-23, track_history=False,
                       max_restarts=3, step_callback=None):
    """Minimise `fg` from `x0` with SSB-II + strong Wolfe.

    tol: convergence tolerance on the gradient norm and on the step size.
    step_callback: optional fn(step, x, f, grad_norm) invoked after each
        accepted step (used by expD16 to log eval trajectories).
    Returns (x, SSBroydenInfo).
    """
    x = x0.detach().clone().to(torch.float64)
    n = x.numel()
    H = torch.eye(n, dtype=torch.float64, device=x.device)
    last_sqrt_arg = 0.5
    did_scale = False

    f, g = fg(x)
    info = SSBroydenInfo(f=f, grad_norm=float(torch.linalg.norm(g)))
    restarts = 0
    if track_history:
        info.history.append(info.grad_norm)

    for step in range(int(max_steps)):
        if init_scale and not did_scale:
            H = H / max(float(torch.linalg.norm(g)), 1e-12)
            did_scale = True

        pk = -(H @ g)
        if float(torch.dot(pk, g)) >= 0:      # not a descent direction -> steepest descent
            pk = -g
        gk_dot_pk = float(torch.dot(g, pk))
        if not math.isfinite(gk_dot_pk) or gk_dot_pk == 0.0:
            break

        alpha, f_new, g_new = _strong_wolfe(x, f, g, pk, gk_dot_pk, fg,
                                            lr, c1, c2, max_ls)
        if alpha == 0.0 or not math.isfinite(f_new):
            # The line search made no progress. This is NOT convergence -- do the
            # standard quasi-Newton restart (reset H to a scaled identity) and
            # retry; only give up once restarts are exhausted.
            if restarts < max_restarts:
                H = torch.eye(n, dtype=torch.float64, device=x.device) / max(
                    float(torch.linalg.norm(g)), 1e-12)
                restarts += 1
                info.restarts = restarts
                continue
            info.stalled = True
            break
        restarts = 0
        x_new = x + alpha * pk
        s = x_new - x
        y = g_new - g
        rhok_inv = float(torch.dot(y, s))

        if abs(rhok_inv) >= EPS_CURV:
            rhok = 1.0 / rhok_inv
            Hkyk = H @ y
            ykHkyk = float(torch.dot(y, Hkyk))
            if abs(ykHkyk) > 0.0:
                h_k = ykHkyk * rhok
                b_k = -alpha * rhok * float(torch.dot(s, g))
                a_k = b_k * h_k - 1.0
                denom = 1.0 + a_k
                sqrt_arg = abs(a_k) / denom if denom != 0.0 else math.nan
                valid = math.isfinite(sqrt_arg) and sqrt_arg >= 0
                use_arg = sqrt_arg if valid else last_sqrt_arg
                if valid:
                    last_sqrt_arg = use_arg
                rho_k_minus = min(1.0, h_k * (1.0 - math.sqrt(abs(use_arg))))

                if abs(rho_k_minus) >= EPS_RHO and a_k != 0.0 and b_k != 0.0:
                    theta_minus = (rho_k_minus - 1.0) / a_k
                    theta_plus = 1.0 / rho_k_minus
                    theta_k = max(theta_minus, min(theta_plus, (1.0 - b_k) / b_k))
                    rho_cap = min(1.0, 1.0 / b_k)
                    sigma_k = 1.0 + theta_k * a_k
                    power = 1.0 / (1.0 - n)
                    sigma_pow = abs(sigma_k) ** power if sigma_k != 0.0 else 0.0
                    if theta_k <= 0.0:
                        tau_k = min(rho_cap * sigma_pow, sigma_k)
                    else:
                        tau_k = rho_cap * min(sigma_pow, 1.0 / theta_k)
                    phi_den = 1.0 + a_k * theta_k
                    if math.isfinite(tau_k) and tau_k != 0.0 and phi_den != 0.0:
                        v_k = rhok * s - Hkyk / ykHkyk
                        phi_k = (1.0 - theta_k) / phi_den
                        H_cand = (H - torch.outer(Hkyk, Hkyk) / ykHkyk
                                  + phi_k * ykHkyk * torch.outer(v_k, v_k)) / tau_k \
                                 + rhok * torch.outer(s, s)
                        if torch.isfinite(H_cand).all():
                            H = H_cand

        step_norm = float(torch.linalg.norm(s))
        x, f, g = x_new, f_new, g_new
        gnorm = float(torch.linalg.norm(g))
        info.steps = step + 1
        info.f, info.grad_norm = f, gnorm
        if track_history:
            info.history.append(gnorm)
        if step_callback is not None:
            step_callback(step + 1, x, f, gnorm)

        if not math.isfinite(f):
            break
        if gnorm <= tol:
            info.converged = True
            break
        if step_norm <= tol:      # stagnated, but the gradient is not at tol
            info.stalled = True
            break

    return x, info
