"""Dense diagnostic variable-projection GN. Not a scalable optimizer proposal.

Head fits and projectors use the same retained SVD. Feature derivatives and
residuals keep sample/output axes explicit until the final flattening. The full
Jacobian differentiates the retained-subspace projection, including nonzero
residual and discarded nonzero singular modes; Kaufman mode isolates the
projector/indexing fixes from that extra correction. Rank must be locally stable
for a classical derivative to exist. No range recalibration occurs mid-solve.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.func import functional_call, jacfwd


def nonlinear_named_parameters(model):
    """All trainable leaves except the solved readout, including shallow bV."""
    # Some older models expose a reshaped view of a head leaf rather than the leaf.
    head_storage = {p.untyped_storage().data_ptr() for p in model.readout()}
    return [(n, p) for n, p in model.named_parameters()
            if p.requires_grad and p.untyped_storage().data_ptr() not in head_storage]


@dataclass
class HeadFit:
    U: torch.Tensor
    s: torch.Tensor
    Vh: torch.Tensor
    keep: torch.Tensor
    weights: torch.Tensor
    prediction: torch.Tensor
    residual: torch.Tensor
    rcond: float


def fit_head(A, Y, rcond=1e-14):
    """The actual head and its retained subspace come from one factorization."""
    rc = max(rcond, 100 * torch.finfo(A.dtype).eps)
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    keep = s > rc * s[0]
    Q, sr, Vr = U[:, keep], s[keep], Vh[keep].T
    weights = (Vr / sr) @ (Q.T @ Y)
    pred = A @ weights
    return HeadFit(U, s, Vh, keep, weights, pred, pred - Y, rc)


def reduced_jacobian(fit, Y, dA, mode="full"):
    """dA[n, features+bias, parameters] -> J[n, outputs, parameters].

    Full mode differentiates P_r(A)Y, with singular-value cross-gap denominators.
    It remains valid for a constant retained rank and a nonzero discarded
    singular spectrum. Exactly repeated singular values straddling the cutoff
    make that derivative undefined and are refused.
    """
    U, s, Vh, keep = fit.U, fit.s, fit.Vh, fit.keep
    Q, sr, Vr = U[:, keep], s[keep], Vh[keep].T
    if mode == "kaufman":
        J0 = torch.einsum("nmp,mq->nqp", dA, fit.weights)
        return J0 - torch.einsum("nr,rs,sqp->nqp", Q, Q.T, J0)
    if mode != "full":
        raise ValueError(mode)
    Dr = torch.einsum("nmp,mr->nrp", dA, Vr)
    Yr = Q.T @ Y
    # Left-null complement, which may be absent from the thin SVD.
    D0 = Dr - torch.einsum("nk,ks,srp->nrp", U, U.T, Dr)
    Y0 = Y - U @ (U.T @ Y)
    J = torch.einsum("nrp,rq->nqp", D0 / sr[None, :, None], Yr)
    J = J + torch.einsum("nr,rqp->nqp", Q,
        torch.einsum("nrp,nq->rqp", Dr, Y0) / sr[:, None, None])
    # Thin-SVD discarded directions, including nonzero truncated modes.
    if bool((~keep).any()):
        Ud, sd, Vd = U[:, ~keep], s[~keep], Vh[~keep].T
        Dd = torch.einsum("nmp,md->ndp", dA, Vd)
        gap = sr[None, :] ** 2 - sd[:, None] ** 2
        if bool((gap <= 0).any()):
            raise ValueError("Retained and discarded singular subspaces have no spectral gap")
        T = (sr[None, :, None] * torch.einsum("nd,nrp->drp", Ud, Dr)
             + sd[:, None, None] * torch.einsum("nr,ndp->drp", Q, Dd)) / gap[:, :, None]
        J = J + torch.einsum("nd,drp,rq->nqp", Ud, T, Yr)
        J = J + torch.einsum("nr,drp,dq->nqp", Q, T, Ud.T @ Y)
    return J


class _Features(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, X):
        F = self.model.feats(X)
        return torch.cat([F, torch.ones(len(F), 1, dtype=F.dtype, device=F.device)], dim=1)


def feature_problem(model, X):
    """Flatten nonlinear leaves only; return a pure feature-map closure."""
    named = nonlinear_named_parameters(model)
    theta = torch.cat([p.detach().reshape(-1) for _, p in named])
    sizes = [p.numel() for _, p in named]
    wrapper = _Features(model)

    def split(th):
        return {"model." + n: v.view_as(p) for (n, p), v in zip(named, th.split(sizes))}

    def features(th):
        return functional_call(wrapper, split(th), (X,))

    @torch.no_grad()
    def assign(th):
        for (_, p), v in zip(named, th.split(sizes)):
            p.copy_(v.view_as(p))

    return theta, features, assign


@torch.no_grad()
def assign_head(model, fit):
    W, b = fit.weights[:-1], fit.weights[-1]
    if hasattr(model, "layers") and hasattr(model.layers[-1].mixer, "set_raw"):
        mx = model.layers[-1].mixer
        mx.set_raw(W.view(mx.M, mx.N, mx.K))
        mx.bias.copy_(b.view_as(mx.bias))
    else:
        wp, bp = model.readout()
        wp.copy_(W.view_as(wp)); bp.copy_(b.view_as(bp))


def gauss_newton(model, Xtr, Ytr, Xte, Yte, *, iters=15, mu=1e-2,
                 rcond=1e-14, jacobian="full", max_trials=12):
    """Monotone training objective; held-out data are logged, never selected on.

    Materializes dA and J and solves dense parameter systems. This utility is
    intentionally a small-problem diagnostic. No learned step affects a range
    tracker, data normalization, centers, or externally stored experiment files.
    """
    Ytr, Yte = Ytr.reshape(len(Ytr), -1), Yte.reshape(len(Yte), -1)
    theta, features, assign = feature_problem(model, Xtr)
    fit = fit_head(features(theta), Ytr, rcond)
    log = dict(step=[], train=[], test=[], rank=[], accepted=[], jacobian=jacobian)

    def record(step, accepted):
        assign(theta); assign_head(model, fit)
        with torch.no_grad():
            log["step"].append(step)
            log["train"].append(float(fit.residual.norm() / Ytr.norm()))
            log["test"].append(float((model(Xte).reshape_as(Yte) - Yte).norm() / Yte.norm()))
            log["rank"].append(int(fit.keep.sum()))
            log["accepted"].append(accepted)

    record(0, False)
    for step in range(1, iters + 1):
        dA = jacfwd(features)(theta).detach()
        J = reduced_jacobian(fit, Ytr, dA, jacobian).reshape(Ytr.numel(), -1)
        residual = fit.residual.reshape(-1)
        H, grad = J.T @ J, J.T @ residual
        diag = H.diagonal().clamp_min(1e-12)
        cost = float(residual.square().sum())
        accepted = False
        for _ in range(max_trials):
            delta = torch.linalg.solve(H + mu * torch.diag(diag), -grad)
            trial_theta = theta + delta
            trial_fit = fit_head(features(trial_theta), Ytr, rcond)
            trial_cost = float(trial_fit.residual.square().sum())
            if torch.isfinite(trial_fit.residual).all() and trial_cost < cost:
                theta, fit, mu, accepted = trial_theta.detach(), trial_fit, mu / 3, True
                break
            mu *= 10
        record(step, accepted)
        if not accepted:
            break
    log["final"], log["final_train"] = log["test"][-1], log["train"][-1]
    return log
