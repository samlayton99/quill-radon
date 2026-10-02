"""qi2: the corrected QI block family for expI02 (Sam's notes of 2026-09-05).

Fixed geometry at every depth: bank centers uniform on [-1, 1], gamma = lambda / h, never moved, no forward
normalization of any kind. Coefficients move where the data land on the mesh (utilization), never lambda.

    layer:   p = z V^T (layer 1 only, V free)  or  p = z ;   H = tanh(gamma (p - c)) in R^{B x M x N}
             z' = b + sum_{r,j} C_{rjk} H_{rj},   C_{r,:,k} = P theta_{r,:,k}      (coordinates P, a fixed N x N map)
             residual layers: z' = z + (...)
    P:       "raw" I | "val" D (function-value / bump coordinates, w_j = (a_j - a_{j-1})/2) |
             "white" capped whitening of the bank Gram on uniform occupancy of [-1, 1] | "val_white" D W
    init:    every profile is the same smooth random function whatever P (theta = P^{-1} w), never zero
    band:    calibrate() puts each channel at mean 0, sd s_star once; band_penalty() is a weak hinge on the values
    head:    the last mixer is solved (QR then SVD, rcond 1e-14) for regression; trained for cross-entropy

Solvers are imported from experiments/expI01_compositional_qi/qiblocks.py (not edited).
"""
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "expI01_compositional_qi"))
from qiblocks import tsvd_solve, lstsq_bias, rel_l2, gauss_newton as _qib_gauss_newton  # noqa: E402

LAM = 0.25
S_STAR = 0.4
BAND = dict(mu_max=0.25, s_min=0.15, s_max=0.6, rho=0.9)
COORDS = ("raw", "val", "white", "val_white")


# ----------------------------------------------------------------------------- bank and coordinates
def bank_centers(N):
    h = 2.0 / N
    return -1.0 + (torch.arange(N, dtype=torch.get_default_dtype()) + 0.5) * h, h


class Bank(nn.Module):
    """H = tanh(gamma (p - c)), p [B, M] -> [B, M, N]. Buffers only; gamma h = lambda, fixed forever."""
    def __init__(self, N):
        super().__init__()
        c, h = bank_centers(N)
        self.register_buffer("c", c)
        self.N, self.h, self.gamma = N, h, LAM / h

    def forward(self, p):
        return torch.tanh(self.gamma * (p.unsqueeze(-1) - self.c))

    def extra_repr(self):
        return f"N={self.N}, gamma={self.gamma:.3g}, lambda={LAM}"


_REF_CACHE = {}


def _ref_bank(N, n_ref=4001):
    """Reference bank on a uniform grid, always in float64 (the whitener and the smooth init need it exact)."""
    key = (N, n_ref)
    if key not in _REF_CACHE:
        x = torch.linspace(-1.0, 1.0, n_ref, dtype=torch.float64)
        c, h = bank_centers(N)
        _REF_CACHE[key] = (x, torch.tanh((LAM / h) * (x.view(-1, 1) - c.to(torch.float64))))
    return _REF_CACHE[key]


def _diff_matrix(N):
    D = 0.5 * torch.eye(N)
    D[torch.arange(1, N), torch.arange(0, N - 1)] = -0.5
    return D


OCCUPIED = 0.8   # the band the calibrated data occupy on the mesh: 1 / 1.25 (the collar convention)


def coord_transform(N, kind, cap=1e3, n_ref=4001, band=OCCUPIED):
    """P [N, N] with C = P theta. Target independent: the whitener uses the bank Gram on uniform occupancy of the
    band [-0.8, 0.8] (whitening on the whole mesh instead costs 20-100x, measured 2026-09-05), amplification capped
    so that (B P) has singular values min(1, cap s / s_max)."""
    return _coord_transform64(N, kind, cap, n_ref, band).to(torch.get_default_dtype())


def _coord_transform64(N, kind, cap, n_ref, band):
    if kind == "raw":
        return torch.eye(N, dtype=torch.float64)
    D = _diff_matrix(N).to(torch.float64)
    if kind == "val":
        return D
    x, B = _ref_bank(N, n_ref)
    B = B[x.abs() <= band]
    if kind == "val_white":
        B = B @ D
    G = B.T @ B / B.shape[0]
    ev, V = torch.linalg.eigh(G)
    s = ev.clamp_min(0).sqrt().flip(0)
    V = V.flip(1)
    amp = torch.minimum(1.0 / s.clamp_min(1e-300), (cap / s[0]) * torch.ones_like(s))
    W = V * amp
    if kind == "white":
        return W
    if kind == "val_white":
        return D @ W
    raise KeyError(kind)


def smooth_profile(N, gen, degree=3, band=0.8):
    """Raw tanh coefficients w [N] of a random smooth function: a degree-3 Chebyshev sum with coefficients
    xi_m / (m + 1), unit sd on the occupied band [-0.8, 0.8] (the collar convention), projected onto the bank by a
    truncated-SVD least squares. Fitting on the occupied band, not the whole mesh, keeps the coefficients moderate."""
    x, B = _ref_bank(N)
    m = x.abs() <= band
    x, B = x[m], B[m]
    xi = torch.randn(degree + 1, generator=gen, dtype=torch.float64)
    g = sum(xi[k] / (k + 1) * torch.cos(k * torch.arccos((x / band).clamp(-1, 1))) for k in range(degree + 1))
    g = (g - g.mean()) / g.std()
    w, _ = tsvd_solve(B, g.to(torch.float64), rcond=1e-10)
    return w.to(torch.get_default_dtype())


def fast_lstsq_bias(F, Y, rcond):
    """Per-step solve: SVD-based gelsd, falling back to pivoted-QR gelsy when LAPACK's SVD fails to converge (wide
    banks with many saturated, identical columns do that). Both are min-norm, rcond-truncated solves."""
    A = torch.cat([F, torch.ones(F.shape[0], 1, device=F.device, dtype=F.dtype)], 1)
    try:
        W = torch.linalg.lstsq(A, Y, rcond=rcond, driver="gelsd").solution
    except Exception:
        W = torch.linalg.lstsq(A, Y, rcond=rcond, driver="gelsy").solution
    return W[:-1], W[-1]


def robust_lstsq_bias(F, Y, rcond):
    """The record's QR-then-SVD solve, with the pivoted-QR fallback if the SVD of R fails to converge."""
    try:
        return lstsq_bias(F, Y, rcond)
    except Exception:
        W, b = fast_lstsq_bias(F, Y, rcond)
        return W, b, -1


def eff_rcond(rcond, dtype):
    """A truncation below 100 x the unit roundoff of the working dtype keeps garbage directions (fp32: 1e-14 is
    meaningless); the floor is dtype-derived, never a hardcoded fp64 constant."""
    return max(rcond, 100.0 * torch.finfo(dtype).eps)


# ----------------------------------------------------------------------------- mixer, layer, stack
class Mixer(nn.Module):
    """z_k = b_k + sum_{r,j} C_{rjk} H_{rj}, C = P theta (full) or C = sum_s a^{(s)} (x) P Phi^{(s)} (rank S).
    The coordinate map P acts on parameters once per step, never on activations: FLOPs per sample are unchanged."""
    def __init__(self, M, N, K, rank=None, coords="raw", cap=1e3, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.M, self.N, self.K, self.rank, self.coords = M, N, K, rank, coords
        self.register_buffer("P", coord_transform(N, coords, cap))
        if rank is None:
            self.theta = nn.Parameter(torch.zeros(M, N, K))
        else:
            a = torch.randn(M, rank, generator=g)
            self.a = nn.Parameter(a / a.norm(dim=1, keepdim=True))
            self.Phi = nn.Parameter(torch.zeros(rank, N, K))
        self.bias = nn.Parameter(torch.zeros(K))

    def raw_profiles(self):
        """Profiles in raw tanh coordinates: theta [M,N,K] -> P theta, or Phi [S,N,K] -> P Phi."""
        src = self.theta if self.rank is None else self.Phi
        return torch.einsum("jn,xnk->xjk", self.P, src)

    def coeffs(self):
        Cp = self.raw_profiles()
        if self.rank is None:
            return Cp
        return torch.einsum("ms,sjk->mjk", self.a, Cp)

    def forward(self, H):
        Cp = self.raw_profiles()
        if self.rank is None:
            return torch.einsum("bmj,mjk->bk", H, Cp) + self.bias
        return torch.einsum("bsj,sjk->bk", torch.einsum("bmj,ms->bsj", H, self.a), Cp) + self.bias

    @torch.no_grad()
    def set_raw(self, C):
        """Write raw coefficients C [M,N,K] (or Phi_raw [S,N,K]) through the inverse coordinate map."""
        src = self.theta if self.rank is None else self.Phi
        Pinv = torch.linalg.inv(self.P) if self.coords != "raw" else None
        src.copy_(C if Pinv is None else torch.einsum("nj,xjk->xnk", Pinv, C))

    def readout_view(self):
        assert self.rank is None, "the solved head needs a full-rank mixer"
        return self.theta.view(self.M * self.N, self.K), self.bias

    def counts(self):
        M, N, K, S = self.M, self.N, self.K, self.rank
        if S is None:
            return dict(params=M * N * K + K, macs=M * N * K)
        return dict(params=M * S + S * N * K + K, macs=M * N * S + S * N * K)


class QILayer(nn.Module):
    """[projection] -> Bank -> Mixer, optional residual. proj=True: M free directions V [M, d_in] (layer 1 only).
    Legacy options reproduce the expI01 notebook recipe and exist only as the baseline rung of the A1 ladder:
    unit_dirs (V row-normalized in the forward pass, scaled to the band 1.25 band_r) and track (range tracking of the
    bank input to [-0.8, 0.8] on update_range)."""
    def __init__(self, d_in, d_out, N, M=None, proj=False, rank=None, coords="raw", cap=1e3, residual=False, seed=0,
                 unit_dirs=False, band_r=None, track=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d_in, self.d_out, self.N, self.proj, self.residual = d_in, d_out, N, proj, residual
        self.unit_dirs, self.band_r, self.track = unit_dirs, band_r, track
        if proj:
            assert M is not None, "M (number of directions) is required with proj=True"
            self.M = M
            self.V = nn.Parameter(torch.randn(M, d_in, generator=g) / math.sqrt(d_in))
            self.bV = nn.Parameter(torch.zeros(M))
        else:
            self.M = d_in
        if residual:
            assert d_in == d_out and not proj, "residual needs d_in == d_out and no projection"
        if track:
            self.register_buffer("m", torch.zeros(self.M)); self.register_buffer("s", torch.ones(self.M))
        self.bank = Bank(N)
        self.mixer = Mixer(self.M, N, d_out, rank=rank, coords=coords, cap=cap, seed=seed + 1)

    def pre(self, z):
        if self.proj:
            if self.unit_dirs:
                return (z @ (self.V / self.V.norm(dim=1, keepdim=True)).T) / (1.25 * self.band_r)
            return z @ self.V.T + self.bV
        return (z - self.m) / self.s if self.track else z

    @torch.no_grad()
    def update_range(self, z, collar=1.25):
        if self.track:
            lo, hi = z.min(0).values, z.max(0).values
            self.m.copy_((lo + hi) / 2); self.s.copy_((collar * (hi - lo) / 2).clamp_min(1e-3))

    def hidden(self, z):
        return self.bank(self.pre(z))

    def forward(self, z):
        out = self.mixer(self.hidden(z))
        return z + out if self.residual else out

    def counts(self):
        c = self.mixer.counts()
        proj = self.M * self.d_in if self.proj else 0
        return dict(params=c["params"] + proj + (self.M if self.proj else 0), units=self.M * self.N,
                    flops=proj + self.M * self.N + c["macs"])


class QIStack(nn.Module):
    """A stack of QILayers; the last layer's mixer is the readout (solvable when its rank is None)."""
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def forward_pres(self, x, solve=None):
        """(output, [bank inputs p of every layer]) in one pass, for the band penalty. solve=(Y, rcond): the head is
        solved on this pass's own last-layer features (detached) before the output is formed -- the Kaufman VarPro
        step at zero extra cost (the head is a constant in the gradient, exactly the variable-projection gradient)."""
        pres = []
        for i, l in enumerate(self.layers):
            p = l.pre(x)
            pres.append(p)
            H = l.bank(p)
            if solve is not None and i == len(self.layers) - 1:
                self._solve_from(H.detach().reshape(H.shape[0], -1), *solve)
            out = l.mixer(H)
            x = x + out if l.residual else out
        return x, pres

    @torch.no_grad()
    def _solve_from(self, F, Y, rcond=1e-14, fast=True):
        last = self.layers[-1].mixer
        rcond = eff_rcond(rcond, F.dtype)
        if fast:
            W, b = fast_lstsq_bias(F, Y.reshape(len(Y), -1), rcond)
        else:
            W, b, _ = robust_lstsq_bias(F, Y.reshape(len(Y), -1), rcond)
        last.set_raw(W.view(last.M, last.N, last.K))
        last.bias.copy_(b.view_as(last.bias))

    def pres(self, x):
        return self.forward_pres(x)[1]

    def feats(self, x):
        """Raw bank features of the last layer, [B, M N]: what the head solve sees (coordinate free)."""
        for l in self.layers[:-1]:
            x = l(x)
        last = self.layers[-1]
        assert not last.residual
        return last.hidden(x).reshape(x.shape[0], -1)

    def readout(self):
        """The head's leaf parameters [theta_last, bias_last] (theta in the head's coordinates)."""
        last = self.layers[-1].mixer
        assert last.rank is None, "the head must be a full-rank mixer"
        return [last.theta, last.bias]

    @torch.no_grad()
    def solve_readout(self, X, Y, rcond=1e-14):
        """The record's head solve: QR then SVD of R, min-norm, rcond truncation (never normal equations)."""
        last = self.layers[-1].mixer
        W, b, rank = robust_lstsq_bias(self.feats(X), Y.reshape(len(Y), -1), eff_rcond(rcond, X.dtype))
        last.set_raw(W.view(last.M, last.N, last.K))
        last.bias.copy_(b.view_as(last.bias))
        return rank

    def nonlinear_params(self):
        ps = []
        for l in self.layers[:-1]:
            ps += list(l.parameters())
        last = self.layers[-1]
        if last.proj:
            ps.append(last.V)
        return ps

    def update_ranges(self, X, momentum=None):
        """Legacy range tracking (no-op unless a layer was built with track=True)."""
        for l in self.layers:
            l.update_range(X); X = l(X)

    def counts(self):
        tot = dict(params=0, units=0, flops=0)
        for l in self.layers:
            for k, v in l.counts().items():
                tot[k] += v
        return tot


def _init_smooth(mixer, gen, scale=1.0):
    """Every profile the same smooth random function in raw coordinates, mapped through P^{-1}."""
    N, K = mixer.N, mixer.K
    rows = mixer.M if mixer.rank is None else mixer.rank
    C = torch.stack([torch.stack([smooth_profile(N, gen) for _ in range(K)], 1) for _ in range(rows)], 0)  # [rows, N, K]
    mixer.set_raw(scale * C / math.sqrt(rows))


def make_block(d, M, N1, K, N2, d_out=1, rank=None, coords="raw", cap=1e3, depth=2, residual=False, seed=0,
               head_coords="raw", N_mid=None, legacy=False, band_r=None):
    """depth 1: the shallow ridge-QI (solved mixer is the readout). depth >= 2: projection layer -> (depth - 2)
    channel layers (residual optional) -> readout layer. Smooth init everywhere, identical across coordinate kinds.
    legacy=True: the expI01 notebook recipe (unit directions on the band 1.25 band_r, range-tracked channels, zero
    profiles, distinct random channel biases); only for the A1 baseline rung."""
    gen = torch.Generator().manual_seed(seed)
    if depth == 1:
        layers = [QILayer(d, d_out, N1, M=M, proj=True, rank=None, coords=head_coords, cap=cap, seed=seed, unit_dirs=legacy, band_r=band_r)]
    else:
        layers = [QILayer(d, K, N1, M=M, proj=True, rank=rank, coords=coords, cap=cap, seed=seed, unit_dirs=legacy, band_r=band_r)]
        for i in range(depth - 2):
            layers.append(QILayer(K, K, N_mid or N2, proj=False, rank=None, coords=coords, cap=cap, residual=residual, seed=seed + 10 + i, track=legacy))
        layers.append(QILayer(K, d_out, N2, proj=False, rank=None, coords=head_coords, cap=cap, seed=seed + 100, track=legacy))
    net = QIStack(layers)
    with torch.no_grad():
        for l in net.layers:
            if l.proj:
                l.V.copy_(torch.randn(l.M, l.d_in, generator=gen) / math.sqrt(l.d_in))
            if legacy:
                if l is not net.layers[-1]:
                    l.mixer.bias.copy_(0.1 * torch.randn(l.mixer.K, generator=gen))
            else:
                _init_smooth(l.mixer, gen)
    return net


# ----------------------------------------------------------------------------- band control
@torch.no_grad()
def _calibrate_mixer(mixer, H, s_target):
    """Scale each output channel's profiles and set its bias so the mixer output has mean 0 and sd s_target on H."""
    raw = mixer(H) - mixer.bias
    mu, sd = raw.mean(0), raw.std(0, unbiased=False).clamp_min(1e-12)
    scale = s_target / sd
    src = mixer.theta if mixer.rank is None else mixer.Phi
    src.mul_(scale.view(1, 1, -1))
    mixer.bias.copy_(-mu * scale)


@torch.no_grad()
def calibrate(model, X, s_star=S_STAR, s_resid=0.1):
    """One representative pass: every bank input gets mean 0 and sd s_star (a projection through V, b_V; a channel
    through the previous mixer's profiles and bias). Residual increments get sd s_resid. The readout is untouched.
    Fixed mesh, fixed lambda: this only decides where the data land on the mesh."""
    if not isinstance(model, QIStack):
        return
    z = X
    layers = list(model.layers)
    for i, l in enumerate(layers):
        if l.proj:
            p = z @ l.V.T
            mu, sd = p.mean(0), p.std(0, unbiased=False).clamp_min(1e-12)
            l.V.mul_((s_star / sd).view(-1, 1))
            l.bV.copy_(-mu * s_star / sd)
        H = l.bank(l.pre(z))
        if i < len(layers) - 1:
            _calibrate_mixer(l.mixer, H, s_resid if l.residual else s_star)
        out = l.mixer(H)
        z = z + out if l.residual else out


def band_penalty(pres, mu_max=BAND["mu_max"], s_min=BAND["s_min"], s_max=BAND["s_max"], rho=BAND["rho"]):
    """Weak hinge on the actual bank inputs: center drift, spread too small, spread too large, excursions past rho.
    Zero inside the tolerances; differentiable through the batch statistics; no forward normalization."""
    total = 0.0
    for p in pres:
        mu, sd = p.mean(0), p.std(0, unbiased=False)
        total = total + (torch.relu(mu.abs() - mu_max) ** 2).sum() + (torch.relu(s_min - sd) ** 2).sum() \
            + (torch.relu(sd - s_max) ** 2).sum() + (torch.relu(p.abs() - rho) ** 2).mean(0).sum()
    return total if torch.is_tensor(total) else torch.zeros(())


# ----------------------------------------------------------------------------- MLP baseline
_ACTS = dict(tanh=torch.tanh, relu=torch.relu)


class MLP(nn.Module):
    """Plain MLP with a solvable last layer (same interface as QIStack)."""
    def __init__(self, d, widths, d_out=1, act="tanh", seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.act_name, self.act = act, _ACTS[act]
        dims = [d] + list(widths)
        self.hidden = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.W = nn.Parameter(torch.zeros(dims[-1], d_out))
        self.b = nn.Parameter(torch.zeros(d_out))
        nn.init.uniform_(self.W, -1 / math.sqrt(dims[-1]), 1 / math.sqrt(dims[-1]))

    def feats(self, x):
        for l in self.hidden:
            x = self.act(l(x))
        return x

    def forward(self, x):
        return self.feats(x) @ self.W + self.b

    def forward_pres(self, x, solve=None):
        F = self.feats(x)
        if solve is not None:
            self._solve_from(F.detach(), *solve)
        return F @ self.W + self.b, []

    @torch.no_grad()
    def _solve_from(self, F, Y, rcond=1e-14):
        W, b = fast_lstsq_bias(F, Y.reshape(len(Y), -1), eff_rcond(rcond, F.dtype))
        self.W.copy_(W.view_as(self.W)); self.b.copy_(b.view_as(self.b))

    def pres(self, x):
        return []

    def readout(self):
        return [self.W, self.b]

    @torch.no_grad()
    def solve_readout(self, X, Y, rcond=1e-14):
        W, b, rank = robust_lstsq_bias(self.feats(X), Y.reshape(len(Y), -1), eff_rcond(rcond, X.dtype))
        self.W.copy_(W.view_as(self.W))
        self.b.copy_(b.view_as(self.b))
        return rank

    def nonlinear_params(self):
        return [p for l in self.hidden for p in l.parameters()]

    def update_ranges(self, X, momentum=None):
        pass

    def counts(self):
        params = sum(p.numel() for p in self.parameters())
        flops = sum(l.in_features * l.out_features for l in self.hidden) + self.W.numel()
        return dict(params=params, units=sum(l.out_features for l in self.hidden), flops=flops)


def mlp_width_for(target, d, key, depth=2, d_out=1, act="tanh"):
    """Smallest width w with MLP(d, [w] * depth, d_out).counts()[key] >= target."""
    def count(w):
        return MLP(d, [w] * depth, d_out, act=act).counts()[key]
    lo, hi = 1, 1
    while count(hi) < target:
        hi *= 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if count(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi if count(lo) < target else lo


# ----------------------------------------------------------------------------- optimizers and the fit loop
def _flat(ps):
    return torch.cat([p.reshape(-1) for p in ps])


def power_iteration_L(loss_fn, params, iters=10, seed=0):
    """Top eigenvalue of the Hessian of loss_fn() with respect to params, by power iteration on Hessian-vector
    products (double backward). Sets the heavy-ball step 1/L; costs 2 * iters passes, once."""
    g = torch.Generator().manual_seed(seed)
    v = [torch.randn(p.shape, generator=g, dtype=p.dtype).to(p.device) for p in params]
    nrm = _flat(v).norm()
    v = [t / nrm for t in v]
    L = 0.0
    for _ in range(iters):
        loss = loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grads = [torch.zeros_like(p) if gi is None else gi for gi, p in zip(grads, params)]
        gv = sum((gi * vi).sum() for gi, vi in zip(grads, v))
        if not gv.requires_grad:
            break
        Hv = torch.autograd.grad(gv, params, allow_unused=True)
        Hv = [torch.zeros_like(p) if h is None else h for h, p in zip(Hv, params)]
        L = float(abs(sum((hi * vi).sum() for hi, vi in zip(Hv, v))))
        nrm = _flat(Hv).norm()
        if float(nrm) == 0.0:
            break
        v = [h.detach() / nrm for h in Hv]
    return max(L, 1e-30)


def _split_proj(model, params):
    """(projection params, the rest) among `params`; only QILayer.V / bV count as projection."""
    proj_ids = set()
    if isinstance(model, QIStack):
        for l in model.layers:
            if l.proj:
                proj_ids.update({id(l.V), id(l.bV)})
    return [p for p in params if id(p) in proj_ids], [p for p in params if id(p) not in proj_ids]


def make_optimizer(model, kind, params, lr=5e-3, L=None, momentum=0.9):
    """kind: 'adam' (all params), 'momentum' (heavy ball, lr = 1/L), 'mixed' (Adam on the projection, heavy ball on
    the rest). Every group carries its base lr in 'base_lr' for the schedule."""
    if kind == "adam":
        groups = [dict(params=params, lr=lr)]
        opt = torch.optim.Adam(groups)
    elif kind in ("momentum", "gd"):
        assert L is not None
        opt = torch.optim.SGD([dict(params=params, lr=1.0 / L)], momentum=momentum if kind == "momentum" else 0.0)
    elif kind == "mixed":
        assert L is not None
        proj, rest = _split_proj(model, params)
        groups = []
        if proj:
            groups.append(dict(params=proj, lr=lr, adam=True))
        if rest:
            groups.append(dict(params=rest, lr=1.0 / L, adam=False))
        opt = _Mixed(groups, momentum)
    else:
        raise ValueError(kind)
    for gp in opt.param_groups:
        gp["base_lr"] = gp["lr"]
    return opt


class _Mixed:
    """Adam on some groups, heavy-ball SGD on the others, behind the torch.optim interface used here."""
    def __init__(self, groups, momentum):
        self.adam = torch.optim.Adam([g for g in groups if g["adam"]]) if any(g["adam"] for g in groups) else None
        self.sgd = torch.optim.SGD([g for g in groups if not g["adam"]], momentum=momentum) if any(not g["adam"] for g in groups) else None

    @property
    def param_groups(self):
        return (self.adam.param_groups if self.adam else []) + (self.sgd.param_groups if self.sgd else [])

    def zero_grad(self, set_to_none=True):
        for o in (self.adam, self.sgd):
            if o:
                o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in (self.adam, self.sgd):
            if o:
                o.step()


def _err(model, X, Y, loss):
    with torch.no_grad():
        out = model(X)
        if loss == "ce":
            return float((out.argmax(1) != Y).to(out.dtype).mean())
        return rel_l2(out, Y.reshape(len(Y), -1))


def _onehot(Y, q, dtype):
    return torch.nn.functional.one_hot(Y, q).to(dtype)


@torch.no_grad()
def refit_eval(model, Xtr, Ytr, Xte, Yte, rcond=1e-14, loss="mse"):
    """Test error with the head solved on the training set, the trained head restored afterwards."""
    saved = [t.clone() for t in model.readout()]
    Yfit = _onehot(Ytr, model.readout()[1].numel(), saved[0].dtype) if loss == "ce" else Ytr
    model.solve_readout(Xtr, Yfit, rcond)
    e = _err(model, Xte, Yte, loss)
    for t, s in zip(model.readout(), saved):
        t.copy_(s)
    return e


@torch.no_grad()
def band_stats(pres, rho=BAND["rho"]):
    """Per layer: [mean |mu|, mean sd, fraction of values past rho]."""
    return [[float(p.mean(0).abs().mean()), float(p.std(0, unbiased=False).mean()), float((p.abs() > rho).to(p.dtype).mean())] for p in pres]


def fit(model, Xtr, Ytr, Xte, Yte, *, optimizer="adam", head="varpro", steps=3000, beta=1e-2, rcond=1e-14,
        log_every=100, loss="mse", batch=None, epochs=None, warmup=50, lr=5e-3, gn_iters=0, calibrate_first=True,
        n_calib=4096, seed=0, verbose=False):
    """Train the nonlinear parameters; the head is solved ('varpro': every step; 'periodic': every 250 steps) or
    trained ('trained': a gradient parameter, the solved-head error still logged). Cross-entropy forces 'trained'.
    Logs train error, test error with the current head, test error with a freshly solved head, and band statistics.
    Ends with the head solved (regression) and, optionally, a Gauss-Newton finisher."""
    t0 = time.time()
    torch.manual_seed(seed)
    if loss == "ce":
        head = "trained"
    q = int(Ytr.max()) + 1 if loss == "ce" else (Ytr.shape[1] if Ytr.dim() > 1 else 1)
    Ytr_m = Ytr if loss == "ce" else Ytr.reshape(len(Ytr), -1)
    Yfit = _onehot(Ytr, q, Xtr.dtype) if loss == "ce" else Ytr_m
    if calibrate_first:
        calibrate(model, Xtr[:n_calib])
    nl = [p for p in model.nonlinear_params() if p.requires_grad]
    params = nl + (model.readout() if head in ("trained", "periodic") else [])
    if head != "trained":
        model.solve_readout(Xtr, Yfit, rcond)                # 'periodic': starts solved, trained by gradient in between, replaced every 250 steps
    elif loss != "ce":
        with torch.no_grad():
            model.readout()[1].copy_(Ytr_m.mean(0).view_as(model.readout()[1]))

    def data_loss(Xb, Yb, solve=None):
        out, pres = model.forward_pres(Xb, solve=solve)
        if loss == "ce":
            dl = torch.nn.functional.cross_entropy(out, Yb)
        else:
            dl = ((out - Yb) ** 2).mean()
        return dl + beta * band_penalty(pres), dl

    L = None
    if optimizer in ("momentum", "mixed", "gd") and params:
        Xc, Yc = Xtr[:n_calib], (Ytr[:n_calib] if loss == "ce" else Ytr_m[:n_calib])
        hb = _split_proj(model, params)[1] if optimizer == "mixed" else params   # the curvature of the group heavy ball steps
        L = power_iteration_L(lambda: data_loss(Xc, Yc)[0], hb, iters=10, seed=seed) if hb else 1.0
    opt = make_optimizer(model, optimizer, params, lr=lr, L=L) if params else None
    n = len(Xtr)
    if batch is not None:
        per_epoch = max(1, n // batch)
        steps = per_epoch * (epochs or 1)
        log_every = per_epoch
    g = torch.Generator(device="cpu").manual_seed(seed)
    log = dict(step=[], train=[], test_trained=[], test_solved=[], band=[], L=L, counts=model.counts())
    lr_at = lambda t: min(1.0, (t + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * t / max(1, steps)))
    for t in range(steps):
        if opt is None:
            break
        if t > 0 and t % 250 == 0:
            model.update_ranges(Xtr)                      # legacy models only; a no-op otherwise
            if head in ("periodic", "varpro"):
                model.solve_readout(Xtr, Yfit, rcond)
                if head == "periodic":                    # the replaced head starts Adam afresh
                    for q in model.readout():
                        opt_state = getattr(opt, "state", None)
                        if opt_state is not None and q in opt_state:
                            del opt_state[q]
        for gp in opt.param_groups:
            gp["lr"] = gp["base_lr"] * lr_at(t)
        if batch is None:
            Xb, Yb = Xtr, (Ytr if loss == "ce" else Ytr_m)
        else:
            idx = torch.randint(0, n, (batch,), generator=g).to(Xtr.device)
            Xb, Yb = Xtr[idx], (Ytr[idx] if loss == "ce" else Ytr_m[idx])
        opt.zero_grad(set_to_none=True)
        total, dl = data_loss(Xb, Yb, solve=((Yfit, rcond) if head == "varpro" and batch is None else None))
        total.backward()
        opt.step()
        if head == "varpro" and batch is not None:
            model.solve_readout(Xtr, Yfit, rcond)
        if t % log_every == 0 or t == steps - 1:
            tr = float(dl.sqrt() * math.sqrt(Yb.numel()) / Yb.norm()) if loss != "ce" else float(dl)
            if head == "varpro":
                model.solve_readout(Xtr, Yfit, rcond)            # the record's solver at the logged points
            te = _err(model, Xte, Yte, loss)
            rf = te if head == "varpro" else refit_eval(model, Xtr, Ytr, Xte, Yte, rcond, loss)
            with torch.no_grad():
                bs = band_stats(model.forward_pres(Xtr[:n_calib])[1])
            log["step"].append(t); log["train"].append(tr); log["test_trained"].append(te); log["test_solved"].append(rf); log["band"].append(bs)
            if verbose:
                print(f"  step {t:5d}  train {tr:.2e}  test {te:.2e}  solved {rf:.2e}  [{time.time() - t0:.0f}s]", flush=True)
    model.update_ranges(Xtr)
    log["final_trained"] = _err(model, Xte, Yte, loss)
    if loss == "ce":
        log["final"] = refit_eval(model, Xtr, Ytr, Xte, Yte, rcond, loss)
    else:
        model.solve_readout(Xtr, Yfit, rcond)
        log["final"] = _err(model, Xte, Yte, loss)
    if gn_iters > 0 and loss != "ce" and nl:
        gl = _qib_gauss_newton(model, Xtr, Ytr_m, Xte, Yte.reshape(len(Yte), -1), iters=gn_iters, rcond=rcond, chunk=64)
        log["final_gn"] = gl["final"]
        log["gn_curve"] = gl["refit"]
    log["time"] = time.time() - t0
    return log
