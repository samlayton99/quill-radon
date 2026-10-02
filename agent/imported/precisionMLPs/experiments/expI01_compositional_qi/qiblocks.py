"""qiblocks: composable ridge-QI layers (checkpoint I). Self-contained: torch only, no repo imports.

Family (see results/checkpoint_I_depth_theory/compositional_qi_theory.md and composable_qi_card.md):

    layer 1:  p = V z            (M learned unit directions; the only place directions exist)
    every layer:  H = act(gamma (u - c)),  u = range-normalized p,  H in R^{B x M x N}      (QIBank, RangeTracker)
                  z' = sum_{r,j} C_{rjk} H_{rj} + b_k,  C = sum_s a^{(s)} (x) Phi^{(s)}  or full      (ChannelMixer)
    later layers: p = z' (the channels ARE the next projections; any extra linear map is absorbed into C)

    gamma = lambda / h with lambda from the activation's aliasing rule (tanh 0.25, gelu 0.707, swish 0.455, sigmoid 0.5).
    Readout of the last layer is solved (truncated SVD), never trained by gradient, whenever a solve is affordable.

Models are configurations of one layer type:
    shallow ridge-QI (checkpoint H):  QINet([RidgeQILayer(d, 1, N, M=M, dirs="learned", rank=None)])
    compositional block (expI01):     QINet([RidgeQILayer(d, K, N1, M=M, dirs="learned", rank=1|S|None),
                                             RidgeQILayer(K, 1, N2, dirs=None, rank=None)])
    deeper / residual:                add layers with dirs=None; residual=True on width-preserving layers
    KAN layer with the QI basis:      RidgeQILayer(d, K, N, dirs=None, rank=None)
    transformer FFN drop-in:          QIFFN(d_model, M, N1, K, N2, rank, activation="gelu", track="ema")

Run `python qiblocks.py` for the self-tests (1-D floor, dense-MLP equivalence, rank algebra, range tracking, a
two-layer fit, the FFN shape and the benchmark).
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- activations and lambda
_ACT = {
    "tanh": (torch.tanh, 0.25),      # expC03 basin, expC07 anchor
    "gelu": (F.gelu, 0.707),         # expC07 aliasing rule (Gaussian-type kernel)
    "swish": (F.silu, 0.455),        # expC07
    "sigmoid": (torch.sigmoid, 0.5), # expC07, exact identity with tanh at lambda/2
}

def lambda_for(activation):
    return _ACT[activation][1]

def alias_tanh(lam):
    x = math.pi ** 2 / lam
    return x / math.sinh(x)

def cell_centers(N, T, device=None, dtype=None):
    h = 2.0 * T / N
    return -T + (torch.arange(N, device=device, dtype=dtype) + 0.5) * h, h

# ----------------------------------------------------------------------------- solvers
def tsvd_solve(A, Y, rcond=1e-14):
    """min-norm least squares by QR then SVD of R; A [n, p], Y [n] or [n, q]. Never form A^T A."""
    Q, R = torch.linalg.qr(A)
    rhs = Q.T @ Y
    U, s, Vh = torch.linalg.svd(R, full_matrices=False)
    keep = s > rcond * s[0]
    z = U[:, keep].T @ rhs
    z = z / s[keep] if z.dim() == 1 else z / s[keep][:, None]
    return Vh[keep].T @ z, int(keep.sum())

def lstsq_bias(Fm, Y, rcond=1e-14):
    A = torch.cat([Fm, torch.ones(Fm.shape[0], 1, device=Fm.device, dtype=Fm.dtype)], 1)
    W, rank = tsvd_solve(A, Y, rcond)
    return W[:-1], W[-1], rank

def pick_rcond(Fm, Y, grid=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14), wmax=100.0):
    """Noisy labels: two-fold validation over the truncation threshold under a readout-norm cap (identifiability rule:
    a resolved readout has O(1) norm; a huge norm interpolates noise and blows up on rare points no split can see)."""
    n = Fm.shape[0]; h = n // 2; best = None
    with torch.no_grad():
        Fs, Ys = [Fm[:h], Fm[h:]], [Y[:h], Y[h:]]
        for rc in grid:
            errs = []
            for a in (0, 1):
                W, b, _ = lstsq_bias(Fs[a], Ys[a], rc)
                if float(W.norm()) > wmax: errs = None; break
                errs.append(float(((Fs[1 - a] @ W + b - Ys[1 - a]) ** 2).mean().sqrt()))
            if errs is None: continue
            e = sum(errs) / 2
            if best is None or e < best[0]: best = (e, rc)
    return best[1] if best else grid[0]

def rel_l2(pred, y):
    return float((pred - y).norm() / y.norm())

# ----------------------------------------------------------------------------- building blocks
class QIBank(nn.Module):
    """Fixed 1-D QI mesh applied to every scalar coordinate: (B, M) -> (B, M, N). Buffers only; gamma h = lambda."""
    def __init__(self, N, T=1.0, activation="tanh", lam=None):
        super().__init__()
        self.N, self.T, self.activation = N, float(T), activation
        self.act, lam_default = _ACT[activation]
        self.lam = lam_default if lam is None else lam
        c, h = cell_centers(N, T)
        self.register_buffer("c", c)
        self.h = h; self.gamma = self.lam / h
    def forward(self, p):
        return self.act(self.gamma * (p.unsqueeze(-1) - self.c))
    def extra_repr(self):
        return f"N={self.N}, T={self.T}, {self.activation}, lambda={self.lam}, gamma={self.gamma:.3g}"

class RangeTracker(nn.Module):
    """Per-channel affine map u = (p - m) / s so that the data occupies [-1, 1] with a collar; the QI mesh lives in u.
    mode="regrid": m, s set from data on demand (precision setting).  mode="ema": running min/max in training (deep learning).
    mode="fixed": identity (use when the band is known, e.g. layer 1 with unit directions and a data ball)."""
    def __init__(self, M, collar=1.25, mode="regrid", momentum=0.01):
        super().__init__()
        self.mode, self.collar, self.momentum = mode, collar, momentum
        self.register_buffer("m", torch.zeros(M)); self.register_buffer("s", torch.ones(M))
    @torch.no_grad()
    def update(self, p, momentum=None):
        lo, hi = p.min(0).values, p.max(0).values
        m, s = (lo + hi) / 2, (self.collar * (hi - lo) / 2).clamp_min(1e-3)
        mu = 1.0 if momentum is None else momentum
        self.m.mul_(1 - mu).add_(mu * m); self.s.mul_(1 - mu).add_(mu * s)
    def forward(self, p):
        if self.mode == "fixed": return p
        if self.mode == "ema" and self.training: self.update(p, self.momentum)
        return (p - self.m) / self.s

class ChannelMixer(nn.Module):
    """z_k = sum_{r,j} C_{rjk} H_{rj} + b_k.  rank=None: full C [M, N, K].  rank=S: C = sum_s a^{(s)} (x) Phi^{(s)},
    a [M, S] shared across channels (KAT at S=1) or per_channel a [M, S, K]."""
    def __init__(self, M, N, K, rank=None, per_channel=False, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.M, self.N, self.K, self.rank, self.per_channel = M, N, K, rank, per_channel
        if rank is None:
            self.C = nn.Parameter(torch.zeros(M, N, K))
        else:
            shape = (M, rank, K) if per_channel else (M, rank)
            self.a = nn.Parameter(torch.randn(*shape, generator=g) / math.sqrt(M))
            self.Phi = nn.Parameter(torch.zeros(rank, N, K))
        self.bias = nn.Parameter(0.1 * torch.randn(K, generator=g))   # distinct per channel: breaks the symmetry at Phi = 0
    def tensor(self):
        if self.rank is None: return self.C
        if self.per_channel: return torch.einsum("msk,snk->mnk", self.a, self.Phi)
        return torch.einsum("ms,snk->mnk", self.a, self.Phi)
    def forward(self, H):
        if self.rank is None: return torch.einsum("bmn,mnk->bk", H, self.C) + self.bias
        if self.per_channel: return torch.einsum("bsnk,snk->bk", torch.einsum("bmn,msk->bsnk", H, self.a), self.Phi) + self.bias
        return torch.einsum("bsn,snk->bk", torch.einsum("bmn,ms->bsn", H, self.a), self.Phi) + self.bias
    def readout_view(self):
        """(W [MN, K], b) when full rank: the solvable form."""
        assert self.rank is None, "a rank-structured mixer is bilinear in (a, Phi); solve needs rank=None"
        return self.C.view(self.M * self.N, self.K), self.bias
    def counts(self):
        M, N, K, S = self.M, self.N, self.K, self.rank
        if S is None: return dict(params=M * N * K + K, macs=M * N * K)
        a = M * S * K if self.per_channel else M * S
        return dict(params=a + S * N * K + K, macs=M * N * S * (K if self.per_channel else 1) + S * N * K)

class RidgeQILayer(nn.Module):
    """[directions] -> RangeTracker -> QIBank -> ChannelMixer, with optional residual.
    dirs: "learned" (M unit directions, parameters), a tensor (fixed directions), or None (identity: the input channels
    are the projections; M = d_in; this is a KAN layer with the QI basis)."""
    def __init__(self, d_in, d_out, N, M=None, dirs=None, rank=None, per_channel=False, activation="tanh", lam=None,
                 track="regrid", collar=1.25, band=1.0, residual=False, eta=1.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d_in, self.d_out, self.N, self.residual, self.eta = d_in, d_out, N, residual, eta
        if dirs is None:
            self.M, self.dir_mode = d_in, "identity"
        elif isinstance(dirs, str) and dirs == "learned":
            assert M is not None, "M (number of directions) is required for learned directions"
            self.M, self.dir_mode = M, "learned"
            V = torch.randn(M, d_in, generator=g); self.V_raw = nn.Parameter(V / V.norm(dim=1, keepdim=True))
        else:
            V = torch.as_tensor(dirs, dtype=torch.get_default_dtype()); self.M, self.dir_mode = V.shape[0], "fixed"
            self.register_buffer("V_raw", V / V.norm(dim=1, keepdim=True))
        self.tracker = RangeTracker(self.M, collar=collar, mode=track)
        self.bank = QIBank(N, T=(band if track == "fixed" else 1.0), activation=activation, lam=lam)
        self.mixer = ChannelMixer(self.M, N, d_out, rank=rank, per_channel=per_channel, seed=seed)
        if residual: assert d_in == d_out, "residual needs d_in == d_out"
    def dirs(self):
        if self.dir_mode == "identity": return None
        return self.V_raw / self.V_raw.norm(dim=1, keepdim=True)
    def project(self, z):
        return z if self.dir_mode == "identity" else z @ self.dirs().T
    def hidden(self, z):
        return self.bank(self.tracker(self.project(z)))                   # (B, M, N)
    def forward(self, z):
        out = self.mixer(self.hidden(z))
        return z + self.eta * out if self.residual else out
    @torch.no_grad()
    def update_range(self, z, momentum=None):
        self.tracker.update(self.project(z), momentum)
    def profiles(self, t):
        """Each direction's learned 1-D function in each channel on the grid t (normalized coordinate if tracked): (M, K, len t)."""
        H = self.bank(t.view(-1, 1))[:, 0, :]                              # (T, N)
        return torch.einsum("tn,mnk->mkt", H, self.mixer.tensor())
    def nonlinear_params(self):
        ps = [self.V_raw] if self.dir_mode == "learned" else []
        return ps + list(self.mixer.parameters())
    def counts(self):
        c = self.mixer.counts(); proj = self.M * self.d_in if self.dir_mode != "identity" else 0
        return dict(params=c["params"] + (self.M * self.d_in if self.dir_mode == "learned" else 0), units=self.M * self.N,
                    macs=proj + self.M * self.N + c["macs"])
    def dense_first_linear(self):
        """The ordinary tanh-MLP form of [directions -> tracker -> bank]: Linear(d_in -> M N) with weight rows gamma v_r / s_r."""
        g, c = self.bank.gamma, self.bank.c
        V = torch.eye(self.d_in, device=c.device, dtype=c.dtype) if self.dir_mode == "identity" else self.dirs().detach()
        m, s = self.tracker.m, self.tracker.s
        if self.tracker.mode == "fixed": m, s = torch.zeros_like(m), torch.ones_like(s)
        W = ((g / s)[:, None, None] * V[:, None, :]).expand(self.M, self.N, self.d_in)   # (M, N, d_in)
        b = -g * (m / s)[:, None] - g * c[None, :]                          # (M, N)
        lin = nn.Linear(self.d_in, self.M * self.N).to(c.device, c.dtype)
        with torch.no_grad(): lin.weight.copy_(W.reshape(-1, self.d_in)); lin.bias.copy_(b.reshape(-1))
        return lin

class QINet(nn.Module):
    """A stack of RidgeQILayers. The last layer's mixer is the readout (solvable when its rank is None)."""
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x
    def feats(self, x):
        for l in self.layers[:-1]: x = l(x)
        last = self.layers[-1]; assert not last.residual
        return last.hidden(x).reshape(x.shape[0], -1)                      # (B, M N) of the last layer
    def readout(self):
        return list(self.layers[-1].mixer.readout_view())
    @torch.no_grad()
    def solve_readout(self, X, Y, rcond=1e-14):
        W, b, rank = lstsq_bias(self.feats(X), Y, rcond)
        Wv, bv = self.readout(); Wv.copy_(W.view_as(Wv)); bv.copy_(b.view_as(bv))
        return rank
    def nonlinear_params(self):
        ps = []
        for l in self.layers[:-1]: ps += l.nonlinear_params()
        last = self.layers[-1]
        if last.dir_mode == "learned": ps.append(last.V_raw)
        return ps
    @torch.no_grad()
    def update_ranges(self, X, momentum=None):
        for l in self.layers: l.update_range(X, momentum); X = l(X)
    def counts(self):
        tot = dict(params=0, units=0, macs=0)
        for l in self.layers:
            for k, v in l.counts().items(): tot[k] += v
        return tot
    def to_mlp(self):
        """The equivalent ordinary dense MLP (Linear -> act -> Linear -> ...). Residual layers are not expanded."""
        mods = []
        for l in self.layers:
            assert not l.residual, "to_mlp does not expand residual layers"
            mods += [l.dense_first_linear(), _Act(l.bank.act)]
            lin = nn.Linear(l.M * l.N, l.d_out).to(l.bank.c.device, l.bank.c.dtype)
            with torch.no_grad(): lin.weight.copy_(l.mixer.tensor().detach().reshape(-1, l.d_out).T); lin.bias.copy_(l.mixer.bias.detach())
            mods.append(lin)
        return nn.Sequential(*mods)

class _Act(nn.Module):
    def __init__(self, fn): super().__init__(); self.fn = fn
    def forward(self, x): return self.fn(x)

# ----------------------------------------------------------------------------- baselines
class MLP(nn.Module):
    """Plain tanh MLP with a solvable last layer (same interface as QINet)."""
    def __init__(self, d, widths, d_out=1, activation="tanh", seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.act = _ACT[activation][0]
        dims = [d] + list(widths)
        self.hidden = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.W = nn.Parameter(torch.zeros(dims[-1], d_out)); self.b = nn.Parameter(torch.zeros(d_out))
        nn.init.uniform_(self.W, -1 / math.sqrt(dims[-1]), 1 / math.sqrt(dims[-1]))
    def feats(self, x):
        for l in self.hidden: x = self.act(l(x))
        return x
    def forward(self, x): return self.feats(x) @ self.W + self.b
    def readout(self): return [self.W, self.b]
    @torch.no_grad()
    def solve_readout(self, X, Y, rcond=1e-14):
        W, b, rank = lstsq_bias(self.feats(X), Y, rcond); self.W.copy_(W.view_as(self.W)); self.b.copy_(b.view_as(self.b)); return rank
    def nonlinear_params(self): return [p for l in self.hidden for p in l.parameters()]
    def update_ranges(self, X, momentum=None): pass
    def counts(self):
        params = sum(p.numel() for p in self.parameters()); macs = sum(l.in_features * l.out_features for l in self.hidden) + self.W.numel()
        return dict(params=params, units=sum(l.out_features for l in self.hidden), macs=macs)

def mlp_width_for(target, d, key, depth=2, d_out=1):
    w = 1
    while MLP(d, [w] * depth, d_out).counts()[key] < target: w += 1
    return w

# ----------------------------------------------------------------------------- training utilities (any model with feats/readout/solve_readout/nonlinear_params/update_ranges)
@torch.no_grad()
def refit_eval(model, Xtr, Ytr, Xte, Yte, rcond=1e-14):
    saved = [t.clone() for t in model.readout()]
    model.solve_readout(Xtr, Ytr, rcond); e = rel_l2(model(Xte), Yte)
    for t, s in zip(model.readout(), saved): t.copy_(s)
    return e

def fit(model, Xtr, Ytr, Xte=None, Yte=None, mode="varpro", steps=2000, lr=5e-3, regrid_every=250, log_every=100,
        rcond=1e-14, final_solve=True, warmup=50, verbose=False):
    """mode='adam': Adam on everything (readout included), final solve.  mode='varpro': Adam on nonlinear params, readout
    re-solved every step.  mode='solve': no training, one solve.  Range tracking on a fixed schedule in every mode."""
    Ytr = Ytr.reshape(len(Ytr), -1) if Ytr.dim() == 1 else Ytr
    if Yte is not None and Yte.dim() == 1: Yte = Yte.reshape(len(Yte), -1)
    nl = model.nonlinear_params(); log = dict(step=[], train=[], test=[], refit=[]); t0 = time.time()
    with torch.no_grad(): model.readout()[1].copy_(Ytr.mean(0).view_as(model.readout()[1]))
    if mode == "solve": steps = 0; opt = None
    elif mode == "adam": opt = torch.optim.Adam(nl + model.readout(), lr=lr)
    elif mode == "varpro": opt = torch.optim.Adam(nl, lr=lr) if nl else None; model.update_ranges(Xtr); model.solve_readout(Xtr, Ytr, rcond)
    else: raise ValueError(mode)
    lr_at = lambda t: lr * min(1.0, (t + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * t / max(1, steps)))
    for t in range(steps):
        if opt is None: break
        if t > 0 and t % regrid_every == 0:
            model.update_ranges(Xtr)
            if mode == "varpro": model.solve_readout(Xtr, Ytr, rcond)
        for gp in opt.param_groups: gp["lr"] = lr_at(t)
        opt.zero_grad(set_to_none=True)
        loss = ((model(Xtr) - Ytr) ** 2).mean(); loss.backward(); opt.step()
        if mode == "varpro": model.solve_readout(Xtr, Ytr, rcond)
        if Xte is not None and (t % log_every == 0 or t == steps - 1):
            with torch.no_grad(): te = rel_l2(model(Xte), Yte)
            rf = refit_eval(model, Xtr, Ytr, Xte, Yte, rcond)
            log["step"].append(t); log["train"].append(float(loss.sqrt() * math.sqrt(Ytr.numel()) / Ytr.norm())); log["test"].append(te); log["refit"].append(rf)
            if verbose and t % (5 * log_every) == 0: print(f"  step {t:5d}  train {log['train'][-1]:.2e}  test {te:.2e}  refit {rf:.2e}  [{time.time() - t0:.0f}s]", flush=True)
    model.update_ranges(Xtr)
    if final_solve: model.solve_readout(Xtr, Ytr, rcond)
    if Xte is not None:
        with torch.no_grad(): log["final"] = rel_l2(model(Xte), Yte)
    log["time"] = time.time() - t0
    return log

def _flat(ps): return torch.cat([p.reshape(-1) for p in ps])
def _unflat(v, ps):
    out, i = [], 0
    for p in ps: out.append(v[i:i + p.numel()].reshape(p.shape)); i += p.numel()
    return out

def gauss_newton(model, Xtr, Ytr, Xte=None, Yte=None, iters=30, mu=1e-2, rcond=1e-14, chunk=None, verbose=False):
    """Variable-projection Levenberg-Marquardt on the nonlinear parameters with the Kaufman Jacobian
    J = -P_perp (dA/dtheta) W*, the readout re-solved at every trial. Scalar or vector output."""
    from torch.func import functional_call, jvp, vmap
    Ytr = Ytr.reshape(len(Ytr), -1) if Ytr.dim() == 1 else Ytr
    if Yte is not None and Yte.dim() == 1: Yte = Yte.reshape(len(Yte), -1)
    names = [n for n, p in model.named_parameters() if any(p is q for q in model.nonlinear_params())]
    ps = [dict(model.named_parameters())[n] for n in names]
    log = dict(step=[], refit=[]); t0 = time.time()
    if not ps: return log
    theta = _flat(ps).detach().clone(); P = theta.numel(); dev = theta.device
    chunk = chunk or (64 if dev.type == "cuda" else 8)
    def set_theta(th):
        with torch.no_grad():
            for p, v in zip(ps, _unflat(th, ps)): p.copy_(v)
    def resid(th): return (functional_call(model, dict(zip(names, _unflat(th, ps))), (Xtr,)) - Ytr).reshape(-1)
    def jacobian(th):
        E = torch.eye(P, device=dev, dtype=theta.dtype)
        return torch.cat([vmap(lambda e: jvp(resid, (th,), (e,))[1])(E[i:i + chunk]) for i in range(0, P, chunk)], 0).T
    model.update_ranges(Xtr); model.solve_readout(Xtr, Ytr, rcond)
    with torch.no_grad(): r = resid(theta); cost = float(r @ r)
    for it in range(iters):
        with torch.no_grad():
            J0 = jacobian(theta)
            A = torch.cat([model.feats(Xtr), torch.ones(len(Ytr), 1, device=dev, dtype=theta.dtype)], 1)
            # Kaufman projector restricted to the RETAINED left singular subspace of A (same truncation as
            # tsvd_solve), not the full column space of an unpivoted QR -- the head is solved on the retained
            # modes, so the projector must use the same effective rank.
            Qa, R = torch.linalg.qr(A); Ur, s, _ = torch.linalg.svd(R, full_matrices=False); Q = Qa @ Ur[:, s > rcond * s[0]]
            q = Ytr.shape[1]
            J = J0 - torch.cat([Q @ (Q.T @ J0[i * len(Ytr):(i + 1) * len(Ytr)]) for i in range(q)], 0) if q > 1 else J0 - Q @ (Q.T @ J0)
            g = J.T @ r; Hm = J.T @ J; accepted = False
            for _ in range(8):
                delta = torch.linalg.solve(Hm + mu * torch.diag(Hm.diagonal().clamp_min(1e-12)), -g)
                set_theta(theta + delta); model.solve_readout(Xtr, Ytr, rcond)
                r_new = resid(theta + delta); c_new = float(r_new @ r_new)
                if c_new < cost: theta, r, cost, mu, accepted = theta + delta, r_new, c_new, mu / 3, True; break
                mu *= 10
            if not accepted: set_theta(theta); model.solve_readout(Xtr, Ytr, rcond)
            model.update_ranges(Xtr); model.solve_readout(Xtr, Ytr, rcond); r = resid(theta); cost = float(r @ r)
            te = rel_l2(model(Xte), Yte) if Xte is not None else float("nan")
        log["step"].append(it); log["refit"].append(te)
        if verbose: print(f"  GN {it:3d}  train {math.sqrt(cost) / float(Ytr.norm()):.2e}  test {te:.2e}  mu {mu:.1e}", flush=True)
        if not accepted and mu > 1e8: break
    log["final"] = log["refit"][-1] if log["refit"] else float("nan"); log["time"] = time.time() - t0
    return log

# ----------------------------------------------------------------------------- transformer FFN drop-in and benchmark
class QIFFN(nn.Module):
    """d_model -> d_model, residual added by the caller (as in a transformer block). EMA range tracking, no solves."""
    def __init__(self, d_model, M, N1, K, N2, rank=None, per_channel=False, activation="gelu", track="ema", seed=0):
        super().__init__()
        self.l1 = RidgeQILayer(d_model, K, N1, M=M, dirs="learned", rank=rank, per_channel=per_channel, activation=activation, track=track, seed=seed)
        self.l2 = RidgeQILayer(K, d_model, N2, dirs=None, rank=None, activation=activation, track=track, seed=seed + 1)
    def forward(self, x):
        shp = x.shape; z = x.reshape(-1, shp[-1])
        return self.l2(self.l1(z)).reshape(shp)
    def counts(self):
        c1, c2 = self.l1.counts(), self.l2.counts()
        return dict(params=c1["params"] + c2["params"], units=c1["units"] + c2["units"], macs=c1["macs"] + c2["macs"])

def dense_ffn(d_model, hidden, activation="gelu"):
    return nn.Sequential(nn.Linear(d_model, hidden), _Act(_ACT[activation][0]), nn.Linear(hidden, d_model))

def bench(module, x, iters=20, compile=False):
    """Forward+backward wall time per iteration (ms). Returns (eager_ms, compiled_ms or None)."""
    def run(m):
        for p in m.parameters(): p.requires_grad_(True)
        for _ in range(3): m(x).sum().backward()
        if x.is_cuda: torch.cuda.synchronize()
        t = time.time()
        for _ in range(iters): m(x).sum().backward()
        if x.is_cuda: torch.cuda.synchronize()
        return (time.time() - t) / iters * 1e3
    eager = run(module)
    comp = None
    if compile:
        try: comp = run(torch.compile(module))
        except Exception as e: print("compile failed:", e)
    return eager, comp

# ----------------------------------------------------------------------------- export for interpretability
@torch.no_grad()
def export_pack(net, X, Y, n_grid=401):
    """Weights plus the interpretable objects: per-layer directions, per-edge 1-D profiles on the normalized coordinate,
    channel values on the given points, and the dense-MLP equivalent's state."""
    t = torch.linspace(-1.0, 1.0, n_grid, device=X.device, dtype=X.dtype)
    layers = []
    z = X
    for l in net.layers:
        layers.append(dict(dirs=None if l.dirs() is None else l.dirs().cpu(), profiles=l.profiles(t).cpu(), range_m=l.tracker.m.cpu(),
                           range_s=l.tracker.s.cpu(), gamma=l.bank.gamma, centers=l.bank.c.cpu(), channels_out=l(z).cpu()))
        z = l(z)
    return dict(state_dict={k: v.cpu() for k, v in net.state_dict().items()}, t_grid=t.cpu(), layers=layers, X=X.cpu(), Y=Y.cpu(),
                pred=net(X).cpu(), counts=net.counts())

# ----------------------------------------------------------------------------- self-tests
def selftest(device=None, verbose=True):
    torch.set_default_dtype(torch.float64)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    g = torch.Generator().manual_seed(0)
    say = print if verbose else (lambda *a, **k: None)

    # 1. the 1-D floor: one fixed direction, fixed band, solved readout on sin(2 pi x) over [-1, 1]
    x = (2 * torch.rand(4096, 1, generator=g) - 1).to(dev); y = torch.sin(2 * math.pi * x[:, 0])
    xt = (1.8 * torch.rand(4096, 1, generator=g) - 0.9).to(dev); yt = torch.sin(2 * math.pi * xt[:, 0])
    net = QINet([RidgeQILayer(1, 1, 128, dirs=torch.ones(1, 1), track="fixed", band=1.25)]).to(dev)
    net.solve_readout(x, y); e = rel_l2(net(xt)[:, 0], yt)
    say(f"[1] 1-D floor, N=128: rel L2 {e:.2e}"); assert e < 1e-11, e

    # 2. dense-MLP equivalence of a two-layer block with range tracking
    d = 3; X = (2 * torch.rand(2000, d, generator=g) - 1).to(dev) * 0.3
    blk = QINet([RidgeQILayer(d, 2, 16, M=6, dirs="learned", rank=1, track="regrid"), RidgeQILayer(2, 1, 32, dirs=None, rank=None, track="regrid")]).to(dev)
    with torch.no_grad():
        blk.layers[0].mixer.Phi.normal_(); blk.layers[1].mixer.C.normal_()
    blk.update_ranges(X); mlp = blk.to_mlp(); diff = float((blk(X) - mlp(X)).abs().max())
    say(f"[2] block == dense MLP: max |diff| {diff:.1e}"); assert diff < 1e-12, diff

    # 3. rank algebra: rank-1 mixer tensor equals a (x) Phi; full C reproduces a rank-1 tensor exactly
    mx = ChannelMixer(6, 16, 2, rank=1).to(dev)
    with torch.no_grad(): mx.Phi.normal_()
    T = mx.tensor(); T2 = torch.einsum("m,nk->mnk", mx.a[:, 0], mx.Phi[0]); assert float((T - T2).abs().max()) < 1e-15
    full = ChannelMixer(6, 16, 2, rank=None).to(dev)
    with torch.no_grad(): full.C.copy_(T); full.bias.copy_(mx.bias)
    H = torch.randn(50, 6, 16, generator=g).to(dev); assert float((mx(H) - full(H)).abs().max()) < 1e-12
    say("[3] rank algebra ok")

    # 4. range tracking puts the data inside the collar
    l0 = blk.layers[0]; u = l0.tracker(l0.project(X)); assert float(u.abs().max()) <= 1 / 1.25 + 1e-9
    say(f"[4] range tracking: max |u| = {float(u.abs().max()):.3f} (collar 1/1.25 = 0.8)")

    # 5. a short two-layer fit on g(rho^2) with VarPro then GN (quick; checks the training path end to end)
    a = torch.tensor([0.2, 0.1, 0.1], device=dev); Yf = torch.exp(-((X - a) ** 2).sum(1) / 0.25)
    Xt = (2 * torch.rand(2000, d, generator=g) - 1).to(dev) * 0.27; Yt = torch.exp(-((Xt - a) ** 2).sum(1) / 0.25)
    net2 = QINet([RidgeQILayer(d, 1, 16, M=6, dirs="learned", rank=None), RidgeQILayer(1, 1, 64, dirs=None, rank=None)]).to(dev)
    log = fit(net2, X, Yf, Xt, Yt, mode="varpro", steps=200, log_every=50); gl = gauss_newton(net2, X, Yf, Xt, Yt, iters=5)
    say(f"[5] two-layer fit: varpro {log['final']:.2e} -> GN {gl['final']:.2e}  ({net2.counts()})"); assert gl["final"] < log["final"] * 1.01 + 1e-12

    # 6. FFN drop-in shape, counts, dense comparison, benchmark
    ffn = QIFFN(64, M=32, N1=8, K=32, N2=8, rank=4).to(dev); xb = torch.randn(256, 10, 64, generator=g).to(dev)
    out = ffn(xb); assert out.shape == xb.shape
    dense = dense_ffn(64, 256).to(dev)
    e_q, _ = bench(ffn, xb, iters=5); e_d, _ = bench(dense, xb, iters=5)
    say(f"[6] QIFFN {ffn.counts()} vs dense FFN {sum(p.numel() for p in dense.parameters())} params: {e_q:.1f} ms vs {e_d:.1f} ms per fwd+bwd on {dev}")

    # 7. KAN layer (identity directions) and export
    kan = QINet([RidgeQILayer(d, 4, 16, dirs=None, rank=None), RidgeQILayer(4, 1, 32, dirs=None, rank=None)]).to(dev)
    kan.update_ranges(X); kan.solve_readout(X, Yf); pack = export_pack(kan, X[:100], Yf[:100])
    say(f"[7] KAN-style stack solved: train rel L2 {rel_l2(kan(X)[:, 0], Yf):.2e}; export keys {list(pack)}")
    say("selftest OK")

if __name__ == "__main__":
    selftest()
