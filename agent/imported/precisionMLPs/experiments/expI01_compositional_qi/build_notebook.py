"""Builds compositional_qi_colab.ipynb (self-contained; no repo imports).

Run:  uv run --extra dev python experiments/expI01_compositional_qi/build_notebook.py
Smoke: uv run --extra dev python experiments/expI01_compositional_qi/build_notebook.py --smoke
       (executes every code cell in-process with QUICK=True; CPU, one thread; output to a scratch dir)
"""
import json, sys, pathlib, tempfile

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "compositional_qi_colab.ipynb"

CELLS = []  # (kind, source)


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


# ============================================================================
md(r"""
# expI01: the compositional ridge-QI block (checkpoint I, depth theory)

Self-contained Colab notebook. It builds the two-level model
$\hat f(x)=b_0+\sum_{k=1}^{K}\hat g_k(\hat P_k(x))$, $\hat P_k(x)=b_k+\sum_{r,j}C_{rjk}\tanh(\gamma_1[v_r^\top(x-x_0)-c_j])$,
with the KAT-style rank-1 tie $C_{rjk}=a_r\Phi_{jk}$ by default, and compares it against the shallow ridge-QI model (checkpoint H)
and a plain tanh MLP at matched parameter count and at matched FLOPs, on targets from $d=2$ up to $d=8$.

**Structural constraints, never penalties:** unit directions (normalized in the forward pass); fixed level-1 centers on the band $[-T,T]$,
$T=1.25\,r_{\rm data}$, $\gamma_1=\lambda/h_1$; fixed level-2 centers on $[-1,1]$ in a per-channel normalized coordinate $u=(P_k-m_k)/s_k$
whose affine map is re-read from the data (range tracking), $\gamma_2=\lambda/h_2$; $\lambda=0.25$ (the tanh aliasing rule); the final readout is solved by truncated SVD.

**Runtime.** GPU (A100). Everything is float64: this is a machine-precision study. TPUs cannot run float64 and are refused.

**Nothing is lost if compute runs out.** Every finished run is appended to `results.json` immediately (on Google Drive if `MOUNT_DRIVE`),
every figure is rebuilt from that file after each target and saved, and re-running a cell skips runs that are already in the file.
Experiments are ordered by value per minute: record checks, the oracle, the headline dimension sweep, then the arm comparison, the knob sweeps, the control, real data.
`BUDGET_MIN` stops the loops gracefully when a wall-clock budget is exceeded.

Plan and checklist: `results/checkpoint_I_depth_theory/expI01_compositional_qi/PLAN.md`. Theory: `results/checkpoint_I_depth_theory/compositional_qi_theory.md`.
""")

code(r"""
import math, time, json, os, sys
import numpy as np
import torch, torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
try:
    from IPython.display import clear_output, display
except Exception:
    clear_output = lambda wait=True: None

torch.set_default_dtype(torch.float64)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if "torch_xla" in sys.modules or os.environ.get("COLAB_TPU_ADDR"):
    raise RuntimeError("TPU runtime detected: float64 is required. Switch to a GPU runtime.")
print("device:", DEV, torch.__version__)

# ---- master switches -------------------------------------------------------
QUICK       = False   # True: tiny sizes and few steps, only to check that the notebook runs end to end
MOUNT_DRIVE = True    # persist results.json and figures to Google Drive (survives a disconnect)
BUDGET_MIN  = None    # e.g. 240: stop starting new runs after this many minutes (finished runs are kept)
RUN_GN      = True    # Gauss-Newton (variable projection) after VarPro
SEEDS       = [0]     # add seeds before claiming any factor under 10x
STEPS       = 300 if QUICK else 4000
GN_IT       = 5 if QUICK else 40

OUTDIR = "expI01_out"
if MOUNT_DRIVE and not QUICK:
    try:
        from google.colab import drive
        drive.mount("/content/drive"); OUTDIR = "/content/drive/MyDrive/expI01_out"
    except Exception as e:
        print("Drive not mounted (", e, "); results stay on the local disk")
os.makedirs(OUTDIR, exist_ok=True)
RES_PATH = f"{OUTDIR}/results.json"
RESULTS = json.load(open(RES_PATH)) if os.path.exists(RES_PATH) else {}
print("results file:", RES_PATH, "| experiments already present:", {k: len(v) for k, v in RESULTS.items()})

def save():
    tmp = RES_PATH + ".tmp"
    with open(tmp, "w") as f: json.dump(RESULTS, f, default=str)
    os.replace(tmp, RES_PATH)
def done(exp, key): return key in RESULTS.get(exp, {})
def put(exp, key, val): RESULTS.setdefault(exp, {})[key] = val; save()
T_START = time.time()
def budget_ok():
    ok = BUDGET_MIN is None or (time.time() - T_START) / 60 < BUDGET_MIN
    if not ok: print("wall-clock budget reached; skipping remaining runs (everything finished is saved)")
    return ok

LAM    = 0.25      # tanh bandwidth of record (expC03 basin, expC07 aliasing rule)
COLLAR = 1.25      # band = 1.25 x data radius (expH01: load-bearing)
RCOND  = 1e-14     # truncated-SVD threshold for wide redundant dictionaries (expH06)

def fp64_speed(reps=10):
    a = torch.randn(2048, 2048, device=DEV); (a @ a)                       # warm-up: the first call pays for CUDA/cuBLAS init
    if DEV.type == "cuda": torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps): (a @ a)
    if DEV.type == "cuda": torch.cuda.synchronize()
    return (time.time() - t) / reps
print(f"fp64 2048^2 matmul: {fp64_speed()*1e3:.1f} ms on {torch.cuda.get_device_name() if DEV.type == 'cuda' else 'CPU'}  (A100: ~1 ms; L4: ~35 ms; T4: ~70 ms)")
""")

md(r"""
## 1. The bandwidth $\lambda$ and the solver

$\lambda=\gamma h$ is set by the activation's first Poisson alias, $A(\lambda)=|\widehat K(2\pi/\lambda)|/|\widehat K(0)|$ with $K=\mathrm{sech}^2$ for tanh,
$A(\lambda)=\frac{\pi^2/\lambda}{\sinh(\pi^2/\lambda)}$; at $\lambda=0.25$ it is $5.7\times10^{-16}$, machine epsilon (expC07, approved).
Both levels use $\gamma=\lambda/h$ with $h$ the center spacing at that level. The readout solve is QR followed by SVD of the triangular factor,
singular values below `RCOND`$\cdot s_{\max}$ dropped; never normal equations (expA01).
""")

code(r"""
def alias_tanh(lam):
    x = math.pi ** 2 / lam
    return x / math.sinh(x)
assert abs(alias_tanh(0.25) - 5.65e-16) < 1e-17, alias_tanh(0.25)
print("A_tanh(0.25) =", alias_tanh(0.25), " A_tanh(0.30) =", alias_tanh(0.30))

def tsvd_solve(A, y, rcond=None):
    rcond = RCOND if rcond is None else rcond          # RCOND is a global so the real-data cell can raise it
    Q, R = torch.linalg.qr(A)
    rhs = Q.T @ y
    U, s, Vh = torch.linalg.svd(R, full_matrices=False)
    keep = s > rcond * s[0]
    z = U[:, keep].T @ rhs
    z = z / s[keep] if z.dim() == 1 else z / s[keep][:, None]
    return Vh[keep].T @ z, int(keep.sum())

def lstsq_bias(F, y, rcond=None):
    A = torch.cat([F, torch.ones(F.shape[0], 1, device=F.device)], 1)
    w, rank = tsvd_solve(A, y, rcond)
    return w[:-1], w[-1], rank

def rel_l2(pred, y):
    return float((pred - y).norm() / y.norm())
""")

md(r"""
## 2. Data and targets

Data uniform in the ball of radius $r$ about $x_0$ (expH05/H06 convention); test on the inner ball of radius $0.9r$, so the collar is never scored.
In $d=2$ the anchors are exactly expH05's and in $d=3$ exactly expH06's, so the shallow arm can be checked against the record; for other $d$ they are generated deterministically.

Targets and what the theory says their shortest factorization is:

| target | form | $d$ | why it is here |
|---|---|---|---|
| gauss bump, radial Runge, fast waves | $g(\rho_a^2)$, one quadratic channel | all | the record's 3-D set; single layer needs 9k units in 3-D and hits a wall in 4-D |
| composition | $\exp(\text{ridge sum})$ | all | different profiles on different directions |
| product peak (Genz) | $\exp\sum_k\log\frac1{1+a_k^2(x_k-b_k)^2}$ | all | a product: one channel of $d$ axis profiles, sharp in every coordinate |
| three bumps | three quadratic channels | all | $K=3$ needed |
| packet | two quadratic channels | 2, 3 | two anchors |
| product sines | four exact ridges | 3 | no dividend: shallow is the shortest form |
| random ridges | 24 random ridges | 3, 5 | control: the block must not win |
""")

code(r"""
def anchors(d):
    if d == 2:
        x0, a1, a2 = [0.35, -0.25], [0.2, 0.1], [0.3, -0.2]                              # expH05
    elif d == 3:
        x0, a1, a2 = [0.35, -0.25, 0.2], [0.2, 0.1, 0.1], [0.3, -0.2, 0.15]               # expH06
    else:
        i = np.arange(d)
        x0 = 0.3 * np.cos(1.3 * i + 0.4); a1 = x0 + 0.15 * np.sin(2.1 * i + 1.0); a2 = x0 + 0.08 * np.cos(0.7 * i + 2.0)
    B = [[.30, -.20, .25, -.15, .10, .20, -.10, .15], [-.40, .35, -.10, .20, -.25, -.15, .30, -.05], [.15, .10, -.30, .05, .30, -.20, .10, .25]]
    t = lambda v: torch.tensor(np.asarray(v, dtype=np.float64)[:d], device=DEV)
    return dict(x0=t(x0), a1=t(a1), a2=t(a2), b1=t(B[0]), b2=t(B[1]), b3=t(B[2]))

def radius(d): return 0.4 if d == 2 else 0.3

def ball(n, d, r, seed, inner=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(n, d, generator=g); u = u / u.norm(dim=1, keepdim=True)
    rad = r * inner * torch.rand(n, 1, generator=g) ** (1.0 / d)
    return (anchors(d)["x0"].cpu() + rad * u).to(DEV)

def rho(x, a): return (x - a).norm(dim=1) / math.sqrt(2.0)

_RIDGE_CACHE = {}
def random_ridges(x, J=24):
    d = x.shape[1]
    if (d, J) not in _RIDGE_CACHE:
        g = torch.Generator(device="cpu").manual_seed(1234)
        U = torch.randn(J, d, generator=g); U = U / U.norm(dim=1, keepdim=True)
        w = 8 + 12 * torch.rand(J, generator=g); ph = 2 * math.pi * torch.rand(J, generator=g)
        _RIDGE_CACHE[(d, J)] = (U.to(DEV), w.to(DEV), ph.to(DEV))
    U, w, ph = _RIDGE_CACHE[(d, J)]
    return torch.sin(w * (x @ U.T) + ph).sum(1) / math.sqrt(J)

GENZ_A = [3, 5, 8, 12, 6, 4, 7, 9]; GENZ_B = [.35, -.25, .10, -.40, .25, .15, -.30, .05]

def target(name, x):
    d = x.shape[1]; A = anchors(d)
    if name == "gauss_bump":    return torch.exp(-((x - A["a1"]) ** 2).sum(1) / 0.5 ** 2)
    if name == "radial_runge":  return 1.0 / (1.0 + 16.0 * rho(x, A["a2"]) ** 2)
    if name == "fast_waves":    return torch.cos(6 * math.pi * rho(x, A["a2"]))
    if name == "composition":
        if d == 2: z = torch.sin(math.pi * x[:, 0]) * torch.cos(math.pi * x[:, 1])
        elif d == 3: z = torch.sin(math.pi * x[:, 0]) * torch.cos(math.pi * x[:, 1]) + 0.5 * torch.cos(math.pi * x[:, 2] + 1.0)
        else: z = (2.0 / d) * (torch.sin(math.pi * x) * torch.cos(math.pi * x.roll(-1, 1))).sum(1)
        return torch.exp(z)
    if name == "product_peak":
        a = torch.tensor(GENZ_A[:d], device=DEV); b = torch.tensor(GENZ_B[:d], device=DEV)
        return torch.prod(1.0 / (1.0 + (a * (x - b)) ** 2), dim=1) * (2.0 ** d)
    if name == "three_bumps":
        return (torch.exp(-((x - A["b1"]) ** 2).sum(1) / (2 * 0.5 ** 2)) + 0.7 * torch.exp(-((x - A["b2"]) ** 2).sum(1) / (2 * 0.3 ** 2))
                + 0.5 * torch.exp(-((x - A["b3"]) ** 2).sum(1) / (2 * 0.2 ** 2)))
    if name == "packet":
        r0 = rho(x, A["x0"])
        return 0.8 * torch.exp(-(r0 / 0.18) ** 2) * torch.cos(10 * math.pi * r0) + torch.exp(-((x - A["a1"]) ** 2).sum(1) / (2 * 0.5 ** 2))
    if name == "product_sines": return torch.prod(torch.sin(2 * math.pi * x), dim=1)
    if name == "random_ridges": return random_ridges(x)
    if name == "sin2pi":        return torch.sin(2 * math.pi * x[:, 0])
    raise KeyError(name)

def make_data(name, d, n_train, n_test=10000, seed=0):
    r = radius(d); Xtr = ball(n_train, d, r, seed); Xte = ball(n_test, d, r, seed + 777, inner=0.9)
    x0 = anchors(d)["x0"]
    return dict(Ztr=Xtr - x0, ytr=target(name, Xtr), Zte=Xte - x0, yte=target(name, Xte), r=r, d=d, name=name)
""")

md(r"""
## 3. Models

All three expose the same interface: `feats(Z)` (what the final readout sees), `readout()`, `regrid(Z)` (range tracking; a no-op except for the
compositional block), `nonlinear_params()`, `counts()` (parameters, tanh units, FLOPs per sample).

* `CompQI`: the compositional block. `rank=1` is the KAT tie $C_{rjk}=a_r\Phi_{jk}$; `rank=S` the low-rank family $\sum_s a^{(s)}_r\Phi^{(s)}_{jk}$;
  `rank=None` the untied tensor. `shared_outer=True` shares one outer function across channels (KAT).
* `ShallowQI`: one ridge-QI layer, the checkpoint-H model, directions learnable.
* `MLP`: plain two-hidden-layer tanh network, standard PyTorch init, last layer solvable.
""")

code(r"""
def cell_centers(N, T):
    h = 2 * T / N
    return -T + (torch.arange(N, device=DEV) + 0.5) * h, h

class CompQI(nn.Module):
    def __init__(self, d, M, N1, K, N2, T, rank=1, shared_outer=False, fixed_dirs=None, seed=0):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.d, self.M, self.N1, self.K, self.N2, self.rank, self.shared_outer = d, M, N1, K, N2, rank, shared_outer
        c1, h1 = cell_centers(N1, T); c2, h2 = cell_centers(N2, 1.0)
        self.register_buffer("c1", c1); self.gamma1 = LAM / h1          # level 1: band [-T, T]
        self.register_buffer("c2", c2); self.gamma2 = LAM / h2          # level 2: normalized coordinate u in [-1, 1]
        V = torch.randn(M, d, generator=g) if fixed_dirs is None else torch.as_tensor(fixed_dirs).cpu().clone()
        V = V / V.norm(dim=1, keepdim=True)
        self.fixed_dirs = fixed_dirs is not None
        if self.fixed_dirs: self.register_buffer("V_raw", V.to(DEV))
        else: self.V_raw = nn.Parameter(V.to(DEV))
        if rank is None:
            self.C = nn.Parameter(torch.zeros(M, N1, K, device=DEV))
        else:
            self.a = nn.Parameter((torch.randn(M, rank, generator=g) / math.sqrt(M)).to(DEV))
            self.Phi = nn.Parameter(torch.zeros(rank, N1, K, device=DEV))
        self.b_z = nn.Parameter((0.1 * torch.randn(K, generator=g)).to(DEV))   # distinct per channel: breaks the symmetry at Phi = 0
        self.register_buffer("cm", torch.zeros(K, device=DEV)); self.register_buffer("cs", torch.ones(K, device=DEV))
        self.Psi = nn.Parameter(torch.zeros(N2 if shared_outer else K * N2, device=DEV))
        self.b0 = nn.Parameter(torch.zeros((), device=DEV))

    def dirs(self): return self.V_raw / self.V_raw.norm(dim=1, keepdim=True)
    def feats1(self, Z):
        proj = Z @ self.dirs().T                                          # [B, M]
        return torch.tanh(self.gamma1 * (proj[:, :, None] - self.c1))     # [B, M, N1]
    def channels(self, Z):
        H1 = self.feats1(Z)
        if self.rank is None: P = torch.einsum("brj,rjk->bk", H1, self.C)
        else: P = torch.einsum("bsj,sjk->bk", torch.einsum("brj,rs->bsj", H1, self.a), self.Phi)
        return P + self.b_z
    def feats(self, Z):
        u = (self.channels(Z) - self.cm) / self.cs
        H2 = torch.tanh(self.gamma2 * (u[:, :, None] - self.c2))          # [B, K, N2]
        return H2.sum(1) if self.shared_outer else H2.reshape(Z.shape[0], -1)
    def forward(self, Z): return self.feats(Z) @ self.Psi + self.b0
    @torch.no_grad()
    def regrid(self, Z):
        P = self.channels(Z); lo, hi = P.min(0).values, P.max(0).values
        if self.shared_outer: lo, hi = lo.min().expand_as(lo), hi.max().expand_as(hi)
        self.cm.copy_((lo + hi) / 2); self.cs.copy_((COLLAR * (hi - lo) / 2).clamp_min(1e-3))
    def readout(self): return [self.Psi, self.b0]
    def nonlinear_params(self):
        ps = [] if self.fixed_dirs else [self.V_raw]
        ps += [self.C] if self.rank is None else [self.a, self.Phi]
        return ps + [self.b_z]
    def counts(self):
        d, M, N1, K, N2, S = self.d, self.M, self.N1, self.K, self.N2, self.rank
        contr = M * N1 * K if S is None else M * N1 * S + S * N1 * K
        return dict(params=sum(p.numel() for p in self.parameters()), units=M * N1 + K * N2,
                    flops=M * d + M * N1 + contr + K * N2 + (N2 if self.shared_outer else K * N2))

class ShallowQI(nn.Module):
    def __init__(self, d, M, N1, T, fixed_dirs=None, seed=0):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.d, self.M, self.N1 = d, M, N1
        c1, h1 = cell_centers(N1, T); self.register_buffer("c1", c1); self.gamma1 = LAM / h1
        V = torch.randn(M, d, generator=g) if fixed_dirs is None else torch.as_tensor(fixed_dirs).cpu().clone()
        V = V / V.norm(dim=1, keepdim=True); self.fixed_dirs = fixed_dirs is not None
        if self.fixed_dirs: self.register_buffer("V_raw", V.to(DEV))
        else: self.V_raw = nn.Parameter(V.to(DEV))
        self.W = nn.Parameter(torch.zeros(M * N1, device=DEV)); self.b0 = nn.Parameter(torch.zeros((), device=DEV))
    def dirs(self): return self.V_raw / self.V_raw.norm(dim=1, keepdim=True)
    def feats(self, Z):
        proj = Z @ self.dirs().T
        return torch.tanh(self.gamma1 * (proj[:, :, None] - self.c1)).reshape(Z.shape[0], -1)
    def forward(self, Z): return self.feats(Z) @ self.W + self.b0
    def regrid(self, Z): pass
    def readout(self): return [self.W, self.b0]
    def nonlinear_params(self): return [] if self.fixed_dirs else [self.V_raw]
    def counts(self):
        return dict(params=sum(p.numel() for p in self.parameters()), units=self.M * self.N1, flops=self.M * self.d + 2 * self.M * self.N1)

class MLP(nn.Module):
    def __init__(self, d, w, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.d, self.w = d, w
        self.l1 = nn.Linear(d, w); self.l2 = nn.Linear(w, w)
        self.W = nn.Parameter(torch.zeros(w)); self.b0 = nn.Parameter(torch.zeros(()))
        nn.init.uniform_(self.W, -1 / math.sqrt(w), 1 / math.sqrt(w))
        self.to(DEV)
    def feats(self, Z): return torch.tanh(self.l2(torch.tanh(self.l1(Z))))
    def forward(self, Z): return self.feats(Z) @ self.W + self.b0
    def regrid(self, Z): pass
    def readout(self): return [self.W, self.b0]
    def nonlinear_params(self): return list(self.l1.parameters()) + list(self.l2.parameters())
    def counts(self):
        d, w = self.d, self.w
        return dict(params=sum(p.numel() for p in self.parameters()), units=2 * w, flops=d * w + w * w + 3 * w)

def mlp_width_for(target_value, d, key):
    w = 1
    while MLP(d, w).counts()[key] < target_value: w += 1
    return w

@torch.no_grad()
def solve_readout(model, Z, y):
    w, b, rank = lstsq_bias(model.feats(Z), y)
    model.readout()[0].copy_(w); model.readout()[1].copy_(b)
    return rank

@torch.no_grad()
def refit_eval(model, Ztr, ytr, Zte, yte):
    W, b = model.readout(); W0, b0 = W.clone(), b.clone()
    solve_readout(model, Ztr, ytr); e = rel_l2(model(Zte), yte)
    W.copy_(W0); b.copy_(b0)
    return e
""")

md(r"""
## 4. Training arms

* `adam`: Adam on every learnable tensor (directions, channel coefficients, readout), zero-initialized function coefficients, structural constraints only; then one final readout solve.
* `varpro`: Adam on the nonlinear parameters only; the readout is re-solved by truncated SVD after every step (never a gradient parameter).
* `gn`: variable-projection Gauss-Newton (Levenberg-Marquardt) on the nonlinear parameters with the Kaufman Jacobian $J=-P^\perp(\partial A/\partial\vartheta)\Psi^\star$,
  readout re-solved at every trial (the expH06 polish, generalized). Run after `varpro`.

Range tracking (`regrid_every`) re-reads the channel ranges from the training data on a fixed schedule and once more before the final solve; identical in every arm.
Full batch throughout (a precision study; the record's optimizer work is full-batch).
""")

code(r"""
def train(model, D, mode="adam", steps=STEPS, lr=5e-3, regrid_every=250, log_every=100, final_lstsq=True, warmup=50, verbose=False):
    Ztr, ytr, Zte, yte = D["Ztr"], D["ytr"], D["Zte"], D["yte"]
    nl = model.nonlinear_params()
    with torch.no_grad(): model.readout()[1].fill_(ytr.mean())
    if mode == "adam": opt = torch.optim.Adam(nl + model.readout(), lr=lr)
    elif mode == "varpro":
        opt = torch.optim.Adam(nl, lr=lr) if nl else None; solve_readout(model, Ztr, ytr)
    else: raise ValueError(mode)
    lr_at = lambda t: lr * min(1.0, (t + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * t / steps))
    log = dict(step=[], train=[], test=[], refit=[]); t0 = time.time()
    for t in range(steps):
        if opt is None: break
        if t > 0 and t % regrid_every == 0:
            model.regrid(Ztr)
            if mode == "varpro": solve_readout(model, Ztr, ytr)
        for gp in opt.param_groups: gp["lr"] = lr_at(t)
        opt.zero_grad(set_to_none=True)
        loss = ((model(Ztr) - ytr) ** 2).mean(); loss.backward(); opt.step()
        if mode == "varpro": solve_readout(model, Ztr, ytr)
        if t % log_every == 0 or t == steps - 1:
            with torch.no_grad(): te = rel_l2(model(Zte), yte)
            rf = refit_eval(model, Ztr, ytr, Zte, yte)
            log["step"].append(t); log["train"].append(float(loss.sqrt() * math.sqrt(len(ytr)) / ytr.norm())); log["test"].append(te); log["refit"].append(rf)
            if verbose and (t % (5 * log_every) == 0 or t == steps - 1): print(f"  step {t:5d}  train {log['train'][-1]:.2e}  test {te:.2e}  refit {rf:.2e}")
            elif t > 0 and t % 1000 == 0: print(f"      ... step {t}/{steps}  refit {rf:.2e}  [{time.time() - t0:.0f}s]", flush=True)
    model.regrid(Ztr)
    if final_lstsq: solve_readout(model, Ztr, ytr)
    with torch.no_grad(): log["final"] = rel_l2(model(Zte), yte)
    log["time"] = time.time() - t0
    return log

def _flat(ps): return torch.cat([p.reshape(-1) for p in ps])
def _unflat(vec, ps):
    out, i = [], 0
    for p in ps: out.append(vec[i:i + p.numel()].reshape(p.shape)); i += p.numel()
    return out

def gauss_newton(model, D, iters=GN_IT, mu=1e-2, chunk=None, verbose=False):
    from torch.func import functional_call, jvp, vmap
    Ztr, ytr, Zte, yte = D["Ztr"], D["ytr"], D["Zte"], D["yte"]
    names = [n for n, p in model.named_parameters() if any(p is q for q in model.nonlinear_params())]
    ps = [dict(model.named_parameters())[n] for n in names]
    if not ps: return dict(final=refit_eval(model, Ztr, ytr, Zte, yte), step=[], refit=[], time=0.0)
    theta = _flat(ps).detach().clone(); P = theta.numel()
    chunk = chunk or (64 if DEV.type == "cuda" else 8)
    def set_theta(th):
        with torch.no_grad():
            for p, v in zip(ps, _unflat(th, ps)): p.copy_(v)
    def resid(th): return functional_call(model, dict(zip(names, _unflat(th, ps))), (Ztr,)) - ytr
    def jacobian(th):
        E = torch.eye(P, device=DEV)
        return torch.cat([vmap(lambda e: jvp(resid, (th,), (e,))[1])(E[i:i + chunk]) for i in range(0, P, chunk)], 0).T
    log = dict(step=[], refit=[]); t0 = time.time()
    model.regrid(Ztr); solve_readout(model, Ztr, ytr)
    with torch.no_grad(): r = resid(theta); cost = float(r @ r)
    for it in range(iters):
        with torch.no_grad():
            J0 = jacobian(theta)
            A = torch.cat([model.feats(Ztr), torch.ones(len(ytr), 1, device=DEV)], 1)
            Qa, R = torch.linalg.qr(A); Ur, s, _ = torch.linalg.svd(R, full_matrices=False); Q = Qa @ Ur[:, s > RCOND * s[0]]
            J = J0 - Q @ (Q.T @ J0)                                       # Kaufman: project off the RETAINED readout span (same truncation as tsvd_solve)
            g = J.T @ r; H = J.T @ J; accepted = False
            for _ in range(8):
                delta = torch.linalg.solve(H + mu * torch.diag(H.diagonal().clamp_min(1e-12)), -g)
                set_theta(theta + delta); solve_readout(model, Ztr, ytr)
                r_new = resid(theta + delta); c_new = float(r_new @ r_new)
                if c_new < cost: theta, r, cost, mu, accepted = theta + delta, r_new, c_new, mu / 3, True; break
                mu *= 10
            if not accepted: set_theta(theta); solve_readout(model, Ztr, ytr)
            model.regrid(Ztr); solve_readout(model, Ztr, ytr); r = resid(theta); cost = float(r @ r)
            te = rel_l2(model(Zte), yte)
        log["step"].append(it); log["refit"].append(te)
        if verbose: print(f"  GN {it:3d}  train {math.sqrt(cost)/float(ytr.norm()):.2e}  test {te:.2e}  mu {mu:.1e}")
        if not accepted and mu > 1e8: break
    log["final"] = log["refit"][-1] if log["refit"] else float("nan"); log["time"] = time.time() - t0
    return log

# ---- one-call runners; every result is a plain dict that goes straight into RESULTS ----
def base_cfg(d): return dict(M=max(6, 2 * d), N1=32, K=2, N2=64)

def run_comp(name, d, cfg, rank=1, mode="varpro", seed=0, gn=True, steps=STEPS, verbose=False):
    n_train = max(4096, 8 * (cfg["M"] * cfg["N1"] + cfg["K"] * cfg["N2"]))
    D = make_data(name, d, n_train, seed=seed)
    m = CompQI(d, cfg["M"], cfg["N1"], cfg["K"], cfg["N2"], COLLAR * D["r"], rank=rank, seed=seed).to(DEV)
    log = train(m, D, mode=mode, steps=steps, verbose=verbose)
    out = dict(name=name, d=d, model="comp", rank=str(rank), mode=mode, seed=seed, cfg=cfg, final=log["final"], curve=[log["step"], log["refit"]], time=log["time"], **m.counts())
    if gn and RUN_GN:
        gl = gauss_newton(m, D, verbose=verbose); out.update(final_gn=gl["final"], gn_curve=[gl["step"], gl["refit"]]); out["time"] += gl["time"]
    out["best"] = min(out["final"], out.get("final_gn", float("inf")))
    return out

def run_shallow(name, d, M, N1, mode="varpro", seed=0, gn=True, steps=STEPS):
    D = make_data(name, d, max(4096, 8 * M * N1), seed=seed)
    m = ShallowQI(d, M, N1, COLLAR * D["r"], seed=seed).to(DEV)
    log = train(m, D, mode=mode, steps=steps)
    out = dict(name=name, d=d, model="shallow", mode=mode, seed=seed, M=M, N1=N1, final=log["final"], curve=[log["step"], log["refit"]], time=log["time"], **m.counts())
    if gn and RUN_GN:
        gl = gauss_newton(m, D); out.update(final_gn=gl["final"], gn_curve=[gl["step"], gl["refit"]]); out["time"] += gl["time"]
    out["best"] = min(out["final"], out.get("final_gn", float("inf")))
    return out

def run_mlp(name, d, w, seed=0, steps=STEPS, lr=3e-3):
    D = make_data(name, d, 4096, seed=seed)
    m = MLP(d, w, seed=seed); log = train(m, D, mode="adam", steps=steps, lr=lr)
    return dict(name=name, d=d, model="mlp", seed=seed, w=w, final=log["final"], best=log["final"], curve=[log["step"], log["refit"]], time=log["time"], **m.counts())

def show(fig, fname, table=None):
    fig.savefig(f"{OUTDIR}/{fname}.png", dpi=120, bbox_inches="tight")
    clear_output(wait=True)
    if table: print(table)
    plt.show(); plt.close(fig)
""")

md(r"""
## 5. Experiment 0: reproduce the record before trusting anything

E0a: the 1-D block (one direction, collar band, fixed centers, solved readout) on $\sin2\pi x$ over $[-1,1]$ must reach $\lesssim10^{-12}$.
E0b: the shallow model with expH05's even directions on expH05's data ball ($r=0.4$, 128 offsets) must reproduce the recorded cliff: fast waves and gauss below $10^{-10}$ at $M=12$, radial Runge at $M=16$.
If either fails, nothing downstream is trustworthy.
""")

code(r"""
def even_dirs_2d(M):
    th = math.pi / (2 * M) + torch.arange(M, device=DEV) * math.pi / M
    return torch.stack([torch.cos(th), torch.sin(th)], 1)

_g = torch.Generator().manual_seed(0)
_Ztr = (2 * torch.rand(4096, 1, generator=_g) - 1).to(DEV); _Zte = (1.8 * torch.rand(4096, 1, generator=_g) - 0.9).to(DEV)
D1 = dict(Ztr=_Ztr, ytr=target("sin2pi", _Ztr), Zte=_Zte, yte=target("sin2pi", _Zte))   # [-1, 1], scored on [-0.9, 0.9]
m1 = ShallowQI(1, 1, 64 if QUICK else 128, COLLAR * 1.0, fixed_dirs=torch.ones(1, 1)).to(DEV)
solve_readout(m1, D1["Ztr"], D1["ytr"]); e0a = rel_l2(m1(D1["Zte"]), D1["yte"])
print(f"E0a  1-D block on sin(2 pi x), N1={m1.N1}: rel L2 = {e0a:.2e}")
put("E0", "1d_sin2pi", e0a)
assert e0a < 1e-9, "1-D block failed to reach the floor: check lambda, centers, solver"

for name in ["fast_waves", "gauss_bump", "radial_runge"]:
    for M in ([6, 12] if QUICK else [4, 8, 12, 16]):
        key = f"h05_{name}_M{M}"
        if done("E0", key): continue
        D = make_data(name, 2, 8 * M * 128)
        m = ShallowQI(2, M, 128, COLLAR * 0.4, fixed_dirs=even_dirs_2d(M)).to(DEV)
        solve_readout(m, D["Ztr"], D["ytr"]); e = rel_l2(m(D["Zte"]), D["yte"]); put("E0", key, e)
        print(f"E0b  {name:13s} M={M:3d}  rel L2 = {e:.2e}")
E0 = RESULTS["E0"]
if not QUICK:
    assert E0["h05_fast_waves_M12"] < 1e-10 and E0["h05_gauss_bump_M12"] < 1e-10 and E0["h05_radial_runge_M16"] < 1e-10, "expH05 cliff not reproduced"
print("E0 passed")
""")

md(r"""
## 6. Experiment 1: the oracle composition (does the dividend exist, and where is the composed floor)

Factorization given: quadratic channel on the coordinate axes for the radial targets, the composition's ridge directions, the Genz product's axis profiles.
The channel readout is solved against the known $P$ (untied $C$, fixed directions), then regrid, then the outer readout is solved. Nothing is learned.
Sweep $N_1$ and $N_2$; $d\in\{2,3,4,5,8\}$. The figure is rebuilt after every target.
""")

code(r"""
def oracle_dirs_and_P(name, d):
    A = anchors(d)
    if name in ("gauss_bump", "radial_runge", "fast_waves"):
        anc = "a1" if name == "gauss_bump" else "a2"
        return torch.eye(d, device=DEV), (lambda Z: ((Z + A["x0"] - A[anc]) ** 2).sum(1) / 2.0)
    if name == "product_peak":
        a = torch.tensor(GENZ_A[:d], device=DEV); b = torch.tensor(GENZ_B[:d], device=DEV)
        return torch.eye(d, device=DEV), (lambda Z: -torch.log1p((a * (Z + A["x0"] - b)) ** 2).sum(1))
    if name == "composition":
        if d == 2: V = torch.tensor([[1., 1.], [1., -1.]], device=DEV)
        elif d == 3: V = torch.tensor([[1., 1., 0.], [1., -1., 0.], [0., 0., 1.]], device=DEV)
        else:
            V = torch.zeros(2 * d, d, device=DEV)
            for i in range(d): V[2 * i, i] = 1; V[2 * i, (i + 1) % d] = 1; V[2 * i + 1, i] = 1; V[2 * i + 1, (i + 1) % d] = -1
        V = V / V.norm(dim=1, keepdim=True)
        def Pfn(Z):
            x = Z + A["x0"]
            if d == 2: return torch.sin(math.pi * x[:, 0]) * torch.cos(math.pi * x[:, 1])
            if d == 3: return torch.sin(math.pi * x[:, 0]) * torch.cos(math.pi * x[:, 1]) + 0.5 * torch.cos(math.pi * x[:, 2] + 1.0)
            return (2.0 / d) * (torch.sin(math.pi * x) * torch.cos(math.pi * x.roll(-1, 1))).sum(1)
        return V, Pfn
    raise KeyError(name)

@torch.no_grad()
def oracle_fit(name, d, N1, N2, seed=0):
    V, Pfn = oracle_dirs_and_P(name, d)
    D = make_data(name, d, max(4096, 16 * (V.shape[0] * N1 + N2)), seed=seed)
    m = CompQI(d, V.shape[0], N1, 1, N2, COLLAR * D["r"], rank=None, fixed_dirs=V).to(DEV)
    H1 = m.feats1(D["Ztr"]).reshape(len(D["ytr"]), -1)
    c, b, _ = lstsq_bias(H1, Pfn(D["Ztr"]))
    m.C.copy_(c.reshape(m.M, N1, 1)); m.b_z.copy_(b.reshape(1))
    chan_err = rel_l2(m.channels(D["Zte"]).squeeze(1), Pfn(D["Zte"]))
    m.regrid(D["Ztr"]); solve_readout(m, D["Ztr"], D["ytr"])
    return dict(err=rel_l2(m(D["Zte"]), D["yte"]), chan_err=chan_err, **m.counts())

E1_TARGETS = ["gauss_bump", "radial_runge", "fast_waves", "composition", "product_peak"]
E1_DIMS = [3] if QUICK else [2, 3, 4, 5, 8]
E1_N1 = [8, 16] if QUICK else [8, 16, 32, 64]
E1_N2 = [32, 64] if QUICK else [32, 64, 128, 256]

def fig_E1():
    E1 = RESULTS.get("E1", {})
    fig, axes = plt.subplots(len(E1_DIMS), len(E1_TARGETS), figsize=(3.8 * len(E1_TARGETS), 3.3 * len(E1_DIMS)), squeeze=False, sharey=True)
    for i, d in enumerate(E1_DIMS):
        for j, name in enumerate(E1_TARGETS):
            ax = axes[i, j]
            for N1 in E1_N1:
                pts = [(N2, E1[f"{d}|{name}|{N1}|{N2}"]["err"]) for N2 in E1_N2 if f"{d}|{name}|{N1}|{N2}" in E1]
                if pts: ax.plot(*zip(*pts), "o-", label=f"N1={N1}")
            ax.set_yscale("log"); ax.set_xscale("log", base=2); ax.set_ylim(1e-15, 1e1); ax.grid(alpha=.3)
            ax.set_title(f"d={d} {name}", fontsize=9); ax.set_xlabel("N2")
            if i == 0 and j == 0: ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=4, fontsize=7, borderaxespad=0)
        axes[i, 0].set_ylabel("test rel L2 (oracle)")
    fig.tight_layout(); return fig

def table_E1():
    E1 = RESULTS.get("E1", {}); lines = ["E1 oracle: best cell per (d, target)   [err @ N1,N2 ; units ; channel err]"]
    for d in E1_DIMS:
        for name in E1_TARGETS:
            rows = [(k, v) for k, v in E1.items() if k.startswith(f"{d}|{name}|")]
            if rows:
                k, v = min(rows, key=lambda kv: kv[1]["err"])
                lines.append(f"  d={d} {name:13s} {v['err']:.2e} @ N1,N2={k.split('|')[2]},{k.split('|')[3]}  units {v['units']:4d}  chan {v['chan_err']:.1e}")
    return "\n".join(lines)

for d in E1_DIMS:
    for name in E1_TARGETS:
        for N1 in E1_N1:
            for N2 in E1_N2:
                key = f"{d}|{name}|{N1}|{N2}"
                if done("E1", key) or not budget_ok(): continue
                put("E1", key, oracle_fit(name, d, N1, N2))
        show(fig_E1(), "E1_oracle", table_E1())
""")

md(r"""
## 7. Experiment 2: the headline. Error against dimension at matched cost

For each $d\in\{2,3,4,5,8\}$ and target: the compositional block (rank 1, and untied, both VarPro then Gauss-Newton, learned from zero), the shallow ridge-QI at the
same unit count (directions learned the same way), a tanh MLP at the same parameter count, and a tanh MLP at the same FLOPs.
Rank 1 with a shared direction weight vector is the most rigid form; it is structurally excluded from three bumps (three channel centers) and the Genz product
(a different profile per axis), which is why the untied block runs alongside it. Cost is matched to the rank-1 block.
Everything from zero; same data, seed, step budget and final solve. This is the figure that answers "does it scale better with $d$" and "is it a better architecture".
The figure is rebuilt after every (d, target).
""")

code(r"""
E2_TARGETS = ["gauss_bump", "fast_waves", "composition", "product_peak", "three_bumps"]
E2_DIMS = [3] if QUICK else [2, 3, 4, 5, 8]
if QUICK: E2_TARGETS = ["fast_waves", "composition"]

def run_matched(exp, name, d, cfg, seed, tag=None):
    key = f"{d}|{name}|{tag}|{seed}" if tag else f"{d}|{name}|{seed}"
    if done(exp, key) or not budget_ok(): return
    row = dict(cfg=cfg); t0 = time.time()
    print(f"{exp}  d={d} {name}{(' ' + tag) if tag else ''}: block rank 1 ...", flush=True)
    comp = run_comp(name, d, cfg, rank=1, mode="varpro", seed=seed); row["comp"] = comp
    print(f"    rank 1 {comp['best']:.2e} [{time.time() - t0:.0f}s]  untied ...", flush=True)
    row["comp_untied"] = run_comp(name, d, cfg, rank=None, mode="varpro", seed=seed)
    print(f"    untied {row['comp_untied']['best']:.2e} [{time.time() - t0:.0f}s]  shallow ...", flush=True)
    row["shallow"] = run_shallow(name, d, M=max(2, comp["units"] // cfg["N1"]), N1=cfg["N1"], seed=seed)
    print(f"    shallow {row['shallow']['best']:.2e} [{time.time() - t0:.0f}s]  MLPs ...", flush=True)
    row["mlp_params"] = run_mlp(name, d, mlp_width_for(comp["params"], d, "params"), seed=seed)
    row["mlp_flops"] = run_mlp(name, d, mlp_width_for(comp["flops"], d, "flops"), seed=seed)
    put(exp, key, row)
    print(f"{exp}  d={d} {name:13s}{(' ' + tag) if tag else ''}  comp r1 {comp['best']:.2e} untied {row['comp_untied']['best']:.2e} ({comp['params']}p/{comp['units']}u)  shallow {row['shallow']['best']:.2e}  "
          f"mlp@params {row['mlp_params']['best']:.2e} (w={row['mlp_params']['w']})  mlp@flops {row['mlp_flops']['best']:.2e} (w={row['mlp_flops']['w']})  [{comp['time']:.0f}s]")

ARMS = [("comp", "compositional (rank 1)", "o-"), ("comp_untied", "compositional (untied)", "D-"), ("shallow", "shallow ridge-QI, same units", "s--"), ("mlp_params", "tanh MLP, same params", "^:"), ("mlp_flops", "tanh MLP, same FLOPs", "v:")]

def fig_E2():
    E2 = RESULTS.get("E2", {})
    fig, axes = plt.subplots(1, len(E2_TARGETS), figsize=(3.8 * len(E2_TARGETS), 3.8), squeeze=False)
    for j, name in enumerate(E2_TARGETS):
        ax = axes[0, j]
        for arm, lab, st in ARMS:
            pts = [(d, E2[f"{d}|{name}|{SEEDS[0]}"][arm]["best"]) for d in E2_DIMS if f"{d}|{name}|{SEEDS[0]}" in E2]
            if pts: ax.plot(*zip(*pts), st, label=lab)
        ax.set_yscale("log"); ax.set_ylim(1e-15, 1e1); ax.set_xlabel("input dimension d"); ax.set_title(name, fontsize=10); ax.grid(alpha=.3)
        ax.set_xticks(E2_DIMS)
        if j == 0: ax.set_ylabel("test rel L2 (best of final / GN)"); ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize=7, borderaxespad=0)
    fig.tight_layout(); return fig

def table_E2():
    E2 = RESULTS.get("E2", {}); lines = ["E2 headline: test rel L2 by d (comp rank1 / untied | shallow | mlp@params | mlp@flops)"]
    for name in E2_TARGETS:
        for d in E2_DIMS:
            k = f"{d}|{name}|{SEEDS[0]}"
            if k in E2: r = E2[k]; lines.append(f"  d={d} {name:13s} {r['comp']['best']:.1e} / {r['comp_untied']['best']:.1e} | {r['shallow']['best']:.1e} | {r['mlp_params']['best']:.1e} | {r['mlp_flops']['best']:.1e}   ({r['comp']['params']} params, {r['comp']['units']} units)")
    return "\n".join(lines)

for d in E2_DIMS:
    for name in E2_TARGETS:
        for seed in SEEDS:
            run_matched("E2", name, d, base_cfg(d), seed)
        show(fig_E2(), "E2_headline_vs_d", table_E2())
""")

md(r"""
## 8. Experiment 3: which arm learns the factorization (Adam, VarPro, VarPro+GN; rank 1, rank 2, untied)

Same block, four targets, $d=3$ and $d=5$. Curves of the refit test error versus step; the Gauss-Newton phase is appended dashed.
Compare with E1: the gap between the oracle and the best learned arm is the optimization problem, not the architecture.
""")

code(r"""
E3_TARGETS = ["fast_waves", "composition"] if QUICK else ["gauss_bump", "fast_waves", "composition", "three_bumps"]
E3_DIMS = [3] if QUICK else [3, 5]

def fig_E3():
    E3 = RESULTS.get("E3", {})
    fig, axes = plt.subplots(len(E3_DIMS), len(E3_TARGETS), figsize=(4.2 * len(E3_TARGETS), 3.6 * len(E3_DIMS)), squeeze=False)
    for i, d in enumerate(E3_DIMS):
        for j, name in enumerate(E3_TARGETS):
            ax = axes[i, j]
            for k, e in E3.items():
                if not k.startswith(f"{d}|{name}|") or e["seed"] != SEEDS[0]: continue
                s, v = e["curve"]; lab = f"{e['mode']} rank={e['rank']}"; ln, = ax.plot(s, v, label=lab)
                if "gn_curve" in e:
                    gs, gv = e["gn_curve"]; ax.plot([s[-1] + 1 + q * max(1, s[-1] // max(1, len(gs))) for q in range(len(gs))], gv, "--", color=ln.get_color())
            ax.set_yscale("log"); ax.set_ylim(1e-15, 1e1); ax.set_xlabel("step (GN iterations appended, dashed)"); ax.grid(alpha=.3); ax.set_title(f"d={d} {name}", fontsize=9)
            if i == 0 and j == 0: ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=7, borderaxespad=0)
        axes[i, 0].set_ylabel("test rel L2 after readout refit")
    fig.tight_layout(); return fig

def table_E3():
    E3 = RESULTS.get("E3", {}); lines = ["E3 arms: final (and +GN) test rel L2"]
    for k, e in E3.items():
        lines.append(f"  {k:32s} final {e['final']:.2e}" + (f"  +GN {e['final_gn']:.2e}" if "final_gn" in e else ""))
    return "\n".join(lines)

for d in E3_DIMS:
    for name in E3_TARGETS:
        for rank in [1, 2, None]:
            for mode in ["adam", "varpro"]:
                for seed in SEEDS:
                    key = f"{d}|{name}|rank{rank}|{mode}|{seed}"
                    if done("E3", key) or not budget_ok(): continue
                    print(f"E3  {key} ...", flush=True)
                    put("E3", key, run_comp(name, d, base_cfg(d), rank=rank, mode=mode, seed=seed, gn=(mode == "varpro")))
        show(fig_E3(), "E3_arms", table_E3())
""")

md(r"""
## 9. Experiment 4: scaling knobs in $d=4$ (where the single layer is walled)

One knob at a time from the base configuration: $N_1$, $N_2$, $M$, $K$. Compositional block versus shallow at the same units versus the two matched MLPs.
Also plotted: error against parameters and against FLOPs for every model.
""")

code(r"""
E4_TARGETS = ["fast_waves", "composition"] if QUICK else ["fast_waves", "composition", "product_peak"]
E4_SWEEP = {"N1": [16, 32], "N2": [32, 64]} if QUICK else {"N1": [16, 32, 64], "N2": [32, 64, 128], "M": [4, 8, 16], "K": [1, 2, 4]}
E4_D = 3 if QUICK else 4

def fig_E4():
    E4 = RESULTS.get("E4", {}); knobs = list(E4_SWEEP.keys())
    fig, axes = plt.subplots(len(E4_TARGETS), len(knobs) + 2, figsize=(3.8 * (len(knobs) + 2), 3.5 * len(E4_TARGETS)), squeeze=False)
    for i, name in enumerate(E4_TARGETS):
        for j, knob in enumerate(knobs):
            ax = axes[i, j]
            for arm, lab, st in ARMS:
                pts = [(v, E4[f"{E4_D}|{name}|{knob}|{v}|{SEEDS[0]}"][arm]["best"]) for v in E4_SWEEP[knob] if f"{E4_D}|{name}|{knob}|{v}|{SEEDS[0]}" in E4]
                if pts: ax.plot(*zip(*pts), st, label=lab)
            ax.set_yscale("log"); ax.set_xscale("log", base=2); ax.set_ylim(1e-15, 1e1); ax.set_xlabel(knob); ax.grid(alpha=.3); ax.set_title(f"{name}: sweep {knob}", fontsize=9)
            if i == 0 and j == 0: ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=7, borderaxespad=0)
        rows = [v for k, v in E4.items() if k.startswith(f"{E4_D}|{name}|")]
        for j, key in enumerate(["params", "flops"]):
            ax = axes[i, len(knobs) + j]
            for arm, lab, st in ARMS: ax.plot([r[arm][key] for r in rows], [r[arm]["best"] for r in rows], st[0], label=lab)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-15, 1e1); ax.set_xlabel(key); ax.grid(alpha=.3); ax.set_title(f"{name}: error vs {key}", fontsize=9)
        axes[i, 0].set_ylabel("test rel L2")
    fig.tight_layout(); return fig

for name in E4_TARGETS:
    for knob, vals in E4_SWEEP.items():
        for v in vals:
            cfg = dict(base_cfg(E4_D)); cfg[knob] = v
            for seed in SEEDS:
                run_matched("E4", name, E4_D, cfg, seed, tag=f"{knob}|{v}")
        show(fig_E4(), "E4_scaling_d4")
""")

md(r"""
## 10. Experiment 5: the control where composition should not help

`random_ridges`: 24 sinusoidal ridges with random directions in $d=3$ and $d=5$. Its shortest factorization is the shallow one (24 exact directions),
so the block has no dividend to collect; the shallow model with learned directions should win or tie. If the block wins here, something is confounded.
""")

code(r"""
for d in ([3] if QUICK else [3, 5]):
    for seed in SEEDS:
        key = f"{d}|random_ridges|{seed}"
        if done("E5", key) or not budget_ok(): continue
        comp = run_comp("random_ridges", d, dict(M=24, N1=16, K=2, N2=64), rank=1, seed=seed)
        sh = run_shallow("random_ridges", d, M=24, N1=16, seed=seed)
        put("E5", key, dict(comp=comp, shallow=sh))
        print(f"E5  d={d} random_ridges seed={seed}  comp {comp['best']:.2e}  shallow {sh['best']:.2e}")
""")

md(r"""
## 11. Experiment 6: real data

California housing (sklearn, 20640 rows, $d=8$): inputs standardized, winsorized at three standard deviations and scaled into the unit ball (so no test point
leaves the band: the record's far-field failure), log target standardized, 80/20 split. Label noise dominates, so the QI arms choose their truncation threshold by two-fold validation on the training set under a readout-norm cap of 100
(the noise floor is $\|w\|\varepsilon_y/\sqrt n$; a $10^{-14}$ truncation on noisy labels interpolates them with a readout norm of $10^9$ that blows up on rare
test points no validation split can see; a resolved readout has $O(1)$ norm), and the question is test RMSE against the shallow model and the param-matched MLP, not precision.
A constant predictor scores 1.0 in these units. If the download fails the built-in diabetes set (442 rows) is used.
""")

code(r"""
def real_data():
    try:
        from sklearn.datasets import fetch_california_housing
        X, y = fetch_california_housing(return_X_y=True); y = np.log(y); nm = "california"
    except Exception as e:
        from sklearn.datasets import load_diabetes
        X, y = load_diabetes(return_X_y=True); nm = "diabetes"; print("fallback:", e)
    X = torch.tensor(X); y = torch.tensor(y, dtype=torch.float64); y = (y - y.mean()) / y.std()
    X = ((X - X.mean(0)) / X.std(0)).clamp(-3, 3) / (3 * math.sqrt(X.shape[1]))   # winsorized: every point inside the unit ball
    g = torch.Generator().manual_seed(0); perm = torch.randperm(len(y), generator=g); ntr = int(0.8 * len(y))
    D = dict(Ztr=X[perm[:ntr]].to(DEV), ytr=y[perm[:ntr]].to(DEV), Zte=X[perm[ntr:]].to(DEV), yte=y[perm[ntr:]].to(DEV))
    D["r"] = float(D["Ztr"].norm(dim=1).max()); D["d"] = X.shape[1]; D["name"] = nm
    return D

def pick_rcond(model, D, grid=(1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12), wmax=100.0):
    # noisy labels: two-fold validation over the truncation threshold, with a cap on the readout norm. The cap is the
    # identifiability rule of the record (a resolved readout has O(1) norm; a huge norm interpolates noise and blows up on
    # rare test points that a validation split cannot see). The noise floor is ||w|| eps_y / sqrt n (expB01).
    n = len(D["ytr"]); h = n // 2; best = None
    with torch.no_grad():
        F = [model.feats(D["Ztr"][:h]), model.feats(D["Ztr"][h:])]; Y = [D["ytr"][:h], D["ytr"][h:]]
        for rc in grid:
            errs = []
            for a in (0, 1):
                w, b, _ = lstsq_bias(F[a], Y[a], rc)
                if float(w.norm()) > wmax: errs = None; break
                errs.append(float(((F[1 - a] @ w + b - Y[1 - a]) ** 2).mean().sqrt()))
            if errs is None: continue
            e = sum(errs) / 2
            if best is None or e < best[0]: best = (e, rc)
    return best[1] if best else grid[0]

def fit_real(m, D, mode):
    global RCOND
    _rc = RCOND; RCOND = pick_rcond(m, D)
    train(m, D, mode=mode, final_lstsq=False); m.regrid(D["Ztr"]); RCOND = pick_rcond(m, D); solve_readout(m, D["Ztr"], D["ytr"])
    chosen, RCOND = RCOND, _rc
    return chosen

if not done("E6", "real") and budget_ok():
    D = real_data(); d = D["d"]; rm = lambda mod: float(((mod(D["Zte"]) - D["yte"]) ** 2).mean().sqrt())
    out = dict(dataset=D["name"], n_train=len(D["ytr"]))
    for K in ([2] if QUICK else [1, 2, 4]):
        m = CompQI(d, 16, 16, K, 64, COLLAR * D["r"], rank=1).to(DEV); rc = fit_real(m, D, "varpro"); out[f"comp_K{K}"] = dict(rmse=rm(m), rcond=rc, **m.counts())
    sh = ShallowQI(d, 16, 16, COLLAR * D["r"]).to(DEV); rc = fit_real(sh, D, "varpro"); out["shallow"] = dict(rmse=rm(sh), rcond=rc, **sh.counts())
    w = mlp_width_for(out["comp_K2"]["params"], d, "params"); mm = MLP(d, w); train(mm, D, mode="adam", lr=3e-3); out["mlp_params"] = dict(rmse=rm(mm), **mm.counts())
    put("E6", "real", out)
    print("E6 ", D["name"], {k: (round(v["rmse"], 4) if isinstance(v, dict) else v) for k, v in out.items()})
""")

md(r"""
## 12. Everything is already saved

`results.json` and the figures live in `OUTDIR` (Drive if mounted). Re-running any cell above resumes from the file. To rebuild every figure from the file: run the next cell.
""")

code(r"""
for fn, nm in [(fig_E1, "E1_oracle"), (fig_E2, "E2_headline_vs_d"), (fig_E3, "E3_arms"), (fig_E4, "E4_scaling_d4")]:
    try: f = fn(); f.savefig(f"{OUTDIR}/{nm}.png", dpi=120, bbox_inches="tight"); plt.close(f)
    except Exception as e: print(nm, "not rebuilt:", e)
print("saved to", OUTDIR, ":", sorted(p for p in os.listdir(OUTDIR)))
print(f"elapsed {(time.time() - T_START) / 60:.1f} min")
""")


def build():
    nb = {
        "cells": [],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}, "accelerator": "GPU"},
        "nbformat": 4, "nbformat_minor": 5,
    }
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell["outputs"] = []; cell["execution_count"] = None
        nb["cells"].append(cell)
    OUT.write_text(json.dumps(nb, indent=1))
    print("wrote", OUT, f"({len(CELLS)} cells)")


def smoke():
    import matplotlib
    matplotlib.use("Agg")
    scratch = tempfile.mkdtemp(prefix="expI01_smoke_")
    ns = {}
    only = None
    if "--cells" in sys.argv: only = {int(x) for x in sys.argv[sys.argv.index("--cells") + 1].split(",")}
    ci = -1
    for kind, src in CELLS:
        if kind != "code":
            continue
        ci += 1
        if only is not None and ci not in only:
            continue
        src = (src.replace("QUICK       = False", "QUICK       = True").replace("plt.show()", "plt.close('all')")
                  .replace('OUTDIR = "expI01_out"', f'OUTDIR = "{scratch}"'))
        exec(compile(src, "<cell>", "exec"), ns)
    print("smoke OK; results keys:", {k: len(v) for k, v in ns["RESULTS"].items()}, "| scratch:", scratch)


if __name__ == "__main__":
    build()
    if "--smoke" in sys.argv:
        smoke()
