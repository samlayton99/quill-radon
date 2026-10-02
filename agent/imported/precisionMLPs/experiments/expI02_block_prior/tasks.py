"""tasks: data, targets and PDEs for expI02. Everything returns CPU tensors in the default dtype; run.py moves them.

Analytic targets are a verbatim port of the expI01 notebook (anchors, data ball about x0, inner-0.9 test ball,
oracle factorizations). Structured regression: Friedman1, a Lorenz flow map (RK4), OpenML kin8nm / concrete.
Fashion-MNIST from the raw IDX files. PDEs live in scaled coordinates on [-1, 1]^d with autograd residuals.
"""
import gzip
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "results" / "checkpoint_I_depth_theory" / "expI02_block_prior" / "data"
sys.path.insert(0, str(REPO / "experiments" / "expF13_bwler_suite"))


# ----------------------------------------------------------------------------- analytic targets (expI01 port)
def anchors(d):
    if d == 2:
        x0, a1, a2 = [0.35, -0.25], [0.2, 0.1], [0.3, -0.2]                              # expH05
    elif d == 3:
        x0, a1, a2 = [0.35, -0.25, 0.2], [0.2, 0.1, 0.1], [0.3, -0.2, 0.15]               # expH06
    else:
        i = np.arange(d)
        x0 = 0.3 * np.cos(1.3 * i + 0.4); a1 = x0 + 0.15 * np.sin(2.1 * i + 1.0); a2 = x0 + 0.08 * np.cos(0.7 * i + 2.0)
    B = [[.30, -.20, .25, -.15, .10, .20, -.10, .15], [-.40, .35, -.10, .20, -.25, -.15, .30, -.05], [.15, .10, -.30, .05, .30, -.20, .10, .25]]
    t = lambda v: torch.tensor(np.asarray(v, dtype=np.float64)[:d]).to(torch.get_default_dtype())
    return dict(x0=t(x0), a1=t(a1), a2=t(a2), b1=t(B[0]), b2=t(B[1]), b3=t(B[2]))


def radius(d):
    return 0.4 if d == 2 else 0.3


def ball(n, d, r, seed, inner=1.0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, d, generator=g, dtype=torch.float64); u = u / u.norm(dim=1, keepdim=True)
    rad = r * inner * torch.rand(n, 1, generator=g, dtype=torch.float64) ** (1.0 / d)
    return (anchors(d)["x0"].to(torch.float64) + rad * u).to(torch.get_default_dtype())


def rho(x, a):
    return (x - a).norm(dim=1) / math.sqrt(2.0)


_RIDGE_CACHE = {}


def random_ridges(x, J=24):
    d = x.shape[1]
    if (d, J) not in _RIDGE_CACHE:
        g = torch.Generator().manual_seed(1234)
        U = torch.randn(J, d, generator=g, dtype=torch.float64); U = U / U.norm(dim=1, keepdim=True)
        w = 8 + 12 * torch.rand(J, generator=g, dtype=torch.float64); ph = 2 * math.pi * torch.rand(J, generator=g, dtype=torch.float64)
        _RIDGE_CACHE[(d, J)] = tuple(t.to(torch.get_default_dtype()) for t in (U, w, ph))
    U, w, ph = _RIDGE_CACHE[(d, J)]
    return torch.sin(w * (x @ U.T) + ph).sum(1) / math.sqrt(J)


GENZ_A = [3, 5, 8, 12, 6, 4, 7, 9]
GENZ_B = [.35, -.25, .10, -.40, .25, .15, -.30, .05]
ANALYTIC = ["gauss_bump", "radial_runge", "fast_waves", "composition", "product_peak", "three_bumps", "packet", "product_sines", "random_ridges"]


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
        a = torch.tensor(GENZ_A[:d], dtype=x.dtype); b = torch.tensor(GENZ_B[:d], dtype=x.dtype)
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


def oracle_dirs_and_P(name, d):
    """The known factorization f = g(P(x)): directions V (rows) and P as a function of the centered input Z."""
    A = anchors(d)
    if name in ("gauss_bump", "radial_runge", "fast_waves"):
        anc = "a1" if name == "gauss_bump" else "a2"
        return torch.eye(d), (lambda Z: ((Z + A["x0"] - A[anc]) ** 2).sum(1) / 2.0)
    if name == "product_peak":
        a = torch.tensor(GENZ_A[:d], dtype=torch.get_default_dtype()); b = torch.tensor(GENZ_B[:d], dtype=torch.get_default_dtype())
        return torch.eye(d), (lambda Z: -torch.log1p((a * (Z + A["x0"] - b)) ** 2).sum(1))
    if name == "composition":
        if d == 2: V = torch.tensor([[1., 1.], [1., -1.]])
        elif d == 3: V = torch.tensor([[1., 1., 0.], [1., -1., 0.], [0., 0., 1.]])
        else:
            V = torch.zeros(2 * d, d)
            for i in range(d): V[2 * i, i] = 1; V[2 * i, (i + 1) % d] = 1; V[2 * i + 1, i] = 1; V[2 * i + 1, (i + 1) % d] = -1
        V = V / V.norm(dim=1, keepdim=True)
        def Pfn(Z):
            x = Z + A["x0"]
            if d == 2: return torch.sin(math.pi * x[:, 0]) * torch.cos(math.pi * x[:, 1])
            if d == 3: return torch.sin(math.pi * x[:, 0]) * torch.cos(math.pi * x[:, 1]) + 0.5 * torch.cos(math.pi * x[:, 2] + 1.0)
            return (2.0 / d) * (torch.sin(math.pi * x) * torch.cos(math.pi * x.roll(-1, 1))).sum(1)
        return V, Pfn
    return None


def analytic(name, d, n_train, n_test=10000, seed=0):
    """Data uniform in the ball of radius r about x0, centered (Z = X - x0); test on the inner 0.9 ball."""
    r = radius(d)
    Xtr = ball(n_train, d, r, seed); Xte = ball(n_test, d, r, seed + 777, inner=0.9)
    x0 = anchors(d)["x0"]
    return dict(Xtr=Xtr - x0, Ytr=target(name, Xtr), Xte=Xte - x0, Yte=target(name, Xte), r=r, x0=x0, d=d, name=name,
                oracle=oracle_dirs_and_P(name, d))


# ----------------------------------------------------------------------------- structured regression
def _split_standardize(X, y, seed, frac=0.8, y_scale=True):
    X = torch.as_tensor(np.asarray(X, dtype=np.float64)); y = torch.as_tensor(np.asarray(y, dtype=np.float64)).reshape(len(X), -1)
    g = torch.Generator().manual_seed(seed); perm = torch.randperm(len(X), generator=g); ntr = int(frac * len(X))
    tr, te = perm[:ntr], perm[ntr:]
    mx, sx = X[tr].mean(0), X[tr].std(0).clamp_min(1e-12)
    my, sy = (y[tr].mean(0), y[tr].std(0).clamp_min(1e-12)) if y_scale else (torch.zeros(y.shape[1]), torch.ones(y.shape[1]))
    f = lambda t: t.to(torch.get_default_dtype())
    return dict(Xtr=f((X[tr] - mx) / sx), Ytr=f((y[tr] - my) / sy), Xte=f((X[te] - mx) / sx), Yte=f((y[te] - my) / sy),
                y_scale=f(sy), d=X.shape[1], n=len(X))


def friedman1(seed=0, n=2000, noise=1.0):
    from sklearn.datasets import make_friedman1
    X, y = make_friedman1(n_samples=n, noise=noise, random_state=seed)
    D = _split_standardize(X, y, seed); D["name"] = "friedman1"; D["noise"] = noise
    return D


def _lorenz_rk4(x, dt, steps, sigma=10.0, rho_=28.0, beta=8.0 / 3.0):
    def f(s):
        return np.stack([sigma * (s[:, 1] - s[:, 0]), s[:, 0] * (rho_ - s[:, 2]) - s[:, 1], s[:, 0] * s[:, 1] - beta * s[:, 2]], 1)
    for _ in range(steps):
        k1 = f(x); k2 = f(x + 0.5 * dt * k1); k3 = f(x + 0.5 * dt * k2); k4 = f(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def lorenz_map(seed=0, n=4000, dt=0.005, sub=10, gap=10, burn=4000):
    """Flow map x(t) -> x(t + sub * dt) on the Lorenz attractor; train and test from different trajectories."""
    rng = np.random.default_rng(seed)
    def traj(m):
        x = rng.normal(size=(1, 3)) * np.array([[5.0, 5.0, 5.0]]) + np.array([[0.0, 0.0, 25.0]])
        x = _lorenz_rk4(x, dt, burn)
        X, Y = [], []
        for _ in range(m):
            y = _lorenz_rk4(x, dt, sub); X.append(x[0]); Y.append(y[0]); x = _lorenz_rk4(y, dt, gap)
        return np.array(X), np.array(Y)
    Xtr, Ytr = traj(n); Xte, Yte = traj(n // 4)
    mx, sx = Xtr.mean(0), Xtr.std(0)
    f = lambda a: torch.as_tensor((a - mx) / sx).to(torch.get_default_dtype())
    return dict(Xtr=f(Xtr), Ytr=f(Ytr), Xte=f(Xte), Yte=f(Yte), y_scale=torch.as_tensor(sx).to(torch.get_default_dtype()), d=3, n=n, name="lorenz_map")


OPENML = dict(kin8nm=[dict(name="kin8nm", version=1)],
              concrete=[dict(data_id=44959)])


def openml_regression(key, seed=0):
    from sklearn.datasets import fetch_openml
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"openml_{key}.npz"
    if cache.exists():
        z = np.load(cache); X, y = z["X"], z["y"]
    else:
        err = None
        for spec in OPENML[key]:
            try:
                d = fetch_openml(as_frame=False, parser="auto", **spec)
                X, y = np.asarray(d.data, dtype=np.float64), np.asarray(d.target, dtype=np.float64)
                np.savez(cache, X=X, y=y); break
            except Exception as e:
                err = e
        else:
            raise RuntimeError(f"OpenML {key} unavailable: {err}")
    D = _split_standardize(X, y, seed); D["name"] = key
    return D


# ----------------------------------------------------------------------------- Fashion-MNIST
_FMNIST = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def fashion_cached():
    return all((DATA_DIR / "fashion" / f).exists() for f in _FILES)


def _idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[:4], "big"); ndim = magic & 0xff
    dims = [int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw[4 + 4 * ndim:], dtype=np.uint8).reshape(dims)


def fashion_mnist():
    """(Xtr uint8 [60000, 784], ytr int64, Xte uint8 [10000, 784], yte int64), downloaded once."""
    d = DATA_DIR / "fashion"; d.mkdir(parents=True, exist_ok=True)
    for f in _FILES:
        if not (d / f).exists():
            urllib.request.urlretrieve(_FMNIST + f, d / f)
    Xtr, ytr, Xte, yte = [_idx(d / f) for f in _FILES]
    t = lambda a: torch.from_numpy(a.copy())
    return t(Xtr.reshape(-1, 784)), t(ytr.astype(np.int64)), t(Xte.reshape(-1, 784)), t(yte.astype(np.int64))


# ----------------------------------------------------------------------------- PDEs in scaled coordinates on [-1, 1]^d
def _grad(u, X):
    return torch.autograd.grad(u.sum(), X, create_graph=True)[0]


def _u(model, X):
    return model(X)[:, 0]


def _res_poisson1d(model, X, f):
    X = X.requires_grad_(True); u = _u(model, X)
    ux = _grad(u, X)[:, 0]; uxx = _grad(ux, X)[:, 0]
    return -uxx - f(X)


def _res_poisson2d(model, X, f):
    X = X.requires_grad_(True); u = _u(model, X)
    g = _grad(u, X); uxx = _grad(g[:, 0], X)[:, 0]; uyy = _grad(g[:, 1], X)[:, 1]
    return -(uxx + uyy) - f(X)


NU_BURGERS = 0.01 / math.pi


def _res_burgers(model, X, f=None):
    """2 u_tau + u u_x - nu u_xx on (x, tau), tau = 2t - 1."""
    X = X.requires_grad_(True); u = _u(model, X)
    g = _grad(u, X); ux, ut = g[:, 0], g[:, 1]; uxx = _grad(ux, X)[:, 0]
    return 2 * ut + u * ux - NU_BURGERS * uxx


def _p1_exact(X):
    x = X[:, 0]; return torch.sin(2 * math.pi * x) + 0.3 * torch.sin(8 * math.pi * x)


def _p1_f(X):
    x = X[:, 0]; return 4 * math.pi ** 2 * torch.sin(2 * math.pi * x) + 0.3 * 64 * math.pi ** 2 * torch.sin(8 * math.pi * x)


def _p2_exact(X):
    x, y = X[:, 0], X[:, 1]
    return torch.sin(math.pi * x) * torch.sin(math.pi * y) + 0.5 * torch.sin(3 * math.pi * x) * torch.sin(2 * math.pi * y)


def _p2_f(X):
    x, y = X[:, 0], X[:, 1]
    return 2 * math.pi ** 2 * torch.sin(math.pi * x) * torch.sin(math.pi * y) + 0.5 * 13 * math.pi ** 2 * torch.sin(3 * math.pi * x) * torch.sin(2 * math.pi * y)


def _burgers_exact(X):
    from problems import _burgers_exact_scaled
    return torch.as_tensor(_burgers_exact_scaled(NU_BURGERS)(X.detach().cpu().double().numpy())).to(X.dtype)


def pde(name, seed=0, n_col=2048, n_bc=256, n_test=None):
    """dict(X_col, X_bc, u_bc, X_test, u_test, exact, residual(model, X), f, linear, d, name, bc_weight)."""
    g = torch.Generator().manual_seed(seed)
    U = lambda n, d: (2 * torch.rand(n, d, generator=g, dtype=torch.float64) - 1).to(torch.get_default_dtype())
    if name == "poisson1d":
        X_col = U(n_col, 1); X_bc = torch.tensor([[-1.0], [1.0]]).to(torch.get_default_dtype())
        X_test = torch.linspace(-1, 1, n_test or 2001).view(-1, 1)
        exact, f, res, d, linear = _p1_exact, _p1_f, _res_poisson1d, 1, True
    elif name == "poisson2d":
        X_col = U(n_col, 2)
        s = U(n_bc // 4, 1)[:, 0]; one = torch.ones_like(s)
        X_bc = torch.cat([torch.stack([-one, s], 1), torch.stack([one, s], 1), torch.stack([s, -one], 1), torch.stack([s, one], 1)])
        gx = torch.linspace(-1, 1, n_test or 101); X_test = torch.stack(torch.meshgrid(gx, gx, indexing="ij"), -1).reshape(-1, 2)
        exact, f, res, d, linear = _p2_exact, _p2_f, _res_poisson2d, 2, True
    elif name == "burgers":
        X_col = U(n_col, 2)
        s = U(n_bc // 2, 1)[:, 0]; one = torch.ones_like(s)
        X_bc = torch.cat([torch.stack([s, -one], 1), torch.stack([-one[: n_bc // 4], s[: n_bc // 4]], 1), torch.stack([one[: n_bc // 4], s[: n_bc // 4]], 1)])
        gx = torch.linspace(-1, 1, n_test or 101); X_test = torch.stack(torch.meshgrid(gx, gx, indexing="ij"), -1).reshape(-1, 2)
        exact, f, res, d, linear = _burgers_exact, None, _res_burgers, 2, False
    else:
        raise KeyError(name)
    return dict(name=name, d=d, linear=linear, X_col=X_col, X_bc=X_bc, u_bc=exact(X_bc), X_test=X_test, u_test=exact(X_test),
                exact=exact, f=f, residual=(lambda model, X, res=res, f=f: res(model, X, f)), bc_weight=10.0)
