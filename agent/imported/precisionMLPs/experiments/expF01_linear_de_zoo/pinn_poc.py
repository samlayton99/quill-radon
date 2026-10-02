"""Proof-of-concept: QI-geometry collocation solve ("PINN without training").

Model: u(x) = sum_m a_m tanh(gamma (x - c_m)) + b0 + b1 x
Geometry: uniform centers with halo, single gamma = lambda/h (the QI geometry).
PDE residual is LINEAR in (a, b) for linear L, so collocation + min-norm lstsq
solves it directly. Nonlinear PDEs: Gauss-Newton, each step is the same lstsq.

Cases:
  1. interpolation control (L = identity)      -- should match repo floor ~1e-13
  2. 1D Poisson  -u'' = f, Dirichlet BC        -- manufactured u* = sin(2 pi x)
  3. 1D Helmholtz u'' + k^2 u = f              -- oscillatory
  4. 1D boundary layer  eps u'' + u' = f       -- steep solution, resolution test
  5. 1D nonlinear  u'' = u^3 + f               -- Gauss-Newton
  6. 2D Poisson on the unit disk, Radon ridges -- Laplacian u = f, BC on circle
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------- tanh feature derivatives (exact closed forms) ----------
def T(z):  return np.tanh(z)
def d0(z): return np.tanh(z)
def d1(z): t = np.tanh(z); return 1.0 - t * t                    # sech^2
def d2(z): t = np.tanh(z); return -2.0 * t * (1.0 - t * t)
def d3(z): t = np.tanh(z); s = 1.0 - t * t; return -2.0 * s * (s - 2.0 * t * t) * 1.0 + 0*z  # -2 s(1-3t^2)? verify below

# verify d3 numerically once
z = np.linspace(-3, 3, 11)
eps = 1e-6
num = (d2(z + eps) - d2(z - eps)) / (2 * eps)
ana = -2.0 * (1 - np.tanh(z)**2) * (1 - 3 * np.tanh(z)**2)
assert np.allclose(num, ana, atol=1e-8), "d3 formula wrong"

# ---------- 1D geometry ----------
def geometry_1d(N, lam=0.25):
    h = 2.0 / N
    R = max(70, int(np.ceil(0.4 * N)))          # default_halo, mpmath flavor
    c = -1.0 + h * np.arange(-R, N + R + 1)
    gamma = lam / h
    return c, gamma

def solve_lstsq(A, y, rcond=1e-13):
    sol, res, rank, s = np.linalg.lstsq(A, y, rcond=rcond)
    return sol, s[0] / s[min(rank - 1, len(s) - 1)] if rank > 0 else np.inf

def eval_model(x, c, gamma, a, b0, b1):
    return T(gamma * (x[:, None] - c[None, :])) @ a + b0 + b1 * x

def report(name, x_eval, u_hat, u_exact):
    err = u_hat - u_exact
    rel_l2 = np.linalg.norm(err) / np.linalg.norm(u_exact)
    linf = np.max(np.abs(err))
    print(f"{name:52s} relL2 {rel_l2:9.2e}   Linf {linf:9.2e}")
    return rel_l2, linf

# dense eval grid
xe = np.linspace(-1, 1, 4096)

def linear_solve_1d(N, lam, L_rows_fn, rhs_fn, bc, n_col=None, weight_bc=None):
    """Generic linear 1D solve.
    L_rows_fn(z, gamma, x) -> (feature block rows [n,W], b0 col, b1 col) for operator L
    bc: list of (x_b, value) Dirichlet conditions on u itself.
    """
    c, gamma = geometry_1d(N, lam)
    W = len(c)
    n_col = n_col or 4 * W
    x = np.linspace(-1, 1, n_col)
    Z = gamma * (x[:, None] - c[None, :])
    F, col0, col1 = L_rows_fn(Z, gamma, x)
    A_pde = np.hstack([F, col0, col1])
    y_pde = rhs_fn(x)
    # row scaling: normalize PDE block to O(1)
    s = max(np.abs(A_pde).max(), 1e-300)
    A_pde, y_pde = A_pde / s, y_pde / s
    if bc:
        xb = np.array([b[0] for b in bc]); yb = np.array([b[1] for b in bc])
        Zb = gamma * (xb[:, None] - c[None, :])
        A_bc = np.hstack([T(Zb), np.ones((len(xb), 1)), xb[:, None]])
        wbc = weight_bc if weight_bc is not None else np.sqrt(n_col / len(xb))
        A = np.vstack([A_pde, wbc * A_bc])
        y = np.concatenate([y_pde, wbc * yb])
    else:
        A, y = A_pde, y_pde
    sol, cond = solve_lstsq(A, y)
    a, b0, b1 = sol[:W], sol[W], sol[W + 1]
    return eval_model(xe, c, gamma, a, b0, b1), cond

print("=" * 80)
for N in (64, 128, 256, 512, 1024):
    lam = 0.25
    # ---- 1. interpolation control ----
    u = lambda x: np.sin(2 * np.pi * x)
    def L_id(Z, gamma, x): return T(Z), np.ones((len(x), 1)), x[:, None]
    u_hat, cond = linear_solve_1d(N, lam, L_id, u, bc=[])
    report(f"[N={N:4d}] interpolation control  sin(2pi x)", xe, u_hat, u(xe))

    # ---- 2. Poisson -u'' = f ----
    f = lambda x: 4 * np.pi**2 * np.sin(2 * np.pi * x)
    def L_poisson(Z, gamma, x):
        return -gamma**2 * d2(Z), np.zeros((len(x), 1)), np.zeros((len(x), 1))
    u_hat, cond = linear_solve_1d(N, lam, L_poisson, f, bc=[(-1.0, u(np.array([-1.0]))[0]), (1.0, u(np.array([1.0]))[0])])
    report(f"[N={N:4d}] Poisson -u''=f          sin(2pi x)", xe, u_hat, u(xe))

    # ---- 3. Helmholtz u'' + k^2 u = f, k=10, u* = sin(7 pi x) exp? keep simple
    k = 10.0
    us = lambda x: np.sin(7 * np.pi * x) + 0.5 * np.cos(3 * np.pi * x)
    fs = lambda x: -(7 * np.pi)**2 * np.sin(7 * np.pi * x) - 0.5 * (3 * np.pi)**2 * np.cos(3 * np.pi * x) + k**2 * us(x)
    def L_helm(Z, gamma, x):
        return gamma**2 * d2(Z) + k**2 * T(Z), k**2 * np.ones((len(x), 1)), k**2 * x[:, None]
    u_hat, cond = linear_solve_1d(N, lam, L_helm, fs, bc=[(-1.0, us(np.array([-1.0]))[0]), (1.0, us(np.array([1.0]))[0])])
    report(f"[N={N:4d}] Helmholtz u''+100u=f    sin(7pi x)+", xe, u_hat, us(xe))

    # ---- 3b. variable coefficients: u'' + sin(pi x) u' + x^2 u = f ----
    uv = lambda x: np.sin(2 * np.pi * x)
    uvp = lambda x: 2 * np.pi * np.cos(2 * np.pi * x)
    uvpp = lambda x: -4 * np.pi**2 * np.sin(2 * np.pi * x)
    fv = lambda x: uvpp(x) + np.sin(np.pi * x) * uvp(x) + x**2 * uv(x)
    def L_var(Z, gamma, x):
        s1 = np.sin(np.pi * x)[:, None]      # coefficient of u'
        s0 = (x**2)[:, None]                 # coefficient of u
        F = gamma**2 * d2(Z) + s1 * gamma * d1(Z) + s0 * T(Z)
        col0 = s0.copy()                     # L[1] = x^2
        col1 = s1 + s0 * x[:, None]          # L[x] = sin(pi x) + x^3
        return F, col0, col1
    u_hat, cond = linear_solve_1d(N, lam, L_var, fv, bc=[(-1.0, uv(np.array([-1.0]))[0]), (1.0, uv(np.array([1.0]))[0])])
    report(f"[N={N:4d}] var-coeff u''+sin(pi x)u'+x^2u      ", xe, u_hat, uv(xe))

    # ---- 4. boundary layer: eps u'' + u' = 1, u(-1)=u(1)=0 ----
    epsl = 0.02
    # eps u'' + u' = 1, u(-1)=u(1)=0. Exact: u = x - 1 + 2 exp(-(x+1)/eps) (layer at x=-1).
    ub = lambda x: x - 1 + 2 * np.exp(-(x + 1) / epsl)
    fb = lambda x: np.ones_like(x)
    def L_bl(Z, gamma, x):
        return epsl * gamma**2 * d2(Z) + gamma * d1(Z), np.zeros((len(x), 1)), np.ones((len(x), 1))
    u_hat, cond = linear_solve_1d(N, lam, L_bl, fb, bc=[(-1.0, 0.0), (1.0, 0.0)])
    report(f"[N={N:4d}] boundary layer eps=0.02             ", xe, u_hat, ub(xe))

print("=" * 80)
# ---- 5. nonlinear: u'' = u^3 + f, manufactured u* = sin(2 pi x) * exp? simple: u* = sin(2pi x)
N, lam = 128, 0.25
c, gamma = geometry_1d(N, lam)
Wd = len(c)
us = lambda x: np.sin(2 * np.pi * x)
fs = lambda x: -4 * np.pi**2 * np.sin(2 * np.pi * x) - us(x)**3
n_col = 4 * Wd
x = np.linspace(-1, 1, n_col)
Z = gamma * (x[:, None] - c[None, :])
Phi, Phi2 = T(Z), gamma**2 * d2(Z)
xb = np.array([-1.0, 1.0]); yb = us(xb)
Zb = gamma * (xb[:, None] - c[None, :])
Phib = np.hstack([T(Zb), np.ones((2, 1)), xb[:, None]])
wbc = np.sqrt(n_col / 2)
theta = np.zeros(Wd + 2)
res_hist = []
for it in range(30):
    u_c = Phi @ theta[:Wd] + theta[Wd] + theta[Wd + 1] * x
    r_pde = Phi2 @ theta[:Wd] - u_c**3 - fs(x)          # residual of u'' - u^3 - f = 0
    J_pde = np.hstack([Phi2, np.zeros((n_col, 1)), np.zeros((n_col, 1))]) \
            - 3 * (u_c**2)[:, None] * np.hstack([Phi, np.ones((n_col, 1)), x[:, None]])
    r_bc = Phib @ theta - yb
    res_hist.append(max(np.max(np.abs(r_pde)), np.max(np.abs(r_bc))))
    s = np.abs(J_pde).max()
    A = np.vstack([J_pde / s, wbc * Phib])
    r = np.concatenate([r_pde / s, wbc * r_bc])
    step, _ = solve_lstsq(A, -r)
    theta = theta + step
    if np.linalg.norm(step) < 1e-14 * max(1, np.linalg.norm(theta)):
        break
u_hat = eval_model(xe, c, gamma, theta[:Wd], theta[Wd], theta[Wd + 1])
print(f"Gauss-Newton iters: {it + 1}")
print("  ||residual||_inf per iteration:", "  ".join(f"{r:.1e}" for r in res_hist[:10]))
report(f"[N={N:4d}] NONLINEAR u''=u^3+f     sin(2pi x)", xe, u_hat, us(xe))

print("=" * 80)
# ---- 6. 2D Poisson on unit disk, Radon ridge geometry ----
# u* = exp(-2(x^2+y^2)) * sin(2x + y) manufactured; Laplacian computed symbolically below.
import numpy as np
def u2(p):
    x, y = p[:, 0], p[:, 1]
    return np.exp(-2 * (x**2 + y**2)) * np.sin(2 * x + y)
def lap_u2(p):
    x, y = p[:, 0], p[:, 1]
    r2 = x**2 + y**2
    E = np.exp(-2 * r2); S = np.sin(2 * x + y); C = np.cos(2 * x + y)
    # u = E S ; u_xx + u_yy with E_x = -4x E, S_x = 2C etc.
    u_xx = E * ((16 * x**2 - 4) * S - 16 * x * C - 4 * S)
    u_yy = E * ((16 * y**2 - 4) * S - 8 * y * C - 1 * S)
    return u_xx + u_yy

def radon_tensor(N):
    J = int(round(np.sqrt(N)))            # directions
    M = int(np.ceil(N / J))               # offsets per direction
    thetas = np.pi * np.arange(J) / J
    collar = 1.25
    ts = np.linspace(-collar, collar, M)
    dirs = np.stack([np.stack([np.cos(th) * np.ones(M), np.sin(th) * np.ones(M)], axis=1) for th in thetas]).reshape(-1, 2)
    offs = np.tile(ts, J)
    return dirs, offs

for N2 in (256, 1024, 2304):
    best = None
    for lam2 in (0.10, 0.15, 0.20, 0.25, 0.30):
        dirs, offs = radon_tensor(N2)
        h_ref = 2.8 / np.sqrt(N2)
        g2 = lam2 / h_ref
        # collocation: area-uniform disk points + boundary circle
        n_int = 6 * len(offs)
        rr = np.sqrt(rng.random(n_int)); th = 2 * np.pi * rng.random(n_int)
        P = np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
        n_b = max(200, int(2 * np.sqrt(N2)))
        tb = np.linspace(0, 2 * np.pi, n_b, endpoint=False)
        B = np.stack([np.cos(tb), np.sin(tb)], axis=1)
        Zi = g2 * (P @ dirs.T - offs[None, :])
        A_pde = g2**2 * d2(Zi)                      # Laplacian of ridge: gamma^2 psi''(z), |w|=1
        A_pde = np.hstack([A_pde, np.zeros((n_int, 3))])   # bias, x, y cols killed by Laplacian... x,y harmonic -> 0 too
        y_pde = lap_u2(P)
        s = np.abs(A_pde).max()
        Zb2 = g2 * (B @ dirs.T - offs[None, :])
        A_bc = np.hstack([T(Zb2), np.ones((n_b, 1)), B])
        wb = np.sqrt(n_int / n_b)
        A = np.vstack([A_pde / s, wb * A_bc])
        y = np.concatenate([y_pde / s, wb * u2(B)])
        sol, _ = solve_lstsq(A, y)
        # eval on disk grid
        gx = np.linspace(-1, 1, 121)
        GX, GY = np.meshgrid(gx, gx)
        mask = GX**2 + GY**2 <= 1.0
        Pe = np.stack([GX[mask], GY[mask]], axis=1)
        Ze = g2 * (Pe @ dirs.T - offs[None, :])
        u_hat = T(Ze) @ sol[:-3] + sol[-3] + Pe @ sol[-2:]
        ue = u2(Pe)
        rel = np.linalg.norm(u_hat - ue) / np.linalg.norm(ue)
        linf = np.max(np.abs(u_hat - ue))
        if best is None or rel < best[1]:
            best = (lam2, rel, linf)
    print(f"[N={N2:5d}] 2D Poisson disk (radon)  best lam={best[0]:.2f}   relL2 {best[1]:9.2e}   Linf {best[2]:9.2e}")
