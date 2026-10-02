"""Quick correctness + timing probe for expF14 (not part of the sweep)."""
import sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import systems, core, reference

print("== RHS / Jacobian verification ==")
systems.verify_all()

print("\n== residual-Jacobian finite-difference check (Lorenz, N=16) ==")
S = systems.System("Lorenz")
T = S.horizon(1.0)
centers, gamma = core.geometry(16)
p = core.n_params(centers)
s = np.linspace(-1, 1, 200)
D0 = core.dict_rows(s, centers, gamma, 0)
D1 = core.dict_rows(s, centers, gamma, 1)
rng = np.random.default_rng(0)
A = 0.02 * rng.standard_normal((S.d, p))
sigma = np.array([8.0, 9.0, 24.0])
R, J = core._assemble(S, A, D0, D1, sigma, T)
h = 1e-7
err = 0.0
for _ in range(12):
    c, m = rng.integers(S.d), rng.integers(p)
    Ap = A.copy(); Ap[c, m] += h
    Am = A.copy(); Am[c, m] -= h
    Rp, _ = core._assemble(S, Ap, D0, D1, sigma, T, need_jac=False)
    Rm, _ = core._assemble(S, Am, D0, D1, sigma, T, need_jac=False)
    fd = (Rp - Rm).T.ravel() / (2 * h)
    err = max(err, np.max(np.abs(fd - J[:, c * p + m])) / max(np.max(np.abs(fd)), 1e-300))
print(f"  max rel |J_analytic - J_fd| = {err:.2e}")
assert err < 1e-6

print("\n== single-cell solve, Lorenz, lambda*T=3 ==")
T = S.horizon(3.0)
ts, Yref = reference.reference(S, T, 4001)
print(f"  T={T:.4f} ({T/S.period:.2f} dominant periods)")
for N in [32, 64, 128]:
    t0 = time.time()
    cell = core.solve_cell(S, T, N)
    Yh = core.model_trajectory(cell, ts)
    Yw = core.warm_trajectory(cell, ts)
    e = core.errors(Yh, Yref); ew = core.errors(Yw, Yref)
    ei = core.interpolation_floor(S, T, N, ts, Yref)
    print(f"  N={N:4d} W={cell['W']:4d} it={cell['iters']:2d} "
          f"warm={ew[0]:.2e} -> solve rel={e[0]:.2e} Linf={e[2]:.2e} "
          f"interp_floor={ei[0]:.2e}  ({time.time()-t0:.1f}s)")
