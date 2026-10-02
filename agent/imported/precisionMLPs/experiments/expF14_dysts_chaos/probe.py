import sys, time
from pathlib import Path
H = Path(__file__).resolve().parent; sys.path.insert(0, str(H))
import numpy as np, systems, core, reference
for name in ["InteriorSquirmer", "DoublePendulum", "MacArthur"]:
    S = systems.System(name); T = S.horizon(3.0)
    ts, Yref = reference.reference(S, T, 6001)
    print(f"== {name} d={S.d} T={T:.4f} ({T/S.period:.2f} periods) lam={S.lyapunov:.3f}", flush=True)
    print("   ref uncertainty:", {k: '%.1e' % v for k, v in
                                  reference.reference_uncertainty(S, T).items()}, flush=True)
    for N in ([96, 192, 256] if S.d < 10 else [96, 128, 192]):
        t0 = time.time()
        try:
            cell = core.solve_cell(S, T, N, warm_rtol=1e-8, warm_atol=1e-11)
            e = core.errors(core.model_trajectory(cell, ts), Yref)
            ew = core.errors(core.warm_trajectory(cell, ts), Yref)
            ei = core.interpolation_floor(S, T, N, ts, Yref)
            print(f"   N={N:4d} W={cell['W']:4d} it={cell['iters']:2d} warm={ew[0]:.2e} "
                  f"solve={e[0]:.2e} interp={ei[0]:.2e} ({time.time()-t0:.1f}s)", flush=True)
        except Exception as ex:
            print(f"   N={N:4d} FAILED: {type(ex).__name__}: {ex}", flush=True)
