# QI Construction

Builds explicit quasi-interpolant tanh MLPs that approximate smooth targets to
high precision. See `papers/practical_implementation.tex` for full details.

## Quick start

```python
from src.construction import construct_qi, evaluate_qi
from src.data.targets import get_target

target = get_target("sine")

# Fast fp64 path (default): ~10ms, L_inf ~ 1e-12
qi = construct_qi(target.fn_numpy, target.deriv_numpy, N=64)

# Machine-epsilon path: ~55s cold, ~0.25s cached, L_inf ~ 3e-15
qi_exact = construct_qi(target.fn_numpy, target.deriv_numpy, N=64,
                        precision="mpmath")

# Evaluate
import numpy as np
x = np.linspace(-1, 1, 2001)
y_pred = evaluate_qi(qi, x)
```

## Two precision regimes

| | fp64 | mpmath |
|---|------|--------|
| `lambda_star` | 0.30 | 0.25 |
| `Kc` | 160 | 160 |
| halo | `default_halo(N, 0.30)` | `default_halo(N, 0.25)` |
| precision achieved | ~1e-12 | ~3e-15 (fp64 eps) |
| cold runtime | ~10ms | ~55s |
| cached runtime | ~10ms | ~0.25s |

Both paths produce valid fp64 coefficients. mpmath is offline precomputation,
not a violation of fp64 training assumptions.

## Cache

Cardinal coefficients `c_j` depend only on
`(lambda_star, Kc, N, precision, mp_dps)` and are target-independent.
Cached at `results/qi_cache/` by default.

- Override cache location: `cache_dir=Path(...)` or `QI_CACHE_DIR` env var.
- Disable: `use_cache=False`.

## Per-experiment recommendations

Use **mpmath** when the construction is a fixed reference:
- expD01 (geometry ladder), exp13 (solution basins).

Use **fp64** everywhere else:
- training runs, sweeps, initialization (all other experiments).

## Files

- `qi_mpmath.py` — `construct_qi`, `evaluate_qi`, `default_halo`, caches.
- `readout.py` — `build_phi`, `solve_readout`, `solve_readout_with_bias`.
- `initialize.py` — `initialize_from_construction`, `initialize_and_freeze`, `initialize_with_readout_solve`.

## Parameter warnings (read before changing defaults)

- `lambda_star >= 1.0` gives intrinsic aliasing ~ `exp(-pi^2/lambda)` > 1e-3. Too large.
- `Kc < 80` is insufficient: cardinal coefficients decay slower than the
  simplified rate `exp(-2*lambda*Kc)` predicts. Default `Kc=160` matches continuous-mlps.
- `halo` must scale with both `N` and `1/lambda`. Use `default_halo(N, lambda_star)`.

## Validation

The fp64 path has been validated against the production continuous-mlps
sweep: both achieve ~1e-12 L_inf, both plateau at the fp64 convolution
cancellation floor. The mpmath path matches the paper's theoretical 10^-15
claim (verified: `TestPrecisionFlag::test_mpmath_path_reaches_machine_eps`).
