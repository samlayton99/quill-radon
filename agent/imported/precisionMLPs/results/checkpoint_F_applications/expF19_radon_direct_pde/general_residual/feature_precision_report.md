# Actual QUILL feature precision

The feature bank can be calibrated far more strictly than its former fixed `1e-8` gate. The important arithmetic improvement is to evaluate the prescribed **anchored tanh differences** before summation. It removes cancellation between saturated halo contributions and a large bias. It does not substitute exact Legendre values or derivatives.

Code changes are isolated to `solver/general_features.py` and `tests/test_quill_general_features.py`. The feature constructor now accepts `encoding_tolerance` and `evaluation='standard'|'anchored'`. Existing defaults remain `1e-8` and `standard`. Strict automatic calibration checks additional center counts and bandwidths. A failed requested tolerance raises an error rather than silently accepting an inaccurate bank.

For the anchored mode the actual representation evaluated is

\[
 P_j(-1)+\sum_k w_{kj}
 [\tanh(\gamma(x-c_k))-\tanh(\gamma(-1-c_k))].
\]

When both tanh arguments are saturated with the same sign, their difference is evaluated through small logistic tails. This avoids subtracting two numbers that have both rounded to ±1. First and second derivatives are still analytic derivatives of the same tanh terms. Halo count is `ceil(sqrt(N))` per side, with **N counting interior centers**.

## Geometry and arithmetic sweep

150 bank constructions: degrees8,12,16; interior counts65,129,257,513,1025; lambda .10,.15,.20,.25,.30; standard and anchored evaluation. Independent validation uses endpoint-concentrated cosine points plus nonuniform interior points. It checks every mode separately for derivative orders0,1,2. The error score is the largest sampled per-mode absolute error divided by `max(1, max_abs_exact_mode_derivative)`.

| Maximum degree | Best standard score | Best anchored score | Anchored geometry | Anchored score / FP64 epsilon |
|---:|---:|---:|---|---:|
| 8 | 1.50e−14 | 5.69e−15 | N257, lambda .20 | 25.6 |
| 12 | 5.21e−14 | 5.69e−15 | N257, lambda .20 | 25.6 |
| 16 | 8.67e−14 | 7.11e−15 | N257, lambda .20 | 32.0 |

These scores are **not one machine epsilon**. The largest absolute second-derivative error for the anchored degree16 bank is2.55e−11 because high-degree Legendre second derivatives have large magnitude. Low-mode error must be considered separately: the degree1 second derivative should be zero, and its sampled noise floor is about5.7e−15 in this configuration. A physical axis of length1 multiplies normalized second derivative errors by4.

More neurons do not monotonically improve FP64 precision: slope gamma grows at fixed lambda, so derivative evaluation increasingly amplifies roundoff. The anchored bank reaches a floor around N257 for this sweep.

These are worst-case bank errors, **not lower bounds on the error of a specific learned field**. Coefficients can emphasize much more accurate modes and errors between modes can cancel. The coordinator's independent analytic-capacity Navier–Stokes control reaches much smaller field and PDE errors than this worst-mode bound. That control is not provided to the PDE solver.

## Arithmetic diagnostic

At degree16,N513,lambda .20, nine representative points were recomputed with55-decimal arithmetic, keeping the constructed coefficients at their original FP64 values. Exact derivatives of Legendre polynomials were also evaluated at55decimals.

- For P1's second derivative, ordinary FP64 summation had5.998e−15 maximum error; `math.fsum` reduced it to2.639e−15;55-digit evaluation reduced it to1.533e−16.
- Higher modes retain coefficient-rounding error even after high-precision evaluation. Across all modes, the maximum absolute second-derivative error decreased from1.091e−11 to8.334e−12.
- Value evaluation using55digits and the unchanged coefficients reached7.86e−16 maximum error across the tested modes and points.

This is a nine-point diagnostic, not a uniform guarantee. It shows both evaluation error and coefficient-rounding error. Compensated summation alone does not eliminate the second derivative floor. On this Mac, `numpy.longdouble` has the same precision as FP64; it is not an extended-precision option. No high-precision coefficient-construction backend was added.

## Checks and artifacts

All10 tests in `tests/test_quill_general_features.py` pass. They cover physical-coordinate derivatives, zero/constant modes, multivariate products, finite differences of the actual network, strict tolerance rejection, configuration validation, and the anchored implementation against45-digit evaluation of its stored tanh coefficients.

- `feature_precision_sweep.json`: standard evaluation, full per-mode errors.
- `feature_precision_anchored.json`: anchored evaluation, full per-mode errors.
- `feature_precision_arithmetic.json`:55-digit diagnostic and compensated sums.
- `feature_precision_adaptive.json`: automatic selections for requested1e−13 and1e−14 tolerances.
- `feature_precision_summary.png`: errors in multiples of FP64 epsilon.
- Reproduce with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python experiments/expF19_radon_direct_pde/feature_precision_study.py`.
