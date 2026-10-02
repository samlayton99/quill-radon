# Native QUILL Navier–Stokes pilot

This small experiment evolves the actual QUILL tanh field from its prescribed initial condition. It consumes no saved trajectory and no external PDE solution. It also has an exact interpretation as a modified pseudospectral method in compressed coordinates; it does not establish a new way to avoid numerical PDE evolution.

The standard interacting Taylor–Green initial condition is `(sin(x)cos(y)cos(z), -cos(x)sin(y)cos(z), 0)` on `[-pi,pi]^3`, with viscosity `.05` and final time `.05`. All runs use 25 RK4 steps of `.002` and lambda `.2`. The K=2/3 runs use a `12^3` analysis grid, 193 interior centers per primitive direction, and 14 halo centers on each side. The appended K=5 refinement uses an `18^3` grid, 257 interior centers, and 17 halo centers per side.

1. Define `E` by applying the existing exact QUILL encoder to the entire real sine/cosine basis profiles. The complex-shift encoder never receives its own meromorphic tanh field.
2. Construct `S E`, its first derivatives, and its Laplacian from actual tanh evaluations at the real analysis nodes. Cache these fixed matrices. The final exported network is also evaluated independently from its ordinary tanh readouts.
3. At every RK4 stage, evaluate the neural velocity, gradient, and Laplacian; form `-u dot grad(u) + nu lap(u)`; use FFT and the Leray projection to obtain the resolved, divergence-free increment. Evolve `b_t = A_K F(S E b)`, equivalently `a_t = E A_K F(S a)` with `a=E b`.
4. After the native runs in each batch finish, generate fresh conventional spectral comparison solutions. The K=2/3 batch precedes the later K=5 refinement, which creates another fresh K=9 comparison after its own evolution. These comparisons are verification data only.

| Quantity | K=2 | K=3 | K=5 |
|---|---:|---:|---:|
| Tanh neurons | 10,829 | 32,045 | 167,907 |
| Primitive directions | 49 | 145 | 577 |
| Relative velocity error vs fresh K=9 | 2.18e-4 | 3.98e-6 | 1.94e-9 |
| Neural vs ordinary spectral at the same K | 1.48e-15 | 1.64e-15 | 1.47e-15 |
| Native dt vs dt/2 difference | 4.95e-16 | 1.96e-15 | Not repeated |
| Maximum neural divergence at analysis nodes | 2.91e-16 | 1.08e-15 | 2.54e-15 |
| Construction and caching seconds | 0.303 | 0.892 | 15.136 |
| Native evolution seconds | 0.029 | 0.058 | 0.780 |
| Ordinary spectral evolution seconds | 0.064 | 0.067 | 0.177 |

Errors use 256 fresh Sobol interior points and 48 cube-face points. Independent K=7 and K=9 spectral references differ by 1.18e-12. The native time-refinement differences are at roundoff and therefore do not establish an observed time-convergence order. Independent collapsed-readout/cache agreement is approximately 1e-15 for values, gradients, and Laplacians.

The initially zero vertical velocity develops RMS 0.00434, and initially absent mode coefficients develop norm 0.00869. Thus this is nonlinear coupled flow, not a vanishing-advection shear or Beltrami special case.

The spectral cutoff dominates error. K=5 reduces the K=3 error by approximately 2,054 times and uses 186.3 MB of cached matrices; peak process memory was 400.2 MB. Its construction plus evolution took 15.916 seconds, and native evolution alone was 4.40 times slower than the ordinary spectral comparison. Cached dense multiplication happens to beat the simple FFT comparison at the two smaller sizes, but these are not optimized performance comparisons. Matrix storage grows as analysis samples times resolved modes. Finite tanh fields are only approximately periodic; the Fourier closure treats their real samples as periodic. Quadratic dealiasing is exact for the retained trigonometric basis and approximate for the tiny non-Fourier tails of its QUILL approximation.

Reproduce from the repository root:

```sh
.venv/bin/python experiments/expF19_radon_direct_pde/native_ns_pilot.py
.venv/bin/python experiments/expF19_radon_direct_pde/native_ns_refine.py
```

`ns_native_metrics.json` contains measured details. `ns_native_K*.npz` stores compressed state and validation fields. `ns_native_network_K*.npz` stores the ordinary tanh geometry/readouts. `ns_native_comparison.png` separates error sources, evolution time, and construction time. No training or readout solve is used.
