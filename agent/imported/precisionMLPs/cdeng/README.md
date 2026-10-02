# Quasi-interpolant benchmark work

This directory contains the quasi-interpolant representation experiments and
results developed by cdeng against the BWLer PDE and Dysts chaotic-system
suites.

## Experiments

- `experiments/expF15_tensor_suites`: separation-rank and constructive
  tensor-QI studies, including the analytic
  `sin(x+y)/sqrt(1+x^2)` comparison.
- `experiments/expF16_full_matched_suite`: the consolidated BWLer vs Radon vs
  tensor-QI benchmark at a common fitted-coefficient budget.

## Headline result

At a cap of 1,156 fitted scalar coefficients, tensor-QI wins all seven PDE
representation tests. On eight Dysts trajectories the result is mixed: BWLer
wins three, Radon wins four, and tensor-QI wins MacArthur.

These are oracle interpolation/representation ceilings, not trained PDE or ODE
solve errors. The full protocol and table are in
`results/checkpoint_F_applications/expF16_full_matched_suite/README.md`.
