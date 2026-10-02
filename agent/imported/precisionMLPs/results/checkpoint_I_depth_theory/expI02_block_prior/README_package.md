# expI02 package: what I did, what worked, what did not

Written 2026-09-05 for Sam. Everything here is single-seed unless stated and is a prior, not a result. Read this file first, then `expI02_results.md` for the full writeup and tables, then the figures.

## What I built

A corrected block family per your notes of 2026-09-05, in `code/qi2.py`:

- fixed mesh at every depth (centers cell-centered on [-1, 1], gamma h = 0.25, never moved), free projection in layer 1 (no unit-norm directions), no forward normalization of any kind;
- one calibration pass at init (every bank input to mean 0, sd 0.4) plus a weak hinge penalty on the actual channel values during training (|mean| <= 0.25, 0.15 <= sd <= 0.6, excursions past 0.9), weight 1e-2;
- coefficient coordinates as a fixed linear map on the parameters: raw, function-value (w_j = (a_j - a_{j-1})/2), capped whitening of the bank Gram (cap 1e3), and both;
- smooth nonzero profile init that is the same function in every coordinate system, so coordinate arms differ only in optimization;
- heads: solved inside the forward pass on that pass's own features (the Kaufman VarPro gradient, zero extra cost), or trained by Adam with the solved-head error logged alongside, or replaced by a solve every 250 steps;
- optimizers: Adam, heavy ball (lr 1/L from a power iteration), mixed (Adam on the directions, heavy ball on the profiles);
- depth 2 to 4, residual streams, rank-S mixers, an MLP baseline with the same interface, and matched-parameter / matched-FLOP width selection.

Twenty-four tests pin the research facts: the mesh, the coordinate maps as exact reparametrizations, the capped spectrum, the 1-D machine-precision floor through the stack, the expH05 cliff cell, calibration, the band penalty, the in-pass solve, the PDE residuals, and the expF01 Poisson floor through the operator solve.

## The studies and the honest outcome

**A0, frozen 1-D features (final, solid).** Your fixed-feature findings reproduce: whitening 3-4 orders, momentum 30-300x over Adam on the whitened problem, function-value coordinates another 3-10x only on top of whitening under momentum. The best first-order number is still six orders above the solved head.

**A1, the block ladder at d = 3, 2000 steps (final, seed 0 with seeds on the top rungs).** None of it transfers. Raw coordinates + Adam + head solved every step won. Whitening was neutral to harmful and seed-unstable under Adam. Heavy ball was 10x worse; mixed 3x worse. The learning rate (0.02 vs 0.005) moved the result 6x, more than any mechanism. Calibration + penalty matched the old range-tracking recipe within 2x, and beta = 0 was indistinguishable from beta = 1e-2. Rank 1 and 2 lost 2.5-3x to the untied tensor. Periodic head replacement under Adam was dead for a reason already on the kill list (Adam's normalized first steps wreck a freshly solved head). Every learned number sat 3-6 orders above its oracle floor.

**A2, scaling at d = 4 (final, seed 0).** No knob (N1, N2, M, K) moved the error more than 3x at fixed steps. Three- and four-layer stacks did not train (0.2-0.8 error); residual streams recovered a little. The plain MLP's width ladder scaled cleanly and overtook the block on composition at about 5k parameters.

**B1, d = 5 head to head at matched params and FLOPs, 4000 steps, 3 seeds (final).** First order, the block was within 3x of the shallow ridge-QI and of the MLP in either direction, with a much wider seed spread. The MLP won the random-ridge control 5x. Dense Gauss-Newton gave the block 1-2.5 orders on product peak, three bumps and composition; that finisher does not scale.

**B2, structured regression, 3 splits (final after a fix).** Block 5x better than the MLP on the noise-free Lorenz flow map, parity on kin8nm, 13-17% worse on Friedman1 and concrete, all far ahead of ridge. The first pass had the block below ridge on three sets: my fp32 solve truncated at 1e-14, which is meaningless at fp32 precision. Fixed with a dtype-derived floor and validation of the truncation before training.

**B3, Fashion-MNIST (final, one seed).** Block 0.6-1.0 points behind matched tanh and ReLU MLPs at 36k and 88k parameters. Its least-squares-refit head already scores 75% on calibrated random features before training.

**B4, PINNs (final for the first protocol).** With Adam on the PINN loss neither model trained in 3000 steps; the block trained worse than the MLP on 2-D Poisson and Burgers. The operator-system head solve on the block's features reached 3e-7 on 1-D Poisson, where the matched MLP's features could not solve at all; on 2-D Poisson it did not help. A second protocol with the head replaced by the operator solve every 250 steps during training was worse everywhere (same kill-list mechanism) and is kept only as a failed variant.

**Profile gifs (in progress when this package was made).** Four runs of the 8 x 8 x 128 block from zero profiles (composition, gauss bump, fast waves seeds 0 and 1); a good and a bad case will be sent separately.

## Scorecard of my seven predictions

1. Whitening the biggest win in the block: held on fixed features, failed in the block.
2. Momentum over Adam, mixed wins: held on fixed features, failed in the block.
3. Calibration + weak penalty matches range tracking: held.
4. Block 3+ orders over the MLP on factorizable d = 5 targets, loses the control: first half failed, control held.
5. Friedman1 both at the floor, Lorenz 1-2 orders, tabular parity: mixed (Lorenz 5x, kin8nm parity, Friedman1 and concrete to the MLP).
6. Fashion-MNIST within a point: held.
7. Linear PDEs 3+ orders via the operator solve: held in 1-D only.

## Mistakes I made and fixed, in the open

- Whitener computed on the whole mesh instead of the occupied band: 20-100x loss on the fixed-feature ladder. Fixed before the ladder consumed it; A0 was re-run.
- The mixed optimizer measured its heavy-ball step from the curvature of all parameters instead of its own group. Fixed; the two mixed rungs re-run (3.6e-2 to 1.5e-2, still worse than Adam).
- The "periodic head" variant first froze the head between solves (not what your notes meant); re-implemented with the head trained in between; both versions are bad, for the kill-list reason.
- fp32 head solves at rcond 1e-14 (see B2). Fixed with a dtype floor; B2 re-run; the first pass is kept as `b2_v1_lr5e-3.json`.
- The first A1 pass crashed at the variants phase on a type check of mine; resumed from the saved rows.
- B4's second protocol was my idea and made things worse; the first protocol is the reported one; the second is kept as `b4_v2_periodic_opsolve.json`.
- The figures use a fixed 1e-15 to 10 axis on data that lives in two decades, so most panels are flat stripes. Bad choice on my part; the numbers are in the tables (`tables.txt`).

## What is not trustworthy

- Single seed everywhere except A1's top rungs and B1/B2. Fast waves is a seed lottery (2e-3 or 1e-1 to 6e-1) in every arm.
- Every block number is at 2000-4000 first-order steps; the record's E2 row reached 1e-6 on d = 3 fast waves with 4000 steps and 40 Gauss-Newton iterations, so budget matters and I did not push it.
- One learning rate per study after one two-point check; whitened coordinates change the parameter scale and were not given their own lr sweep.
- Wall-clock numbers in the JSON were partly taken while another study shared the machine.

## What I would run next

The open lever is the optimizer for the channel layer (the checkpoint-D step-2 solver applied to the channel tensor, or a scalable Gauss-Newton), not coordinates or normalization. Depth needs its own init and lr study before it means anything. The block's clear wins are all "a head solved on QI-resolved, noise-free structure"; that is the use case to build on.

## Package contents

- `expI02_results.md`: the full writeup (design, every table, per-figure how-to-read, scorecard).
- `tables.txt`: every table printed from the JSON files.
- `figures/`: fig1 (fixed-feature ladder), fig2 (block ladder and solved-head gap), fig3 (scaling), fig4 (head to head), fig5 (Fashion-MNIST), fig6 (PINNs); profile gifs and their final frames if finished.
- `results/`: the study JSON files (`a0` .. `b4`), the superseded passes (`*_v1_lr5e-3`, `b4_v2_periodic_opsolve`), and the quick-mode smoke files.
- `code/`: `qi2.py`, `tasks.py`, `run.py`, `plots.py`, `tables.py`, `profile_gif.py`, the tests, the spec and the plan.
- Logs of every run.
