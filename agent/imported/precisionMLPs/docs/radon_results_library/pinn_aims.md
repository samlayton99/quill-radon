# What PINNs aim to offer, and what our methods actually offer

This is the consolidated requirements and evidence record for the Radon-PINN program, dated October 1, 2026. It separates the field's motivations, our additional requirements, and observed results. “PINNs promise this” means a research aim or advertised capability; it is not a theorem that every implementation delivers it.

The original PINN framework joins PDE constraints with observations for forward solution and inverse identification. The broader physics-informed-learning literature emphasizes combining imperfect data with physics, reducing dependence on conventional meshes, and addressing parametric/high-dimensional problems. [Raissi, Perdikaris and Karniadakis, 2019](https://doi.org/10.1016/j.jcp.2018.10.045), [Karniadakis et al., 2021](https://doi.org/10.1038/s42254-021-00314-5).

Uncertainty quantification requires additional machinery. For example, Bayesian PINNs model uncertainty from noisy observations using a probabilistic inference procedure; ordinary residual fitting does not automatically provide calibrated uncertainty. [Yang, Meng and Karniadakis, 2021](https://arxiv.org/abs/2003.06097). Successful examples also do not imply reliable optimization across equations: published failure studies demonstrate failures even on relatively simple PDE families. [Krishnapriyan et al., 2021](https://arxiv.org/abs/2109.01050).

## Core forward and data capabilities

The success criteria below are our operational interpretation of these aims, not quotations or claims that the literature guarantees them. Status refers to the **current single-hidden-layer native Radon route** unless another route is explicitly named.

| Aim | What would count as delivering it | Existing evidence | Current status |
|---|---|---|---|
| Solve from physics without interior solution labels | Use only equation, forcing, IC/BC and physical gauges; validate independently | Smooth elliptic and NS cases; wall-driven disk flows with zero body force | **Demonstrated in selected cases**; hard cases fail |
| Handle nonlinear and coupled equations | Same residual interface for genuinely nonlinear multi-field systems | NS advection/coupling, semilinear elliptic, Allen–Cahn attempts | **Partial**; declaring a PDE is broader than reliably solving it |
| Fuse sparse/incomplete observations with physics | Measurements at specified locations/components constrain otherwise unknown fields/parameters | Five-location manufactured NS inverse; 16-location wall-NS inverse | **Demonstrated narrowly**; not arbitrary sensor coverage |
| Handle noisy measurements | Fit to the declared noise model while retaining valid physics | Reduced diffusion inverse; specialized Burgers noise studies | **Partial**; noisy wall NS unfinished, general noisy-data robustness unestablished |
| Mesh-free sampling | Evaluate constraints at scattered/boundary points without a volume mesh | Collocation-based native implementation | **Implemented**, but sampling/quadrature and spatial coverage still matter |
| Arbitrary/complex domains | Declare geometry and BCs without a hand-derived basis or equation-specific solver | Disks, boxes, affine variants; older mapped-annulus route uses noncompliant architecture | **Limited**; cornered cavity remains hard, arbitrary geometry unproved |
| Continuous queryable solution and derivatives | One network returns fields and derivatives anywhere in its domain | Ordinary tanh export plus held-out derivative checks | **Implemented**, with cancellation and derivative-error limits |
| Enforce boundary, initial, periodic and gauge constraints | Distinguish exact algebraic constraints from approximate sampled penalties | Mixed constraints, pressure gauges, IC pairing controls | **Partial**; exact symbolic identities can still lose accuracy in deployed arithmetic |
| Generalizable physical consistency between samples | Held-out residuals, refinement and ideally an error bound | Richer-model checks and ordinary-MLP audits | **Sampled evidence**, no general continuum certificate |

## Inference and repeated-use capabilities

| Aim | What would count as delivering it | Existing evidence | Current status |
|---|---|---|---|
| Infer physical parameters | Identify coefficients from sparse observations, with true/reference error and identifiability checks | Reduced native wrapper; analytical heat tensor; native Burgers tangents | **Demonstrated narrowly**; failures and same-method references disclosed |
| Infer unknown fields, sources or initial conditions | Recover a function under declared prior/regularization, not just one scalar | D03's 17-coordinate initial profile | **Restricted**; no general unknown coefficient-field inverse |
| Discover governing equations / hidden physics | Infer equation structure or unknown constitutive laws, not just supplied-law parameters | No adequate current demonstration | **Not demonstrated** |
| Quantify uncertainty / detect nonidentifiability | Calibrated intervals/posteriors or sensitivity-based diagnostics validated under repeated noise | Some specialized sensitivity/noise controls | **Not general calibrated UQ**; point-estimate accuracy is insufficient |
| Solve parameterized families in one surrogate | One network accepts physical parameters and position without refitting each query | Parameterized diffusion family | **Demonstrated in one simple family**; not a universal neural operator |
| Reuse/transfer across related problems | Reduce total new-solve effort without importing forbidden solution labels | Native continuation between degrees/viscosities | **Useful restricted continuation**, not validated broad transfer |
| Efficient repeated inference and sensitivities | Count construction plus query cost; distinguish parameter-input derivatives from differentiating an entire solve | Compact final MLP; analytic/AD jets; parametric derivative audit | **Implemented**; no universal speed advantage over established solvers |
| Combine heterogeneous/multifidelity data | Handle different sensors/noise fidelities and retain sound weighting | Observation abstraction supports data blocks; little systematic validation | **Not adequately benchmarked** |

## Robustness and scaling ambitions

| Aim | What would count as delivering it | Existing evidence | Current status |
|---|---|---|---|
| Avoid high-dimensional mesh costs | Show actual memory/time/accuracy scaling against dimension and representation | Streamed coordinates; real 5D control; separate low-degree/known-sparse high-d examples | **Memory progress**, not escape from generic angular/approximation cost |
| Handle stiffness, multiscale dynamics and long time | Fixed/adaptive general policy succeeds across distinct difficult cases | Contrast normalization helps; Allen–Cahn, low-viscosity Burgers and hard flows remain problematic | **Unmet generally** |
| Handle discontinuities and entropy solutions | Weak/conservation formulation selects the physical solution in a suitable norm | Tiny supplied-front C09 study | **Unmet generally**; smooth strong-form residual is not sufficient at a shock |
| Detect blowup / singular behavior | Solve the actual PDE/forcing up to a justified time and distinguish refinement from imposed scaling | Only supplied concentrating-field representation control | **Not demonstrated** |
| Routine setup with reliable stopping | User supplies mathematical problem data, tolerance and budget; solver selects stable resolution/policies and reports honest failure | General declaration API, native gates, budget and validation records | **Partial**; current studies still expose tuning, stalls and soft-budget overruns |
| Accuracy improves predictably with resolution | Separate angular, scalar, solve and arithmetic errors; establish a usable rate for a stated target class | Known-function theory and selected ladders; ordinary-arithmetic counterexamples | **Conditional**, not arbitrary-precision assurance for every PDE |
| Compete with standard PINNs/classical methods | Matched tasks, information, precision, hardware and end-to-end costs | Isolated historical baselines, not a complete matched campaign | **Not established broadly** |

## Additional requirements from this project

These are stronger or more specific than the usual PINN aims. They must remain visible rather than being retroactively attributed to all PINN papers.

| Project requirement | Why it matters here | Present accounting |
|---|---|---|
| One global single-hidden-layer MLP | Preserve the actual Radon ridge-network object | Current route2 outputs comply; deep/product/time-slab routes are marked outside scope |
| No solve-elsewhere-then-encode as a blind-PDE result | Construction precision must not disguise external solution discovery | Main native route complies; external evolution/compiler branches remain explicit controls |
| No unnecessary extra dimensional explosion | Memory is the main bottleneck; numerical help is allowed when structurally scalable | Streaming removes global arrays, but O(MpP) conversion and iteration growth still count |
| Roughly 1e-14 when mathematically/statistically appropriate | Accuracy and useful scaling matter more than minimal runtime | Selected smooth cases meet or approach it; no general guarantee; noisy recovery judged statistically |
| Generality without supplied hidden solution structure | Avoid success only on hand-picked low-degree, sparse or analytical families | Such families are retained and clearly labeled; broad hard-case robustness remains incomplete |
| Evidence before broad claims | Ordinary deployed model, independent sampling, reference provenance, all failure outcomes | Current archive separates these; older claims must be read with their qualified evidence |

## What should not be conflated

A no-fit construction for a known function answers **how to represent supplied information**. A PINN answers **how to infer an unknown solution from equations and constraints**. They share an encoder but do not supply the same missing information.

Similarly, fewer readout unknowns is not zero numerical solving; matrix-free is not dimension-free; low noise error is not uncertainty quantification; a supplied analytical solution is not blind inference; and a low sampled residual is not a proof of the true error. A well-posed PDE and identifiable observations are necessary parts of the problem specification, not optional solver features.

Current evidence is organized by the corresponding use case in [rankings](rankings.md), with detailed [forward](evidence/forward_and_controls.md), [NS/dimensional](evidence/navier_stokes_and_dimensions.md), [inverse](evidence/inverse_problems.md) and [failure](evidence/failures_and_limits.md) pages. No new numerical work was performed while consolidating this aims document.
