# Audit: route 2 / streamed native residual solver (Oct 1-2, 2026)

Agent-side document (preliminary). Read-only audit subagent, Oct 2, 2026; JSON values spot-checked; no reruns. Paths relative to `agent/imported/precisionMLPs/`. Results paths under `results/checkpoint_F_applications/expF19_radon_direct_pde/`.

## 0. Bottom line
1. **Route 2 is a Legendre/Zernike polynomial spectral least-squares collocation solver whose iterate is compiled exactly into a one-hidden-layer tanh MLP.** Unknowns are total-degree-p polynomial coefficients (P = binom(p+d,d) per field in the box chart; (p+1)(p+2)/2 disk polynomials in 2D). An exact Radon (Funk-Hecke/Gegenbauer) identity over a fixed tensor sphere rule with M = (p+1)^(d-1) directions turns the polynomial into ridge profiles C_n^(d/2)(omega . z); each profile is QUILL-encoded into H = N + 2 ceil(sqrt N) tanh neurons. The export really is `Linear/Tanh/Linear` on the original inputs, but every inner weight is fixed by quadrature and QUILL; only the readout varies, and only through the linear map a -> v. The function space is exactly polynomials of degree <= p, up to ~1e-15 encoding error. In the native preset the Krylov Jacobian is the polynomial ("ideal") Jacobian; the tanh network only evaluates the outer residual, accepts steps, and validates. Network and polynomial agree to 1e-11 to 1e-15. The imported docs never say this plainly ("analytical readout coordinates", "polynomials are not in the deployed model" are true but hide it).
2. **~1e-15 accuracy is shown only on smooth, low-frequency, polynomial-friendly problems.** Harder cases fail or exhaust budgets.
3. **Requirements:** one hidden layer PASS; no external solver FORMALLY PASS (no outside trajectory, but it is a classical spectral solve -- Sam's call whether that is "dressing up another method"); not brittle in d FAIL; memory bound by inference PARTIAL; plug-and-play PARTIAL (no named-PDE dispatch, but many hand-tuned settings).

## 1. Algorithm
- **Chart.** z = (x - m) * s, s_i = 2/((b_i - a_i) sqrt d): box into its enclosing unit ball.
- **Ideal field.** u_ideal = sum_{|alpha| <= p} a_alpha prod_i P_{alpha_i}(sqrt(d) z_i) -- the "product features" / "polynomial coordinates".
- **Radon step (exact).** u_ideal = beta + sum_m sum_{n=1..p} E_mn(a) C_n^(d/2)(omega_m . z), omega_m, w_m from a folded positive product cubature exact through degree 2p (`ridge_frame_dimension_study.sphere_rule`). E via exact ball moments (`ball_ridge_features`, mpmath for p > 8) or a factored Wick/Laplacian recursion (`route2_box_profiles._projection_factors`), never storing the M x p x P map.
- **QUILL step (exact).** `quill_boundary.encode`: t_j = -1 + jh, gamma = lambda/h, lambda = 0.2, halo ceil(sqrt N) per side, complex-shift contour readouts with rational halo correction. Deployed u(x) = b + sum_{m,j} v_mj tanh(gamma omega_m . z(x) - gamma t_j), B = M H neurons.
- **Neurons vs unknowns** (B/P grows with d; accuracy set by p, not neurons; N fixed at 257/129, so QUILL's geometric convergence in N plays no role; no neuron scaling law measured):

| Case | Neurons B | P per field | B/P |
|---|---|---|---|
| 2D box p36 | 10,767 | 703 | 15 |
| 2D disk p40 | 11,931 | 861 | 14 |
| 3D NS p16/N257 | 84,099 | 969 | 87 |
| 3D NS p20 | 128,331 | 1,771 | 72 |
| 4D NS p16/N129 | 751,689 | 4,845 | 155 |
| 4D semilinear p10 | 387,321 | 1,001 | 387 |
| 5D semilinear p8/N129 | 1,003,833 | 1,287 | 780 |

- **Residual.** User supplies `ResidualDeclaration(domain, make_blocks(count, seed), fields, params)` with `ResidualBlock`s (points, pointwise torch callback F(x, jets, theta), derivative multi-indices, weight, scale, eq/le/ge). Jets from the actual tanh network in bounded batches. Q_b = max(256, oversampling * P), oversampling 6 default, 2 in NS runs. Optional `residual_scaling.normalize_equations` (caller picks blocks).
- **Solver.** `solve_route2`: degree ladder; per degree memory preflight, embed previous coefficients, `solve_residual` = damped GN/LM, each step min ||J d + r||^2 + lam ||d||^2 by scipy LSMR, right-preconditioned by block-Jacobi Cholesky of regularized Gram blocks (parity groups, width s = 1024 / 640 / 2805), line search, stationarity checks. Native preset: J_ideal = callback sensitivities at the tanh state x exact Legendre/Jacobi jets of polynomial coordinates; sum factorization when d <= 8 and (p+1)^d fits 4 MiB. Acceptance by fresh normalized residuals, raw residuals by torch autograd on the exported MLP, successive-degree agreement.
- **Inverse.** `route2_inverse.solve_route2_inverse`: outer GN over a few physical parameters with finite-difference sensitivities; each theta needs a full validated physics-only forward before its sensor misfit counts.
- **Costs.** Memory O(F P s) block factors (~435 MB in 4D NS) + O(Q E J) sensitivities + model O(B(d+F)) (dominates in 5D). Each tanh pass O(Q B (d+J)) + O(F M p P) conversion. Each ideal Krylov product O(Q P J). With Q ~ P, dominant ~K P^2 (K total Krylov products: hundreds to the 4000 cap; no growth law measured). "Streamed" holds for storage, not inference-bound arithmetic.
- **Not brittle in d: FAIL.** Tensor rule M = (p+1)^(d-1), ~(d-1)! more directions than the harmonic dimension needs; sparse rules only p <= 3 with signed weights; B/P 14 -> 780; Gegenbauer magnitudes and monomial cancellation worsen with p and d (cavity p32 encoding perturbs PDE by 5.2e-6; p40 readout L1 2.8e5); docs' own forecast p16 four fields: 103 MB 4D, 1.94 GB 5D, 36 GB 6D.
- **Plug-and-play: PARTIAL.** Hand-picked per run: chart (box/disk/affine_disk/sparse; disk 2D only), degree ladder, N and lambda (257/0.2 vs 129/0.25), damping (1e-30 "linear class" vs 1e-6 "nonlinear class"), which blocks normalized, oversampling (6 vs 2), block size, inner iterations (100/150/300/4000), inexact_newton, refine_on_ideal_stall, reflection_orbits, tolerance (1e-13 to 1e-16), soft budgets; every experiment writes its own `make_blocks`. Domains: boxes, 2D disk/ellipse; annulus only via older cached features.

## 2. What the saved results show
- **3D steady NS p16** (`route2_hardening/ns/trig_p16_n257_native_ideal_correction*`): velocity 7.934e-15, pressure 1.557e-14, 84,099 neurons, `resolution_limit`; single-degree resume of a resume chain with tuned settings; quoted 211 s is final stage only.
- **Cold frozen-policy NS** (`route2_transfer_2026_10_02/N001`): velocity 1.376e-14, pressure 1.184e-14 at p20, 128k neurons, 169 s, 548 MB, all checks passed -- best-provenance NS evidence. Same family (frequency 0.7, nu 0.2). Harder N002 (bubble, freq 1.5, nu 0.03): 9.2e-10, corner max 3.8e-9, out of time.
- **4D transient NS** (`ns/transient_trig_p16_n129_native_fullgroups*`): velocity 1.845e-15, pressure 6.37e-14, corner field 1.32e-13, `time_budget` 1999.6 s; N and lambda changed mid-chain; single case; N003 never finished.
- **Disk wall-driven NS** (no known truth; residual and refinement checks only): nu 0.1 p40 momentum RMS 3.93e-15; nu 0.03 p48 6.71e-15.
- **Counterrotation nu 0.1 p64**: momentum 2.46e-8, `budget_exhausted` (120 s budget became 3,749 s); nu 0.03: 7.8e-3.
- **Parametric diffusion (x, mu)**: 1.63e-15, single simple family.
- **5D semilinear**: 2.73e-9, PDE RMS 1.73e-8, corner max 5.4e-7, `time_budget`. Target exp(x^T B x) with B ~ 0.02-0.03 so u ~ 1 + O(0.1): relative to the varying part, 20-50x worse. Transfer D001 3D 4.86e-15; D002 4D 1.11e-11 on time budget.
- **Wall-NS inverse** (`route2_battletest/inverse/wall_ns_reduced_inverse/summary.json`): nu = 0.029999999630522526 vs 0.03 in 1889 s; observations from a finer run of the same method (inverse crime); tolerance 1e-8; one scalar. Noisy sigma 1e-3 interrupted after 12 forwards.
- **Burgers**: inverse nu=0.02 relative parameter error 2.07, held-out 0.199; forward gates fail for nu 0.04 and 0.02; Oct 1 route-2 Burgers all `line_search_failed`.
- **Diffusion pair**: G024 7.63e-16 all checks; G025 9.74e-16 stopped on time.
- **Hard 2D transfer**: Helmholtz n2 7.45e-12; n4 8.6e-4; Allen-Cahn 0.53; H004 never finished.
- **Robustness**: multi-run evidence only for freq-0.7 NS, disk stirring at two viscosities, diffusion pair. Single cases: 4D NS, 5D, parametric, wall-NS inverse. Interrupted: noisy wall-NS inverse, recurrence campaign, first 5D baseline, D003/H004/N003. Soft time budgets not enforced.

## 3. Overfitting / quietly added structure
1. Manufactured targets suit polynomials (exp(x^T B x) with tiny B; NS freq 0.7 amp 0.2; quadratic NS control in p2 space; 10D/20D sparse examples inside the polynomial space). Quality drops at freq 1.5 or with stiffness.
2. Headline NS numbers from tuned chained resumes, not one frozen policy.
3. Oct 2 recipe developed on the same diffusion pair it was tested on (box chart, damping 1e-30, inner tol 1e-15, 4000 LSMR); manual linear/nonlinear damping policy.
4. Equation normalization added after the contrast-1000 failure; caller picks blocks.
5. Symmetry exploited: parity preconditioner blocks and `reflection_orbits` assume centered axis-symmetric boxes.
6. Active-axis prior: `route2_scaling_features.py` uses declared active axes and signed sparse rules for 10D/20D demos.
7. Inverse evidence: same-method reference, 1e-8 tolerance, single scalar.
8. Chart chosen per case; disk vs box conditioning differ ~1e7 at p32.

## 4. Oct 2 "coefficient access / representation-to-solve gap"
Files: `route2_gap_{diagnosis,native,preconditioner,summary}.py`; results `route2_gap_diagnosis/` (25 G runs, 7 P controls); `docs/radon_results_library/evidence/coefficient_access_2026_10_02.md`, `current_findings/scratchpad.md`. Question: 2D variable diffusion (contrast 10 and 1000) -- the space contains a ~1e-15 solution but the native solver plateaued ~1e-11 to 1e-12 (4.3e-3 at contrast 1000 before normalization). Findings: an earlier "p32" result had done zero p32 corrections (embedded p24 passing a loose stop test); chart conditioning dominates (p32 disk 1.23e11 vs box 4.82e4; worst parity block 5.1e10 -> 3.7e3); QR vs Gram Cholesky no difference; tight inner solves essential (loose G019 1.4e-5 with readout L1 2e8; tight 4.7e-15, 2.2e-15; full cold ladder G024 7.6e-16); higher degree not monotone (p40 loses digits); raw PDE residual stays 1e-12 to 1e-11 even at field error 1e-15. Interpretation: the "gap" was conditioning and stopping in a polynomial spectral LS solve; the combined fix was never ablated and was validated on one 2D linear pair; the transfer batch only partly carried it over.

## 5. Code dependencies
- `solver/route2.py` imports `solver/{ball_ridge_features, ridge_pinn_features, general_residual, general_adaptive}` (`general_residual` lazily imports `streamed_residual`) and top-level experiment scripts `route2_disk_profiles`, `route2_affine_disk_profiles`, `route2_box_profiles`, `route2_directional_operator`, lazily `route2_box_tensor`, `route2_scaling_features`. Those depend on `quill_boundary.encode`, `ridge_frame_dimension_study.sphere_rule` (a study script that also defines an OUT path), `solver/general_features` helpers.
- `solver/__init__.py` eagerly imports all legacy backends (pulls in native_burgers, native_tanh with matplotlib + MPLCONFIGDIR). Full closure 32 files ~500 KB; minimal core with emptied `__init__` 22 files ~379 KB. External: numpy, scipy, torch, mpmath, optional sympy. Nothing from `src/`.
- All 46 relevant tests `sys.path.insert(0, .../experiments/expF19_radon_direct_pde)`; several import experiment scripts as libraries (driver -> route2_parametric_study; hardening_ns; physical_flow + streamed_study; battletest_*; constraints -> route2_constraints_common, general_symbolic). No test reads `results/`.
- Every script sets `OUT = parents[2]/'results/checkpoint_F_applications/expF19_radon_direct_pde/<sub>'`; `route2_transfer_*` import ROOT/logging from `route2_gap_diagnosis`; saved configs hold absolute `/Users/sam/...` resume paths; `solver/pinn_baseline.py` expects `/private/tmp/codex-quill-deps`.
- `solver/README.md` stale (no route2/streaming); `solver/__init__` does not export `solve_route2`.

## 6. File tags
**Solver package:** route2.py CORE; general_residual.py CORE (83 KB; cached mode stores Q x P); streamed_residual.py CORE; general_adaptive.py CORE (solve_declared HISTORY); general_domains.py CORE; general_features.py CORE helpers / HISTORY products; ball_ridge_features.py CORE; route2_inverse.py CORE; residual_scaling.py SUPPORT; ridge_pinn_features.py SUPPORT; general_problem.py, general_symbolic.py SUPPORT; general_sparse.py HISTORY; adaptive_ridges.py HISTORY; `__init__.py` replace; README.md stale; base, tensor_box, elliptic, verification, ns_extended, inverse_profile, highdim, hyperbolic, reaction, pure_mlp HISTORY; pinn_baseline.py SUPPORT.

**Experiment-dir library modules:** route2_box_profiles CORE (high-p cancellation); route2_directional_operator CORE; route2_disk_profiles, route2_affine_disk_profiles CORE (2D only); route2_box_tensor CORE; route2_scaling_features SUPPORT (special structure); route2_disk_recurrence SUPPORT; route2_initial_plane_profiles HISTORY; quill_boundary CORE; ridge_frame_dimension_study CORE (extract sphere_rule).

**Route-2 scripts:** gap_diagnosis, gap_native CORE evidence; gap_preconditioner, gap_summary SUPPORT; transfer_highdim CORE (near-constant target; D003 unfinished); transfer_ns (+audit) CORE (N001 best NS evidence); transfer_hard2d SUPPORT (failures); hardening_ns CORE evidence (tuned chains); battletest_inverse_ns CORE (only nontrivial inverse); other hardening/physical/battletest/normalized/constraints/scaling/benchmark scripts SUPPORT; route2_burgers_*, frame_iteration, shock_diagnostic, initial_plane_study HISTORY.

**`general_*` scripts** (Oct 1 product-architecture solver): HISTORY (product gates violate one-layer rule); general_baseline.py (DeepXDE) SUPPORT.

**Tests:** CORE (20 files): test_route2_{driver, box_profiles, box_tensor, directional_operator, directional_batching, disk_profiles, affine_disk_profiles, disk_ideal_jets, multifield, multifield_engine, inverse}; test_quill_{ideal_correction, ideal_preconditioner, parity_preconditioner, streamed_residual, residual_stationarity, general_residual, residual_scaling, sphere_rule}; test_ball_ridge_features. SUPPORT: hardening, physical, battletest, recurrence, encoding-sweep, scaling, constraints. HISTORY: initial_plane, burgers_protocol, frame_iteration, test_quill_general_* (product), verified_solver, ns_guards, pure_mlp.

**Results:** route2_gap_diagnosis CORE; route2_transfer_2026_10_02 CORE; route2_hardening CORE (ns, physical_flow, 5D, parametric); route2_battletest SUPPORT (wall_ns_reduced_inverse CORE); route2_generality SUPPORT/HISTORY; general_solver HISTORY (PINN baseline SUPPORT); general_residual HISTORY.

**Docs:** radon_results_library README/rankings/requirements_and_costs/pinn_aims/components SUPPORT (fairly honest but omit the spectral-solve framing); methods/native_solver.md CORE; methods/analytical_construction.md CORE; methods/constrained_inverse.md SUPPORT; evidence/coefficient_access_2026_10_02.md CORE; other evidence SUPPORT; current_findings/scratchpad.md SUPPORT (ends mid-transfer); quill_streamed_hardening_status.md SUPPORT (contains Sam's Oct 2 requirements); quill_route2_generality_status.md, radon_catalogue_pure_routes_audit.md SUPPORT; quill_general_solver_{status,checkpoint,theory}.md HISTORY; docs/superpowers specs/plans IRRELEVANT to route 2 (pre-program global-lstsq Radon PINN).
