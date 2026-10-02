# expF19 early and middle phases: audit

Agent-side document (preliminary). Read-only audit subagent, Oct 2, 2026. Numbers spot-checked against saved JSON unless marked. Paths relative to `agent/imported/precisionMLPs/`. `R/` = `results/checkpoint_F_applications/expF19_radon_direct_pde/`, `X/` = `experiments/expF19_radon_direct_pde/`.

Scope: everything in `X/` that is not `route2_*`, not `general_*`, and not the known-function encoding scripts (`scaling*.py`, `quill_boundary.py`, `quill_sweep.py`). The earlier Codex self-audit `docs/radon_catalogue_early_pde_audit.md` (E01-E24) is largely accurate; exceptions in section 5.

## Summary
- **Phase A** (Sep 30; linear*, nonlinear*, inverse.py): PDEs solved outside the network -- closed forms (advection, Airy, heat, d'Alembert wave, Cole-Hopf Burgers) or a classical Fourier-Galerkin RK4 solver (3D Taylor-Green NS) -- then encoded into ridge readouts. Sam rejected this as cheating. `R/status.md` still lists them as "Validated outcomes".
- **Phase B** (native_*, solver modules): evolve amplitudes a' = A F[u] on QUILL-encoded Fourier/sine bases. Data shows spectral equivalence to classical Galerkin (~1e-15 agreement), slower. Later switched to product gates (tensor_box, ns_extended, elliptic, fluid animation): not one hidden layer. HISTORY.
- **Phase C**: depth/polarization compiler (pure_mlp; 2 tanh layers; rejected; its only PDE test has a quadratic exact solution inside the 15-coordinate basis) -> HISTORY. **Ridge PINN** (`solver/ridge_pinn_features.py`) -> CORE: genuine Linear/Tanh/Linear, analytic readout map w = E a, Gauss-Newton on true tanh jets from zero. 4.05e-16 with 5529 neurons / 190 coordinates; clean p/M/N sweeps showing an M >= p+1 cliff and separate angular and center floors. Failed space-time Burgers IVP (~2.4e-7, iteration_limit, cond ~3e9 on a square).
- **quill_ns_sweep** is a legitimate known-function encoding result: 3D velocity at 2.24e-15 with 358,872 neurons (64% fewer than long-halo); max-of-floors rule predicts held-out grid within 0.8% (velocity) / 10.3% (Laplacian).
- Also surviving: fixed 4-input wave MLP (ladder to 7.2e-15 at 514k neurons); exact per-neuron divergence-free Fourier-ridge encoding; alias amplification law sinh(Ak)/sinh(A omega); oracles (Kirchhoff, periodic Cole-Hopf, PeriodicNS); verification negative controls; DeepXDE baseline harness; inverse negative controls.

## 1. Narrative
**Phase A.** Linear: closed-form propagation of each Radon profile (constant-coefficient linear PDE => spokes evolve independently). Classical but legitimate demonstration. Nonlinear: Burgers via Cole-Hopf; 3D TG NS via classical dealiased Fourier-Galerkin/RK4 (`PeriodicNS` in `nonlinear_pde.py`), Fourier state grouped by collinear wavevectors into ridge profiles and encoded. Docstrings disclose this honestly.

**Phase B.** RHS evaluated on the actual QUILL-encoded field; amplitudes evolve a' = A F[u], A an explicit trigonometric quadrature analysis matrix, E the QUILL encoding of a fixed basis, weights w = E a. Fourier-Galerkin with sines replaced by tanh encodings. Domain/basis-specific (periodic analysis, Leray FFT projection, Laplacian eigenfunctions). Not plug-and-play.

**Phase C.** After Sam flagged product gates: route 1 `solver/pure_mlp.py` (polarization squares into 2-tanh-layer MLP; rejected). Route 2 prototype `solver/ridge_pinn_features.py`: `Linear(2,B)/Tanh/Linear`; readouts w = E a with E built analytically from Logan-Shepp ridge polynomials plus QUILL; P coordinates solved by GN/LSMR on the true neural PDE residual from zero. Seed of everything `route2_*`. Still a global (small) coordinate solve with P = binom(d+p, p).

## 2. Per-method entries
**`X/linear.py`** -- advection, Airy, 1D heat, analytic 3D anisotropic heat. Uses an OLD encoder (`profile_encode`: complex-shift density Im f(c+ia)/a, 25/gamma long halo, lambda .25), not the September boundary correction -> inflated neuron counts. Corrected advection h=.02: 1.0e-16; Airy values ~1e-16 but scaled PDE RMS 3e-12 to 7.6e-11; uncorrected readouts 3-70% error. Defines `OUT`, `sphere_rule`, `profile_encode`, `heat_*` (imported by linear_numerical, linear_wave, linear_verify, inverse). SUPPORT (encoding demo); PDE part HISTORY.

**`X/linear_numerical.py`** -- FFT-evolved filtered Radon heat profiles 3D. 1024 dirs 1.6-2.6e-15 (262k-1.05M neurons); 400 dirs 1.9e-12 to 2.6e-10; 144 dirs 1e-7 to 4e-6; flat in N (angular floor). HISTORY.

**`X/linear_wave.py`** -- fixed four-input tanh MLP, 3D wave IVP (d'Alembert per profile; direction w -> spacetime pair (w, +-c)). Directions 16/64/144/400/1024: 8.1e-2, 8.7e-4, 4.7e-6, 4.3e-10, 7.2e-15 (514,048 neurons). Uncorrected at 1024: 0.77. Kirchhoff self-refinement 4.5e-15. Only early result that is a single fixed spacetime one-hidden-layer MLP. SUPPORT.

**`X/linear_verify.py`** -- dense PyTorch autograd re-evaluation. Wave 32,128 neurons: dense vs factorized 5.9e-16, autograd PDE RMS 4.6e-15. **Caveat:** checked model has solution error 8.7e-4; the 1e-14 model was not dense-checked. SUPPORT (export + autograd pattern).

**`X/nonlinear_pde.py`** -- Cole-Hopf Burgers, Fourier-Galerkin TG NS, then encoding; readout a_vj = h Re sum_k (i sinh(A|k|)/A) u_k e^{i|k| c_j}. NS grids 16/24/32/48: 337k/669k/801k/1.003M neurons, errors vs refined 9.8e-5, 4.1e-6, 5.1e-8, 3.66e-11; encoding adds 1.4-1.8e-15. Burgers 309 neurons 5.83e-15. Useful: (i) per-neuron divergence-free (v . a_vj = 0; measured 2e-15); (ii) alias amplification sinh(Ak)/sinh(A omega_l): h=.16 with modes to 49 -> error 1.29e4, max readout 7.1e6; cut k <= 19 -> 9.9e-7 (`burgers_bandwidth_ablation.json`). PDE claim HISTORY; oracles and encoding facts SUPPORT. Results 86 MB (mostly NS npz).

**`X/nonlinear_validation.py`, `nonlinear_postprocess.py`** -- 64^3 reference, time refinement, positive controls, pruning table. HISTORY (pruning mild SUPPORT).

**`X/inverse.py`** -- (a) SPD diffusion tensor from 36 observations, analytic heat forward + 6-param least_squares: 2.3e-5 at 144 dirs, 2.79e-9 at 400 dirs; 1% noise 1.2-5.3%. (b) Backward heat: FFT + discrepancy cutoff then encoding; unregularized 2.7e173, cutoff 6 -> 1.62e-3, conversion 4.2e-16. (a) HISTORY; (b) SUPPORT as baseline/negative control.

**`X/native_tanh.py`** -- Burgers theta' = E A F(S theta). K 8/16/32/64: 4.8e-3, 1.0e-4, 5.7e-8, 2.50e-13 (= FFT baseline, time-step limited). **"2.52e-14 at dt .0005" is in no saved JSON.** Hub dependency of solver/base. HISTORY.

**`X/native_burgers.py`** -- B-spline/ReLU^3 dynamics, second order. IRRELEVANT but imported (fft_baseline, eval_fourier).

**`X/native_ns_pilot.py`, `native_ns_refine.py`, `native_taylor.py`** -- flat-ridge native 3D TG. K 2/3/5: 2.18e-4, 3.98e-6, 1.94e-9; K5 cache 186 MB. Taylor deg 8: total 1.97e-9. HISTORY.

**`X/native_elliptic.py`** -- -Delta u + 5u^3 = f on [0,pi]^2, sine basis. K12: 6.37e-8 (122,400 neurons). Shift 0 diverges. Eigenbasis-specific. HISTORY.

**`X/native_inverse.py`** -- Burgers nu + 2 IC amplitudes from 24 sensors. K64: nu 1.14e-11 (K32: 2.06e-6); 0.1%/1% noise median 1.21%/12.1%; IC-only data rank 2. HISTORY (identifiability control design SUPPORT).

**`X/native_ns_extended.py` + `solver/ns_extended.py`** -- product-gate tensor NS to T=1, 2.4-7.2e-6, "867 tanh" but 36k-59k coefficients. HISTORY.

**`fluid_animation_*`** -- runs ns_extended; validation.json mislabels as "Native QUILL PDE dynamics"; `_preview.py` needs macOS pyobjc. 44 MB; junk file `preview.mp4.sb-9e8052a1-Symy1I`. HISTORY.

**Solver modules** -- base.py (dispatcher), tensor_box.py (product gates), elliptic.py (product gates, dense Cholesky), reaction.py, hyperbolic.py (Rusanov FV; shock baseline at most), highdim.py (supplied-ridge linear + Fourier-then-encode), inverse_profile.py (17-coef IC inverse; inverse-crime rejection SUPPORT): HISTORY. verification.py (agreeing truncations with 70.7% error): SUPPORT. pinn_baseline.py (DeepXDE 1.15 harness, coupled to native_elliptic): SUPPORT.

**`solver/ridge_pinn_features.py` + `ridge_pinn_theory.md` + `ridge_pinn_{study,conditioning,burgers,box_followup}.py`** -- CORE. Disk Logan-Shepp basis U_n(v_nk . x)/sqrt(pi), exact common-grid angular reproduction (M >= p+1); E[(m,j),(n,k)] = W[j,n] C_n[m,k]; GN/LSMR over P = (p+1)(p+2)/2 from zero on tanh jets; torch Sequential export with autograd audit. p 4/8/12/16/18: 8.99e-3, 2.80e-6, 8.72e-11, 8.67e-16, 2.74e-16; p18 MLP (5529 neurons) 4.05e-16, fresh PDE RMS 1.0e-15, 5 GN iterations. M sweep at p12 (2/4/8/13): 4.4e-2, 1.3e-3, 1.5e-7, 8.7e-11. N sweep (33/65/129/257): 6.2e-7, 1.2e-9, 8.7e-11, 8.7e-11. Burgers space-time p28 2.44e-7 (iteration_limit); box p24 1.05e-7; conditioning p28 disk 3.98 vs square ~3e9. Concerns: global P-coordinate solve, P ~ binom(d+p,p), disk-specific basis, manufactured solution.

**`pure_mlp_depth_study.py` + `solver/pure_mlp.py`** -- 2 tanh layers; quadratic exact solution in span; 6.85e-16 with 9,312 units. HISTORY.

**`R/pure_mlp_repair/`** -- Gegenbauer ball-frame capacity d2-4; 4D manufactured NS degree 4, 19,125 neurons, 7.67e-14 (easy: face traces determine it). SUPPORT (route2 precursor).

**`quill_ns_sweep.py`** -- corrected-halo re-encoding of saved Fourier NS state: N128 lambda .25 358,872 neurons 2.24e-15; N192 lambda .20 Laplacian 1.61e-15; plain sqrt(N) halo 1e-3 to 1e-2; two-floor prediction within 0.80%/10.3%. Reads `R/nonlinear/ns_snapshot48.npz`. SUPPORT/CORE for the known-function story.

## 3. What remains genuinely useful
1. Ridge-PINN construction (CORE): w = E a, one hidden layer, physics solved from zero; M >= p+1 angular cliff; separate angular/center floors.
2. Encoding facts: per-neuron divergence-free encoding; alias amplification; corrected-halo NS encoding with tested max-of-floors rule; fixed 4-input wave MLP ladder.
3. Oracles: `kirchhoff()`, `burgers_solution()`, `PeriodicNS` (48 vs 64 diff 3.4e-11), `native_burgers.fft_baseline`.
4. Negative controls (belong in the paper-2 evaluation protocol): agreeing truncations wrong by 70.7%; periodic probes miss a pulse (33%); IC-only data zero viscosity sensitivity; inverse-crime fit (3e-20) fails refinement; unregularized backward heat 2.7e173; Airy tiny values with large high-derivative residual.
5. Baselines: DeepXDE harness (needs decoupling), backward-heat discrepancy, Rusanov FV.
6. Export/audit pattern: ordinary `torch.nn.Sequential` + dense autograd vs stable evaluator.

## 4. Portability notes
Scripts need `X/` on sys.path (top-level `from quill_boundary import ...`); outputs hardcoded via `Path(__file__).parents[2]`; MPLCONFIGDIR set to `/private/tmp/...`; manifest says `.venv/bin/python` (use `uv run --extra dev`); `solver/__init__.py` imports the whole package; `ridge_frame_dimensions.json` holds absolute paths. ~180 MB NS npz regenerable (48^3 in ~11 s).

## 5. Overclaims and inconsistencies in imported docs
1. `R/status.md` "Validated outcomes": Cole-Hopf and "actual interacting periodic 3D NS ... no training or readout LS" are an analytic formula and a classical solve, then encoded. Same framing in `nonlinear/report.md`.
2. `docs/quill_pde_checkpoint.md` headline "dramatically more accurate and faster than three ... PINNs": winner is tensor_box (product gates, sine eigenbasis matched to box and forcing); PINNs had one fixed recipe; classical solver faster still. Native results are spectrally equivalent to classical.
3. "Native Burgers 2.52e-14 at dt .0005" absent from saved JSON; best saved 2.50e-13 = FFT baseline.
4. NS "867 tanh units" / "Native QUILL PDE dynamics": product-gate tensor spectral solver with FFT/Leray closure (59,049 coefficients).
5. Catalogue E01 rates the depth compiler Req 7/10 "Successful pure-MLP repair" (2 tanh layers; quadratic solution in span); `docs/quill_pure_mlp_status.md` "Both architecture gates passed".
6. Wave dense-autograd check (5.9e-16) quoted beside "~1e-14" but run on the 8.7e-4 model.
7. `X/README.md` stale and mixed; `R/expF19_results.md` overwritten by route2 hardening, so early phases have no writeup.
8. "No global readout solve" language elides that ridge PINN is a global GN/LSMR over P coordinates, and native_elliptic needs box Laplacian eigenvalues.
9. Inverse headlines: "2.8e-9 tensor" only at 400 dirs (144 -> 2.3e-5); "nu 1.14e-11" is K64 (main K32 2.1e-6).
10. Phase-A "corrected" encoder is the long-halo complex-shift density, not the September correction; its neuron counts mixed with current ones without saying so.
