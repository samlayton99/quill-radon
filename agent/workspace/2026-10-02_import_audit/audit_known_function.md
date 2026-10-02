# Known-function Radon / ridge approximation: audit for paper 2

Read-only audit, 2026-10-02. Scope: checkpoints E, H, I (ridge parts), the expF19 encoding-side scripts, and catalogue rows A01-A21 (plus E04/E05) in `docs/radon_method_catalogue.md` and `docs/radon_catalogue_construction_audit.md`. Paths are repo-relative to `/Users/sam/my-repos/research/collaborations/precisionMLPs`. Numbers marked "json" were reloaded from saved data by the auditors. Part 2 is the full per-experiment detail, written by three sub-audits and merged here.

## 1. What is established

### 1.1 The two families of evidence

**(a) Analytic construction, no fit (the result Sam called "brilliant").** The representation is f(x) ≈ b + Σ_m Σ_j a_mj tanh(γ(v_m·x − c_j)), a single hidden layer on the original inputs. The profile is the filtered back-projection / Fourier-slice profile q_v, for example q_v = −(1/2π)∂²Rf in 3-D, or the closed form det(B)^{-1/2} s_v^{-d/2} ₁F₁(d/2; 1/2; −t²/s_v) for a Gaussian exp(−xᵀBx). The readouts are a_mj = (h w_m/2) ρ_m(c_j), with ρ(c) = Im q(c + iA)/A and A = π/(2γ). This "complex-shift" density inverts the tanh (sech²) smoothing exactly in the continuum. Directions come from a sphere quadrature with weights w_m. There is no least squares, no SVD and no training.

Two versions of the boundary handling exist:
- A long plain grid of 201 centers on [−4,4] (I04, F19 `scaling*`).
- Paper 1's finite-contour correction on an R = ⌈√N⌉ halo (F19 `quill_boundary.py` / `quill_sweep.py`).

**(b) Fixed ridge geometry with one global truncated-SVD least squares (checkpoints E, H).** This is the earlier body of work. It gives good M-vs-N tradeoff evidence and shows that a flat ridge MLP can reach the floor in d = 2 and 3. However, it uses the global dense readout solve that Sam has ruled out for the deployed method, and its precision is local to a small data ball.

### 1.2 Best results table

| d | Method (family) | Target(s) | M directions × centers/dir | Neurons | Error | Source (json) |
|---|---|---|---|---|---|---|
| 2 | Analytic filtered-Radon readouts + 80-center halo per spoke, no solve (a) | 4 radial targets (gauss, Runge, fast waves, packet) | 32 × 288 | 9,216 | rel L2 1.9e-15 – 5.4e-14 | `results/checkpoint_H_highdim/expH05_direction_cliff_2d/spoke_profiles/radon_prediction/forward_check/comparison.json` |
| 2 | Target-derived profiles, one shared 1-D lstsq per spoke (no 2-D fit) | same 4 + composition (angular interpolation) | 32 × 64 | 2,048 | 5.7e-16 – 2.6e-14; composition 4.1e-15 | same folder, `angular_interpolation.json` |
| 3 | Analytic, long plain grid (a) | 3 Gaussian-family targets (isotropic, anisotropic, shifted mixture) | 2304 × 201 | 463,104 | rel L2 4e-15 – 9e-15, max abs ~5e-15 | `results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/radon/metrics.json` (gitignored) |
| 3 | Analytic, corrected finite-contour √N halo (a), current paper-1 encoder | anisotropic Gaussian | 1600 × (129+2·12) | 244,800 | value 6.4e-16, gradient 1.5e-15, Laplacian 7.9e-15 (512 fresh pts) | `results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review/allocation_optimized_selected.json`, `boundary_validation.json` |
| 4 | Analytic, Gauss–Jacobi tensor sphere, long plain grid (a) | full-rank rotated anisotropic Gaussian | 32,768 × 201 | 6,586,368 | rel L2 8.15e-15, max abs 6.3e-14 (1,024 fresh pts) | `.../expF19_radon_direct_pde/scaling/structured_validation.json` (checked by me) |
| 4 | same | isotropic Gaussian | 4,096 × 201 | 823k | 1.1e-15 | `.../scaling/structured*.json` |
| 5 | same | anisotropic Gaussian | 331,776 × 81 (and × 201) | 26.9M (66.7M) | 6.45e-11 (6.43e-11), limited by angular resolution; 128 pts only | `.../scaling/structured_push.json` (no generating script) |
| 5 | same | anisotropic, cheaper | — | 13.2M | 1.19e-7 (1,024 fresh pts) | `.../scaling/structured_validation.json` |
| 4–32 | Analytic + Sobol-QMC directions | anisotropic Gaussian | 16,384 × 201 | 3.29M | median 0.063% (d4) → 2.34% (d32) | `.../scaling/dimension.json` |
| 32, 64 | Analytic + known-metric warped directions | anisotropic Gaussian | 16,384 × 201 | 3.29M | 0.44–0.61% | `.../scaling/adapted.json` |
| 256 | Analytic, known sparse ridge support | sum of 4 ridge sines (a ridge sum by construction) | 4 × 201 | 804 | 5.9e-16 | `.../scaling/sparse.json` |
| 2–4 | Gegenbauer ridge frame, literal torch `Linear-Tanh-Linear` export (a) | a single Gegenbauer ridge, degree 4/6 | (p+1)^{d−1} × (257+34) | 1,455 – 99,813 | degree 4: ≤4.4e-15 all d; degree 6: 1.5e-14 – 1.2e-13 | `.../pure_mlp_repair/ridge_frame_dimensions.json` |
| 2 | Joint global lstsq, Radon tensor (b) | Gaussian bump | ~20 × 51 | 1,024 | L∞ 1.4e-14 (unit disk) | `results/checkpoint_E_2d/expE01_geometry_zoo_2d/data.json` |
| 3 | Joint global lstsq, even nested, rcond 1e-14 (b) | 4 of 6 suite targets | 256 × 48 | 12,288 | 1.3e-14 – 3.0e-13 (ball r = 0.3); Runge 9.6e-14 after Gauss–Newton polish | `results/checkpoint_H_highdim/expH06_ridge_hierarchy/rcond_scan_d3.json` |
| 4 | Joint global lstsq, even (b) | gauss, composition | 1024 × 12 | 12,288 | 3.8e-11, 4.5e-11; no 4-D target reaches the floor | `.../expH06_ridge_hierarchy/push_d4.json` |

The best fully analytic floor results are d = 4 (Gaussian, 6.6M neurons) and d = 3 with derivatives (244,800 neurons, using the current corrected encoder).

### 1.3 Scope of the known-function claim

- Every floor-level result with d ≥ 3 is on Gaussian-family targets. Their Radon profiles are entire functions in closed form, so the complex-shift density can be evaluated exactly. No non-Gaussian target with d ≥ 3 has been tried with the analytic construction. The behavior is untested for profiles with a finite analytic strip, numerically computed Radon transforms, or non-smooth targets.
- The high-d "precision" results (d = 256) are ridge sums by construction. Generic targets with d ≥ 8 sit at the percent level.
- The 4D/5D runs use the old long [−4,4] grid with λ = 0.25 fixed. The corrected encoder was run only in 1-D, 3-D and 2–4-D Gegenbauer.
- Errors are sampled norms on 128–1,024 points (unit ball or small cube), not domain-wide L∞. The E/H least-squares results hold only on the data ball; off the ball the error is O(1) to 1e9.

### 1.4 Direction-count vs center-count (M vs N) tradeoff evidence

1. **Exact error split.** For the analytic construction, error ≤ angular quadrature error + profile-conversion error, and every row in F19 `scaling/` stores both parts. Conversion stays at 1e-15 throughout the dimension sweep, so the remaining error is angular.
2. **Angular leg is spectral.** 3-D Gauss–Legendre × φ: about 0.42 decades per unit of quadrature order (M ≈ order²), consistent with exp(−a M^{1/(d−1)}). 4-D/5-D Gauss–Jacobi behaves the same way, but the constant degrades with d: 5-D needs 3.3e5 directions for 6e-11. QMC gives slow algebraic decay.
3. **Center leg.** It is exponential with a cliff when there are too few centers: the sinh(Aω) amplification blows up below resolution (3-D 52k budget: 2304 × 21 gives 4e13). With the corrected encoder: N = 33 → 1e-7, N = 65 → 1.7e-12, N = 129 → 3e-16. The least-squares experiments (H05) instead measured a power law, roughly N^-10, with a 25% collar and no √N halo. See 4.1.
4. **Fixed-budget U-curves (3-D, analytic).**

   | Budget | Best (M, N) | Error | Other cells |
   |---|---|---|---|
   | ~52k | 576 × 89 | 2.5e-10 | 64 × 811: 1e-3; 1600 × 31: 2e5 |
   | ~208k | 1600 × 129 | 1.9e-15 | — |

   Source: `scaling/equal_budget.json`.
5. **Max-of-two-floors model.** Predicting the error as max(e_M, e_N) from separately measured floors:
   - F19 `quill_review/allocation_prediction.json`: within 3% for value, 5% for gradient, 15% for Laplacian.
   - H05 2-D least squares: within 21%.
   - H06 3-D: within 10% on 46 of 48 cells.
6. **Optimal allocation under least squares (H05 split-exact).** M* ∝ B^α with α = 0.27–0.44, never the balanced 1/2. The optimal path alternates doubling M and doubling N, with N/M between 1/2 and 8. In H06 4-D, N saturates at 12 and e_M falls 1.3–1.5 orders per doubling (about 2.5 in 3-D).
7. **The direction cliff (H05).** Error drops 5–9 orders over 2–3 steps of M. The required M grows with data radius, roughly (k r)^{d−1}. Doubling N at r = 0.8, M = 16 changes nothing, so directions are the binding constraint. Even angles are a sharp optimum, and the widest gap orders the error (`angular_gradualness/`).

### 1.5 What failed

- **Nearest-direction snapping (theory doc Step 2):** composition error 7e-4. Repaired by trigonometric angular interpolation, which gives 4e-15.
- **Coefficients without the smoothing inverse.** Leading derivative-sample readouts give 18–36% error. A short band with no halo gives 1e-7 to 4e-6. A naive √N halo without the contour correction gives 4e-5 (3-D) and 1e-4 (sine).
- **Under-resolved angular rule:** 0.82 error (Gegenbauer degree-6 target on a degree-3 rule).
- **QMC / metric-warped directions in d ≥ 8:** no precision.
- **5-D anisotropic:** stuck at 6.4e-11 because of angular resolution.
- **Least-squares-side placement heuristics:**
  - Angle-density placement: no effect.
  - Active subspace on ridge + background: 3e-3.
  - Iterated active-subspace fit: stalled at 5e-6 on rerun.
  - Greedy grower on isotropic targets: worse than an even mesh.
  - Offset grading, spikes and water-filling: no gain.
  - Projection-pursuit atoms without a joint polish: 1e-2 to 1e-1.
  - Beta spacing jump: stalls.
- **Hard targets:**
  - Steps and kinks: 1e-1 to 1e-5.
  - Narrow Runge never gets below 1e-10.
  - 3-D isotropic targets at 4,096 units: 1e-7 to 1e-2.
  - No 4-D least-squares floor.
- **Off the data ball:** every ridge geometry gives O(1) or worse.
- **Checkpoint I deep / compositional blocks:** far from precision, with a best of 4.7e-5. They also violate the one-hidden-layer rule.

### 1.6 Most useful next steps (not done)

1. Rerun 4-D and 5-D with the corrected `quill_boundary.encode` on [−1,1] and a √N halo, report fresh-point L∞, and measure an M/N tradeoff there.
2. Add non-Gaussian analytic targets with known Radon profiles, for example sums of anisotropic Gaussians, or rational / Poisson kernels whose analytic strip is finite, to test the complex-shift tube limit.
3. Test whether H05's power law in N becomes exponential once the 25% collar is replaced by a √N halo.
4. Pick a generic target class whose profiles can be computed numerically from f, and price that step.

## 3. Copy list for a paper-2 repo

Sizes are from `du -sh`. Everything under `results/` is gitignored except the `*_results.md` writeups and some PNGs and READMEs, so json and npz files must be copied by hand.

### 3.1 Code

| Path | Size | Role | Dependencies / fixes needed |
|---|---|---|---|
| `experiments/expF19_radon_direct_pde/quill_boundary.py` | 12K | **CORE** 1-D corrected QUILL encoder (`encode`); paper 1's construction | numpy, scipy. Optional `validate_reference()` loads `experiments/expD06_fixed_center_scales/construction_reference.py` (mpmath) via `parents[1]` (copy it too). Default `--out` is relative. |
| `experiments/expF19_radon_direct_pde/scaling.py` | 12K | **CORE** analytic Gaussian Radon profiles, sphere rules, d-sweep, allocation, sparse | numpy, scipy, matplotlib. `ROOT=parents[2]`; hard-coded `MPLCONFIGDIR=/private/tmp/codex-f19-scaling-mpl` |
| `experiments/expF19_radon_direct_pde/scaling_structured.py` | 4K | **CORE** Gauss–Jacobi tensor sphere for 4-D/5-D | imports `scaling` (bare sibling import) |
| `experiments/expF19_radon_direct_pde/scaling_controls.py` | 4K | **CORE** equal-budget M/N U-curves; plain-q′ control | imports `scaling` |
| `experiments/expF19_radon_direct_pde/quill_sweep.py` | 12K | **CORE** 3-D allocation and λ sweep with the corrected encoder; max-of-floors prediction | imports `quill_boundary`, `scaling` |
| `experiments/expF19_radon_direct_pde/scaling_adapted.py` | 4K | SUPPORT known-metric direction warp | imports `scaling` |
| `experiments/expF19_radon_direct_pde/ridge_frame_dimension_study.py` | 8K | SUPPORT literal torch one-hidden-layer export, Gegenbauer frame | torch, scipy, `quill_boundary` |
| `experiments/expF19_radon_direct_pde/feature_precision_study.py` | 8K | SUPPORT 1-D derivative-precision study | needs `solver/general_features.py` (private helpers), mpmath |
| `experiments/expI04_codex_geometry_unification/radon.py` | 12K | **CORE** first analytic 3-D construction (long grid); about 5 s to regenerate | numpy, scipy, matplotlib; `parents[2]`. Skip `deep*.py`, `theory_*.py` and `__pycache__` (164K). |
| `experiments/expH05_direction_cliff_2d/` (run.py, spoke_profiles.py, spoke_radon_prediction.py, theory_forward_check.py, composition_angular_interpolation.py, angular_gradualness/) | 360K | **CORE** 2-D cliff, split, tradeoff, 2-D analytic forward check | imports `h01suite` from expH01 via sys.path. Forward-check scripts load saved npz files (`spoke_profiles/endpoint_direction_check/solution_M32_N*.npz`, `radon_prediction/predictions_N*.npz`); copy those. The direction rule changed (commit b8ffdf0), so pass `--angle-rule recorded` to reproduce. |
| `experiments/expH01_highdim_suite/h01suite/` (+ run.py, viz.py) | 320K | SUPPORT target suite and even-geometry baseline; dependency of H04/H05 | `even_directions` uses the post-b8ffdf0 endpoint rule; saved data used half-step |
| `experiments/expH06_ridge_hierarchy/` (h06/ package + run.py) | 260K | CORE (3-D/4-D floors, M vs N) / SUPPORT (atoms, grower) | self-contained numpy; `parents[2]` |
| `experiments/expH04_mesh_finding/` | 200K | SUPPORT (directions_vs_radius, split_d3) | `h01suite` |
| `experiments/expH02_nonuniform_spacing_1d/` | 60K | SUPPORT (approved by Sam; non-uniform offsets) | `src.construction.qi_mpmath.default_halo` |
| `experiments/expE01_geometry_zoo_2d/` | 60K | SUPPORT (2-D geometry zoo) | `src/data/targets2d.py` (82 lines), `src/data/sampling2d.py` (32 lines), `src/construction/hex_geometry.py` (HISTORY; only needed to rerun E01) |
| `tests/test_expH01_highdim_suite.py`, `tests/test_expH04_mesh_finding.py`, `tests/test_expH06_ridge_hierarchy.py` | small | tests | — |

Repo-wide portability issues:
- The F19 scripts use bare sibling imports and must be run from inside their folder.
- READMEs cite a `.venv` that this project does not have.
- `parents[2]` roots are used everywhere.
- No code depends on `results/qi_cache`.

### 3.2 Docs

| Path | Tag | Notes |
|---|---|---|
| `docs/ridge_quadrature_theory.md` (8K) | CORE | Theory of record: Fourier-polar ridge representation, snapping certificate, two-floor model. Needs the corrections in 4.1. |
| `docs/radon_catalogue_construction_audit.md` (rows K1–K17) | CORE | Best cost and accuracy ledger for A01–A21 |
| `docs/radon_method_catalogue.md` rows A01–A21, E04, E05 (lines 71–445 plus intro 1–70) | SUPPORT | Historical; scores predate the strict one-layer rule |
| `docs/radon_results_library/methods/analytical_construction.md` | SUPPORT | Short summary of the analytic route |
| `results/checkpoint_F_applications/expF19_radon_direct_pde/scaling/report.md`, `quill_review/boundary_method.md` | CORE | Derivations |
| `results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/radon/report.md` | CORE | Includes the gauge example: coefficients change 7.07× while the function changes 2e-13 |
| `results/checkpoint_I_depth_theory/compositional_qi_theory.md` §2–3 | SUPPORT | Summary of the one-layer ridge extension. Its halo formula, R = max(⌈35/(2λ)⌉, ⌈0.4N⌉), differs from paper 1's ⌈√N⌉. |
| `docs/highdim_open_questions.md` | SUPPORT/HISTORY | Q7 is stale |
| `results/checkpoint_H_highdim/expH0{1,2,4,5,6}*/expH0*_results.md`, `results/checkpoint_E_2d/expE01_geometry_zoo_2d/expE01_results.md` | SUPPORT | Writeups. All are drafts except H02, which is approved. |

### 3.3 Result folders

| Path | Size | Copy? | Large binaries (>1 MB) |
|---|---|---|---|
| `results/checkpoint_F_applications/expF19_radon_direct_pde/scaling/` | 1.3M | all | none |
| `.../expF19_radon_direct_pde/quill_review/boundary_*`, `allocation_*` (json, png, md) | 0.97M | yes | none |
| `.../expF19_radon_direct_pde/quill_review/ns_*` | ~100M | no (encodes a classical NS solve) | `ns_quill_snapshot.npz` 48M, `ns_quill_compact_snapshot.npz` 33M, `ns_quill_balanced_snapshot.npz` 18M |
| `.../expF19_radon_direct_pde/pure_mlp_repair/ridge_frame_*` | ~12M | json yes; .pt optional | `.pt` files: d4_p6_n257 4.8M, d4_p6_n129 2.5M, d4_p4_n257 1.7M. Skip `flat_ridge_ns4d_*.pt` (PDE side). |
| `.../expF19_radon_direct_pde/general_residual/feature_precision_*` | small part of 7.7M | optional | — |
| `results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/radon/` | 7.1M | yes | `construction.npz` 6.7M (gitignored; regenerable in 5 s) |
| `results/checkpoint_H_highdim/expH05_direction_cliff_2d/` | 15M | yes | `spoke_profiles/` 12M: about 0.6–0.7M npz each and ~1M six-panel PNGs, none over 1.1M. `radon_prediction/` is 3.6M. |
| `results/checkpoint_H_highdim/expH06_ridge_hierarchy/` | 5.1M | yes | `spikes/figures/error_disk_maps.png` 1.3M |
| `results/checkpoint_H_highdim/expH04_mesh_finding/` | 8.6M | json only | none >1M |
| `results/checkpoint_H_highdim/expH01_highdim_suite/` | 10M | json + 1 gallery | gallery_d2..d5.png 2.1–2.5M each |
| `results/checkpoint_H_highdim/expH02_nonuniform_spacing_1d/` | 6.0M | optional | residual_N128_beta.png 1.4M, residual_N128_bimodal.png 1.3M |
| `results/checkpoint_E_2d/expE01_geometry_zoo_2d/` | 26M | data.json + writeup only | `weights/` 13M (192 npz), geometry_viz.ipynb 4.3M, error_vs_lambda_anim.html 2.6M, geometry_viz_sample.png 2.2M, .mp4 1.1M, targets_3d.png 1.0M |
| `results/checkpoint_I_depth_theory/` (everything else) | ~125M | no (HISTORY) | expI02 data 79M (Fashion-MNIST 29M, profile npz 49M), GIFs 3.8M and 4.9M, investigation_20260906/depth 11M |

A minimal CORE kit is about 25–30 MB without the optional .pt files and the E01 weights.

## 4. Overclaims, contradictions, superseded items

### 4.1 Theory vs data

1. **`docs/ridge_quadrature_theory.md` Step 3** says the offsets floor is exponential, e^{−αN}. H05 measured a power law, about N^-8.7 to N^-12.9, with a 25% collar. The F19 analytic construction with a long grid or the corrected halo does look exponential. The likely cause is the collar, but this is untested.
2. **Same doc, Step 5:** "max(e_M, e_N) ≤ e ≤ √2 max" is labeled a theorem. The upper bound follows from the certificate. The lower bound, that the joint least squares can do no better than each floor, does not follow in general, because the doc's own Step 4 says least squares often lands below the certificate. The max law is a measured regularity, not a theorem.
3. **Step 2** presents snapping as the construction. The forward check shows snapping fails (7e-4) and angular interpolation or proper quadrature is required. Step 4 says "the construction is never used as weights", which is superseded by the later analytic constructions.
4. **Halo convention conflict:** `compositional_qi_theory.md` uses R = max(⌈35/(2λ)⌉, ⌈0.4N⌉), paper 1 uses ⌈√N⌉, the long-grid runs use about 3N/4 outside the data, and H05/H06 use a 25% collar.

### 4.2 Overclaims in docs

5. **"No universal maximum input dimension" / d = 256 at 804 neurons** (F19 `scaling/report.md`, catalogue A18). The target is a known 4-ridge sum, so this is trivial in d. Generic d ≥ 8 is at percent level. Sam's "arbitrarily large functions with higher d" should be scoped to the Gaussian d ≤ 4 floor results plus ridge-sum targets.
6. **"4-D/5-D headline uses the current construction."** It does not: the old long grid with fixed λ. README says `quill_review` supersedes that protocol, yet it was never rerun in 4-D/5-D. For decaying Gaussian profiles the old long grid is actually as neuron-efficient per direction or better:
   - Old: 121 neurons/direction → 3.5e-15.
   - Corrected: 83 neurons → 1.7e-12; 153 neurons → 2.8e-16.

   No doc says this. The correction is essential for non-decaying profiles.
7. **`scaling/allocation_selection.json` "selector"** picks max M and max N in all 9 budgets, so it is a no-op. The report's wording ("often retains more centers") hides that.
8. **`quill_review` allocation and per-N λ** are chosen against the known truth (oracle), then re-verified on fresh points. They are a demonstration, not a rule.
9. **5-D numbers** are 128-point empirical norms. `structured_push.json` and `structured_allocation.json` have no generating script (they came from ad hoc `scaling.cell` calls).
10. **H01 TL;DR** "every smooth target in d = 1, 2 reads 3e-14 – 2e-12": false for the hotspot-data tasks (2e-11 to 2e-10). It holds only on the dense-region test set.
11. **H04:**
    - "Geometric convergence 8e-4 → 5e-6 → 1e-9" of the iterated active-subspace fit comes from the first (bw1.5) ladder. The rerun stalled at 5e-6.
    - The code-and-data list cites files that exist only in `bw1.5/`.
    - The manifold / patch theory is argued, not built.
12. **H06:**
    - "Every 3-D target within 3.3e-12 of the floor" uses test-selected (M, N, rcond, polish) per target, a single seed and a local r = 0.3 ball.
    - Its header says "nothing committed", but it is committed.
    - The 9216-unit (192, 48) cell numbers come from `log_push.txt`, not `push_d3.json`.
13. **E01:**
    - N = 8192 rows come from `extra_analysis.py`, not `run.py`.
    - λ and rcond were picked on the eval grid.
    - Radon sine2d is non-monotone.
14. **`docs/highdim_open_questions.md` Q7** ("nothing at the floor in 3-D") is stale.
15. **Checkpoint I:**
    - expI02's "within 3× of shallow on every target" is false (Gaussian bump 11.6×).
    - "3–6 orders above oracle" is false for product peak (12×).
    - The one-layer "shallow ridge-QI" baseline is tainted by two unfixed bugs: `qi2.py:314` leaves the projection bias out of training, and `run.py:315` never passes the lr.
    - `README_package.md` cites a nonexistent `code/qi2.py`.
    - The I01 card claim "block 1e-6 vs 1e-2 on d = 3 fast waves" has no saved data behind it.

### 4.3 Reproducibility

16. **Direction-rule drift.** Commit `b8ffdf0` (2026-09-01) changed `h01suite.even_directions` and E01 `geometries.py` from the half-step rule π(j+½)/J to the endpoint rule πj/J. All saved E01, H01, H04-2D and H05 data used the half-step rule, and the writeups still describe it. Reruns will not reproduce.
17. **rcond.** H01, H04, H05 and E01 used rcond 1e-13. H06 showed that 1e-14 lowers 3-D cells from 1e-11 to 1e-14, so some earlier "floors" are truncation-limited.
18. **Unexplained timestamps.** F19 `scaling/` file mtimes all fall within 10 minutes on Sep 30. The manifest hashes match the current code, so there is no integrity problem, but there is no run log for the push cells.

### 4.4 Requirement violations near this slice

19. **Product gate.** `moonshot_geometry.py` multiplies three 1-D encodings. It is also trivial, because the scales are supplied, so it amounts to coordinate rescaling.
20. **Dressing up another method.** `quill_ns_sweep.py` encodes a saved classical Fourier–Galerkin NS state. It is valid as encoding-precision evidence (velocity 2.2e-15 at 358,872 neurons; Laplacian 1.6e-15 at 519,420), but it is not a PDE solve.
21. **Global dense solve.** All E/H results (the least-squares family) use a global dense truncated SVD, up to 49,152 × 12,289. H06 direction learning is Gauss–Newton training. These are acceptable as approximation evidence only. None of the F19 known-function scripts use global least squares.
22. **Deep / compositional architecture.** All of checkpoint I except the expI04 `radon/` construction and the expI01 E0b / expI04 `manifold` one-layer heads is deep or compositional, so it is HISTORY.

---

# 2. Per-experiment entries

The detail below is from the three sub-audits, merged verbatim.


## 2A. Checkpoints E and H (+ src modules, docs)
## Audit: Checkpoint E (2-D) and Checkpoint H (high-d ridge/QI), for paper 2

Read-only audit, 2026-10-02. All numbers below were re-read from the saved json/npz unless marked "(writeup only)". Paths are repo-relative.

### 0. Cross-cutting findings (read these first)

1. **Strongest known-function Radon result lives in a Sept-29 subfolder of expH05, not in any writeup.** `results/checkpoint_H_highdim/expH05_direction_cliff_2d/spoke_profiles/radon_prediction/forward_check/` (README + `comparison.json`, `angular_interpolation.json`). In d=2, profiles derived from the target alone (filtered Radon/Abel inverse for radial targets; trigonometric angular interpolation of Fourier atoms for the composition), placed on 32 directions:
   - Profiles converted to tanh by **independent per-direction 1-D least squares** (8N+1 samples of the known profile, no 2-D fit): rel L2 at 32x128 = 4096 neurons: gauss 4.0e-15, Runge 2.5e-15, fast waves 1.4e-14, packet 1.2e-14; at 32x64 = 2048: 1.4e-15, 5.7e-16, 2.6e-14, 1.8e-15. Composition via angular interpolation: 4.1e-15 (2048), 3.4e-14 (4096). Verified in json.
   - **Fully analytic coefficients, no solve at all** (a_mj = (h/2) x sinh-corrected q'_m(c_mj), the tanh-smoothing deconvolution), on the 4096 band: 2e-7..4e-6 (truncation of the coefficient tail). Adding 80 halo centers per spoke (9216 neurons): 6.2e-15, 3.3e-15, 5.4e-14, 1.9e-15 (json key `corrected_extended_halo`); at N=64 + halo (7168 neurons) 6.3e-16..6.9e-15. This is the "no training, no fit" result Sam calls first-class; it is local (train disk r=0.4, scored on r=0.36 about x0=(0.35,-0.25)), d=2 only, 4 radial targets (composition not tested with the analytic+halo route; only snapping, which fails at 7e-4).
   - Nearest-direction snapping (theory doc Step 2) gives 7.0e-4 on the composition: a negative result for the snapping construction, fixed by angular interpolation.
   - Raw joint-lstsq coefficients do NOT equal the Radon coefficients (composition 278% interior discrepancy), but 99.991% of the squared difference lies in the SVD-discarded subspace; projected discrepancy 2.6%, radial targets 0.02-0.1%.
2. **Direction-rule drift breaks reproducibility.** Commit `b8ffdf0` (2026-09-01) changed `even_directions` (`experiments/expH01_highdim_suite/h01suite/baseline.py`) and expE01 `radon_tensor/interlaced` (`geometries.py`) from half-step angles pi(j+1/2)/J to endpoint pi j/J. All saved E01 (June), H01 (Aug 28), H04 2-D, H05 (Aug 29-31) data were produced with the half-step rule; the writeups still describe half-step. Rerunning `run.py` now gives different numbers (e.g. the H05 product-sines sawtooth would move from M = 2 mod 4 to M = 0 mod 4). Only `expH05/spoke_profiles.py` has a `--angle-rule recorded` shim.
3. **Data is untracked.** `.gitignore: results/**` except `*_results.md` / pngs / a few READMEs. Every json/npz cited here (incl. the forward-check npz that the Radon scripts load) exists only on this machine. Copy them explicitly when migrating.
4. **Every fit except the forward-check constructions is one global dense truncated-SVD lstsq** over all neurons (32768-49152 rows x up to 12289 cols in H06). Fine as approximation-theory evidence; it is exactly the "global readout solve" Sam rules out for the deployed method. H06's direction learning is Gauss-Newton training, not a construction.
5. **Theory vs data contradiction.** `docs/ridge_quadrature_theory.md` Step 3 states the offsets floor is exponential, e^{-alpha N}. expH05's fine tradeoff measured e_N as a power law (~N^-8.7..-12.9, with a shoulder near N=20); e_M exponential or faster. Plausible (unverified) cause: H05/H06 use a 25% collar (T=1.25r) instead of paper-1's sqrt(N) halo; the forward check shows a halo is what restores the analytic coefficients to the floor. Worth a direct test before paper 2 states a rate in N.
6. **rcond.** H06 found rcond 1e-13 (inherited from 1-D) cuts needed modes on wide redundant dictionaries; 1e-14 drops 3-D cells from 1e-11 to 1e-14. H01/H04/H05/E01 all used 1e-13 (E01 via lstsq default), so some of their "floors" may be truncation-limited. rcond and lambda selections in E01/H06 were chosen on the test/eval set (no separate validation), a mild selection effect.

### 1. src modules

| module | exists | importers | tag |
|---|---|---|---|
| `src/data/targets2d.py` (82 lines; sine2d, gauss_bump, runge2d, mixed2d, planewave) | yes | expE01 run/extra_analysis; also expD17, D19, D21, `_viz_targets_3d.py`, archived D11-D13 | SUPPORT |
| `src/data/sampling2d.py` (32 lines; disk_uniform, disk_grid) | yes | same as above | SUPPORT |
| `src/construction/hex_geometry.py` (186 lines; hex_pack) | yes | only expE01 `geometries.py`; `tests/test_hex_geometry.py` | HISTORY (point/tangent geometry lost to Radon tensor) |
| `src/construction/center_geometry.py` (151 lines) | yes | 1-D only: expB01, C04, C05, D02, G02, G05; `tests/test_center_geometry.py`. Not used by E/H | IRRELEVANT |

H experiments import almost nothing from `src/`: only expH02 uses `src.construction.qi_mpmath.default_halo`. H04 and H05 import `h01suite` (via `sys.path.append(REPO_ROOT/experiments/expH01_highdim_suite)`); H06's `h06/` package is self-contained (numpy only). All use `REPO_ROOT = parents[2]` (H05 angular_gradualness uses parents[3]). No hard-coded absolute paths in code; READMEs reference `.venv/bin/python` and `/Users/sam/Downloads/review_compositional_ridge_QI.md`, `depth_theory_response.md` (outside the repo). No `results/qi_cache` dependency anywhere in E/H.

### 2. Per-experiment

#### expE01_geometry_zoo_2d -- SUPPORT
- **Status:** draft, conclusions pending Sam.
- **Question:** does fixed geometry + fp64 lstsq reach the floor in R^2 -> R, and which ridge geometry?
- **Method:** neurons tanh(gamma(w.x - t)); six geometries (hex/mode_radial/sixn_rings tangent ridges, random_ridges, radon_tensor = J directions x M offsets, radon_interlaced = half-step shift on alternate directions). Radon: J = round(sqrt(N/2.5)), M = N/J offsets over [-2.5,2.5] (e.g. N=8192 -> J=57, M=144). gamma = lambda/h_ref, h_ref = 2.8/sqrt(N), lambda swept over 9 values, best eval-Linf cell kept. Global lstsq with bias, 8000 disk samples (12000 for N=8192), eval on 120x120 grid clipped to the unit disk.
- **Key numbers (data.json, matches writeup):** gauss_bump radon_tensor Linf 7.3e-14 (N=576), 1.4e-14 (1024), 5.9e-14 (8192); best non-Radon at 8192 = 3.6e-12 (writeup "~60x" correct). runge2d random_ridges 7.9e-9 at 8192 (correct). At 8192 Radon is best on mixed2d (2.9e-11), tied on sine2d (8.4e-12 vs sixn_rings 7.5e-12).
- **Caveats:** N=8192 rows come from `extra_analysis.py --add8192`, not `run.py` (whose WIDTHS stop at 4096); N=4096 fit on 8000 rows (~2 rows/col); lambda picked on the eval grid; Radon sine2d non-monotone (2.0e-11 at 256, 2.6e-7 at 576); direction rule changed after the data (see 0.2).
- **Deps:** `src.data.{targets2d,sampling2d}`, `src.construction.hex_geometry`.
- **Size:** experiments 60K; results 26M (weights/ 13M, 192 npz; geometry_viz.ipynb 4.3M; error_vs_lambda_anim.html 2.6M; geometry_viz_sample.png 2.2M; error_vs_lambda_anim.mp4 1.1M; targets_3d.png 1.0M).

#### expH01_highdim_suite -- SUPPORT (h01suite is a code dependency of H04/H05)
- **Status:** draft, suite build, pending Sam.
- **Question:** a d=1..5 benchmark (80 tasks, 16/dim) whose targets are not sums of ridges (mixed-second-difference check), with exact gradients, three test sets.
- **Reference model:** B tanh units: max(3, round(B^{1/d})) centers/direction, round(B/that) directions; d=2 even angles (half-step at run time), d=3 spherical Fibonacci, d>=4 Gaussian draw (placeholder); T = 1.25||v||_1, lambda=0.25, truncated-SVD lstsq rcond 1e-13. Random-feature control.
- **Key numbers (smoke.json, B=4096, n=8B, one seed; table matches exactly):** d=1 (1 dir x 4096, rank 3288) smooth rel L2 2e-13..6e-13; d=2 (64 dirs x 64, rank ~1990 of 4097) smooth 3.6e-13..1.9e-12 on even/uniform data, curved sheet 3.2e-14 on-sheet but 0.5 off; d=3 (256 x 16) fast waves 1.8e-7, bursts 0.25-0.73. Max-abs on uniform is 1e-11..2e-10 even where rel L2 ~1e-13.
- **Overclaim:** TL;DR "every smooth target in d=1,2 reads 3e-14..2e-12" -- false for the hotspot-data tasks on same_as_train/uniform (2.11 2.0e-10, 2.13 1.3e-10, 2.12 2.2e-11, 1.13 8.4e-12); true only on dense_region. "Kinks at 1e-5" holds in d=1 only (2.10 kink ring 4.9e-3).
- **Tests:** `tests/test_expH01_highdim_suite.py` (55).
- **Size:** experiments 320K; results 10M (gallery_d2..d5.png 2.1-2.5M each; smoke.json 137K).

#### expH02_nonuniform_spacing_1d -- SUPPORT (1-D; informs non-uniform offsets along a ridge)
- **Status:** conclusions approved by Sam (2026-08-28).
- **Method:** centers c_j = Q_s^{-1}(j/N), mixture density (1-s)/2 + s q; gamma_j = 0.25/h_j (local spacing); halo default_halo(N,0.25) (N=512 -> R=204, W=921); truncated-SVD lstsq.
- **Key numbers (data.json):** at N=512 uniform data n=16W: even 4e-14..1e-13; halfgauss s=1 5e-14..2e-13; bimodal s=1 2e-14..1e-13; beta s=2/3 8e-14..9e-13; beta s=1 stalls 3e-7 (sin), 6e-9 (Runge), 1e-2, 3e-2, neighbor ratio 4.45. Table matches (even sin 4e-14 vs writeup 6e-14, negligible).
- **Negative:** a spacing jump that does not shrink with N stalls the error. Halo vs jump not separated.
- **Deps:** `src.construction.qi_mpmath.default_halo`.
- **Size:** experiments 60K; results 6.0M (residual_N128_beta.png 1.4M, residual_N128_bimodal.png 1.3M).

#### expH03_distribution_matching -- IRRELEVANT (placeholder; run.py raises SystemExit). 4K each.

#### expH04_mesh_finding -- SUPPORT (directions_vs_radius and split_d3 are M-vs-N evidence; bw1.5/ is HISTORY)
- **Status:** draft, pending Sam.
- **Question:** do monitor-driven center placements (data density, true/estimated slope, residual, frequency) and direction placements (angle density A(theta)^{1/3}, active subspace) beat the even mesh?
- **Method:** h01suite even geometry, per-direction monitor -> graded density (floor s=2/3, grading 0.15, smoothing 5.8 gaps derived from the measured 12-gap mesh-map limit); global lstsq.
- **Key numbers (verified):** 1.16 at B=128 dense: even 8e-7, est. slope 2e-14, residual 1e-14. split_d3 (B=4096, even, d=3): 3.3 fast waves U-shaped, best 1e-7 at 24/dir (171 dirs); 3.16 sheet best 2e-11 at 32/dir. directions_vs_radius d=2 (48 offsets/dir): r=0.1 M=12 -> 3e-13; r=0.8 M=16 6e-9, M=24 5e-12. d=3 (32/dir): r=0.15 M=64 2e-10; r=0.3 M=128 3e-10; r=0.6 M=256 5e-10. Active subspace d=5: 5.5 composition 8e-4 -> 5e-6, 5.16 sheet 1e-6 -> 5e-11 (rows for d=5 live in `ladder_d3.json`).
- **Negatives:** angle-density rule = even mesh on every task; active subspace on a pure ridge+bump known-answer target is worse than even (direction bias 0.01 deg costs 3e-3); iterated active fit stalled in the rerun (5e-6) -- the "geometric convergence 8e-4 -> 5e-6 -> 1e-9" in the TL;DR is from the first (bw1.5) ladder only; d=3 isotropic content unresolved at 4096; steps unresolved in uniform data; 2% jitter of a gap costs 1e-15 -> 7e-11.
- **Provenance mismatch:** "Code & data" lists `ladder_d1freq.json`, `ladder_d2{c,d}.json`, `ladder_d5.json` at top level; they exist only in `bw1.5/` (first ladder, sigma=1.5); the top-level `log_d1freq/d2c/d2d/d5.txt` are also first-ladder logs. The sigma=6 rerun's d=2 tasks are in `ladder_d2{a,b}.json`, d=3/5 in `ladder_d3.json`.
- **Theory overreach:** the (kL)^m manifold cost, patching, zero-sum profiles and global coarse mesh are argued, not built.
- **Tests:** `tests/test_expH04_mesh_finding.py`. **Size:** experiments 200K; results 8.6M (no file >1M).

#### expH05_direction_cliff_2d -- CORE
- **Status:** draft, pending Sam (main + split + tradeoff); angular_gradualness draft; spoke_profiles/radon_prediction are Sept-29 Codex-era READMEs with no status line.
- **Method:** d=2, data uniform in ball radius r about x0=(0.35,-0.25), n_train = 8 x units; M even angles (half-step at run time), N cell-centered offsets on [-1.25r, 1.25r] (25% collar, no sqrt(N) halo), gamma = 0.25/h; one truncated-SVD lstsq rcond 1e-13; scored on inner 0.9r ball.
- **Cliff (data.json, 288 rows, N=128 fixed; table reproduced exactly):** smallest M below 1e-10 at r = 0.1/0.2/0.4/0.8: gauss 8/12/12/16; fast waves 8/12/12/16; slow waves 4/6/6/8; radial Runge 8/12/16/--; composition 12/12/--/--; packet 8/12/--/--; narrow Runge never (M<=16); polynomial 4 everywhere; product sines 2 (sawtooth, exact two-ridge function). Gauss r=0.1: 6e-2, 6e-3, 3e-4, 2e-6, 4e-9, 2e-12, 4e-14 over M=1..12 (matches). Floor 3e-15..9e-14. Readout norm 1e7-1e9 below the cliff, O(0.1) at the floor.
- **Control (control_n_per.json):** r=0.8, M=16, N 128 -> 256: Runge 1.505e-6 -> 1.571e-6, composition 7.36e-6 -> 7.46e-6, narrow Runge 9.17e-3 -> 9.42e-3 (matches): directions bind.
- **Split-exact (split_exact_2d.json, r=0.4, MN<=4096; best cells match):** B=1024 best 16x64 (fast waves 7.3e-13), 32x32 (Runge 8.0e-12, composition 4.2e-11, packet 4.3e-11); B=2048 32x64 Runge 7.4e-15; B=4096 32x128 composition 1.3e-14, packet 1.9e-14. Optimal path alternates doublings of M and N; N/M in [1/2,4], fast waves up to 8.
- **Tradeoff (tradeoff_2d.json, 147 cells):** e_M exponential to super-exponential (q = 0.92-2.1); e_N power-law ~N^-10; max(e_M, e_N) predicts all above-floor exact cells within 21%; M* ~ B^alpha, alpha = 0.274 (fast waves), 0.420 (Runge), 0.343 (composition), 0.437 (packet) (verified), predicted 0.25-0.41. Exponent fits use only B = 64..1024 (above the 1e-13 floor). Never balanced 1/2.
- **angular_gradualness/ (SUPPORT):** even angles are a sharp optimum; widest gap, not gradualness, orders the error (radial Runge M=32: even 1.4e-14, smooth density 9.4e-12, one-lobe 1.7e-9; verified). No angular analogue of the Beta stall.
- **spoke_profiles/ + radon_prediction/ + forward_check/ (CORE):** see 0.1. Also: 2a/h vs g' correlation 0.9987-0.99999 on solved coefficients; collar coefficients differ 12-58% (interior-only claim). README notes original runs saved no coefficients; reproduced fits saved as npz.
- **Caveats:** rel L2 denominator is the ball norm; r and h confounded in the r sweep; rcond 1e-13 (see 0.6); direction rule drift (0.2).
- **Deps:** `h01suite.baseline.{even_directions,_solve_svd,LAMBDA,RCOND}`, `h01suite.metrics.error_metrics`; spoke scripts chain `spoke_profiles -> run`, `spoke_radon_prediction -> spoke_profiles`, `theory_forward_check/composition_angular_interpolation -> spoke_radon_prediction` and LOAD saved `spoke_profiles/endpoint_direction_check/solution_M32_N*.npz` and `radon_prediction/predictions_N*.npz` (gitignored); scipy (`dawsn`, `iv`, `leggauss`).
- **Size:** experiments 360K (run.py 1336 lines); results 15M (spoke_profiles 12M: npz 0.6-0.7M each, six_panel pngs ~1M; no single file >1.1M).

#### expH06_ridge_hierarchy -- CORE (3-D/4-D floors, M-vs-N law) / SUPPORT (atoms, grower, polish = training)
- **Status:** draft-pending-Sam; header says "nothing committed" but it is committed (e1d8116). TL;DR heading typo "TL;DRw".
- **Method:** d=3,4; data ball r=0.3 about x0; nested farthest-point directions; block = one direction, n even offsets on 1.25 x projected band, gamma = 0.25/h; global truncated-SVD lstsq (QR-first for big cells). Atoms = projection pursuit + variable-projection Gauss-Newton on directions; greedy grower (open_bg / open_atom / refine).
- **Key numbers (verified):** 3-D even (192,48) = 9216 units: fast waves 7.8e-13, composition 1.9e-13, gauss 2.2e-14, product sines 9.4e-14 (from log_push.txt; this cell is not in push_d3.json; rcond_scan_d3.json has it at 49152 rows). (256,48) = 12288 units, rcond 1e-14: fast waves 3.0e-13, composition 2.7e-14, gauss 1.3e-14, product sines 5.9e-14 (vs 1.2e-11, 1.4e-12, 1.6e-13, 3.1e-13 at 1e-13). Radial Runge (320,32) 7.4e-13 -> 9.6e-14 after 8 GN polish iterations (6662 s); packet (384,32) 4.0e-12 -> 3.3e-12. 4-D best even (1024,12) = 12288 units: composition 4.5e-11, gauss 3.8e-11, Runge 2.1e-9, fast waves 7.9e-9, packet 1.5e-6 -- no 4-D target at the floor. Ridge recovery: 24 cases at 2e-13..1e-12, direction errors 0..2e-8 rad except one 0.5 rad stuck direction. Grower: ridge4+bump 7.9e-14 (3-D), 9.1e-12 (4-D); gauss 4.9e-13; radial Runge 3.1e-10 (worse than even's 1.5e-11); packet 2.3e-8.
- **M vs N:** 3-D exact cells 46/48 within 10% of max(e_M,e_N); in 3-D at 4096 units 256-512 directions x 8-16 offsets beat 64-128 x 32-64; 4-D offsets floor saturates by N=12, e_M falls 1.3-1.5 orders per doubling (3-D ~2.5); 4-D floor crossing estimated at 25-50k units (estimate, not measured).
- **Negatives:** greedy hierarchy cannot find the balanced split on isotropic targets; packet fails at 4096 everywhere; offset grading / spikes / water-filling not earned at uniform data; off-center packet clean negative for learned arms.
- **Overclaim risk:** "every 3-D target within 3.3e-12 of the floor" uses a different, test-selected (M,N,rcond,polish) per target, a local r=0.3 ball, single seed, and 12288-unit dense solves.
- **Deps:** self-contained `h06/`; `spikes/angular_gaps_2d.py` writes `results/.../spikes/`. Tests `tests/test_expH06_ridge_hierarchy.py` (7).
- **Size:** experiments 260K; results 5.1M (spikes/figures/error_disk_maps.png 1.3M).

### 3. Docs

- `docs/highdim_open_questions.md` (Sam, 2026-08-29/31) -- SUPPORT/HISTORY. Q7 "3-D to the floor ... nothing at the floor" is stale (H06 reached 1e-14 on the r=0.3 ball). Records Sam's program (1-D..4-D fat-trimming, learned directions, moonshot block design).
- `docs/ridge_quadrature_theory.md` (2026-08-31, cited by H05 READMEs) -- CORE theory: Fourier-polar ridge representation, direction-snapping certificate e_M <= r Theta_V int|xi| d|mu|, 1-D leg e_N, certificate-to-projection, max-of-two-floors. Conflicts with data: e_N stated exponential (measured power law, 0.5); snapping as the construction (forward check shows snapping fails at 7e-4 and angular interpolation is needed).

### 4. Synthesis for paper 2

**Strongest known-function results**
| d | construction | neurons (M x N) | error | domain | file |
|---|---|---|---|---|---|
| 2 | analytic filtered-Radon coefficients, no solve, +80 halo/spoke | 32 x (128+160) = 9216 | rel L2 1.9e-15..5.4e-14 (4 radial targets) | disk r=0.36 | forward_check/comparison.json |
| 2 | target-derived profiles, per-direction 1-D lstsq | 32 x 64 = 2048 | 5.7e-16..2.6e-14; composition (angular interp.) 4.1e-15 | disk r=0.36 | forward_check/*.json |
| 2 | joint global lstsq, Radon tensor | 1024 (J~20 x ~51) | Linf 1.4e-14 gauss | unit disk | expE01 data.json |
| 2 | joint lstsq, even 64x64 | 4096 | rel L2 ~2e-13..2e-12 smooth | cube [-1,1]^2 | expH01 smoke.json |
| 3 | joint lstsq, even nested, rcond 1e-14 | 256 x 48 = 12288 | 1.3e-14..3.0e-13 (4 of 6); Runge 9.6e-14 after polish | ball r=0.3 | expH06 rcond_scan_d3.json, polish_d3_big_runge.json |
| 4 | joint lstsq, even | 1024 x 12 = 12288 | 3.8e-11 (gauss), 4.5e-11 (composition) | ball r=0.3 | expH06 push_d4.json |
| 5 | active subspace (m=2) + lstsq, composition on noisy sheet | 4096 | 5e-11 dense region (2e-13 iterated, first ladder) | data on a 2-D sheet | expH04 ladder_d3.json, bw1.5 |

**M vs N evidence:** H05 cliff + N-doubling control (directions bind at fixed N=128), split-exact (alternating path, N/M 1/2..8), tradeoff (e_M exp/super-exp, e_N ~N^-10, max-model within 21%, alpha 0.27-0.44), angular gradualness (even angles optimal, widest gap governs); H04 split_d3 (U-shape, optimum 16-24/dir at B=4096; low-active-dim target wants 24-32) and directions_vs_radius (M grows with data radius, ~(kr)^{d-1}); H06 floors (max-model holds in 3-D and 4-D, coarse N wins, 4-D N saturates at 12). E01 Radon tensor used J ~ sqrt(N/2.5).

**What failed:** snapping construction on composition (7e-4); 1-D spacing jumps (Beta stall); angle-density placement (null); active subspace on ridge+background (3e-3); iterated active fit (stalled in rerun); greedy grower on isotropic targets; offset grading/spikes/water-filling; steps/kinks (1e-1..1e-5); narrow Runge (never <1e-10 at M<=16, 2-D); 3-D isotropic at 4096 units (1e-7..1e-2); 4-D floor (not reached at 12288 units); far field off the data ball O(1)..1e9 for every ridge geometry (local precision only); coefficient-level identity between lstsq readouts and Radon profiles (only in the retained subspace).

## 2B. expF19 encoding side
## expF19 known-function / encoding side: audit

Scope: the encoding-side scripts in `experiments/expF19_radon_direct_pde/` and their results under `results/checkpoint_F_applications/expF19_radon_direct_pde/`. This audit was read-only. Numbers marked "json" were reloaded from the saved data. All 11 scripts exist. The manifest's sha256 hashes for `scaling*.py` and `quill_*.py` match the current files, so the saved data comes from the current code.

### 0. Headline for paper 2

- **Strongest analytic, no-fit result (CORE):** a full-rank rotated anisotropic Gaussian $f(x)=e^{-x^\top Bx}$ (eigenvalues 1.5–6) in **d=4**. It uses a Gauss–Jacobi tensor sphere rule with M=32,768 hemisphere directions × N=201 centers (6,586,368 tanh neurons) and reaches **rel L2 8.15e-15, max abs 6.3e-14 on 1,024 fresh points** (json `scaling/structured_validation.json`). With N=121 (3.96M neurons) it reaches 6.91e-15 on 128 points. The isotropic 4D Gaussian reaches 1.5e-15 (validation set) and 1.1e-15 already at M=4,096 (823k neurons).
- **d=5:** the anisotropic case is **angular-limited at 6.43e-11** at M=331,776 (Gauss–Jacobi order 24). The value is unchanged from N=81 to N=201, i.e. 26.9M to 66.7M neurons. Peak process memory is 2.2 GB, and these numbers come from 128 points only. The 13.2M-neuron 5D case validated on 1,024 fresh points at 1.19e-7. The isotropic 5D case is at the floor (2.6e-15).
- **d=3 with the current paper-1 boundary correction** (`quill_review`): M=1600, N=129 interior + R=12 halo gives **244,800 neurons: value 7.2e-16, gradient 1.8e-15, Laplacian 9.5e-15**. On a fresh 512-point set: value 6.4e-16, Laplacian 7.9e-15.
- **Construction (all of the above):** the Fourier-slice / filtered-back-projection identity in closed form, $q_v(t)=\det B^{-1/2}s_v^{-d/2}\,{}_1F_1(d/2;1/2;-t^2/s_v)$ with $s_v=v^\top B^{-1}v$. The profile is turned into tanh readouts by the complex-shift QI density $\rho(c)=\mathrm{Im}\,q(c+iA)/A$, $A=\pi/(2\gamma)$, with weights $a_{mj}=\tfrac{h}{2}w_m\rho_m(c_j)$. There is **no least squares, no training and no SVD**, and coefficients are computed in chunks. The derivation is written out in `scaling/report.md`; I did not re-derive it.
- **Above d=5 it is not precise.** With 16,384 Sobol-QMC directions (3.29M neurons), the anisotropic median error is 0.063% (d=4), 0.079% (d=5), 0.265% (d=8), 1.17% (d=16) and 2.34% (d=32). Using the known metric to adapt the directions only improves this to 0.44–0.61% at d=32 and 0.46–0.57% at d=64. Precision in high d is reached only when the ridge directions are known: a sum of 4 ridge sines in d=256 uses 804 neurons and reaches 5.9e-16, but that target is a ridge sum by construction.

### 1. Per-script / result-group findings

#### 1a. `scaling.py` → `scaling/` (dimension.json, pilot.json, allocation.json, allocation_selection.json, sparse.json, figures) — **CORE**
- **Question:** How do direct analytic Radon encodings of Gaussians scale with dimension and with the M/N allocation?
- **Math:** see section 0. Direction rules: d=2 uses a uniform half-circle; d=3 uses Gauss–Legendre(z) × periodic φ on the hemisphere (M≈order²); d≥4 uses scrambled Sobol mapped through inverse-normal and normalized, with 2 scrambles. Centers are on **[-4,4]**, h=8/(N-1), γ=0.25/h, with **no finite-halo correction**: the long grid acts as the halo, and three quarters of the centers lie outside the projected range [-1,1]. Evaluation uses 128 points in the unit ball, uniform in radius rather than in volume. Targets are the isotropic Gaussian B=3I and the rotated anisotropic one.
- **Modes:**
  - dimension: d ∈ {2,3,4,5,8,16,32}, M up to 16,384, N=201.
  - allocation: d ∈ {2,3,4}, M ∈ {16,...,1024}, N ∈ {41,81,121,201}.
  - sparse: 4 known ridge sines, d up to 256.
- **Spot checks (all match report.md):**
  - Dimension medians 0.063/0.079/0.265/1.17/2.34%.
  - Center conversion error stays at 1e-15 to 2.5e-14 throughout, so the error is angular.
  - Allocation, 3D anisotropic: the angular floor goes 4.0e-2 (M16) → 1.0e-3 (64) → 6.4e-7 (256) → 7.3e-14 (1024). The center floor goes 0.38–0.60 (N41) → 5e-13 (N81) → 2e-15 (N121).
  - 4D allocation (Sobol, not tensor) is stuck at 4.5e-3 for M=1024.
- **`allocation_selection.json` is degenerate.** The "refinement-difference selector" picks N=201 and the largest M in every one of the 9 budgets. It is a no-op, and the report's wording ("often retains more centers") understates this.
- **Deps:** numpy, scipy (hyp1f1, ndtri, qmc, leggauss), matplotlib. Uses `ROOT=parents[2]` and a hard-coded `MPLCONFIGDIR=/private/tmp/codex-f19-scaling-mpl`. It is self-contained and is imported by `scaling_structured.py`, `scaling_adapted.py`, `scaling_controls.py` and `quill_sweep.py`.

#### 1b. `scaling_structured.py` → `scaling/structured*.json`, `structured_dimensions.png` — **CORE (headline)**
- Recursive Gauss–Jacobi tensor sphere rule, antipodally folded. M=n^(d-1)/2: n ∈ {4,...,32} for d=4 and n ∈ {4,...,24} for d=5. N=201 on [-4,4].
- json, 4D anisotropic: 2.6e-2 (M64), 7.3e-4 (512), 1.2e-5 (1728), 2.8e-7 (4096), 6.9e-11 (13824), 5.6e-15 (32768).
- json, 5D anisotropic: 2.5e-2 (256), 4.6e-4 (4096), 9.8e-6 (20736), 1.8e-7 (65536), 6.4e-11 (331776).
- This is spectral angular convergence. Conversion error stays at about 1e-15.
- `structured_allocation.json` (N=81/121 at the top orders) and `structured_push.json` (5D, N=201) are **not produced by any script**. report.md says they came from ad hoc calls to `scaling.cell(...)`, so the reproduction gap is small but real.
- **Report vs data:** every quoted number matches the json (8.15e-15 and 30.3 s; 6.91e-15, 3,964,928 and 7.0 s; 6.45e-11 and 20.1 s; 6.43e-11 and 39.5 s; 1.19e-7; 521.4 MiB/target; 2197.8 MiB peak).
- **Caveats:**
  - Only Gaussians were tested, which are entire functions with closed-form ${}_1F_1$ profiles.
  - Results above 4D are 128-point empirical norms.
  - The "timing" covers only 128 evaluation points.
  - The isotropic case is angularly trivial because its profile does not depend on direction.

#### 1c. `scaling_adapted.py` → `scaling/adapted.json`, `adapted_geometry.png` — **SUPPORT**
- Uses the known $B^{1/2}$ to warp Sobol directions: $z=uB^{1/2}$, $v=z/|z|$, with the profile written in terms of the scaled $z$. This is a Gaussian-specific change of variables; it is not learned and does not generalize.
- json (d=4..64): 3.8e-4 to 4.6e-4, 1.1e-3 to 1.2e-3, 3.9e-3 to 4.1e-3, 4.4e-3 to 6.1e-3, 4.6e-3 to 5.7e-3. Conversion error rises to 5.2e-13 at d=64.
- Report matches. The gain is at most about 4x, and every case stays at 0.04–0.6%, so this shows no route to precision in high d.

#### 1d. `scaling_controls.py` → `scaling/equal_budget.json`, `plain_control.json`, `equal_budget.png` — **CORE (M vs N tradeoff evidence)**
- 3D, fixed budgets of about 52k and 208k neurons, sweeping M=order² against N≈budget/M.
- json, anisotropic, 52k budget:

  | (M,N) | error |
  |---|---:|
  | (64,811) | 9.97e-4 |
  | (144,361) | 2.86e-5 |
  | (256,203) | 6.36e-7 |
  | (400,129) | 1.21e-8 |
  | **(576,89)** | **2.50e-10** |
  | (1024,49) | 1.07e-5 |
  | (1600,31) | **2.3e5** |
  | (2304,21) | **4.2e13** |

- json, anisotropic, 208k budget: **(1600,129) gives 1.92e-15**, (1024,203) gives 7.3e-14, and (4096,49) gives 1.07e-5.
- This is a clean U-shape. Coarse N does not merely set a floor: the inverse $\sinh$ deconvolution makes it catastrophic, which the report correctly flags.
- Plain control (readout $=q'$ samples, no complex-shift density): 14.7%/18.2% (2D) and 19.2%/22.7% (3D). These match the report.
- The allocation model "$e_{ang}\sim e^{-aM^{1/(d-1)}}$, $e_{ctr}\sim e^{-bN}$ ⇒ $M\propto P^{(d-1)/d}$" is stated as conditional, which is fair. In 3D the data fit roughly 0.42 decades per unit of quadrature order (√M), so it is consistent with the model.

#### 1e. `quill_boundary.py` → `quill_review/boundary_*.json`, `boundary_method.md`, `boundary_comparison.png` — **CORE (the 1-D leg; this is paper 1's construction)**
- Implements the 09_26_sl.tex finite-contour correction: density $\rho=(f(z+id)-f(z-id))/2id$, contour moments μ and ν, and a stable scaled partial-fraction map onto the outer m=⌊(R+1)/2⌋ halo readouts with R=⌈√N⌉. There is no solve.
- `encode()` is **the shared 1-D encoder used by nearly every later expF19 script**: solver/highdim, general_features, moonshot, ridge_frame, quill_sweep and quill_ns_sweep.
- json at λ=.25, max value error: sin 2πx goes from 1.14e-4 (plain) to 5.55e-16 (corrected) at N=128, and from 3.15e-5 to 7.77e-16 at N=256. sin 12πx has corrected L2 2.19e-14 at 128 and 2.11e-15 at 256. The Gaussian reaches 1.1e-16.
- The λ sweep at N=128 has an interior optimum: λ=.10 gives 5.0e-8, λ=.25 gives 2.3e-16, λ=.40 gives 2.5e-11. All of this matches `boundary_method.md`.
- Independent check: against the 60-digit mpmath reference at N=512, weights agree to 2.5e-16 and corrections to 5.0e-16. Quadrupling the quadrature changes outputs by 4.4e-16.
- Note the convention: the diagnostic counts N as **cells**, while the integrations count interior centers. The doc says so.
- **Deps:** numpy and scipy only for `encode`. `validate_reference()` loads `experiments/expD06_fixed_center_scales/construction_reference.py` (mpmath) through a `parents[1]` path. The default `--out` is a relative path, so the script must be run from the repo root.

#### 1f. `quill_sweep.py` → `quill_review/allocation_*.json`, `allocation_summary.png` — **CORE (cleanest M/N two-floor evidence with the current construction)**
- 3D anisotropic Gaussian, unit ball. The d=3 closed-form profile is $q=\text{pref}\cdot e^{-t^2/s}(1-2t^2/s)$. Encoding uses `quill_boundary.encode` on [-1,1] with N interior centers and R=⌈√N⌉ halos per side, at total cost M(N+2R).
- Evaluation: 128 ball points plus 31 sphere points plus the 6 poles. The error measure covers value, gradient and Laplacian.
- λ sweep (json, value conversion error):
  - N=33: best λ=.40 (3.1e-10).
  - N=65: λ=.30 (2.4e-14).
  - N=129/257: λ=.25 reaches the floor.
  - The Laplacian prefers smaller λ: .18 at N=257.
  - λ=.25 is not universally optimal at small N.
- Allocation grid at λ=.25 (json):
  - Angular: M16 5.8e-2 → M144 5.9e-5 → M400 3.3e-8 → M1024 3.1e-13 → M1600 7.6e-16.
  - Centers: N33 9.9e-8, N65 1.7e-12, N129 2.8e-16.
- **Max-of-floors prediction holds** (`allocation_prediction.json`): actual/predicted ratios are 1.000–1.026 for value, 1.000–1.048 for gradient and 0.969–1.155 for Laplacian across the held-out cells. This matches status.md.
- **Oracle caveat:** `allocation_selected*.json` choose (M,N,λ) by **error against the known truth** on the grid, then re-verify on 512 fresh points. That is an oracle allocation, not a rule. λ is also chosen against the known target (conversion error).
- **Not stated anywhere:** for this decaying Gaussian profile, the old uncorrected long grid of `scaling.py` is *more neuron-efficient per direction* than the corrected √N-halo grid.
  - Old: 121 neurons/direction on [-4,4] (h=0.067) → conversion 3.5e-15. Old: 81 neurons → 5.9e-13.
  - New: 83 neurons (N=65, R=9) → 1.7e-12. New: 153 neurons (N=129, R=12) → 2.8e-16.
  - The correction is essential for non-decaying profiles (sines, Fourier NS data). For decaying Radon profiles of localized targets, extending the grid is competitive. README calls `quill_review/` a supersession of the earlier geometry-selection protocol; that holds for the convention, not for efficiency in this case.
- The corrected construction was **never rerun in 4D/5D**. The 4D/5D headline therefore still uses the old [-4,4] uncorrected grid and fixed λ=.25.

#### 1g. `quill_ns_sweep.py` → `quill_review/ns_*.json`, `ns_report.md`, `ns_quill_*_snapshot.npz` (18/33/48 MB) — **SUPPORT (encoding precision of non-decaying trig data) / HISTORY as a PDE claim**
- Re-encodes a **saved 48³ Fourier-Galerkin Navier–Stokes state** (from `nonlinear_pde.py`, a classical spectral solve). Each primitive integer frequency direction $k/\gcd$ gets one ridge, giving 2,361 directions. A finite Fourier series is exactly a finite ridge sum, so there is **no angular quadrature here**.
- json matches ns_report:
  - N128, λ.25 (358,872 neurons): velocity 2.24e-15, Laplacian 6.53e-14, pressure 9.60e-15.
  - N192, λ.20 (519,420): 2.62e-15, 1.61e-15, 2.68e-15.
  - Old long-halo baseline (1,003,425 neurons): 1.65e-15, 8.38e-14, 7.38e-14.
  - Plain √N halo without correction: 1e-5 to 1e-2.
- The M×N two-floor check under pruning is 2.6e-7 at M393 (prediction ratio 1.00005).
- **Requirement flag:** this is "dressing up another method" if it is ever cited as solving NS. As an encoding-precision demonstration on real trig data it is useful. It is a set of spatial snapshots with time-varying readouts, not a space-time MLP.
- **Deps:** `nonlinear_pde.py` (PeriodicNS, ns_fields, ...) and the saved `results/.../nonlinear/ns_spectral_states.npz` (4.8 MB) and `ns_snapshot48.npz` (48 MB). It monkey-patches `qb.leggauss`.

#### 1h. `ridge_frame_dimension_study.py` → `pure_mlp_repair/ridge_frame_*.json/.pt` (12 MB of .pt, largest 4.8 MB) — **SUPPORT**
- Known-function check of the reproducing-kernel identity $E_\omega[C_n^{d/2}(\omega\cdot x)C_n^{d/2}(\omega\cdot\nu)]=C_n^{d/2}(\nu\cdot x)$ (Kerkyacharian et al. 2010), using a positive product sphere rule exact to degree 2n. It is exported as a literal `torch.nn.Sequential(Linear, Tanh, Linear)` and derivatives are checked by autograd.
- json: d=2..4, degrees 4 and 6, angular error ≤ 8.9e-16. Scaled derivative errors up to second order are 2e-15 to 4e-14 at N=257 and up to 2.6e-11 at N=129.
- Negative control: under-resolved angular degree 3 for a degree-6 target gives **0.82**, which is good evidence that the quadrature exactness degree matters.
- **Weakness:** the target $C_n(\nu\cdot x)$ is itself a single ridge, so this tests quadrature exactness for polynomials rather than general d-dimensional approximation. It is the cleanest "literal one-hidden-layer MLP file" artifact.
- **Deps:** torch, scipy, `quill_boundary`. Writes to `pure_mlp_repair/`, a folder shared with PDE-side `flat_ridge_ns4d_*` files (pt 1.4–2.6 MB each).

#### 1i. `feature_precision_study.py` → `general_residual/feature_precision_*.json/.md/.png` — **SUPPORT (1-D numerics)**
- A 1-D QUILL bank of Legendre $P_0..P_p$ (p=8, 12, 16), N ∈ {65..1025}, λ ∈ {.10..30}, with "standard" and "anchored" evaluation (tanh differences computed through logistic tails).
- json matches the report. Best anchored scaled error across values, first and second derivatives: 5.7e-15 (p8 and p12) and 7.1e-15 (p16), all at N=257, λ=.20.
- Error **rises** past N=257 (1.3e-14 at 513, 3.8e-14 at 1025) because γ grows and amplifies rounding in the derivatives. A 55-digit diagnostic shows coefficient rounding sets the second-derivative floor.
- Feeds the product-feature solver (`solver/general_features.py`), which is not a ridge construction. It matters for paper 2 only as evidence of derivative precision. **Deps:** `solver.general_features` (private `_evaluate_encoding`, `_exact_legendre`), `quill_boundary`, mpmath.

#### 1j. `moonshot_geometry.py` → `adaptive_followup/moonshot_*` — **HISTORY / IRRELEVANT (requirement violation)**
- Synthetic Gaussian swirl with core scales borrowed from an OpenAI NS-blowup PDF. The representation is a **product of three 1-D QUILL encodings**: $u_x=-X_0(x)\,Y_1(y)\,Z_0(z)$. That is a product gate and not a ridge sum, so it violates the one-hidden-layer requirement.
- json: scaled geometry with 1,275 units holds 1.4e-15 core error for every τ. Fixed grids blow up, reaching 1.5e38 and 2.3e100 at small τ.
- The result is trivial: the scales are supplied, so this is a coordinate rescaling. The script's own audit admits it is not a PDE solve and that the required force diverges.

#### 1k. `adaptive_ridge_study.py` → `adaptive_followup/adaptive_*` — **HISTORY (PDE side, out of scope)**
- Periodic 2D reaction–diffusion $u_t=\nu\Delta u+u^2$. It uses adaptive selection of integer-frequency ridge directions, i.e. spectral thresholding in neural coordinates.
- json: K18 fixed (80,056 neurons) gives 1.53e-7. Replay-adaptive (19,098 neurons) gives 2.18e-7. The identical classical trig algorithm takes 0.79 s vs 3.3 s and returns the same answer.
- No precision floor and no QUILL advantage. **Deps:** `solver/adaptive_ridges.py`, `solver/highdim.py`.

#### 1l. `capabilities/` (native_inverse.py, native_elliptic.py) — **HISTORY / out of scope**
- PDE-side work: a Burgers 3-parameter inverse and a 2D elliptic problem in a QUILL-encoded sine basis. Not encoding-side; listed only for completeness (492 KB).

#### 1m. `adaptive_followup/` other files (ns_*, inverse_profile_*, verification_*) — **out of scope**
- Native NS here uses **product gates and a tensor readout** (ns_report.md says so). This violates the flat-MLP requirement. 3.9 MB total.

### 2. M vs N tradeoff: consolidated evidence

- **The error decomposition is exact:** error = angular quadrature error + profile-conversion error, with the triangle inequality. Each saved row stores the `angular` and `conversion` parts separately, and this structure is verified everywhere.
- **Angular error (fixed target, spectral rules):**
  - 3D hemisphere GL rule: about $10^{-0.42\cdot\text{order}}$ with M≈order², consistent with $e^{-aM^{1/(d-1)}}$.
  - 4D/5D Gauss–Jacobi: the same spectral behaviour, but the anisotropic constant degrades with d. 5D needs 3.3e5 directions for 6e-11.
  - QMC in d≥8 is algebraic and slow, at percent level.
- **Center error:** roughly exponential in N with a sharp cliff below a resolution threshold. Old grid: 3D N=41 → 0.4, N=81 → 5e-13. Corrected: N=33 → 1e-7, N=65 → 1.7e-12, N=129 → 3e-16. Too-coarse N blows up because $\sinh(A\omega)$ amplifies.
- **Fixed-budget U-curves:** in 3D at 52k and 208k neurons the optimum is interior (576×89 and 1600×129). The max-of-floors rule predicts held-out cells to within 3% for value and 15% for the Laplacian.
- **Not measured:** an M/N tradeoff in d≥4 with the corrected construction; any target besides Gaussians (or exact trig/ridge sums) in d≥3; L∞ over the whole domain (all norms are sampled).

### 3. Overclaims, contradictions, superseded items, violations

1. **"Old dense-solve barrier bypassed" (report.md):** true for these encodings, which use no LS. But the precise results cover only Gaussians, whose analytic ${}_1F_1$ profile is known in closed form. No general-target known-function path exists for f without an analytic Radon transform. The complex-shift density also needs an analytic continuation of q.
2. **"There is no universal maximum input dimension" / 256-D at 804 neurons:** the target is 4 known ridge sines, so this is trivial in d. The honest high-d result is percent-level error at d=8–64.
3. **`allocation_selection.json` "selector":** always returns max N and max M (9/9 budgets), so it carries no information.
4. **`quill_review` allocation selection and λ choice are oracle-based**, using errors against the known truth. They are fine as a demonstration but not as an automatic rule.
5. **Supersession is only partial.** README says `quill_review` supersedes the fixed-λ, long-halo protocol. The 4D/5D headline numbers are still from the superseded protocol and were not redone. For decaying Gaussian profiles the old long grid was actually comparably or more neuron-efficient (see 1f).
6. **Reproducibility gaps:**
   - `structured_push.json` and `structured_allocation.json` have no generating script.
   - README says to run with `.venv/bin/python`, but project memory says there is no project venv and to use `uv run --extra dev`.
   - Hard-coded `/private/tmp/...` MPLCONFIGDIR (harmless).
   - All scripts rely on `ROOT=parents[2]` and on running from inside `experiments/expF19_radon_direct_pde/`, because imports are bare (`from quill_boundary import encode`, `from scaling import ...`) and `solver/highdim.py` inserts `parents[1]` into `sys.path`.
7. **File timestamps:** all of `scaling/` has mtimes between 20:25 and 20:35 on Sep 30, although runs such as 5D M=331k must have taken longer. These are probably rewrites of the outputs. Hashes match the current code, so I see no integrity problem, but there is no run log for the push cells beyond `structured.log` and `allocation.log`.
8. **Requirement violations within scope:**
   - `moonshot_geometry.py` uses a product of 1-D encodings.
   - `quill_ns_sweep.py` encodes a classical Fourier-Galerkin solution ("dressing up another method" if cited as PDE solving).
   - `adaptive_followup` native NS uses product gates.
   - None of the known-function scripts use global dense LS. `ridge_frame` and `scaling*` give flat one-hidden-layer nets; `scaling*` stores them in shared-direction compressed form, and an untied export would be 3.56 GB for the 5D push.
9. **Per-case tuning:** λ=.25 is fixed in `scaling*`. `quill_sweep` tunes λ per N against the known target. The Gaussian adaptation in `scaling_adapted` is target-specific by design (labelled as such).

### 4. Sizes and large binaries

| Folder | Size | Files over 1 MB |
|---|---|---|
| `scaling/` | 1.3 MB | none (largest png 243 KB) |
| `quill_review/` | 101 MB | `ns_quill_snapshot.npz` 48 MB, `ns_quill_compact_snapshot.npz` 33 MB, `ns_quill_balanced_snapshot.npz` 18 MB (all NS encodings) |
| `adaptive_followup/` | 3.9 MB | none |
| `capabilities/` | 492 KB | none |
| `pure_mlp_repair/` | 21 MB | ridge_frame .pt total 12 MB (d4_p6_n257_a6.pt 4.8 MB, d4_p6_n129 2.5 MB, d4_p4_n257 1.7 MB); `flat_ridge_ns4d_*.pt` 1.4–2.6 MB (PDE side) |
| `general_residual/` | 7.7 MB | mostly not encoding-side |

External inputs: `results/.../nonlinear/ns_snapshot48.npz` (48 MB) and `ns_spectral_states.npz` (4.8 MB), needed by quill_ns_sweep.

### 5. Copy-into-new-repo kit (minimal, for the paper-2 known-function section)

- `quill_boundary.py`: depends only on numpy and scipy. The optional `validate_reference` needs `experiments/expD06_fixed_center_scales/construction_reference.py` and mpmath.
- `scaling.py`, `scaling_structured.py`, `scaling_controls.py` (and `scaling_adapted.py` as a control): depend on numpy, scipy and matplotlib. Fix `OUT` and the bare-module imports.
- `quill_sweep.py`: needs `quill_boundary` and `scaling`.
- `ridge_frame_dimension_study.py`: needs torch and `quill_boundary`.
- Result data to keep: everything in `scaling/` (1.3 MB) and the `quill_review` boundary_* and allocation_* json/png files (under 1 MB). Leave the 99 MB of NS npz behind.
- **Most valuable next step (not done):** rerun 4D/5D with corrected `encode` on [-1,1] plus a fresh-point L∞, and add non-Gaussian analytic targets with known Radon profiles. Examples: sums of anisotropic Gaussians, or a rational/Poisson kernel with finite analytic strip, to test the complex-shift tube limit.

## 2C. Checkpoint I
## Section I audit: Checkpoint I (depth theory) as it touches ridge/Radon approximation

Read-only audit, 2026-10-02. Scope: `experiments/expI01..I04`, `results/checkpoint_I_depth_theory/`. Numbers below marked "verified" were recomputed from the saved json/npz; all spot-checks matched the writeups unless flagged.

### Bottom line for paper 2

- Checkpoint I is almost entirely about **deep/compositional** QI blocks (two-level ridge-QI block, stacks, encoders). Those architectures violate paper 2's one-hidden-layer requirement and are HISTORY/SUPPORT.
- **One CORE item:** `expI04_codex_geometry_unification/radon/`, the analytic, no-fit 3-D Radon-to-tanh construction. It is a flat one-hidden-layer tanh MLP on the original inputs, x -> sum_{m,j} a_mj tanh(gamma(v_m.x - c_j)) + b, with closed-form readouts from the filtered Radon profile and an imaginary-shift bandwidth correction. It reaches the fp64 floor (relative L2 2e-15 to 9e-15, max abs 4-5e-15) on three Gaussian-family 3-D targets at 2304 directions x 201 centers = 463,104 neurons. This is the direct predecessor of the "known-function analytic Radon construction" result. All numbers verified from `metrics.json` and recomputed from `construction.npz`.
- A few incidental one-layer ridge numbers live inside I01/I02/I04-manifold (listed below). They are fitted by LS or first-order training, not constructed.
- The Radon result's data (`metrics.json`, `construction.npz`) is **gitignored**: only `report.md` and the PNGs are tracked. It regenerates in about 5 s from `radon.py`.

### Per-item audit

#### 1. `results/checkpoint_I_depth_theory/compositional_qi_theory.md` (24K, tracked). Tag: SUPPORT (sections 2-3), HISTORY (sections 4-10)
- **What it is:** the checkpoint I "theory of record" (2026-09-01), Version 2. Sections 2-3 summarize the 1-D QI theory and the **one-layer ridge extension**. Sections 4-10 cover the two-level compositional block, KAT, the conditional theorem and depth.
- **Ridge content useful for paper 2 (all cited from expH04/H05/H06, not measured here):**
  - Finite-direction certificate: ||F - F_V|| <= r * integral of ||xi|| theta(xi, V) d|mu|, with Theta_V ~ c_d M^{-1/(d-1)}.
  - Fourier-slice statement: the profile's 1-D leg costs ||mu|| C e^{-alpha N}, with no factor of M or d ("e_N flat in M, identical in 2, 3, 4-D").
  - The angular error is a cliff: e_M ~ e^{-a M^q} with q = 0.9 to 2.1, which is exact in 2-D by Jacobi-Anger. A doubling of M buys 2.5 orders in 3-D and 1.3-1.5 orders in 4-D.
  - Direction count: M ~ max(C(p+d-1, d-1), c(k r)^{d-1}) with p ~ 12. 2-D needs 12-24 directions; 3-D needs 64/128/256 at k r = 1/2/4.
  - Two-floor law: e(M, N1) = max(e_M, e_N), with the optimal split M* ∝ B^{0.27-0.44}.
  - The 3-D floor sits at (M, N1) = (192, 48), i.e. 9216 units. The 4-D crossing is near 25-50k units: "4-D needs packing, not scale".
  - These are the direction-count vs center-count evidence, but the primary data is in checkpoint H.
- **Flags:**
  - It cites `~/Downloads/SQIN_corrections.md`, an external absolute path outside the repo.
  - The fp64 composed-floor and "dividend" claims are explicitly tagged [derived, unmeasured].
  - The halo formula R = max(ceil(35/(2 lambda)), ceil(0.4 N)) differs from paper 1's R = ceil(sqrt N). Reconcile this before reuse.

#### 2. `results/checkpoint_I_depth_theory/composable_qi_card.md` (8K, tracked). Tag: HISTORY
- **What it is:** the design card for the low-rank compositional layer (KAT rank 1, then rank S, then full C), with a two-hidden-layer MLP equivalence and GPU notes.
- **Status:** explicitly "not on record as law; Sam does not agree with all of it."
- **No ridge results.** Its annotation "expI01 d=3 fast waves QUICK: block 1e-6 vs shallow/MLP 1e-2" **cannot be verified locally**: `expI01_out/results.json` holds only E0 and E6, and expI03's audit says the same.

#### 3. expI01_compositional_qi. Tag: HISTORY (the E0b one-layer cliff numbers are SUPPORT)
- **Question:** can a two-level block beat the shallow ridge-QI model and a matched MLP?
- **Architecture:**
  - Layer 1: M unit directions x N1 fixed centers (lambda = 0.25, 25% collar).
  - Rank-S channel tensor C into K channels, range-tracked into an outer QI bank of N2 units.
  - Readout by QR+SVD (rcond 1e-14).
  - Arms: oracle, Adam, VarPro and GN.
  - **Deep: violates the one-layer requirement.**
- **Code:**
  - `build_notebook.py` writes the self-contained Colab notebook `compositional_qi_colab.ipynb` (57K, **0 saved outputs**).
  - `qiblocks.py` (29K) is the layer library: RidgeQILayer, QINet, tsvd_solve, gauss_newton and to_mlp.
  - No `src/` imports. The notebook mounts Google Drive when run on Colab.
  - expI02 imports `qiblocks` via a `sys.path` insert of `parents[1]/expI01_compositional_qi`.
- **Saved data:** only `experiments/expI01_compositional_qi/expI01_out/results.json` (1 KB), containing E0 and E6.
  - **E0a:** 1-D sin 2πx reaches rel L2 7.6e-15 with a fixed bank and solved head (verified).
  - **E0b, a one-layer ridge fit with fixed even 2-D directions** (expH05 geometry: d=2, data ball r=0.4, 128 offsets per direction, LS head; metric rel L2, not Linf). Verified values:

    | M directions | fast waves | gauss bump | radial Runge |
    |---|---|---|---|
    | 4 | 6.7e-2 | 1.2e-3 | 2.4e-3 |
    | 8 | 1.5e-6 | 1.0e-7 | 7.3e-6 |
    | 12 | 4.9e-13 | 7.5e-13 | 2.4e-8 |
    | 16 | 2.3e-14 | 9.3e-15 | 8.3e-11 |

    This reproduces the expH05 angular cliff, so it is direction-count evidence: 12-16 directions x 128 offsets reach the floor in 2-D at r = 0.4.
  - **E6 (California housing):** comp K1 0.494, shallow 0.559, MLP 0.462. The writeup matches.
  - **E1-E5 (the learned-composition headline) have no local data.**
- **Additional flag:** expI03's audit found that I01's MLP arm uses a fixed 4096 rows while the QI arms scale rows with units. That violates the plan's same-data rule.
- **Sizes:** 252K, with 110K of that in `__pycache__`. Nothing over 1 MB.

#### 4. expI02_block_prior. Tag: HISTORY (the B1 shallow-ridge column is weak SUPPORT and is tainted)
- **Question:** which coordinates, preconditioners, optimizers, heads, rank and depth make the compositional block train, and where it beats an MLP.
- **Architecture:**
  - Fixed mesh at every depth, with a free (un-normalized) projection V in layer 1.
  - Calibration plus a weak band penalty, a VarPro in-pass solved head, and depth 2-4.
  - **Deep: violates the one-layer requirement.**
  - The baselines include "shallow ridge-QI" (one hidden layer, learned directions, Adam with a solved head), which is the only one-layer content.
- **Key results, spot-checked against `b1.json`, `a2.json` and `b4.json` (all match `expI02_results.md` and `tables.txt`):**
  - **B1 at d=5,** 4000 steps, median of 3 seeds. Block 831 params; shallow M=14, N1=32, 448 units, 533 params:

    | target | block | shallow ridge-QI | MLP |
    |---|---|---|---|
    | gauss bump | 3.5e-2 | 3.0e-3 | 1.1e-2 |
    | fast waves | 0.23 | 0.17 | 0.093 |
    | composition | 2.0e-3 | 2.2e-3 | 7.2e-3 |
    | product peak | 4.0e-2 | 6.5e-2 | 6.1e-2 |
    | three bumps | 7.8e-3 | 8.3e-3 | 2.1e-2 |
    | random ridges | 0.34 | 0.20 | 0.058 |

    Block plus GN on seed 0: product peak 1.4e-4, three bumps 2.1e-4.
  - **A2 oracle floors at d=4:** fast waves 6.8e-6, composition 1.6e-7, product peak 1.3e-3. The writeup matches.
  - **B4 PINN:**
    - 1-D Poisson with the block plus operator solve: 3.43e-7.
    - The MLP operator solve gives 5.86.
    - 2-D Poisson block: 0.77.
    - Burgers block: 0.79.
- **Overclaims (confirmed by expI03's `evidence_audit.md` and by my recomputation):**
  - "B1 block within 3x of shallow on every target" is false for the gauss bump, where the block is 11.6x worse.
  - "Every learned number 3-6 orders above oracle" is false for A2 product peak, which is 12x.
  - "No size knob moves it" hides fast waves N1 = 8 -> 64, which improves about 10x.
  - The three-bumps GN gain mixes a 3-seed median with a seed-0 endpoint; the true same-seed gain is about 5x.
- **Bugs that taint the one-layer "shallow ridge-QI" baseline** (from expI03's audit; the files were not fixed):
  - `qi2.py:314` omits the shallow projection bias bV from the trained parameters.
  - `run.py:315` (`fit_shallow`) does not pass the lr, so shallow ran at 0.005 while the block ran at 0.02.
  - The shared GN (`qiblocks.py:359`) projects with untruncated QR.
  - Treat every B1 shallow number as a weakened baseline.
- **Other flags:**
  - `README_package.md` says the code lives in `code/qi2.py`; it is actually in `experiments/expI02_block_prior/`.
  - Several figures use a fixed 1e-15..1e1 axis on data spanning two decades; the author admits this.
  - The writeup's PINN section is a block-plus-operator-LS solve on tiny models, not relevant to paper 2's PINN.
- **Code dependencies:**
  - `qi2.py` imports `../expI01_compositional_qi/qiblocks` via `sys.path` from `parents[1]`.
  - `tasks.py` uses `REPO = parents[2]` and inserts `experiments/expF13_bwler_suite` (it imports `problems._burgers_exact_scaled`).
  - It also needs sklearn (`fetch_openml`, `make_friedman1`). Tests are in `tests/test_expI02_block_prior.py` (24 tests).
- **Sizes:** experiment folder 316K (155K pycache); results 91M.
  - `data/` is 79M: Fashion-MNIST gz is 29M (`train-images` 25M, `t10k-images` 4.2M); `profiles/*.npz` is 49M, eleven files of 4.1-5.8 MB.
  - `figures/` is 9.9M, including the gifs `profiles_bad_fast_waves.gif` (3.8M) and `profiles_best_from_zero_composition.gif` (4.9M).
  - The JSON files are 250-440K each.

#### 5. expI03_investigation and `results/.../investigation_20260906/`. Tag: HISTORY (deep); the practical `FixedQIMLP` is IRRELEVANT to paper 2
- **Question:** which checkpoint-I results reflect real compositional approximation versus initialization or fitting failure, and does depth beyond two layers help.
- **What was done:**
  - Two-stage blocks at d=3 with arms raw, poly3, quadratic, quadratic_free and MLP. These use a per-step ridge-regularized SVD head (alpha 1e-6), seeds 3-5, and **no halo** on the banks.
  - Depth continuation from A2.
  - The practical `FixedQIMLP` (12 channels x 11 centers per stage, plain Adam) on airfoil, Parkinsons and bike.
  - F04 initialization probes.
  - A corrected dense-GN reference.
  - All of it is deep or compositional. Note that `ridge_screen.json` is a ridge-*regularized head* screen, not ridge functions.
- **Verified medians from `confirm_d3.json` and `quadratic_release.json`** (inner-ball rel L2; all match `expI03_results.md`):

  | arm | fast waves | composition | product peak | random ridges |
  |---|---|---|---|---|
  | raw | 0.0749 | 3.02e-3 | 2.26e-2 | 0.255 |
  | quadratic | 4.00e-4 | 6.05e-3 | 1.49e-2 | 0.372 |
  | quadratic_free | 9.77e-4 | 1.86e-3 | not run | 0.0760 |
  | MLP | 0.0174 | 1.14e-3 | 9.25e-3 | 0.0243 |

  - The best analytic number is 4.7e-5, from seed-3 L-BFGS refinement in `confirm_d3_refine.json`. The program is nowhere near precision.
- **Canonical real-data results** (`f_init/canonical_summary.json`, block_random vs dense_qi2, test MSE / training variance) match the writeup:
  - Airfoil: 0.0941 vs 0.1256.
  - Parkinsons: 0.277 vs 0.368.
  - Bike: block_random 0.0812 vs dense_qi2 0.0782.
- **Status:** completed by Codex on 2026-09-06, with a self-audit (`evidence_audit/`) that is the most reliable critique of I01/I02.
- **Code dependencies:**
  - `probe.py` and `depth/run.py` insert `experiments/expI02_block_prior` (qi2, tasks) into `sys.path`.
  - `f_init/probe.py` inserts `experiments/expF04_qi_init_real_data` (model.py) and reads `results/checkpoint_F_applications/expF04_qi_init_real_data`.
  - Roots come from `parents[2]` / `parents[3]`. The README commands use `.venv/bin/python`, which exists.
  - Tests: `tests/test_expI03_investigation.py`, `f_init/test_fixed_qi_mlp.py`, `evidence_audit/test_solver_audit.py`.
  - `evidence_audit/mpl_cache/fontlist` is junk.
- **Sizes:** experiment folder 304K; results 28K for the md plus 19M for `investigation_20260906`.
  - `depth/` is 11M; `depth/screen.json` is 4.0M.
  - There are 121 `.pt` checkpoints totalling 7.7M across investigation folders.
  - `f_init` is 2.3M.

#### 6. expI04_codex_geometry_unification (Codex, 2026-09-30). Mixed tags per subfolder
- **Question:** a five-part "geometry unification":
  1. learned second-layer geometry;
  2. uniform placement;
  3. a learned chart plus a ridge head;
  4. the meaning of Radon coefficients;
  5. readout interpretability.
- **Code:** `deep.py`, `deep_figures.py`, `manifold.py`, `manifold_summary.py`, `radon.py`, `theory_relu_checks.py` and `theory_learned_relu.py` (248K total, all tracked).
  - Each uses `ROOT = parents[2]`, with output under `results/.../expI04_codex_geometry_unification/<sub>`.
  - `manifold.py` inserts `experiments/expH01_highdim_suite` to import `h01suite.baseline.even_directions`.
  - `radon.py` sets a hard-coded `MPLCONFIGDIR=/private/tmp/codex-radon-mpl` (harmless), and otherwise uses only numpy, scipy and matplotlib. It has no `src/` imports and is **fully self-contained** (about 300 lines).
- **Results size:** 22M. Large binaries:
  - `radon/construction.npz` 6.7M;
  - `manifold/coordinates_s0.npz`, `coordinates_s0_large.npz` and `coordinates_s1_large.npz`, 1.8M each;
  - `manifold/encoder_s*.pt`;
  - `deep/*.npz` and `theory/*.npz`, all under 1M.
- **Git:** md and PNG files are tracked; json, npz, pt and pdf files are not.

**6a. `radon/` (CORE).** This is the analytic 3-D Radon-to-tanh construction with no LS of any kind.
- **Construction:**
  - **Profile:** q_v(t) = -(1/2π) d²/dt² Rf(v, t), giving f(x) = (1/4π) ∫_{S²} q_v(v·x) dΩ. For a Gaussian component, q_v(t) = a/(sqrt(det B) s^{3/2}) (1 - 2y²/s) e^{-y²/s}, with s = vᵀB⁻¹v and y = t - v·μ, in closed form.
  - **Corrected readout:** a_mj = (h w_m/(2A)) Im q_{v_m}(c_j + iA), with A = π/(2γ). This deconvolves the sech² smoothing by analytic continuation. It is **not** paper 1's cardinal-coefficient (Kc = 160 Toeplitz) QI.
  - **Bias:** set from q at the origin, so nothing is fitted.
  - **Leading readout:** (h w/2) q'(c_j), which equals the naive derivative sample, gives 29-36% error. The correction is essential.
- **Geometry:**
  - d = 3, h = 0.04, γ = 6.25 (λ = 0.25).
  - **201 centers on [-4, 4]**, which is a very long halo: the data projections only span [-0.91, 0.99], so only 49 of 201 centers lie inside (verified).
  - **Directions:** Gauss-Legendre in z (nz nodes) x 2nz azimuths, restricted to the hemisphere. This gives M = nz² ∈ {16, 64, 144, 400, 1024, 2304}.
- **Test:** 512 scrambled Sobol points in [-0.6, 0.6]³, rel L2, on three targets:
  - isotropic Gaussian B = 3I;
  - rotated anisotropic diag(1.5, 3, 6);
  - a signed two-Gaussian shifted mixture.
- **Results (verified):**

  | M | neurons | isotropic | anisotropic | mixture |
  |---|---|---|---|---|
  | 16 | 3,216 | 3.36e-3 | 3.99e-2 | 6.47e-2 |
  | 64 | 12,864 | 2.87e-7 | 1.10e-3 | 1.80e-3 |
  | 144 | 28,944 | 8.79e-12 | 2.40e-5 | 3.87e-5 |
  | 400 | 80,400 | 2.07e-15 | 1.18e-8 | 1.57e-8 |
  | 1024 | 205,824 | 2.26e-15 | 5.28e-14 | 1.23e-13 |
  | 2304 | 463,104 | 4.24e-15 | 6.45e-15 | 8.78e-15 |

  - At 2304 directions, max abs error is 3.9e-15 / 4.9e-15 / 4.0e-15. I recomputed this from `construction.npz` predictions vs truths.
  - The tanh-vs-continuous-angular-sum discrepancy is about 2e-15 at every M. **All error above the floor is angular quadrature error.** The 1-D leg sits at the floor with N fixed, so this is clean M-only (direction-count) evidence. The angular rate is spectral: isotropic loses about 4 orders per step of nz.
- **Caveats to carry into paper 2:**
  - It needs a whole-space, decaying, analytic target with closed-form plane integrals that extend to complex t. That is substantial information.
  - Only the Gaussian family was tested, and only d = 3.
  - The neuron count is inflated about 4x by the [-4, 4] band. The `docs/radon_method_catalogue` entry A14 notes that a "longer halo fixed short-band failure" occurred, so a short band failed elsewhere (F19).
  - No center-count (N) sweep was done here.
  - The metric is rel L2 on an inner cube, not Linf over the ball.
  - The gauge example is exact. Writing ||x||² = (1-τ)Σx_i² + τΣ(Rx)_i² changes the coefficients by 7.07x (verified 7.0711) while the function changes by 2.2e-13. So per-neuron Radon readouts are not identifiable from function values; only invariants (here Q = Σ α_m v_m v_mᵀ = I) are.
- **Overclaims:** none found. The report is careful about cost and assumptions.
  - It cross-references a 2-D analogue at `results/checkpoint_H_highdim/expH05_direction_cliff_2d/spoke_profiles/radon_prediction/forward_check/README.md`. That file exists and is out of scope here. The cross-reference reports 3.41e-14 at 4096 neurons from analytic filtered profiles plus per-spoke 1-D solves.

**6b. `manifold/` (SUPPORT for the one-layer numbers, HISTORY for the encoder).**
- **Setup:** a 24-D embedding of a 3-D manifold. The arms are a learned autoencoder chart (deep, violates the requirement), PCA, the true coordinates, and an ambient 24-D input.
- **Head:** a frozen ridge bank with **128 directions (spherical Fibonacci) x 16 centers = 2048 neurons**, a 25% collar, and an LS head with λ and rcond selected on validation, using 4096 training points.
- **One-hidden-layer numbers** (test rel L2, verified from `metrics_s0_large.json`):
  - True 3-D coordinates: product wave 6.3e-7, composition 1.7e-5, gauss bump 3.1e-7.
  - Ambient 24-D ridge bank on the original inputs: 4.1e-3 / 2.4e-4 / 1.1e-3.
  - The learned chart gives 1.5e-4 to 1.5e-3, and PCA gives 0.1-0.4 (it folds the manifold).
- **Caveat:** the embedding is favorable (a linear inverse chart exists). There is no floor and no neuron rate.

**6c. `deep/`, `theory/`, and the top-level `report.md`, `theory.md`, `status.md` (IRRELEVANT / HISTORY).**
- `deep/` covers 1->48->48->1 trained tanh models on 1-D input.
  - Their test rel L2 is 6.72e-6 / 8.53e-6. Refit and arclength-uniform placement give 5.4e-7 / 1.35e-6.
  - The saved `seed*_metrics.json` holds only the random-placement medians (2.10e-6 / 5.64e-6 for seeds 0/1 — 4.70e-6 / 1.06e-5 are the single-zero variant); those match the report's random-placement row. I did not trace the main-table values beyond that.
- `theory/` covers ReLU chart coefficient prediction (0.0584%), the folding obstruction, and the observation that trained models' readouts are not local-derivative samples (93.8-95.3% error).
- `theory.md` is mostly projection/conditional-expectation decomposition. Its section 8 links to Radon and adds nothing beyond `radon/report.md`.
- The "geometry unification" claims are scoped modestly. The status file explicitly disclaims a universal theory and an Adam decoder.
- **Not a ridge-geometry unification result in the paper-2 sense.**

### Cross-cutting findings
- **One-hidden-layer ridge evidence found in scope:**
  - I04 radon: analytic, d = 3, floor at 2304 directions x 201 centers.
  - I01 E0b: LS, d = 2, floor at M = 16 x 128 offsets.
  - I04 manifold: LS, d = 3, 128 x 16 gives about 3e-7; d = 24 ambient gives about 1e-3.
  - I02 B1 shallow: Adam, d = 5, 1e-3 to 2e-1, with a bugged baseline.
- **Direction vs center evidence:**
  - Only the I04 radon M-sweep (N fixed, wide) and I01 E0b (N = 128 fixed).
  - The quantitative M-N tradeoff law is in expH04/H05/H06 and is merely summarized in `compositional_qi_theory.md` section 3.3.
- **Failures:**
  - Deep blocks do not train first order; they sit 1-6 orders above the oracle.
  - Depth 3-4 fails from scratch.
  - Block PINNs fail beyond 1-D.
  - The MLP wins the random-ridge control.
- **Unverifiable claim:** the card's and expI02's "record's E2 row reached 1e-6 on d = 3 fast waves" has no local data.
- **Venv:** the repo `.venv/bin/python` exists and has numpy; system python3 lacks numpy.
