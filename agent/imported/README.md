# Imported from precisionMLPs (agent-side evidence)

`precisionMLPs/` is a **verbatim snapshot** of every Radon-relevant file from `~/my-repos/research/collaborations/precisionMLPs` (GitHub `samlayton99/precision-mlps`) at commit `ca6dc9d` ("Record Radon solver experiments and documentation", Oct 2, 2026). The original repo is untouched. The relative layout is preserved, so imported scripts and doc links work in place. **Do not edit files here**; new work goes in `agent/workspace/` or `human/experiments/`. All docs here were written by agents and contain overclaims; read the audit (`agent/workspace/2026-10-02_import_audit/`) before trusting any headline.

Running: `cd agent/imported/precisionMLPs && OMP_NUM_THREADS=1 MPLCONFIGDIR=$TMPDIR/mpl ~/venv/quill-radon/bin/python -m pytest -q tests/test_route2_driver.py` (scripts write into this snapshot's `results/` via `Path(__file__).parents[2]`). Verified Oct 2: all 574 imported tests pass in place (`-m "not slow"`). Large binaries (npz, pt, npy, gif, mp4) exist locally but are gitignored.

Tags below: **CORE** (paper 2 builds on it), **SUPPORT** (baseline, tooling, context), **HISTORY** (superseded, failed, or violates the requirements; kept for the record). Paths relative to `precisionMLPs/`.

## 1. Known-function Radon construction (no PDE)

| Path | What | Tag |
|---|---|---|
| `experiments/expF19_radon_direct_pde/quill_boundary.py` | Corrected 1-D QUILL encoder (sqrt N halo + finite-contour rational correction); every Radon profile uses it. Counts **cells**, not centers. Its `validate_reference` uses `experiments/expD06_fixed_center_scales/construction_reference.py` (copied). | CORE |
| `experiments/expF19_radon_direct_pde/quill_sweep.py` | 3-D corrected Radon construction, lambda and (M,N) allocation sweeps, held-out allocation test | CORE |
| `results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review/` (`boundary_method.md`, `allocation_*.json`, `boundary_*.json`, PNGs) | Data behind the 6.4e-16 headline; NS re-encoding (`ns_*`, encodes an external Fourier solve: encoding demo only) | CORE |
| `experiments/expF19_radon_direct_pde/scaling.py`, `scaling_structured.py` | General-d Gaussian profiles (1F1), sphere rules; 4-D/5-D ladder (**old long halo**) | CORE (rules, profiles) |
| `results/.../expF19_radon_direct_pde/scaling/` (`report.md`, `structured*.json`, `sparse.json`, ...) | 4-D 8.15e-15 at 6.59M; 5-D angular-limited; d=256 sparse-ridge control | CORE/SUPPORT |
| `experiments/expF19_radon_direct_pde/scaling_adapted.py`, `scaling_controls.py` | QMC / metric-adapted controls | SUPPORT |
| `experiments/expF19_radon_direct_pde/ridge_frame_dimension_study.py` + `tests/test_quill_sphere_rule.py` | `sphere_rule` (Gauss-Jacobi product, folded), Gegenbauer ball frames | CORE |
| `experiments/expI04_codex_geometry_unification/radon.py`, `results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/radon/report.md` | Origin of the training-free construction (Oct 1): 205,824 neurons, 2.26e-15; 3-D derivation; coefficient-gauge example | CORE |
| `docs/ridge_quadrature_theory.md` | Checkpoint-H ridge theory (least-squares era); see theory audit §3 for corrections | CORE (with corrections) |
| `results/checkpoint_H_highdim/expH05_direction_cliff_2d/` (+ `spoke_profiles/radon_prediction/forward_check/README.md`) | 2-D direction cliff, two-floor law (LS), spoke-coefficient vs Radon prediction, angular interpolation | CORE/SUPPORT |
| `results/checkpoint_H_highdim/expH06_ridge_hierarchy/`, `expH04_mesh_finding/`, `expH01_highdim_suite/`, `expH02_nonuniform_spacing_1d/`; `experiments/expH0*` | Least-squares-era ridge geometry, hierarchy, mesh finding, 80-task high-d suite | SUPPORT |
| `results/checkpoint_E_2d/`, `experiments/expE01_geometry_zoo_2d/` | First 2-D geometry zoo | HISTORY |
| `results/checkpoint_I_depth_theory/compositional_qi_theory.md` §3, §11-12; `experiments/expI01..I03` | Ridge extension + depth/composition theory | SUPPORT |
| `docs/radon_catalogue_construction_audit.md` | Codex's audit of construction methods K1-K17 with costs | CORE (reference) |
| `docs/highdim_open_questions.md`, `docs/geometry_readout_theory_codex/` | Sam's high-d questions; readout-lens interpretability theory | SUPPORT |
| `docs/lambda_rule_theory.md`, `theory_lambda_rule.md`, `lambda_theorem_compatibility/` | 1-D lambda rule | SUPPORT |

## 2. PDE / physics with the Radon network (expF19, Oct 1-2)

| Path | What | Tag |
|---|---|---|
| `experiments/expF19_radon_direct_pde/solver/route2.py`, `route2_inverse.py`, `general_residual.py`, `streamed_residual.py`, `general_adaptive.py`, `general_domains.py`, `general_features.py` (helpers), `ball_ridge_features.py`, `residual_scaling.py` | Route 2: streamed Gauss-Newton/LSMR on polynomial coordinates compiled to a flat tanh MLP; reduced inverse wrapper. See proposal P003 for what it is. | CORE (code) |
| `experiments/expF19_radon_direct_pde/route2_{box_profiles,box_tensor,directional_operator,disk_profiles,affine_disk_profiles}.py` | Coordinate-to-profile maps and streamed tanh jets (library modules living in the experiment dir) | CORE |
| `experiments/expF19_radon_direct_pde/solver/ridge_pinn_features.py`, `ridge_pinn_*.py`, `ridge_pinn_theory.md` | Route-2 prototype: disk Logan-Shepp basis, M >= p+1 cliff, separate angular/center floors (4.05e-16, 5529 neurons) | CORE |
| `route2_transfer_*` + `results/.../route2_transfer_2026_10_02/` | Frozen-policy battle test (N001 3-D NS 1.38e-14 cold; failures Helmholtz n4, Allen-Cahn; three orphaned runs) | CORE evidence |
| `route2_gap_*` + `results/.../route2_gap_diagnosis/`, `docs/radon_results_library/evidence/coefficient_access_2026_10_02.md`, `current_findings/scratchpad.md` | Oct 2 conditioning/stopping diagnosis | CORE evidence |
| `route2_hardening_ns.py` + `results/.../route2_hardening/` | Tuned-chain NS 3-D/4-D, 5-D semilinear, parametric | CORE evidence (tuned) |
| `route2_battletest_inverse_ns.py` + `results/.../route2_battletest/inverse/wall_ns_reduced_inverse/` | Only nontrivial inverse (single scalar, same-method reference) | CORE evidence |
| other `route2_*` scripts, `results/.../route2_generality/` | Hardening, physical flows, constraints, scaling studies | SUPPORT |
| `route2_burgers_*`, `route2_frame_iteration.py`, `route2_shock_diagnostic.py`, `route2_initial_plane_*` | Failed or superseded | HISTORY |
| `general_*.py` (scripts), `solver/{base,tensor_box,elliptic,ns_extended,reaction,hyperbolic,highdim,inverse_profile,pure_mlp,general_sparse}.py`, `native_*.py`, `fluid_animation_*`, `pure_mlp_depth_study.py` | Product-gate / native-spectral / depth-compiler solvers: violate one hidden layer or are spectral equivalents | HISTORY |
| `linear*.py`, `nonlinear*.py`, `inverse.py` + `results/.../{linear,nonlinear,inverse}/` | Closed-form or external-solver solutions **encoded** into the network (Sam: "cheating"). Encoding facts and oracles remain useful (fixed 4-input wave MLP; per-neuron divergence-free encoding; alias amplification; Kirchhoff/Cole-Hopf/PeriodicNS oracles) | HISTORY (PDE claim) / SUPPORT (encoding, oracles) |
| `solver/verification.py`, `verification_study.py`, `solver/pinn_baseline.py`, `general_baseline.py` | Negative controls; DeepXDE PINN baseline harness | SUPPORT |
| `docs/radon_results_library/`, `docs/radon_method_catalogue.{md,csv}`, `docs/radon_catalogue_*_audit.md`, `docs/quill_*.md` | Codex's own organization (60-method catalogue, per-class rankings, PINN aims, requirements/costs). Useful index; ratings and some headlines are overstated | SUPPORT |
| `experiments/expF19_radon_direct_pde/README.md`, `results/.../expF19_radon_direct_pde/{expF19_results.md,status.md}` | Stale/mixed: lists encoded solves as validated outcomes | HISTORY |
| `tests/test_route2_*.py`, `test_quill_*.py`, `test_ball_ridge_features.py`, `test_ridge_pinn_features.py` | 20 core + support tests; all `sys.path.insert` the expF19 dir | CORE/SUPPORT |

## 3. Pre-Radon physics (checkpoint F) and baselines

| Path | What | Tag |
|---|---|---|
| `experiments/expF13_bwler_suite/` (`problems.py`, `ref/`) + results | BWLer five + poisson_man; canonical oracles incl. Cole-Hopf Burgers | CORE (benchmark) |
| `experiments/expF17_method_scaling_suite/` (`f17/`) + results (`SPEC.md`, `cells.jsonl`, `plot2_table.md`) | Matched-column multi-method harness: QI-Radon, tensor-QI, ELM, spectral, RBF, 1-layer PINN + refit certificate; no writeup yet | CORE (harness) |
| `experiments/expF18_ns_spacetime/` + results | 2-D+t drifting Taylor-Green NS, d=3; the 12.3k-unknown / 7 GB wall. Imports `experiments/expH06_ridge_hierarchy/h06/core.py` | CORE (benchmark) |
| `experiments/expF01_linear_de_zoo/` (`problems.py`, `PINN_FEASIBILITY.md`), `expF02_nonlinear_de_zoo/`, `expF03_ablation_and_baselines/` (+ `heldout_problems.py`) | 18-problem zoo + 3 held-out; no-oracle width selector; ELM/RBF/spectral/PINN/FD baselines (expF03 `cache/` not copied, regenerable) | CORE (benchmark, protocol) |
| `experiments/expF08_darcy_sweep/` (`darcy_problems.py`, `core.py`), `expF09_navier_stokes/` | darcy_man, Stokes | SUPPORT |
| `experiments/expF12_tensor_ns3d/` | 3-D+t Beltrami on a tensor basis (product architecture; capped ~sqrt eps) | SUPPORT (negative) |
| `experiments/expF14_dysts_chaos/` | Chaotic ODEs (1-D input) | SUPPORT (optional) |
| `cdeng/` | Catherine Deng's tensor-QI vs Radon vs BWLer audits (expF15/F16) | SUPPORT |
| `results/checkpoint_F_applications/expF_results.md` | Stale checkpoint-F summary | HISTORY |

## 4. Other

- `src/` (copied whole, minus the 87 MB `precision/_build`): the 1-D library used by the H/E experiments.
- `CLAUDE.md`, `README.md`, `docs/{INDEX,ORIENTATION,REQUIREMENTS}.md`, `pyproject.toml`, `uv.lock`: precisionMLPs context (its REQUIREMENTS/ORIENTATION are for the 1-D optimizer program, not this one).
- `docs/appendix_notes/paper_v5_replacements/organized_v7_package/`: paper 1 appendix source (also in `papers/paper1_quill/appendix_source_v7/`).

Not imported: the 1-D optimizer program (checkpoints A-D, G), expF04-F07 and F10-F11 (irrelevant or superseded), `docs/superpowers` (July designs), paper-1 appendix working drafts, and the raw Codex session files (paths in `agent/workspace/2026-10-02_import_audit/sessions/RAW_SESSIONS.md`).
