# precisionMLPs

## Research Question

Can we find a training/optimization strategy that learns QI-like solutions, closing the gap between explicit construction (~10^-15) and training (~10^-10)?

Three violations in trained networks explain the gap:
1. **Gamma scaling**: gamma stays O(1) instead of growing as O(N)
2. **Weight blowup**: outer weights diverge instead of staying O(1)
3. **Rank saturation**: features collapse instead of uniform utilization

## Before you propose any optimizer mechanism

**Read `docs/REQUIREMENTS.md` and run its section-8 checklist on paper first.** It holds the hard requirements, the complexity budget in numbers, how the computation actually hits a GPU, and a kill list of designs already measured dead. Feasibility, not correctness, is what has killed the most work on this project: mechanisms that worked numerically and could never scale, and mechanisms rebuilt after being measured dead months earlier. The checklist takes ten minutes and it is not optional.

`docs/INDEX.md` maps every document in the repo.

## How to Succeed
1. Always get context of the research! The papers/theory guiding the experiments are in the /papers/ folder. The main paper is `papers/QIs_workshop.pdf` ("Constructing Machine-Precision Neural Networks with Quasi-Interpolants"). NOTE: Section 3 (the construction) is NOT yet updated in that PDF -- read `papers/Section_3_Rewrite.pdf` IN ADDITION to `papers/QIs_workshop.pdf`, for the current construction (do not just read the rewrite on its own, it is only a single section rewritten), and `papers/practical_implementation.tex` for the fp64/mpmath implementation details. We are trying to complete this paper by finding an optimization strategy.
2. Use the additional repo 'continuous-mlps' (next door neighbor to this repo in the file structure) as inspiration or a resource when you need it (it is a correct implementation of the paper), but not as something to just copy exactly.
3. **read `docs/REQUIREMENTS.md` (the gate) and `docs/ORIENTATION.md` (the state) every time.** ORIENTATION is the entry point for the optimizer program: where it stands, the three open questions, the test matrix, and which claims in the record were later found wrong. `docs/INDEX.md` maps everything else. `docs/future_experiments.md` holds the open questions for the non-optimizer checkpoints (A/B/C/E/F/G).
4. When implementing new machinery or experiments, always write and clearly communicate to me the tests that verify your implementation actually matches the research (e.g. show me the QI construction reaches machine eps precision after being first built, etc.)

## Architecture

```
papers/                       Source material guiding everything
src/                          Core library (PyTorch, all computation in float64)
  config/
    schema.py                 ExperimentConfig and sub-configs (dataclasses)
    loader.py                 YAML load/save, sweep expansion
  models/
    layers.py                 GammaLinear, GammaExpLinear, StandardLinear
    mlp.py                    QIMlp: single-hidden-layer tanh MLP
    freeze.py                 requires_grad freezing utilities
  construction/
    qi_mpmath.py              High-precision QI via mpmath Toeplitz solve
    readout.py                Feature matrix Phi, exact readout solve (numpy/scipy)
    initialize.py             Project construction into model params
  data/
    targets.py                TargetFn registry (6 categories)
    sampling.py               Sampling functions (equispaced, uniform, Chebyshev, QI grid)
    dataset.py                build_dataset() -> Dataset dataclass
  training/
    optimizers.py             Optimizer dispatch (Adam, LBFGS, SGD)
    losses.py                 MSE, Lp, hybrid boundary
    train_loop.py             Multi-stage training orchestration
    metrics.py                MetricsCollector: uniform metric set across experiments
  precision/                  No-leakage p-bit arithmetic (expC11): a binary format with a p-bit significand, every
                              +,-,*,/,sqrt correctly rounded (emulator, MPFR-verified), the tanh model, and reference
                              LAPACK 3.12.1 DGELSS ported statement by statement (lapack_gelss.h). Bit-identical to
                              netlib SGELSS/DGELSS (reference_lapack.py builds them from source with gfortran).
                              Tests: tests/test_pbit.py (needs the 'precision' extra for gmpy2).

experiments/                  One FLAT folder per experiment (expXNN_name), each with config.yaml + run.py.
                              Flat (not nested under checkpoints) because run.py uses REPO_ROOT = parents[2].
  # Checkpoint A -- numerical validation, method justification
  expA01_numerics_sanity/        Numerics sanity checks
  expA02_qi_vs_lstsq/            QI construction vs least-squares readout (lstsq is superior)
  expA03_coeff_nullspace/        Coefficient closeness / readout null space
  expA04_activation_conditioning/ tanh O(1) vs GELU O(N) null-space regimes
  expA05_weight_blowup/          QI vs lstsq readout norm (no blowup; norm decays with width)
  # Checkpoint B -- scaling laws + noise robustness
  expB01_sampling_and_noise/     Centers vs samples; y-noise; 1/sqrt(n) law
  expB02_scaling_laws/           Width + data scaling, multi-activation (linear-then-floor)
  # Checkpoint C -- how much the geometry matters
  expC01_lambda_tradeoff/        U-shaped error curve in lambda (QI + lstsq)
  expC02_lambda_vs_frequency/    Optimal lambda constant across frequency
  expC03_lambda_basin/           Robust basin: lambda* ~ 0.25, gamma*/N ~ 0.10
  expC04_center_geometry/        Center-placement comparison (uniform vs others)
  expC05_geometry_interpolation/ center/weight/bandwidth perturbation; one-way coupling; reparam argument
  expC06_soft_neuron_interp/     soft-neuron hump (low-degree polynomial basis); cascaded-geometry lead
  expC07_lambda_energy_rule/     the aliasing rule: lambda* solves |Khat(2pi/lambda)|/|Khat(0)| = eps per activation (tanh 0.25,
                                 gelu 0.707, swish 0.455); fiber_check.py verifies the exact fiber-theorem floor
  expC07_lambda_energy_rule/lambda_rule/  consolidated ex-expC08/C09/C10 (2026-09-08): fiber-floor evidence on tones and general targets,
                                 the frequency-resolved rule's failure, the pre-registered two-wall test, the halo ladder;
                                 READ results/.../expC07_lambda_energy_rule/lambda_rule/hardened_rule.md (what survives the
                                 skeptic review) before citing any lambda claim; superseded writeups in its archive/
  expC08_anchor_rule/            the lambda anchor rule (2026-09-10): lambda from B * R_{K,r}(lambda, N, omega) <= eps_p (first ghost pair
                                 of the fiber theory at one representative frequency); exploratory sweeps (max vs mean frequency,
                                 4 activations incl. gaussian r=0) and a hashed pre-registered test on 10 fresh targets, widths 48-384,
                                 precisions 11-53 bits + native fp32; anchor_rule_v1.md is the frozen statement; lands at the
                                 valley corner except under-resolved high-frequency content; v1n = B/||f||_inf (amplitude fix)
  expC11_true_precision_law/     panel (c) of the expC09 figure redone with the model AND the least-squares solve at p bits
                                 (src/precision): chirp, W=1024, p=8..53, refined-rule lambda; SPEC.md is normative;
                                 run.py --sweep/--anchors/--standard, plot.py
  expC13_five_method_comparison/ five constructions (QUILLS, Mhaskar, staircase, Costarelli-Spigler, ChebNet) with every
                                 construction and inference operation at p bits via pfloat (~/my-repos/projects/tools/pfloat),
                                 format (p, -958, 959), 3073-parameter budget, validation selection, 320-bit error meter;
                                 SPEC.md is normative; run.py --sweep/--extended, plot.py; tests/test_expC13_*.py;
                                 headline: tradeoff_data.py + tapout_plot.py (tap-out width and floor, QUILLS vs ChebNet, N<=1024, p<=53)
  # Checkpoint D -- can optimizers find the geometry, and the optimizer program
  #   Entry point for D07 onward: docs/ORIENTATION.md
  expD01_geometry_ladder/        Adam on frozen geometry stalls; lstsq solves
  expD02_adam_geometry/          Init x training-regime cube (QI-init + refit wins)
  expD05_scale_init_story/       fp64 initializer matrix (scale-aware families win)
  expD06_gd_width_scaling/       GD width scaling
  expD07_lstsq_optim_suite/      optimizer bake-off + the dl_test real-data litmus
  expD08_qi_init_nlcg/           11-iteration optimizer campaign. ONLY iteration_11 survives
  expD09_2nd_order_regime/       step-2 recipe: machine eps on frozen Phi at O(d) state
  expD10_step2_hardening/        step-2 hardening ledger, tier structure, lessons T1-T12
  expD11_batching/               NEGATIVE: no O(d*k) solver reaches the fp64 floor
  expD12_mu_ladder/              the mu ladder on a frozen Phi   (writeup: results/checkpoint_D_optimizers/expD12_mu_ladder/STEP2_SOLVER_SPEC.md)
  expD13_drift_ladder/           the mu ladder on a drifting Phi (writeup: results/checkpoint_D_optimizers/expD12_mu_ladder/STEP2_SOLVER_SPEC.md)
  expD14_lobotomy/               step 3, first stitching attempt. READ ITS CORRECTION HEADER
  expD15_inclusion_score/        which parameters enter L (writeup: results/checkpoint_D_optimizers/expD15_inclusion_score/METHOD_L_selection.md)
  expD03/expD04/exp13            stubs, never run
  # Checkpoint E -- extend to 2D
  expE01_geometry_zoo_2d/        Six 2D ridge geometries head-to-head (hex folded in; ex-exp11)
  # Checkpoint F -- applications
  #   depth, higher input/output dim, non-MSE losses, physics tasks, transformer init
  expF01..expF12/               DE zoo (linear + nonlinear), ablations/baselines, spline ridge,
                                Newton-Burgers, PINN finisher, Darcy, Navier-Stokes, operator
                                learning, FNO init, 3-D tensor NS. Synthesis: checkpoint_F_applications/expF_results.md
  expF18_ns_spacetime/          2-D + time incompressible Navier-Stokes (drifting Taylor-Green, exact) on the
                                expH06 space-time ridge mesh (d = 3), primitive (u, v, p) on one shared geometry,
                                Gauss-Newton collocation lstsq, width ladder + oracle ceiling; gif / error-vs-time /
                                residual heatmaps. problem.py, ridge3d.py, run.py (--oracle --tune --solve --plot --gif)
  # Checkpoint G -- generalization
  #   precision-vs-generalization; mask-the-data; soft-weight tradeoff; data-poor regions
  expG01_interactive_explorer/  Interactive Dash explorer (lambda, N, target, hold-out mask; live lstsq refit)
  expG03_extrapolation/         Extrapolation sweep
  expG04_cascade_multiband/     Band-count ablation
  expG05_geometry_reader_claude/ Interactive geometry reader (Claude version; stdlib server + numpy + canvas JS):
                                drag centers/gammas/readout, live lstsq, recordable Adam with staged edits, inject,
                                replay and forks; Full reset; presets incl. expD06 trained-Xavier arrays. README.md there; runs on
                                the laptop (~/venv/precisionMLPs/bin/python app.py); tests/test_expG05_geometry_reader.py
  # Checkpoint H -- high-dimensional ridge/QI extension
  #   80-task suite over [-1,1]^d, d=1..5, 16 per dimension (spec: SUITE_SPEC.md, Version 3). Targets are
  #   genuinely multivariate (no target is a sum of 1-D profiles).
  expH01_highdim_suite/         Self-contained: h01suite/ package (basis, targets, densities, tasks, metrics,
                                even-geometry reference baseline) + run.py/viz.py (gallery, smoke, sweeps)
  expH02_nonuniform_spacing_1d/ Smoothly non-uniform center spacing in 1-D with gamma_j = 0.25/h_j: 3 center
                                distributions x 4 levels of non-uniformity x 4 targets x 4 data settings
  expH03_distribution_matching/ PLACEHOLDER (not specified yet): ideal center placement for a data distribution
  expH04_mesh_finding/          The mesh-finding ladder: monitor -> graded density -> centers (mesh.py), from
                                data-only through true/estimated slope, residual, frequency, to the active
                                subspace (gradient covariance) for directions; d = 1, 2, 3, 5 on suite tasks
  expH05_direction_cliff_2d/    The 2-D direction cliff on 9 targets: data ball of radius r about an off-center
                                point, ridge system recentered on it, sweep the direction count M (1..16) at a
                                fixed 128 offsets per direction; error scored on the inner 90% of the ball.
                                --split-exact: the directions-vs-offsets budget split on a power-of-two
                                grid at r = 0.4 (exact iso-budget diagonals + the steepest-descent path)
  expH06_ridge_hierarchy/       The hierarchical ridge mesh, d = 3, 4: nested background directions with coarse
                                even blocks, atoms (directions found by projection pursuit on the residual and
                                polished jointly by Gauss-Newton with the readout eliminated), and refine-vs-open
                                arbitration by trial fits (the two-floor law in operation). h06/ package
                                (core, targets, atoms, grow); --floors (3-D floor curves), --ridges (hidden
                                ridge recovery), --grow (hierarchy vs even), --push (3-D to the floor)
  # Checkpoint I -- depth theory: composing QI blocks
  #   theory: results/checkpoint_I_depth_theory/compositional_qi_theory.md (the conditional theorem,
  #   KAT as universality only, the untied compositional ridge-QI block)
  expI01_compositional_qi/      Self-contained Colab notebook (build_notebook.py writes it; --smoke runs it at
                                QUICK size on CPU): the two-level block (rank-1 KAT tie, range-tracked outer
                                QI, solved readout) vs shallow ridge-QI vs tanh MLP at matched params/FLOPs;
                                oracle / adam / varpro / Gauss-Newton arms. Plan + checklist:
                                results/checkpoint_I_depth_theory/expI01_compositional_qi/PLAN.md
                                qiblocks.py: the layer library (RidgeQILayer, QINet, QIFFN, solvers, fit, GN,
                                to_mlp, export); `python qiblocks.py` runs its self-tests. Directions exist only
                                in layer 1; later layers are QI banks on their input channels + the rank-S tensor.

tests/                        Unit tests
results/                      Experiment results output, grouped: results/checkpoint_<A..H>_*/exp*/
                              The global cross-experiment summary is results/results.md.
```

## QI Construction: Critical Facts

The QI construction in `src/construction/qi_mpmath.py` has two precision regimes.
**Read `papers/practical_implementation.tex` before touching construction code.**

- **fp64 path** (default, `precision="fp64"`, lambda=0.30): ~10ms, L_inf ~ 1e-12.
  Limited by fp64 cancellation in the convolution (c_j reach |c_0|~338 with alternating signs).
- **mpmath path** (`precision="mpmath"`, lambda=0.25): ~55s cold, ~0.25s cached, L_inf ~ 3e-15 (machine eps).
  Required because fp64 Toeplitz solve is ill-conditioned at lambda=0.25.

**Cardinal coefficients `c_j` are target-independent and cached to disk** at
`results/qi_cache/`, keyed by `(lambda_star, Kc, N, precision, mp_dps)`.
Second call at same config completes in ~0.25s even for mpmath.

**Use mpmath for baseline experiments (expD01 geometry ladder, exp13 solution basins)**
where QI is a fixed reference point. **Use fp64 everywhere else**
(training runs, sweeps, initialization). Both paths produce valid fp64 coefficients.

**mpmath does NOT violate fp64 assumptions.** The construction is an offline
precomputation that produces fp64 coefficients. The model, training loop, and
evaluation all run in fp64. Analogy: `numpy.pi` is computed at high precision
once and stored as a fp64 constant.

**Parameter warnings:**
- `lambda_star=1.5` does NOT work (intrinsic aliasing too large).
- `Kc=12` does NOT work (cardinal coefficients don't decay that fast).
- Use `Kc=160` (matches continuous-mlps) and `halo=default_halo(N, lambda_star)`.

## The Practicality Gate (binding, from Sam)

Every proposed optimizer mechanism must make the method MORE practical at real-world scale, not less. Concretely:
- No state that grows with dataset size: no per-sample caches, no per-batch parameter-sized vectors.
- Optimizer state stays O(m) in Adam's class (Adam stores 2m).
- Never patch one impracticality by adding another.
- Battle-tested mechanisms (EMA, momentum) win any design toss-up over novel ones.
Judge every design idea through this gate BEFORE proposing it.

## Key Abstractions

**QIMlp** (`src/models/mlp.py`): Single-hidden-layer tanh MLP (`nn.Module`). Always `inner_layer -> tanh -> readout`. Exposes `features(x)` for the Phi matrix and accessors for gamma, centers, readout weights.

**QIResult** (`src/construction/qi_mpmath.py`): Immutable dataclass holding construction output. Pure data -- never references a model.

**MetricsCollector** (`src/training/metrics.py`): Logs a fixed set of metrics at every eval step: train/eval loss, L_inf, relative L2, gamma/lambda stats, outer weight norms, feature rank. Writes JSONL.

**ExperimentConfig** (`src/config/schema.py`): Top-level dataclass. Every field has a default. YAML files only override what they need.

## Conventions

- **PyTorch with fp64.** `torch.set_default_dtype(torch.float64)` in `src/__init__.py`.
- **Device selection.** CUDA -> MPS -> CPU. Set in `src/__init__.py` as `DEVICE`.
- **Single hidden layer only.** No depth experiments until 1-layer is understood.
- **Freezing via requires_grad.** `param.requires_grad_(False)` to freeze.
- **Construction uses numpy/mpmath, training uses PyTorch.** Readout solve uses numpy/scipy. Model forward passes and gradients use PyTorch.
- **Multi-stage training.** Adam -> LBFGS is the default. LBFGS uses PyTorch's built-in closure pattern.
- **Analysis is per-experiment.** No pre-built analysis module. Each experiment's `run.py` implements its own analysis using PyTorch directly.
- **Results format.** JSONL for metrics. Config YAML saved alongside.
- **Experiment writeups.** Every experiment gets its OWN results file at `results/checkpoint_<X>_<name>/exp<X>NN_<name>/exp<X>NN_results.md` (e.g. `results/checkpoint_A_numerics/expA03_coeff_nullspace/expA03_results.md`) -- one per experiment, never a shared/per-experiment global doc (the only global doc is `results/results.md`). `results/` is gitignored EXCEPT `*_results.md`, `results/results.md`, and the one pinned notebook (see `.gitignore`), so writeups are version-controlled while data/figures are not.
  **Standard structure (every writeup, in this order):**
  1. **Title + Status** -- one line (approved / data-obvious / draft-pending-Sam).
  2. **TL;DR** -- 2-4 lean bullets, the takeaways; numbers only where they carry the point.
  3. **Question / hypothesis** -- get to the heart in 1-2 sentences; do not pad. Example (expA02): "Does solving the readout matrix directly beat the QI convolution, at both fp64 and mpmath?"
  4. **Experiment design** -- THE section that earns depth: a reader should come away knowing *exactly* what was tested. State the actual math (feature-matrix definitions like $\Phi_{ik}=\tanh(\gamma(x_i-c_k))$, the augmented $[\Phi,\mathbf 1]$, $\gamma=\lambda/h$; any estimator / interpolation / construction formulas), the key params (sweeps, widths, $K_c$, seeds), and the metric definitions ($L_\infty=\max_x|\hat f-f|$, rel $L_2=\|\hat f-f\|/\|f\|$). Use sub-bullets when there are several variants/checks. End with a single **Code & data** block -- the ONLY place file paths appear (run.py/config, data files, figures).
  5. **Results** -- the signal in plain language, sparse numbers. Then a **Figures** subsection with ONE bullet per figure (never one lumped paragraph): name the figure, give its layout (axes, what each line/color is), and what to look for.
  6. **Additional details** -- flexible; include only if it earns its place (derivations, confounds, caveats). Goes ABOVE Conclusions; omit the section entirely if there is nothing load-bearing.
  7. **Conclusions** -- synthesize the signed-off claim in 1-2 sentences; do NOT re-list the TL;DR.
  8. **Open questions** -- conservative (few); the global `results.md` re-aggregates them per checkpoint.

  **Voice / calibration:** aim for ~5/10 fluff -- plain language, actively cut redundancy (the biggest offender is the TL;DR restated in Conclusions). Keep the references and the per-figure how-to-reads; cut hedging and restatement. If a sentence is not adding information, delete it. The design section is the one place to add depth, not subtract it.
  **Number discipline:** spend raw numbers like currency -- only where they matter; do not saturate prose with `N=128, 1.2e-12, ...`. Use tables only when they tell a clean story.
  **Conclusions are special:** a statement may go in the Conclusions section ONLY if it is plainly obvious from the data, OR it was proposed, discussed, and explicitly approved by Sam. Do not write conclusions before Sam has reviewed the numbers and signed off; keep proposed-but-unapproved conclusions out (or clearly marked as pending). Write conservatively: state only what the data shows, and flag any metric that is not independent evidence.
- **Every experimental claim ships with a plot.** Any experiment or probe that generates data MUST produce well-thought-out figures BEFORE the results are reported to Sam: informative and uncluttered, fixed/meaningful axes (no misleading autoscale), trajectories not just endpoints, never a lone datapoint. If Claude claims a result from data it generated, the claim comes with a plot that shows it clearly -- the only escape hatch is when a plot is obviously useless. When Sam gives plot directions, follow them exactly (his plot intuition is good). Unplotted results are lazy science and hide what Sam would catch.
- **Figure legends.** Every legend on a subplot goes OUTSIDE the chart, above it -- never inside the axes (it occludes data). Use a per-axes legend placed above the axes (e.g. `ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=..., borderaxespad=0)`) or a single shared figure legend across the top (`fig.legend(..., loc="upper center", bbox_to_anchor=(0.5, ...), ncol=...)` with `subplots_adjust`/`tight_layout(rect=...)` reserving the top margin). Applies to all multi-line subplots.
- **Research-summary formatting.** Write all math in research summaries/writeups in LaTeX (`$...$` inline, `$$...$$` display, KaTeX-safe -- no `\*`, no `\emph`). Do NOT hard-wrap prose: write each paragraph as a single line and let markdown/the editor wrap it. Manual line breaks mid-paragraph (short fixed-width lines) make the docs hard to read and edit.

## Experiment Workflow

Each experiment's `run.py` is a self-contained script that imports from `src/`:

```python
from src.config import load_config, expand_sweep
from src.data import build_dataset
from src.models import QIMlp
from src.construction import construct_qi, initialize_from_construction
from src.models.freeze import freeze_gamma, freeze_centers
from src.training import run_training

config = load_config("experiments/expC01_lambda_tradeoff/config.yaml")
for cfg in expand_sweep(config):
    for seed in cfg.seeds:
        for width in cfg.widths:
            dataset = build_dataset(cfg)
            model = QIMlp(cfg.model)
            # ... construct, freeze, train, analyze, save
```

## Dependencies

PyTorch, mpmath, numpy, scipy, PyYAML.

## Success Criterion

An optimization method has been found if, as widths N increase (e.g. {32, 64, 128, 256, ...}) on the target-family matrix (6 categories), over 3-5 seeds, it's error falls at O(log(1/eps)) and eventually reaches eval relative L2 <= 1e-13 and eval L_inf consistent with machine epsilon precision, without initialization from the exact constructive solution.
