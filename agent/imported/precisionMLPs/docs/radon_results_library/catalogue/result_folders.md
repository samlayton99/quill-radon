# Where the result folders belong

Original artifacts stay in place. This is a logical hierarchy by utility, not a move of files or a new experiment. The [CSV inventory](result_inventory.csv) lists every file currently present in expF19; predecessor checkpoints are linked from the [60-method catalogue](README.md). File counts include interim states and failures, not just completed runs.

| Existing folder | Utility class | How to read it | Files |
|---|---|---|---|
| [quill_review](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review) | Known-function construction / halo and allocation | A19/A20; derivative/boundary/bandwidth evidence; old NS-encoding comparisons require provenance. | 28 |
| [linear](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/linear) | Analytical / linear IVPs | B02-B04; supported operators, not general nonlinear discovery. | 17 |
| [nonlinear](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/nonlinear) | Analytical nonlinear shortcuts and hybrids | B05/B06; distinguish Cole-Hopf from external evolution and encoding. | 17 |
| [native_dynamics](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/native_dynamics) | Native coefficient evolution | B07-B11; native state evolution, snapshot vs global spacetime distinction. | 26 |
| [adaptive_followup](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup) | Adaptive evolution controls | B13/B14; include failed no-replay variants and replay limits. | 35 |
| [inverse](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/inverse) | Earlier specialized inverse methods | D01-D04; distinct from later generic route2 inverse campaign. | 5 |
| [scaling](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/scaling) | Known-function dimension controls | Supplied sparse support and generic angular rules have different meanings. | 24 |
| [capabilities](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/capabilities) | Early capability demonstrations | Mixed formulations; do not infer current flat architecture without checking method. | 7 |
| [fluid_animation](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/fluid_animation) | Visualization / architecture control | C05 shared-product dynamics; attractive animation is not pure flat-MLP evidence. | 14 |
| [general_solver](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver) | Earlier equation-specific/generalization prototypes | Historical native/mixed solver stages; superseded by residual engine/route2. | 36 |
| [general_residual](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/general_residual) | Broad product/polynomial residual prototype | C06-C10; valuable controls but architecture departures. | 263 |
| [pure_mlp_depth](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/pure_mlp_depth) | Depth-route repair | E01 genuine deep tanh; outside current single-hidden-layer requirement. | 10 |
| [pure_mlp_repair](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/pure_mlp_repair) | Early pure-MLP / coordinate repairs | E01-E03 lineage; check actual export and cache costs per artifact. | 40 |
| [ridge_pinn](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/ridge_pinn) | Native dense ridge PINN baselines | B01 and related direct feature solves; memory/conditioning controls. | 62 |
| [route2_generality](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality) | Native flat solver variants and constraints | E02-E11; cached/streamed, matched-frame, parameterized and specialized high-d studies. | 239 |
| [route2_hardening](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening) | Current native PDE and dimension evidence | U01/U02/U07-U09/U11; NS, physical flow, encoders, 5D and forecasts. | 275 |
| [route2_battletest](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest) | Stress tests, controls and inverse validation | U03-U06/U10; keep failures, budgets and administrative interruptions visible. | 390 |

## Current campaigns with detailed evidence

- `route2_battletest/forward`: original paired-seed stress suite, `accuracy_v2`, and `normalized_diffusion_precision_v3`.
- `route2_battletest/normalized`: all six normalized controls; older comparison figures do not contain every later result.
- `route2_battletest/initial_plane`: failed/partial exact-IC controls.
- `route2_battletest/inverse`: joint Burgers, longer budgets, reduced diffusion, wall NS and failed forward gates.
- `route2_battletest/flows`: older hard-flow continuation, including a runner error after completed checkpoints.
- `route2_battletest/flows_recurrence`: two completed budget exits, one user-interrupted stirring case, one unstarted case.
- `route2_hardening/ns`: actual native solves plus later expanded ordinary-MLP audits.
- `route2_hardening/physical_flow`: disk stirring successes and cavity limits.
- `route2_hardening/five_dimensional*`: interrupted baseline versus separate completed cold refinement control.
- `route2_hardening/hard_encoding`: fixed-native re-encoding and clearly separated known-target capacity study.

Inventory cut: 2026-10-01; 1491 existing files in expF19. Roles are descriptive only; do not count a preflight/checkpoint as a completed solve. See [status](../STATUS.md).
