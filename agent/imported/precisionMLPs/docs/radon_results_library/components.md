# Components, extensions and unsuccessful controls

These entries account for changes after the original 60-row catalogue. They are **not twelve new independent solvers**. Each belongs to an existing family; the rankings use problem classes rather than counting every implementation change as a method.

| ID | Component/extension | Parent | Present utility | Evidence |
|---|---|---|---|---|
| U01 | Native streamed disk/box platform with independent coordinates and multi-field residuals | E09–E11, replacing E03's stored map | Main general forward-PDE platform; still a numerical solve | [Method](methods/native_solver.md), [NS/dimension evidence](evidence/navier_stokes_and_dimensions.md) |
| U02 | Ideal-coordinate preconditioner and approximate correction Jacobian | U01 | Major reduction in expensive actual-neural work on smooth cases; actual residuals still control acceptance | [Hardening record](../quill_streamed_hardening_status.md) |
| U03 | Reduced physics-constrained inverse wrapper | U01 plus earlier D-series ideas | Preferred small-parameter inverse formulation when forward gates pass | [Method](methods/constrained_inverse.md), [inverse evidence](evidence/inverse_problems.md) |
| U04 | Fixed equation-derived residual normalization | U01 | Large contrast-diffusion improvement; not a universal repair or raw-tolerance certificate | [Six-case evidence](evidence/forward_and_controls.md) |
| U05 | Universally tighter inner/outer precision control | U01/U04 | Unsuccessful tested extension: spends budget on coarse stages without attaining requested floor | [Forward controls](evidence/forward_and_controls.md) |
| U06 | Initial-plane pairing in a flat tanh model | U01 | Algebraic IC constraint is compatible with one layer; tested Allen–Cahn remains poor, cancellation damages ordinary arithmetic | [Forward controls](evidence/forward_and_controls.md) |
| U07 | Shared jets, bounded tensor products and disk recurrences | U01/U02 | Reusable operator acceleration; microbenchmarks do not establish an equal full-solve speedup | [NS/dimension evidence](evidence/navier_stokes_and_dimensions.md) |
| U08 | Refine after an uncertified ideal-correction stall | U01 | Optional policy reached p8 in full-rank 5D control; no stationarity/convergence certificate | [5D evidence](evidence/navier_stokes_and_dimensions.md) |
| U09 | Encoding/capacity separation and halo cancellation audit | A20/U01 | Explains an ordinary-arithmetic limit; known-target capacity checks are diagnosis, not blind PDE solutions | [Encoding evidence](evidence/navier_stokes_and_dimensions.md) |
| U10 | Native continuation on harder physical flows | U01/U07 | Some improved counterrotation residuals; lower-viscosity case fails budget, later cases paused | [Physical evidence](evidence/navier_stokes_and_dimensions.md) |
| U11 | Richer-resolution/ordinary-MLP validation and resource checks | U01/U03 | Required evidence discipline; forecasts, planned bytes and measured RSS separated | [Requirements](requirements_and_costs.md), [inverse evidence](evidence/inverse_problems.md) |
| U12 | Earlier direct noisy-regression evidence newly indexed here | Fixed QI/SVD, related to A01 | Actual 1D denoising evidence; not a new run or multidimensional proof | [Problem-class ranking](rankings.md#3-noisy-measurements-of-a-function-without-a-pde) |

## Construction choices to keep with A20

- Balance M-direction and N-center errors instead of interpreting neuron count alone.
- Use the corrected finite contour with sqrt(N) halos; merely truncating at sqrt(N) was a failed method (A19).
- Record lambda and derivative order. Value accuracy and PDE-derivative accuracy can prefer different bandwidths.
- Keep analytical-profile availability explicit. “Known samples,” “known transform,” “known sparse support,” and “known full function” are different input contracts.

## Improvements that do not change the scientific task

Chunking, shared profile banks, preconditioning, sign/parity grouping, recurrences and native checkpoint continuation can improve cost without altering the deployed MLP. They do not by themselves establish noise robustness, a better approximation rate, higher-dimensional tractability or a universal solver. Hardware timings and historical snapshots are retained with their corresponding source version where saved.

The [historical catalogue](catalogue/README.md) separately retains discarded attempts: uncorrected halos, nearest-direction snapping, poor geometry heuristics, external-solution compilation, product architectures and deep repair. Their utility as controls is preserved even when they no longer meet the target method's requirements.
