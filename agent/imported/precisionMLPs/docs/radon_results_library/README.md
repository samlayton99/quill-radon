# Radon results library

**Historical catalogue evidence cut: October 1, 2026.** New experiments resumed with explicit authorization on October 2; the historical inventory remains a dated snapshot.

Current investigation: [native coefficient-access findings](evidence/coefficient_access_2026_10_02.md), with every run recorded in [current findings/scratchpad.md](current_findings/scratchpad.md).

Start with the problem you need to solve. There is no single overall winner: constructing a known function, recovering a function from noisy samples, and discovering an unknown PDE solution require different information and different algorithms.

| What you want to find | Where to go |
|---|---|
| Best three options for each problem class, with reasons and limits | **[Rankings by problem](rankings.md)** |
| Everything PINNs are supposed to offer, and what we actually deliver | **[PINN aims and evidence](pinn_aims.md)** |
| Every historical attempt, grouped by present usefulness | [Complete method catalogue](catalogue/README.md), [spreadsheet](catalogue/methods.csv) |
| Which newer changes are solver improvements rather than new methods | [Components and extensions](components.md) |
| Memory, construction cost, inference cost, and current requirements | [Requirements and costs](requirements_and_costs.md) |
| Where all result folders belong | [Result-folder map](catalogue/result_folders.md), [file inventory](catalogue/result_inventory.csv) |
| Existing figures, including failures | [Figure guide](figures.md) |
| What was paused and what this inventory did | [Status and provenance](STATUS.md) |

## Problem classes come first

1. **Known function, with usable analytical ridge profiles:** preserve the corrected QUILL–Radon construction as a first-class result. It generates readouts without fitting an unknown function. This is the origin of the program, not a failed PDE solver.
2. **Known function through samples:** use the sample-fitting methods; their costs differ from analytical construction. Sparse-direction methods require corresponding target structure.
3. **Noisy function measurements:** distinguish direct regression from using a known PDE to infer an unobserved state. We do not yet have a validated three-way Radon ranking for generic noisy regression.
4. **Blind forward PDEs:** native neural residual solving is the main current platform. Analytical readout coordinates make numerical coefficient discovery more manageable; they do not make discovery unnecessary.
5. **Clean inverse problems:** compare general native parameter inference with narrower analytical or native-evolution methods.
6. **Noisy inverse problems:** judge physical consistency, statistical error, and identifiability separately. Noise-aware parameter estimation is not a machine-precision recovery claim.
7. **Shocks, singularities, and difficult geometry:** retain restricted demonstrations and unsuccessful cases without promoting them into a general solver.

## Three principal building blocks

These are a dependency map, **not a global top three**. Per-problem rankings are on the next page.

| Building block | What it does | Main documentation |
|---|---|---|
| Analytical QUILL–Radon construction | Converts available directional functions into one hidden-layer tanh readouts, including finite-halo corrections | [Construction](methods/analytical_construction.md) |
| Streamed native residual solver | Finds unknown readout coordinates from PDE, boundary and initial constraints without storing a full global feature/Jacobian matrix | [Native solver](methods/native_solver.md) |
| Physics-constrained inverse wrapper | Proposes physical parameters, validates a native forward solution, then compares with observations | [Inverse wrapper](methods/constrained_inverse.md) |

The current single-hidden-layer requirement changes the relevance of older successes. Deep compilers, product networks, time-slab collections, and externally solved trajectories remain documented as controls or specialized alternatives, not compliant global-PINN winners. The original 60-method catalogue and its historical ratings are preserved; current classifications and newer evidence are added here.

## Evidence pages

- [Forward benchmarks and controls](evidence/forward_and_controls.md): all six hard-forward cases, normalization, initial-condition treatment, precision failures, and earlier streamed variants.
- [Navier–Stokes and dimensions](evidence/navier_stokes_and_dimensions.md): successful smooth 3D/4D results, physical flows, difficult failures, measured 5D progress, and dimension forecasts.
- [Inverse problems](evidence/inverse_problems.md): clean/noisy results, failed forward gates, interrupted noisy NS, and reference-data limitations.
- [Failure and limitation summary](evidence/failures_and_limits.md): the boundaries of the successful claims in one table.

Raw data, model files and figures remain in their original experiment folders. This library supplies organization and links; it does not relocate or duplicate the result payloads. Older reports are snapshots of evolving implementations. When a newer expanded audit changes the interpretation, the evidence pages identify it explicitly.
