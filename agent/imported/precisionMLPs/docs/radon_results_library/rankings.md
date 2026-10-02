# Best methods by problem class

**Current utility ranking, October 1, 2026.** Ranks mean “which existing option is most useful for this information/setup?” They are not matched benchmark scores. Every entry includes its scope. There is no universal top-three list. Where three credible options do not exist, the empty slot is recorded rather than filled with an unrelated success.

The links below lead to descriptions and evidence. A/B/C/D/E IDs refer to the [complete historical catalogue](catalogue/README.md). Latest extensions are in the [component register](components.md).

## At a glance

| Problem class | First choice | Second choice | Third choice | What this ranking actually supports |
|---|---|---|---|---|
| Known analytical function / directional profiles | A20 corrected analytical QUILL–Radon | A10/A11 shared scalar-profile solve with angular interpolation | A18 explicit sparse Fourier-ridge construction | Strong construction results; second fits scalar profiles, third requires supplied sparse structure |
| Clean function samples, no analytical profiles | A01 uniform Radon + SVD | A06/A07 learned-direction hierarchy | A02 smooth-monitor centers | Three geometry strategies sharing readout fitting; dense solve remains a limit |
| Noisy direct function measurements | Fixed-QI/SVD regression, B01 evidence | Cutoff regularization, supporting control only | **No third validated method** | Proper direct noise evidence is 1D, not a high-dimensional Radon comparison |
| Blind smooth forward PDE | Current streamed native residual solver | E02 cached native disk solver | E08 matched-frame residual iteration | Main general platform, small-memory-permitting baseline, then a restricted-operator method |
| Clean inverse parameters | Reduced native inverse wrapper | D02 native Burgers + tangent sensitivities | D01 analytical heat-tensor inverse | Only the first targets a general interface; all need identifiable data and feasible forwards |
| Noisy inverse parameters / initial profiles | Reduced native inverse wrapper | D02 noise-aware native Burgers parameters | D03 regularized unknown initial profile | First is preferred formulation with limited completed noisy evidence; others are specialized |
| Analytically tractable IVPs | B02 analytical linear propagation | B05 Cole–Hopf Burgers construction | B04 analytical wave construction | Equation-specific shortcuts, not a general nonlinear PDE method |
| Shocks / weak solutions / blowup | C09 restricted weak/entropy front construction | **No second compliant general method** | **No third compliant general method** | Only a supplied small front family has useful evidence; no demonstrated NS blowup solver |

## 1. Known function: preserve the original construction result

**1 — Corrected analytical QUILL–Radon (A20).** The strongest general construction mechanism in this program when suitable profiles are available: explicit readouts, sqrt(N) halos with the actual finite-boundary correction, and one hidden layer. The 3D anisotropic Gaussian example reached 6.40e-16 relative value error and 7.87e-15 Laplacian error. Its value is removing the global readout fit, not claiming every function has a cheap transform or few directions. [Method, derivation and raw evidence](methods/analytical_construction.md).

**2 — Shared scalar-profile solve plus angular interpolation (A10/A11).** Useful when the directional representation is known but direct complex-profile encoding is inconvenient. It avoids one global all-neuron solve by doing smaller reusable scalar work. Angular interpolation fixes the failed nearest-direction snapping approach. It does contain numerical scalar fitting, so it should not be presented as the no-fit theorem. [Full method and costs](../radon_method_catalogue.md#a11-angular-interpolation--scalar-profile-ls), [construction audit](../radon_catalogue_construction_audit.md).

**3 — Explicit sparse Fourier-ridge construction (A18).** Best when the exact target has supplied sparse directional content. Four prescribed ridges were constructed through ambient d=256 with 804 neurons and roughly 5.89e-16 error. This is a powerful high-dimensional result within that class; it does not discover sparse structure or solve a generic dense 256D target. [Construction audit](../radon_catalogue_construction_audit.md), [historical row A18](../radon_method_catalogue.md).

A14's successful long-halo complex-shift construction remains an important backup/reference within the first family. The corrected finite-halo version outranks it for present usefulness. A15 dense sphere rules and A16 Sobol rules remain dimension controls. The original construction's success is not discounted because it cannot independently discover an unknown PDE solution: these are different problems.

## 2. Clean samples: choose geometry, then fit

**1 — Uniform Radon geometry with truncated-SVD readout (A01).** Most direct baseline when only evaluations are available. It has broad low-dimensional approximation evidence and some smooth floor results. It is memory-limited by the global fit.

**2 — Learned directions and hierarchical ridge placement (A06/A07).** Worth using when the target contains a small set of important directions. Joint variable projection and direction refinement rescue the stagewise method. They require repeated global readout fits; reported hidden-ridge precision was generally around 1e-13–1e-12, not a universal 1e-14 floor.

**3 — Smooth-monitor center adaptation (A02).** Useful for localized spikes/nonuniform difficulty. Selected sharp examples needed far fewer neurons; rough monitors failed. It does not replace the global fitting bottleneck.

These are not three unrelated solvers: their main difference is how geometry is chosen. Evidence: [construction/placement audit](../radon_catalogue_construction_audit.md), [H04 placement results](../../results/checkpoint_H_highdim/expH04_mesh_finding/expH04_results.md), [H06 hierarchy results](../../results/checkpoint_H_highdim/expH06_ridge_hierarchy/expH06_results.md).

## 3. Noisy measurements of a function, without a PDE

**1 — Fixed-QI geometry and sample-based SVD regression.** The cleanest actual batch evidence is [B01 sampling/noise](../../results/checkpoint_B_scaling/expB01_sampling_and_noise/expB01_results.md), [saved data](../../results/checkpoint_B_scaling/expB01_sampling_and_noise/data.json): fixed 1D geometry, label-noise SD 1e-3, three seeds. Mean relative error against the clean target drops from 7.49e-4 at 256 observations to 3.59e-5 at 131,072. This is denoising by oversampling, not numerical-floor recovery. The separate width study selected bandwidth using clean evaluation error, so it is not a deployable noise-only selection rule.

**2 — Stronger SVD cutoff for unstable gaps: provisional control.** The [G01 generalization experiment](../../results/checkpoint_G_generalization/expG01_interactive_explorer/expG01_results.md) documents sensitivity to noise and readout cutoff. It motivates regularization but is not a matched statistical benchmark or a validated second general method.

**3 — Unestablished.** No completed three-way noisy multidimensional Radon study was found. Unknown-initial-state inference from PDE measurements belongs below, not here. Likewise “noisy sheets” in H01/H04 thicken input clouds while keeping response labels exact; they are not evidence of noisy-label recovery.

## 4. Blind forward PDEs

**1 — Current streamed native residual solver.** The main platform: independent readout coordinates, actual-neural physics, bounded-memory products/preconditioners, optional analytical correction operators and ordinary-MLP validation. Smooth 3D NS reaches near-floor values; physical disk flow reaches near-floor sampled residuals. Harder cases still fail. [Procedure](methods/native_solver.md), [forward results](evidence/forward_and_controls.md), [NS and dimensions](evidence/navier_stokes_and_dimensions.md).

**2 — Cached analytical disk coordinates (E02).** Useful when the problem is small enough to store the features: simpler, strong smooth disk precision and a good verification baseline. It shares the analytical coordinate idea with the current streamed method, but retains the very memory dependence we are trying to avoid. [Historical description](../radon_method_catalogue.md), [pure-route audit](../radon_catalogue_pure_routes_audit.md).

**3 — Matched-frame residual iteration (E08).** An instructive, successful native route for a specially matched coercive operator. It avoids global least squares using an analytical frame action. The operator match is its advantage and its restriction; it is not an established general NS or arbitrary nonlinear solver. [Saved frame study](../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/frame_iteration), [audit](../radon_catalogue_pure_routes_audit.md).

Native reduced ridge dynamics (B09/B10) remains useful for restricted periodic evolution and inverse studies. It integrates neural coordinates and exports a flat network at a fixed time. It does not currently replace a single global space-time MLP. Dense spacetime GN (B01) remains a native baseline with a memory wall. Deep/product solvers and external evolution followed by encoding are outside the required architecture/provenance.

## 5. Clean inverse problems

**1 — Reduced physics-constrained native inverse.** Best current match to the desired general interface: validate a PDE solution before scoring sensors. Wall-driven NS recovered viscosity .0299999996305 from 16 velocity locations for reference .03, with momentum RMS 2.82e-9 at requested tolerance 1e-8. This is a completed useful inverse, not a floor result; measurements use a finer same-method reference. [Method](methods/constrained_inverse.md), [evidence](evidence/inverse_problems.md).

**2 — Native Burgers parameter/tangent inverse (D02).** Stronger precision in a narrower problem: refined viscosity error about 1.14e-11, withheld field about 2.81e-13. Uses native coefficient time evolution and tangent equations; export is a spatial snapshot rather than one global space-time network. [Early PDE/inverse audit](../radon_catalogue_early_pde_audit.md).

**3 — Analytical heat-tensor inverse (D01).** A small inverse over a supported analytical forward model. Useful when that model is the real problem; six tensor entries from 36 observations achieved about 2.8e-9 noiseless parameter error. Not a recipe for an arbitrary nonlinear equation. [Same evidence audit](../radon_catalogue_early_pde_audit.md).

The joint native readout/parameter solve remains a baseline: one manufactured NS case recovered viscosity to the same float64 value, but harder Burgers campaigns exposed its finite-penalty failure. That single floor control does not outweigh broader robustness concerns.

## 6. Noisy inverse problems

**1 — Reduced physics-constrained native inverse.** Preferred general formulation because noisy data cannot purchase a fit by violating the forward PDE gate. Completed noisy diffusion matches its statistical optimum while keeping physics near roundoff. Noisy wall NS was interrupted: its best feasible candidate is saved, but there is no completed final inverse result. This first-place utility rank is **not** a claim of broad noisy-inverse validation. [Evidence](evidence/inverse_problems.md).

**2 — Native Burgers inverse with tangents (D02).** Completed noise and sensitivity controls for a small physical parameter vector. Useful in that family; it inherits the forward integrator and identifiability restrictions.

**3 — Regularized unknown-initial-profile recovery (D03).** Solves a different valuable inverse: a 17-coordinate initial profile from later observations. Noise/late-time information loss limit recovery; reported noisy profile errors are percent-scale. This supplies actual functional-inverse evidence, but not an unrestricted coefficient-field inverse.

**Special fallback — D04 backward heat with spectral regularization.** A sound example of stabilizing an ill-posed inverse, then encoding the recovered function (error .001624; neural conversion 4.20e-16). Keep it for that use, with the classical inverse dependency explicit. Conversion precision is not recovery precision. [D02–D04 details](../radon_catalogue_early_pde_audit.md).

## 7. Analytical IVPs and nonsmooth problems

For supported linear heat/diffusion, use B02 analytical propagation. For the tested Cole–Hopf Burgers family, B05 is a useful nonlinear closed-form transformation. For the analytical wave setup, B04 constructs its space-time solution. They rank as useful equation-specific routes, not competitors on a common benchmark; changing equations may remove the shortcut entirely. [Early PDE methods](../radon_catalogue_early_pde_audit.md).

For shocks, C09 demonstrates conservation/entropy checks in a supplied tiny front family. There is no established second or third compliant general shock method. C04 is a finite-volume reference, C10 a supplied shrinking-core representation, and C05's fluid animation uses product architecture. None demonstrates a general one-hidden-layer solver for shocks, singularities or NS blowup. [Failures and limits](evidence/failures_and_limits.md).

## High-dimensional use is a cross-cutting qualification

There is no dimension-only winner. A18 is excellent for supplied sparse ridges; E04 is excellent for declared low-degree polynomial structure; E05 uses supplied active axes. These are distinct assumptions. Generic tensor-direction construction and native PDE solving retain the curse of dimensionality. The latest genuine full-rank 5D control reaches 2.73e-9 with about a million neurons; 6D p16 storage forecasts already exceed 36 GB for four fields. Do not substitute specialized 20D/256D examples for that general limit. [Measured evidence and forecast distinction](evidence/navier_stokes_and_dimensions.md).
