# Revision of choosing_optimal_lambda_bundle (2)

This bundle revises the lambda subsection and its appendix. The reference manuscripts `QIs_workshop.pdf` and `theorem_for_sam.pdf` are unchanged. The latter, dated September 13, 2026, supplies the representation theorem used here.


## Iteration 7: GELU and more varied targets

The figure now compares tanh and exact GELU, using sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(10*pi*x) and exp(sin(3*pi*x)). Both targets have existing expC08 width and precision sweeps. The 2-by-4 layout, viridis curves, equal halves, green basic lines and red refined lines are retained. The precision panels show p=24,32,40,48,53: p=16 is omitted for both activations because GELU's refined predictions exceed the measured lambda interval. No observed data were changed or new fits run.

The appendix uses the GELU second-derivative transform (1+xi^2)*exp(-xi^2/2), updates the basic lookup values, and explains monotonicity on the selected search interval. Its frequency-scale examples, small-angle sensitivity illustration and numerical recipe match the displayed targets. The standalone selector now supports GELU. At fp64 the basic GELU choice is about 0.69857. Across displayed refined choices the omitted tail ratio is below 2.29e-7.

Validation covers the GELU scalar equations, the standalone recipe, the logarithmic sensitivity argument, and the existing bound and pole-bridge checks. The main section is unchanged from iteration 6. The two reference manuscripts and all original source observations remain unchanged.

## Iteration 6: balance the original motivation with the polished presentation

Restored the original progression in the opening: the competing error terms suggest pushing lambda upward, the alias budget limits that choice, and its Fourier description makes selection practical. A short third paragraph previews the basic and refined rules. The overlap explanation again describes narrower bumps and dips between neighbors, and the Fourier paragraph connects those wiggles to repeated frequency components.

The original's unsupported claims about universal applicability and numerically unstable non-alias terms are not restored. The current theorem, simplified selection equations, rough-frequency interpretation, proof, numerical implementation and figures are unchanged. Paragraph spacing remains 7 pt, and the main text still fits one page without imposing a page target.

## Iteration 5: polish the argument without changing the recipe

The opening now focuses on lambda; definitions of the fixed constants stay in the appendix. Removed the preview of the two rules, repeated statements about amplitude normalization, and the main-text explanations of integration and boundary anchoring. The mixture extension stays in the proof. The transition into the refinement explicitly says to retain the target-frequency shifts. Main paragraph spacing increases from 4 to 7 points.

The tail ratio rho_K is correct under the lemma's log-concavity condition, which tanh satisfies. It compares the first two points on the nearer alias sequence; concavity bounds every later ratio on both sequences by that value. The proof now spells this out. The minimum in the theorem denominator was optional: the exact central transform gives a valid sharper bound and has the same value for tanh. The revised numerical checks use that denominator and directly check 800 successive alias ratios.

The budget uses a non-strict inequality so the continuous maximum is well-defined, removing the distracting finite-grid qualification. The practical selection equation, standalone selector and all figure predictions remain unchanged. No page-count constraint is imposed.

## Iteration 4: the practical refinement needs a frequency scale, not an amplitude

The specialized rule now displays the first-pair formula directly and instructs the reader to solve it for lambda by bisection, then set gamma=lambda/h. It keeps the frequency powers and exact central Fourier denominator. The negligible geometric tail correction is omitted; amplitude and the fixed anchoring factor are absorbed into the chosen effective tolerance. These are practical simplifications; the theorem and proof retain the factors needed for a bound.

Bar omega is now a rough angular frequency scale, approximately 2*pi divided by a typical wavelength. An amplitude-weighted mean is one possible estimate, not a required input definition. The selector no longer takes spectral mass or output scale. Machine epsilon is a reproducible starting tolerance rather than a mandatory value. The plots use that choice and the existing expC08 frequency estimates; their red prediction locations are recomputed from the new formula. Raw observations remain unchanged.

The omitted later-pair ratio is at most 0.0001265 across the displayed predictions. Numerical checks compare the implemented first-pair score with direct high-precision evaluation. Earlier iteration descriptions below are retained as revision history.

## Iteration 3: restore the intended flow and explain the approximation

The main text now follows Sam's eight steps: strategy with an inline schematic lambda bound and budget formula; overlap intuition using tanh(gamma(x-c)); brief Fourier connection; the alias-bound theorem without a box; a short interpretation; a two-sentence basic-rule reduction; the brief mean-frequency refinement; then the plots. The opening follows the spirit of the original bundle. All Fourier notation uses Khat directly. Page compression is no longer a build requirement.

The approximation is now about the selected bandwidth. The appendix proves a logarithmic-sensitivity bound showing how the frequency prefactor moves the threshold, states when Fourier shifts can be neglected, and explains why this is not a uniform large-width limit at fixed precision. In tanh's small-angle regime, the amplitude-weighted mean reproduces the leading weighted sum because it is linear in frequency. The finite-angle substitution remains heuristic. The independent analytic-class estimate supports the quarter-scale tanh alias guarantee.

The figure follows the actual expC08 viridis figures: full log-log curves over lambda=0.03 to 1.5, light grids, five widths and six precision levels. Two equal halves have a divider at the exact figure midpoint. Basic choices are green and dashed; refined choices are red and solid. All raw fits and the prediction formulas are retained. Prediction markers have measured vertical coordinates.

The appendix adds explicit kernel equations, a tanh/Gaussian lookup table, target summaries for both plotted functions, and the conversion gamma=lambda/h. `choose_lambda.py` implements the basic and refined selectors with Python's standard library, stable log arithmetic, and an explicit output-scale convention. Its outputs reproduce the existing figure roots. The strict-budget maximum is defined on finite candidates; the continuous version uses a supremum or a non-strict maximum.

## Iteration 2: Sam's presentation changes

The main section now displays the actual representation bound, including both appearances of lambda in the prefactor and exponents. It introduces the theorem with Sam's requested sentence and uses no box. The basic rule is derived from the small-grid-angle limit, and the more specific rule is displayed separately. At fixed target angular frequency omega, theta=2|omega|/N tends to zero as width increases. The full output bound retains its frequency factor; the kernel score is the normalized limit, with negligible later replicas. The appendix states and proves that limit, and the numerical checks now include it.

The figure is a 2-by-4 comparison: tanh and Gaussian, a sine mixture and 1/(1+25x^2), width sweeps on the left and precision sweeps on the right. Green and red traces connect the basic and refined choices on the observed curves. Points predict lambda only; their error coordinates are interpolated measurements. The old two separate figures are replaced by this single grid, shown on a landscape review page and supplied as a full-size vector PDF.

## Why the section still belongs

The section explains how to select the scale parameter exposed by the representation theorem. Its progression remains: error tradeoff, overlap intuition, Fourier repetition, compact theorem, interpretation, basic rule, frequency-dependent rule, plots. The appendix proves the connection and explains the approximations.

The new result strengthens the reason for selecting the largest feasible lambda. At fixed admissible width, its combined resolution-and-boundary envelope decreases with lambda while its replica envelope increases. Maximizing lambda within a replica budget therefore minimizes that non-replica envelope over the feasible set. It does not claim to minimize measured least-squares error or the sum of all error bounds.

## Changes required by the new proof

1. **Replace the old four-term opening.** The old stencil term disappears. The new representation has resolution, corrected boundary, and replica contributions. The main theorem combines the first two. The opening is adapted to this change while retaining Sam's reasoning and order.
2. **Use the new coefficient construction.** For a Fourier mode, the local density in (A.2) exactly divides out the normalized sech-squared transform. Thus it matches the retained amplitude. The old cardinal normalization deficit no longer belongs in this section.
3. **Prove a direct bridge to the finite construction.** Expanding the tanh replica sum and the new contour pole sum gives the same positive series. The finite pole contribution is bounded by twice the old unanchored replica sum. The revised compact bound includes this factor two, caused by pinning the output at the left endpoint.
4. **Retain the width factor.** The new uniform replica bound is proportional to `exp(-pi^2/lambda)/(lambda W)`. At fixed lambda it continues to decrease with width. A width-independent basic rule is a practical default, not the exact width dependence of the theorem.
5. **Keep assumptions distinct.** The finite-construction replica theorem is proved for tanh. General activation kernels retain the Fourier-ratio calculation and conditional geometric tail estimate; the new manuscript does not prove the same boundary construction for all activations. No periodicity or global Fourier decay is imposed on the general analytic target class.
6. **Keep the frequency sum in the theorem.** A coefficient-weighted mean frequency is useful in the selection rule but generally cannot replace the weighted nonlinear bound in a theorem. A finite Fourier model is exact for a finite mixture; for a general local-analytic target, a rigorous replacement requires controlling its error on the complex contour. The class-wide replica bound already applies without a Fourier model.

The infinity-norm theorem is an absolute error statement. The complex-neighborhood bound `B` and spectral coefficient mass `S = sum |b_j|` are different quantities. Relative budgets require an explicitly chosen target normalization.

## Quarter-scale rule

For tanh, the basic score at lambda = 1/4 is approximately 5.651e-16. This score alone is not an equality with binary64 epsilon. The new proof provides a stronger independent justification: at admissible widths, its sharper replica bound is at most 2.863e-17 times the complex-neighborhood bound B. This controls the approximation's replica contribution; sampling, fitting, and evaluation remain separate.

## Figures and validation

The source data contain the 11,520 existing least-squares fits supplied with bundle (2); the figure displays the selected width and precision curves. Their raw data are unchanged. Prediction locations use the revised definitions, including the anchoring factor two. The basic score uses the single kernel ratio displayed in the text. The practical refinements for other activations remain examples of the spectral score rather than consequences of the tanh boundary theorem.

The experiments report relative sampled L2 error and use least-squares readouts. They do not implement the new explicit boundary-corrected coefficients or establish its full numerical certificate. The new manuscript's ordinary-LS result applies through the returned residual and readout norm, not merely because a solver was called.

The written proof is checked independently against both Fourier and residue derivations. `check_math.py` additionally checks deconvolution, the residue/replica identity, direct finite pole sums, geometric tail envelopes, and the quarter-scale constants at high precision. These calculations supplement the proof.

Page counts and typesetting checks are recorded in `verification.json` after building. The original long main section and appendix have been replaced in this bundle; no reference manuscript is edited.
