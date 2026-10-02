# Choosing optimal lambda — iteration 7

Revises `choosing_optimal_lambda_bundle (2).zip` against the supplied `theorem_for_sam.pdf`. Both reference manuscripts remain unchanged. This iteration replaces Gaussian with GELU in the comparison figure and uses a higher-frequency sine mixture and exp(sin(3*pi*x)). It updates the appendix examples and standalone selector for GELU. The main section retains iteration 6’s balance of original motivation and concise explanation, including the increased paragraph spacing.

- `choosing_optimal_lambda_section.pdf`: main section following Sam's argument sequence, with the error bound inline, no theorem box, and Khat notation.
- `choosing_optimal_lambda_review.pdf`: the section, the 2-by-4 figure, and appendix together.
- `choosing_optimal_lambda_appendix.pdf`: proofs, the logarithmic sensitivity argument, approximation conditions, and an implementable recipe.
- Matching `.tex` files: sources for insertion into the paper.
- `figures/lambda_rule_grid.pdf` and `.png`: vector and raster figures.
- `choose_lambda.py`: standalone basic/refined selection using only Python's standard library.
- `revision_notes.md`: mathematical compatibility and changes in this iteration.

The theorem bounds the actual finite-contour alias contribution for tanh modes and finite mixtures, including the factor two from boundary anchoring. The general rule retains the leading kernel dependence; the appendix proves when a frequency prefactor only modestly shifts the selected bandwidth. The representative frequency is a rough scale estimate, not a prescribed statistic. The practical refinement uses the explicit first-pair score and absorbs amplitude and anchoring constants into the effective tolerance. General activation predictors do not inherit the finite-contour tanh theorem.

The figure follows expC08's viridis plots, including full logarithmic axes and light grids. Left: N=32,64,128,256,512 at binary64. Right: p=24,32,40,48,53 at N=128. Both halves compare tanh and exact GELU for sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(10*pi*x) and exp(sin(3*pi*x)). At p=16, the refined GELU predictions exceed the measured lambda range, so that precision is omitted from both activations. All original observations, including those omitted from the figure, remain available in source_data/. Green/basic and red/refined choices sit on measured error curves; their error coordinates are not theoretical predictions. The red choices have been recomputed from the current simplified rule. Observed source data are unchanged. No network fits were rerun.

## Use the recipe

```python
from choose_lambda import choose_lambda
from math import pi

# N=128 grid spacings on [-1,1], binary64 by default.
basic = choose_lambda("tanh", spacing=2/128)
refined = choose_lambda("tanh", spacing=2/128,
                        omega_scale=30*pi/7)
# Use "gelu" for the same rules with the GELU second-derivative kernel.
# Each returns lambda and gamma=lambda/spacing, plus search status.
```

The default `e_tol` is binary64 machine epsilon. Use `e_tol=2**-23` as a binary32 starting value, or choose another effective tolerance. The refinement needs only `omega_scale`, a rough angular frequency in radians per input unit (about 2*pi divided by a typical wavelength). Amplitude and fixed constants are assumed into `e_tol`; there is no separate amplitude input. A search-limit result is not a located threshold. Representation-theorem admissibility must still be checked separately.

## Reproduce the documents and figures

Python dependencies: `mpmath`, `numpy`, `scipy`, `matplotlib`, and `pypdf`. A LaTeX engine (`pdflatex` or `tectonic`) is required for PDFs.

```sh
python reproduce_figures.py
python check_math.py
python build_review.py
```

Alternatively specify `python build_review.py --engine /path/to/tectonic`. The builder checks overfull boxes and unresolved references, and records the actual page counts in `verification.json`; it imposes no main-section page target.

High-precision checks cover the replica/pole bridge, tail bound, normalized small-angle limit, logarithmic threshold sensitivity, and the standalone selector. They supplement the written proofs.
