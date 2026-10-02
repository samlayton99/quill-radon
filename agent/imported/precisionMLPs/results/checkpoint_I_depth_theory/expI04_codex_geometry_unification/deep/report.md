# Learned deep geometry: two trained models and controlled placement interventions

The second-layer neurons do have a concrete geometric meaning: each is a hyperplane transition restricted to the curve produced by the first layer. In these two accurate networks, nearly every transition crosses that curve once. This is compatible with the kernel interpretation. It does **not** make the learned readouts samples of the target derivative: that direct prediction fails badly.

A constructive intervention does help: evenly distribute the actual crossing neurons along the first hidden curve, retain the other units, and refit the head. Test error improves by 6.0x and 2.2x over refitting the original geometry. Preserving each moved neuron's local width gives essentially the same result. Forcing *all* neurons into an evenly spaced set is worse in one seed. Uniform placement is therefore a useful controlled construction, not a universal rule.

![Summary](deep_summary.png)

**Figure caption.** Rows are independently trained seeds. Left: PCA of the second hidden activations, colored by input. The input is one-dimensional, so the existence of a curve is guaranteed; its existence is not a discovery of a hidden manifold. Middle: the actual Adam-plus-L-BFGS readouts, with transition orientation made positive, compared with an independent target-derivative prediction. Right: independent test errors after interventions and readout refits. “Uniform crossings” moves only neurons with one detected zero crossing; “uniform all” also forces noncrossing units into the domain. Random is the median of ten matched crossing-only placements. These refits have coefficient norms around 10^7–10^8 and depend on the singular-value cutoff: the improvement is in attainable noiseless approximation, not established numerical conditioning or noise robustness.

## Reproducible setup

Both models use `1 -> 48 tanh -> 48 tanh -> 1 linear`, with biases, Xavier normal weights, and float64 CPU computation. The noiseless target is

\[
f(x)=\sin(2\pi x)+0.4\exp[-80(x-0.35)^2],\qquad -1\le x\le1.
\]

Training uses 513 uniform points. Validation uses 1,024 different midpoint locations; test uses 4,096 independent uniform random locations, seed 123456. Adam runs for 10,000 steps with cosine LR decay from 0.002 to 0.0001. Standard L-BFGS then polishes. A positive loss multiplier prevents absolute curvature thresholds from prematurely suppressing small updates; it changes neither the loss minimizer nor the model class. The exact stages, including an ineffective retained seed-0 pilot, are recorded in `training_protocol.json`. This is an exploratory mechanistic test, not a broad benchmark or a claim about Adam alone.

Head refits use a finite-precision SVD solve, choosing a cutoff from 1e-10, 1e-12, 1e-14, 1e-15 using validation only. We call these empirical readout floors, not mathematically exact minimum errors. Gamma selection for constructed features also uses validation, never test.

## The geometric object actually measured

For scalar input x, let g(x) in R^48 be the first hidden activation vector. Second-layer neuron j has direction a_j in R^48, bias b_j in R, preactivation q_j(x)=a_j^T g(x)+b_j, and activation tanh(q_j(x)). Thus q_j=0 is the intersection of an affine hyperplane with the hidden curve. If this intersection occurs at c_j and q'_j(c_j) is nonzero, its local inverse width in *input coordinates* is

\[
\gamma_{\mathrm{eff},j}=|a_j^Tg'(c_j)|.
\]

This chain-rule factor matters. The norm of a_j alone is not its physical width. The derivative contribution is exactly v_j sech^2(q_j(x)) q'_j(x); beyond a locally straight curve it need not be a translated constant-width bump.

The final models have 44/48 and 45/48 single-crossing neurons. The remaining 4 and 3 have no detected crossing in the domain. No multiple crossings or sign-reversing derivative lobes were detected on the 4,097-point diagnostic grid. Crossing positions use zero interpolation and analytic derivatives. These counts are measured here, not a general restriction on deep networks.

At initialization one PC captures 99% of variance in each hidden layer. After training the first layer needs two PCs and the second needs four. Training expands the feature variation needed for a linear output, rather than simply compressing this already-one-dimensional input. In the full hidden representations, linear decoding recovers x to about 1e-11–1e-12 relative error on interleaved held-out points. Consequently, these models have not merely collapsed to the scalar target. A low PCA dimension is not by itself a theorem about the learned data manifold.

## Measured independent test errors

| Geometry / readout | Seed 0 | Seed 1 |
|---|---:|---:|
| Actual trained readout | 6.71641e-6 | 8.53194e-6 |
| Same geometry, refit | 3.23227e-06 | 3.01803e-06 |
| Uniform arclength, all neurons | 1.44207e-06 | 6.55602e-06 |
| Uniform arclength, crossing neurons | 5.38619e-07 | 1.35185e-06 |
| Uniform crossings, local widths preserved | 4.98501e-07 | 1.35928e-06 |
| Tangent QI, gamma=4 | 4.27569e-07 | 1.35555e-06 |
| Direct 1D QI, gamma=4 | 1.41863e-06 | 1.41863e-06 |
| Random crossing-only placement, median of ten | 4.70000e-6 | 1.06357e-5 |

Uniform arclength is computed along g(x), using s(x)=integral ||g'(t)|| dt. Crossing neurons are sorted by their original centers and assigned equal s-quantiles; their directions a_j stay fixed and biases become -a_j^T g(new c_j). Noncrossing neurons stay unchanged in the crossing-only policy. Width preservation additionally scales each moved row and bias by old gamma_eff / new gamma_eff.

The best individual random placements reach 6.99143e-7 and 1.00294e-6, so uniform placement is not the best possible geometry. The median comparison supports its value as a deterministic placement rule, within these two models. Dropping the noncrossing units entirely gives 2.27108e-6 / 3.06179e-6 on the native geometry and 6.92198e-7 / 1.45963e-6 after crossing-only uniformization. Their retention helps the uniformized construction modestly; it is not evidence that a special indispensable global basis has been discovered.

The strongest constructive formula is a local-tangent QI layer. Place centers c_j uniformly in input coordinates, including six halo centers per side, and set

\[
a_j=\Gamma\frac{g'(c_j)}{\|g'(c_j)\|^2},\qquad b_j=-a_j^Tg(c_j).
\]

Then **exactly** q_j(c_j)=0 and q'_j(c_j)=Gamma. Taylor expansion gives q_j(x)=Gamma(x-c_j)+O((x-c_j)^2), so tanh(q_j(x)) locally has the intended QI width. This constructs second-layer features from the learned first-layer curve without optimizing their geometry. The final head is solved. Six candidate Gamma values, 1,2,4,8,12,20, are compared using validation; both seeds choose 4. All constructions use 48 readout features, including halos. The direct one-layer QI reference has the same count.

## Independent derivative prediction: a negative result

For all single-crossing neurons, let h_j be the average gap to neighboring crossing centers (one-sided at the two endpoints), and canonicalize orientation by multiplying the learned readout by sign(q'_j(c_j)). The independent QI prediction is

\[
v_j^{\mathrm{pred}}=\frac{h_j}{2}f'(c_j).
\]

No learned coefficients, fitted amplitude, or derivative reconstruction enter that predictor. Its relative coefficient errors are **93.8% and 95.3%**, with correlations **0.411 and 0.330**. Refitted coefficients have errors essentially 100% and correlations 0.009 / -0.030. The latter are extremely cancellation-heavy, so their amplitudes cannot be inferred from local target samples alone. Interior-only errors are also recorded in the metrics, to expose boundary effects rather than hide them.

This falsifies the claim that a very accurate ordinary two-layer tanh network automatically settles into the simple local QI encoding. It does not falsify QI's derivative formula under its regular-grid, common-kernel, boundary, and bandwidth assumptions; these learned features do not satisfy those assumptions. The useful positive statement is narrower: we can explicitly turn the learned curve into a layer with prescribed local transitions, and doing so improves approximation in this controlled test.

## Numerical limits and the answer to Q1/Q2

The original refitted coefficient norms are 2.70779e8 and 1.66797e7. Crossing-uniformized norms are 2.02912e7 and 2.25681e7. At a relatively aggressive 1e-10 solve cutoff, seed 1's crossing-uniformized error is 3.20214e-4, worse than the original 5.14827e-6; it only wins when weaker directions are retained. Therefore “more uniform” does not mean “uniformly better conditioned.” Target alignment and numerical stability must be evaluated separately.

**Q1:** Yes, the second layer admits an explicit hyperplane-on-hidden-curve interpretation, and the trained examples exhibit coherent single transitions. The experiment does not support a claim that training discovered a universal simple manifold or that their raw readouts became local derivative samples.

**Q2:** Yes, a defined crossing-only arclength uniformization plus refit improves both seeds, and a tangent-QI construction gives another working intervention. No, blindly making every unit uniform is not a generally reliable improvement. The improvement is target- and solve-resolution-dependent; robustness to noise is not established by this noiseless experiment.

Files: `deep_summary.png/.pdf`, individual seed geometry/placement figures, final/initial/Adam checkpoints, `seed*_metrics.json`, `training_protocol.json`. Reproduce training with the command in the protocol; rerunning `deep.py` without `--reproduce-final` reanalyzes the saved final states. `deep_figures.py` produces the compact summary.
