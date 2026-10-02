# Independent review of the new checkpoint-I probes

Status: read-only review, September 6, 2026. Reviewed `probe.py`, `refine.py`, `f_init/fixed_qi_mlp.py`, and the current `f_init/canonical_run.py`. No blocking correctness or test-selection fault found in the scalar confirmation or canonical fixed-bank runs. No other agent's files were edited.

## Findings that affect interpretation

1. **The polynomial arms change the learned function class, not only initialization.** `probe.py:43`–52 replaces each inner profile's $N$ independent coefficients with $D$ trainable coefficients mapped through fitted Chebyshev basis vectors. The restriction persists during training. “Low-degree mixer restriction plus initialization” is accurate; “initialization alone” is not. The basis consists of QI-bank approximations to low-degree functions, not exact polynomials. The quadratic arm additionally uses a radial structural prior, as already acknowledged by the coordinator.

2. **The analytic MLP matches the raw block's parameter budget, not the smaller polynomial arms.** At the currently saved confirmation configuration, actual parameter counts are raw 539, poly3 191, quadratic 179, MLP 568. This favors the MLP in capacity relative to the compact polynomial models; it does not invalidate their positive results. Report these counts rather than calling every pair exactly matched. The 24 completed confirmation rows inspected during this review had one consistent configuration: 2,000 steps, 2,560 training points, 4,096 inner test points, $d=3$, learning rate 0.005, band penalty 0.01, ridge $10^{-6}$.

3. **Per-step ridge fitting is an explicit compute cost.** `probe.py:101`–106 materializes and factors the centered full feature matrix each step, and `refine.py` repeats this per objective/gradient evaluation. For $n$ rows and $H$ head features with $n\ge H$, dense SVD has approximately $O(nH^2+H^3)$ arithmetic and stores the $n\times H$ features. The scalar block has 128 head features while its MLP has 21. Same Adam updates and roughly matched total parameters do not match training cost. The fixed-bank canonical model has no solve during ordinary Adam training and is the relevant separate practical test.

4. **Refinement's evaluation budget is soft, and its memory comment understates the constant.** `refine.py:58` checks `max_eval` only between complete `LBFGS.step` calls, then adds the explicit before/after closure calls. A stage may exceed the stated threshold; use recorded `calls` for comparisons. `history_size=10` stores ten displacement vectors and ten gradient-difference vectors, plus working state, rather than ten total vectors. This remains $O(P)$ for fixed history size, but is not Adam's two-array constant. The routine is correctly labeled a small dense diagnostic. Its training-monotonicity guard applies to the regularized objective within each ridge stage; objectives differ between stages.

5. **Canonical seeds share the existing outer test split.** Each seed changes model randomness, the training/validation partition, minibatches, and the capped training subset, but uses the same cached test points. These are three seeded runs on each existing data split, not three independent test splits. Feature min/max scaling was fitted to the original cached training split before the new validation subdivision; target standardization is recomputed only from the actual training subset. No outer-test information enters calibration or checkpoint selection. These cached Parkinsons and bike-sharing tasks use random-row splits, not held-out-patient or future-time validation; conclusions should concern those interpolation tasks.

6. **Fixed-bank equivalence is a function identity, not an optimizer identity.** `FixedQIMLP.expanded_mlp()` correctly uses repeated rows $\gamma V_r$ and biases $\gamma(b_r-c_j)$, preserving flattening order and the readout. Training that expanded dense copy freely would release the direction/bias ties and change the architecture. The canonical dense-QI arm also differs in bank counts and sampled-center initialization, so it is an approximately parameter-matched architecture/initialization comparison, not a single-variable test of tying alone.

7. **Keep reported checkpoint and diagnostic times aligned.** Canonical `selected_test_mse` is selected by validation only, which is correct. Its `final` occupancy diagnostic is measured at step 5,000, which need not be `selected_step`; do not explain selected-checkpoint performance using final occupancy as though they were the same state. The saved `precision_head`/`precise` diagnostic in the analytic probes is a different, unregularized head refit and must remain distinct from the regularized final result. `refine.run` leaves the in-memory head in that diagnostic unregularized state on return, although its returned `final` metrics describe the regularized state; the current runner saves metrics only, so this is not corrupting its recorded result.

## The ridge envelope gradient is correct for the current scalar runs

The solve minimizes

$$
\frac1n\|FW+b-y\|_2^2+\lambda\|W\|_2^2
$$

with unpenalized bias. Centering $F,y$ and using the filter $s/(s^2+n\lambda)$ is the correct solution. In `probe.train`, the solved head is frozen while differentiating the prediction loss. The ridge term has no partial derivative with respect to the nonlinear parameters when that head is held fixed, so the resulting gradient is exactly the gradient of the reoptimized regularized objective by the envelope theorem. Omitting the ridge value from that backprop expression is therefore harmless. `refine.py` correctly includes the ridge **value** for line-search comparisons.

The current runners are explicitly scalar. If generalized to $q>1$ outputs while retaining `loss.square().mean()` over all entries and penalty $\lambda\|W\|_F^2$, the solve denominator needs $nq\lambda$, not $n\lambda$; otherwise the API's stated objective changes. Likewise equal numerical $\lambda$ in different native feature bases is a shared regularization policy, not identical function-space regularization.

## Checks and outcome

The root-provided ridge tests cover the augmented-system solution and reoptimized finite differences. The fixed-bank tests were independently run: **2 passed**, verifying arbitrary-depth dense forward equivalence before and after training, constant midpoint spacing and $\gamma h$, immutable geometry buffers, and nonzero projection gradients. The three cached target arrays were checked to be one-dimensional, ruling out accidental batch-by-batch broadcasting in the canonical MSE loss.

Canonical actual parameter counts are close in the intended direction: airfoil 1,801 block versus 1,834 dense; Parkinsons 1,969 versus 1,996; bike sharing 1,885 versus 1,925. All 36 canonical runs existed when inspected. Calibration and initialization use training inputs only, samples/minibatches are shared across arms within a seed, and the analytic full-ball diagnostic uses independently generated points with no test-based state selection.

The code supports the intended bounded empirical claims when the compact polynomial restriction, solve cost, radial prior, and shared outer test splits are stated explicitly. It does not yet support a general initialization-only or equal-compute claim.
