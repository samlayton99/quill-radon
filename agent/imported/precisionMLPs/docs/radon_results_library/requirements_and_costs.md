# Current requirements and cost conventions

Evidence cut: October 1, 2026. These are the current user requirements. Earlier catalogue scores used a broader architecture allowance and remain historical; they must not silently override this page.

## Required representation and information boundary

- The final global representation is one hidden-layer MLP on the original input, with one or several linear outputs. No product gates, chained hidden layers, learned input encoder or piecewise time-slab routing in the deployed result.
- Numerical correction is allowed. It must not introduce avoidable dimension dependence beyond the chosen representation; both arithmetic and memory count.
- A blind PDE solve uses the equation, domain, boundary/initial conditions, declared forcing, observations and physical priors. No external reference trajectory supplies the solution coefficients or interior fitting targets.
- Exact manufactured functions may declare forcing and boundary/initial data and supply postfit error checks. Reference-generated synthetic observations are legitimate inverse inputs, but same-method references must be disclosed.
- The long-term goal is a routine adaptive method across equations. Numerical sweeps and hand-tuned studies are evidence, not proof that current defaults are plug-and-play.
- With noise, assess recoverable statistical accuracy and physical consistency separately. Do not demand parameter error below the information supplied by the observations.
- No new experiments are authorized in this organization phase.

Native time evolution is different from encoding an external solver: it evolves coordinates of the neural representation itself. However, an exported spatial snapshot is not automatically the requested single global space-time network. Both facts appear in the catalogue.

## Symbols

| Symbol | Meaning |
|---|---|
| d | Number of model inputs, including time or parameters if supplied as inputs |
| M | Ridge directions |
| N, R, H | Interior centers, halo per side, total centers H=N+2R; current corrected construction uses R=ceil(sqrt(N)) |
| B=MH | Actual hidden neurons |
| p, P | Maximum coordinate degree; full total-degree count P=binomial(p+d,d) per field |
| F, J | Output fields and derivative channels needed by the PDE |
| Q | Collocation/residual sampling size; derivative/field factors shown separately when material |
| b, s | Bounded point batch and preconditioner block width |
| K, G | Inner Krylov products and outer nonlinear iterations; neither assumed constant as resolution grows |
| k, Qobs | Unknown inverse parameters and observation count |

## Construction versus inference

| Path | Construction/solve arithmetic | Main storage | Inference after construction |
|---|---|---|---|
| Global dense least squares, Q>=B | Feature generation plus O(Q B²); multiple nonlinear steps repeat work | O(QB+B²), plus inputs/model | Ordinary MLP O(dB+FB) per point |
| Analytical profile construction | Profile evaluation, contour/quadrature and finite-boundary correction, then angular assembly; no global readout fit | Profile/contour work plus model; shared banks avoid encoding every direction separately | Same ordinary MLP; structured direction sharing can reduce redundant dot products |
| Streamed native coordinate solve | Setup + all correction products + preconditioner setup + actual neural acceptance/audits | Model O(B(d+1+F)); coordinates/Krylov O(FP); block factors roughly O(FPs); bounded feature tiles; residual/input storage | Same final MLP; no inner solve at inference |
| Reduced native inverse | Repeated validated native forwards; finite-difference sensitivity costs grow with k, plus small Qobs-by-k algebra | Forward workspace plus observations/sensitivities, O(Qobs k+k²), and retained checkpoints | Recovered field MLP; another inverse solve only if refitting parameters |

These are summaries, not substitutes for the [historical row-specific cost formulas](../radon_method_catalogue.md). In particular:

1. A generic box coordinate-to-profile conversion still requires O(M p P) arithmetic per field. Avoiding its stored tensor does not remove that arithmetic.
2. A prepared neural residual pass costs approximately O(Q M(H+d)) for fixed field/derivative counts. Repeating it K or G times is not inference-equivalent construction.
3. Bounded preconditioner panels remove a dense allocation, but building them can still require many operator passes. Blocks have factorization costs as well as O(Ps) storage.
4. Current analytical construction includes quadrature-node setup and explicit partial-fraction work. With q contour nodes and m≈R/2, conservative setup terms include O(q³+m³), plus profile/contour contractions. These can be reused in favorable cases but are not absent by definition.
5. Total RSS includes interpreter/libraries, derivative arrays, audit batches and factorization workspace. A model's byte size or planned block allowance is not a process memory measurement.

For the present full tensor angular rule, M≈(p+1)^(d−1). A conditional spectral approximation model is error≈exp(−c B^(1/d)), with assumptions on angular/scalar regularity and enough boundary correction. It is not a measured universal law, and singular targets need different rates. Known sparse ridges and low-degree polynomial cases have different scaling because their classes are smaller.

## Accuracy labels that must remain distinct

| Label | What was actually measured |
|---|---|
| Relative field error | Network versus a known/declared reference on held-out points |
| Raw PDE RMS / maximum | Equation residual, in the original units, on stated samples |
| Scaled residual | Residual after weighting/normalization; not numerically interchangeable with raw PDE error |
| Encoding error | Ordinary neural function versus the ideal function represented by the same coordinates |
| Refinement difference | Change after increasing the representation; useful consistency evidence, not automatically true error |
| Parameter error | Recovered parameter versus known synthetic truth, or statistical optimum when explicitly stated |
| Converged / stationary / budget / interrupted | Different stopping outcomes; a small metric does not rewrite the recorded exit status |

“Near floor” always names a quantity. An RMS near 1e-15 is compatible with larger corner errors. An excellent compiler can reproduce a poor PDE solution. A noisy inverse can solve the PDE precisely while retaining statistical parameter error.
