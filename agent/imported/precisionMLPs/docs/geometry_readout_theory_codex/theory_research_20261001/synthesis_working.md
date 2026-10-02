# Working synthesis — not a claim of a universal learning theorem

This is root's mathematical scratch record for the >=45-minute theory pass. The user receives the conclusions in chat. Existing empirical results are not new experiments in this pass.

## 1. The object must include more than a coefficient vector

Fix an input distribution μ, a representation h, and a dictionary of restricted features φ_θ∘h. The representation determines fibers, pushed-forward data measure, and the feature family. A coefficient measure produces a function, but distinct measures can encode the same function. The observable object is a function together with a chosen encoding convention; raw coefficients alone have no universal target-only interpretation.

The nontrivial research content is to exploit structure of this synthesis map: local Green operators, total positivity, scale refinement, and slowly varying geometry. Merely defining a pseudoinverse or differentiating the fitted function is not sufficient progress.

## 2. Stationary scale families and a common potential

For normalized even derivative kernels K_s and density fields ρ_s, d=D^r f=ΣK_s*ρ_s. On a finite resolved Fourier band, or under an explicit dual-range condition, a specified quadratic encoding norm ∫Σw_s|ρ_s|² has canonical solution ρ_s=w_s^-1 K_s*ψ. Existence of a minimum-norm coefficient field alone does not ensure an L² potential for a compact infinite-dimensional operator. For constant weights and widths, ψ̂=d̂/S, S=Σw_s^-1|k_s|². Thus K_s*(w_tρ_t)=K_t*(w_sρ_s). This compatibility is a target-independent condition on entire branches, more restrictive than checking their summed function.

At constant center density n_s, raw tanh readouts satisfy ρ_s=2n_s v_s, and raw coefficient-square cost Σv_j² becomes ∫ρ_s²/(4n_s). Density factors cannot be omitted.

For Gaussian normalized kernels K_t with Fourier transform exp(-tω²), the common-potential fields K_t*ψ satisfy ∂_t u=∂_c²u exactly. For tanh with τ=γ^-2, k_τ=t/sinh t, t=πω√τ/2. Then ∂_τ log k=(1-t coth t)/(2τ), which has leading multiplier -π²ω²/24. Hence its scale relation has heat diffusion as its low-frequency leading term, with computable higher-order corrections. This concerns canonical branch fields, not arbitrary trained weights.

## 3. Quantitative weak readout interpretation

For tanh model fhat=b+Σv_j tanh(γ_j(x-c_j)), let α_j=2v_j and K_j=γ_j sech²(γ_j·)/2. The point-charge measure is ν=Σα_jδ_cj; Dfhat=Σα_jK_j(·-c_j).

For a compactly supported C² probe q, integration by parts and Cauchy–Schwarz give |<Dfhat-Df,q>|≤||fhat-f||_2||q'||_2. Since K_j has unit mass, zero first moment, and variance π²/(12γ_j²), Taylor's theorem gives

|<ν-Df,q>| ≤ ||fhat-f||_2||q'||_2 + (π²/12)||q''||_∞ Σ_j |v_j|/γ_j².

This is a useful sufficient condition for *coarse* readout mass to track target variation. It loses force for broad kernels with huge cancelling coefficients; therefore function accuracy alone does not imply pointwise derivative-like coefficients. It applies only with integrals/boundaries well-defined and q supported away from a finite fitting boundary. Analogues for other activations use their normalized derivative kernel and absolute second moments when signed.

## 4. Exact positive width transport for tanh

For 0<γ<Γ,

ν_{γ→Γ}(x) = (Γ/π) sin(πγ/Γ) / [cosh(2γx)+cos(πγ/Γ)].

It is a probability density and has Fourier transform (Γ/γ)sinh(πω/(2Γ))/sinh(πω/(2γ)). Therefore K_γ=K_Γ*ν_{γ→Γ}, and tanh(γx)=∫ν_{γ→Γ}(u)tanh(Γ(x-u))du. Width transport composes by convolution. Narrowing the synthesis kernel permits broad atoms to spread their charges positively; reverse transport is an ill-conditioned inverse filter. This is exact on the whole line with continuum centers, not exact on a finite lattice.

At Γ=2γ, ν=2γ sech(2γx)/π. This positive refinement shows width is not identifiable from the represented function without a selection convention. It also implies width-unpenalized continuum variation-norm models have no need for broad atoms when the finest width is available everywhere; finite neuron budgets and bounded centers change this conclusion.

## 5. Generalized splines under deep geometry

For monotone scalar h=g, the operator L_g=D_x(g'^{-1}D_x)^{r-1} maps pulled-back truncated-power atoms [(g(x)-t)_+]^{r-1}/(r-1)! to point masses at g^-1(t). Its nullspace consists of polynomials in g of degree<r. For r=1 this is just D_x: the first derivative as a measure is coordinate invariant, while latent density and smoothing change.

For curved embeddings h(s), restricted ridge ReLUs need not be Green atoms of any common second-order scalar operator. Along the unit circle, L=D³+D annihilates 1,cos s,sin s, so L ReLU(w·h+b) is supported on crossings as a tied δ/δ' source. This is stronger than 'take an rth derivative': geometry can choose a differential operator and tie several sources to one coefficient. A high-order Wronskian annihilator is formally available for sufficiently regular independent coordinate functions, but is not useful unless its order, coefficients, and stability are controlled.

## 6. Precision, complexity, and depth

A neuron-count comparison alone is misleading. Smooth shallow activations can approximate polynomial operations with fixed-size finite-difference stencils while coefficients diverge. ReLU deep multiplication has a genuine accuracy-vs-depth construction with bounded weights, but its shallow ReLU lower bound does not transfer to unbounded-weight tanh. Any efficient-representation theory must count stability/weight scale as well as number of atoms and approximation error.

## Completed proof audits

- Strict total positivity and deletion alternation for finite-interval equal-γ tanh function LS with free bias.
- Mixed-width counterexample to universal alternation.
- Density-aware nonuniform moment predictor and quantified remainders.
- Discrete alias selection can defeat continuum coefficient limits even when function errors are tiny.
- Explicit deep operator and compositional cost proofs.

These results are recorded with proofs and independent review in nonuniform.md, variational_branches.md, and deep_operators.md. The audits identified and repaired the infinite-dimensional dual-range caveat, a variable-geometry periodic mean-mode issue, and the distinction between derivative-loss and function-loss regularization. No new numerical experiment has been used in this research pass.

## 7. An elementary global-polynomial example

For fixed a>0, the two-neuron function

F_{γ,a}(x)=[tanh(γ(x-a))-tanh(γ(x+a))+2tanh(γa)]/(2aγ³)

converges uniformly to x² on bounded x intervals as γ→0. Expanding tanh through fifth order gives

F_{γ,a}(x)=x²-(2/3)γ²[x⁴+2a²x²]+O(γ⁴).

Both readouts diverge as γ^-3 and the bias diverges as γ^-2. This is an explicit target→weights construction, not a fitted plot. It explains how broad neurons can represent global polynomial structure through cancellation. It also shows why a raw center/readout plot need not resemble samples of f′ outside the QI regime. Higher-order fixed-center constructions and the exact polynomial-space limit are established in variational_branches.md.

## 8. The exact information-geometry connection under a specified readout metric

For feature vector φ(x)∈R^m and positive definite coefficient penalty matrix M, the induced feature kernel is k(x,y)=φ(x)^T M^-1φ(y). Under Gaussian observation noise variance σ², readout Fisher is E_x[φφ^T]/σ². Averaging the input Fisher over centered random readouts with covariance M^-1 gives Jφ^T M^-1Jφ/σ², the pullback metric of that same kernel. This is a precise bridge between feature-manifold geometry and statistical distinguishability, not a claim that one scalar target reveals the entire input manifold.

Under a basis change φ_new=Tφ and v_new=T^-T v, one must also transform M_new=T M T^T to preserve the kernel/norm/metric. A bare feature span does not determine this geometry without a coefficient metric. Optimization and regularization can choose different metrics on the same approximation space.

## 9. A chart bottleneck gives a precise precision budget

Suppose the target factors as f=F∘g, a learned encoder approximates this specified (or aligned) chart with sup error δ, and an outer approximator approximates F uniformly by ε on a set containing the encoder outputs. If F is L-Lipschitz there, the triangle inequality gives

||Fhat∘ghat−F∘g||∞≤ε+Lδ.

This requires alignment to a true chart; arbitrary autoencoder coordinates are identifiable only up to transformations, and reconstruction error or explained variance alone does not supply δ or L. The formula explains why improving the outer QI/Radon solve cannot compensate for a fixed chart-error floor. An actual information-destroying fold has the stronger conditional-variance lower bound in deep_operators.md.

A low intrinsic dimension does not imply a single global Euclidean chart of that dimension. A compact boundaryless d-manifold cannot continuously embed injectively into R^d: invariance of domain would make its image open, while compactness makes it compact. Multiple charts, a higher-dimensional embedding, or a task-compatible quotient are needed. The prior synthetic three-dimensional graph example had a global chart and is therefore a favorable special case.

## 10. An exact broad-tent example of coefficient nonuniqueness

On the infinite uniform grid jh, let T_j(x)=max(1−|x−jh|/(2h),0). The even-indexed tents form the ordinary piecewise-linear partition of unity at spacing 2h; so do the odd-indexed tents. Therefore Σ_even T_j=Σ_odd T_j=1 and Σ_j(−1)^j T_j=0. Both v_j=1/2 and v_even=0,v_odd=1 represent the constant one exactly. Deleting any even-indexed tent is compatible with the second encoding.

The same statement is finite-dimensional on a periodic interval with an even number of appropriately periodized tents. For support radius mh, the m residue classes each partition unity, giving at least m−1 independent periodic exchange directions. This is exact geometry redundancy and not target oscillation. Infinite constant sequences are not ℓ², which is why the finite periodic version is the appropriate Hilbert-space example. Finite nonperiodic boundaries can remove exact degeneracy.

## 11. New predictive results selected for the chat explanation

- With common positive tanh slope and ordered distinct centers, ordinary full-rank function least squares with a free bias has a strictly totally positive centered Gram. Its inverse has checkerboard signs. Removing a positive coefficient therefore forces alternating correction signs, for any target and any nonuniform center grid. Mixed slopes and numerical regularization can break this law.
- The sharp limit gives exact half-differences of adjacent target cell means. This provides an explicit local derivative predictor rather than defining a numerical inverse.
- The broad fixed-N common-width limit gives polynomial approximation. Target polynomial degree r is encoded in a center-polynomial pattern of degree r−1, with sensitivity proportional to γ^(2r−1). If the degree-N target projection coefficient a_N is nonzero, v_j is asymptotic to a_N γ^(−(2N−1))/∏_{l≠j}(c_j−c_l). This predicts simultaneous center movements analytically.
- Broad mixed-width features have explicit Taylor coordinate matrices. Away from a stated rank loss, target polynomial moments computed once predict readouts under multiple geometry changes, with an O(ε²) error in normalized polynomial coordinates. This is a geometric moment conversion, not an unconstrained feature-data least-squares solve.
- A tuned three-neuron example proves how sensitive the limiting space can be: centers −a,0,a; outer slope γ; middle slope Γ²=γ²[3sech²(γa)−2]. As γ→0 the feature span tends to {1,x,x²,x⁵}, whereas equal slopes give P3. For target x³ under uniform fitting on [−1,1], the tuned LS limit is (7/30)x+(77/90)x⁵ and has relative L2 error exactly 4/45. Equal slopes have error tending to zero. This is an exact-arithmetic asymptotic, not a measured experiment or a finite-precision claim.

## 12. Literature position and claims to avoid

Generalized-TV splines, ridge/Radon spline representer theorems, compositional function spaces, and neural Hilbert ladders already supply much of the function-space language. Our useful specialization in this pass is the explicit geometry-dependent coefficient predictions and counterexamples. No priority or novel-universal-theory claim is justified.

Do not claim that arbitrary Adam branches are minimum-norm fields, that a learned low-dimensional representation is automatically a good coordinate chart, that common-width total positivity extends to arbitrary gammas, that a good forward fit identifies its raw coefficients, or that neuron count alone measures computational efficiency. The true residual gap is explaining actual heterogeneous learned coefficient allocation under its actual optimization and numerical selection rule. A representation theorem alone does not close that gap.

Exact-identifiability correction: finite sums of distinct real tanh atoms with positive slopes are linearly independent along with a constant, even with unequal slopes. In a putative zero combination, a maximal-slope atom has a nearest complex pole that no smaller-slope atom can share; different centers separate pole real parts. Its coefficient must vanish, then recurse. Exact interval equality extends analytically and fixes the finite representation up to duplicates/permutation. The severe obstruction here is approximate identification from finite-accuracy or finite-sample data, and continuum representation nonuniqueness, not an exact finite-tanh gauge.

## 13. A precise activation/architecture failure on a manifold

On the fixed circular representation h(s)=(cos s,sin s), consider a single hidden layer of bias-free ReLU units and a free output bias. Because ReLU(z)−ReLU(−z)=z, every represented function obeys f(s)−f(s+π)=A cos s+B sin s. Thus all odd Fourier harmonics of order at least three are absent. For target cos(3s), the best relative L2 error under uniform circle measure is exactly one, for any width. The same obstruction holds for any activation satisfying σ(z)−σ(−z)=z, including standard GELU, softplus and SiLU. Bias-free odd tanh instead misses nonconstant even harmonics (apart from the output constant).

This identifies a real failure of the architecture/activation/representation combination with no experiment. It does not assert that the activation fails generally, or that the obstruction survives trainable changes to the representation, biases, or additional layers. It illustrates why the candidate theory must retain source-range constraints, not only approximation capacity or a local manifold metric.

## 14. Final conceptual distinctions

The hidden map may be a task-compatible quotient, not an embedding of the full input manifold. A radial target exp(−||x||²) on a full-dimensional domain factors through the cheap scalar statistic ||x||²; retaining all input coordinates is unnecessary for that task. This supplies a concrete depth construction rather than the vacuous choice h=f: count d square modules, their sum, and a scalar exponential module, with propagated approximation errors. It does not guarantee learning those modules from data.

Keep atomic and quadratic source spaces distinct in the explanation: a finite ReLU spline has measure-valued second derivative and typically does not have a square-integrable second-derivative density. Generalized TV permits its atoms. The quadratic Green-kernel construction instead uses L² source densities with fixed boundary conditions; finite readout metrics give another exact finite-feature kernel construction. Sharing a source operator does not make these different penalties or function spaces identical.

The universal organizing language is learned approximation spaces; the derivative/source interpretation is its stronger specialization when the chosen features have a tractable Green operator or transform. It should not be asserted literally for arbitrary discrete-input architectures. The most useful results from this pass are the explicit coefficient laws, source-range obstructions, stability bounds, and constructive depth savings, not the organizational language by itself.

For a fixed geometry, exact full-rank least squares defines unique readouts. A good approximate fit, a continuum redundant frame, or finite samples need not. Any discussion of the actual observed Adam branches must distinguish those situations. We have proved interpretable limiting modes and conditional branch laws; we have not matched all those modes to the existing heterogeneous Adam runs.
