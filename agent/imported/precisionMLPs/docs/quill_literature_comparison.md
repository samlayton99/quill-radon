# QUILL literature comparison audit

Status: source audit and independent mathematical cross-checks complete, 2026-09-24. The starred entries below are deductions in this audit, not published bit-complexity theorems.

Scope: univariate analytic targets on a fixed compact interval, with fixed analytic-class constants. QUILL's supplied guarantees are accepted as established; this audit checks external comparison rows. No experiments were run.

## Recommended quantitative table

Bounds are asymptotic as epsilon tends to zero. Count nonzero weights and biases, and count only hidden layers. The derivations below occasionally abbreviate log(1/epsilon) by ell; the paper table spells out all logarithms.

| Specified construction | Parameters | Hidden layers | Maximum parameter: upper bound | Sufficient evaluation bits |
|---|---:|---:|---:|---:|
| Elementary smoothed-step baseline | O(epsilon^-1) | 1 | O(epsilon^-1 log(1/epsilon)) | O(log(1/epsilon)) |
| Mhaskar (1996), explicit tanh specialization | O(log(1/epsilon)) | 1 | exp(O(log^2(1/epsilon) log log(1/epsilon)))* | O(log^2(1/epsilon) log log(1/epsilon))* |
| De Ryck–Lanthaler–Mishra (2021) | O(log(1/epsilon)) | 2 | exp(O(log^2(1/epsilon) log log(1/epsilon)))* | O(log^2(1/epsilon) log log(1/epsilon))* |
| Opschoor–Schwab–Xenophontos (2025) | O(log(1/epsilon)) | O(log log(1/epsilon)) | epsilon^-O(1)* | O(log(1/epsilon) log log(1/epsilon))* |
| QUILL | O(log(1/epsilon)) | 1 | O(log(1/epsilon)) | O(log(1/epsilon) + log log(1/epsilon)) |

All entries are sufficient upper bounds for specified constructions, not necessary costs for every network or implementation. The elementary baseline is the group's explicit construction, not a quantitative theorem attributed to Cybenko. It may be omitted to keep the main table focused on analytic approximation methods.

*The weight specializations and floating-point estimates are derived below. Evaluation precision covers rounding the completed network's parameters and evaluating it with ordinary floating-point arithmetic, sufficient exponent range, and accurately rounded tanh. It does not cover the precision needed to compute those parameters from target data. If the paper's column includes preprocessing as well, do not substitute these evaluation estimates without further analysis. Keep the upper-bound wording: in particular, the Opschoor precision estimate is coarse and may be improvable.

QUILL's comparison is the simultaneous guarantee of logarithmic parameter count, one hidden layer, logarithmic parameter magnitude, and logarithmic evaluation precision. This audit does not establish a lower bound on competitors' necessary precision.

## De Ryck specialization

Published ingredients: Corollary 5.5, equation (110), gives one-dimensional widths 3 ceil(s/2)+N-1 and 6N, with error at most (1+delta) Q [3/(2RN)]^s. Theorem 5.1 and equation (106) quantify full-network weight growth; Lemma 3.2 treats univariate monomials.

Our calculation: fix N > max(3/2, 3/(2R)), independently of epsilon. Set s = O(ell) to reserve epsilon/2 for approximation. The second width is constant, so even fully connected parameter count is O(ell).

Use the analytic-class remainder majorant C_s = Q [3/(2R)]^s in the proof's tolerance choices. This avoids inverting a target-dependent derivative seminorm that could vanish. Specializing the published weight estimate to d=1, k=0 gives

    B <= O(C_s^(-s/2) N^((1+s^2)/2) [s(s+2)]^(3s(s+2))).

Consequently log B = O(s^2 log s) = O(ell^2 log ell). Polynomial-coefficient factors on the fixed partition do not change that order.

For a two-hidden-layer tanh network of width at most W, with all parameters bounded by B>=1 and unit roundoff u, propagating rounding errors through the affine layers and tanh's Lipschitz bound gives the conservative estimate

    evaluation error <= C u W^2 B^3.

Therefore b >= log_2(1/epsilon) + 2 log_2 W + 3 log_2 B + O(1) mantissa bits suffice. With W=O(ell), this yields the starred bit entry. This is a new forward-error estimate, not an attribution to the authors.

## Opschoor specialization

Published ingredients: Proposition 7.12 gives error C exp(-beta p) in W^{1,infinity}, O(p) nonzero parameters, and total depth ceil(log_2 p)+1. Definition A.1 recursively changes tolerance delta to delta/(4m^2). Lemma 7.9 uses fixed-arity tanh identity/product networks; Definition 7.8 merges adjacent affine maps.

Our calculation: choose p=O(ell) and delta=exp(-beta p). Across O(log p) recursion levels, the smallest tolerance satisfies

    log(1/tau_min) = beta p + O((log p)^2) = O(p).

The cited De Ryck primitives have weight bounds O(tau^-1/2) for identity and O(tau^-1) for a two-input product. Merging affine maps multiplies neighboring primitive scales, not all scales across depth; tanh layers separate the successive stages. Thus a conservative bound is

    B <= poly(p) tau_min^-2 = exp(O(p)) = epsilon^-O(1).

A generic depth-D forward perturbation estimate has log amplification O(D log(WB)). With D=O(log ell), W=O(ell), and log B=O(ell), sufficient bits are O(ell log ell). This is conservative. A calculation using bounded derivatives of the identity/product blocks could improve it; no necessary precision separation from QUILL is established here.

## Mhaskar: verified result and limits

Theorem 2.3, equation (2.13), gives geometric analytic approximation; in one dimension it yields O(log(1/epsilon)) parameters in one hidden layer. Lemma 3.2, equations (3.19)–(3.21), constructs monomials using divided differences. Its readouts contain inverse powers of the difference step and inverse activation derivatives. The step depends on degree-dependent constants that the paper does not quantify into the proposed epsilon-only maximum-weight or mantissa-bit bound.

The original preprint's theorem and relevant proof pages were available through the web search index; the complete PDF download was unavailable. Those pages verify the parameter-count claim and construction, but do not establish the screenshot's Theta(epsilon^(-1/k)) or tilde-Omega precision claims. The 1993 multilayer, order-k-activation paper is distinct from this 1996 paper.

Update: keep Mhaskar in the table. The explicit tanh instantiation below supplies conservative derived upper bounds; the absence of a published precision theorem is not a reason to omit this central comparator. This derivation was independently checked in the source-audit task. It does not validate the original screenshot's claimed precision lower bound.

### Explicit tanh instantiation of the divided-difference construction

Let phi=tanh and b0=(log 2)/2. For p>=1,

    phi^(p)(b0) = 2^(p+2) A_p(-2) / 3^(p+1),

where A_1(z)=1 and A_(p+1)(z)=(1+pz)A_p(z)+z(1-z)A_p'(z). Induction gives integer coefficients and constant term 1. Hence A_p(-2) is odd and nonzero. Thus |phi^(p)(b0)|^-1 <= exp(Cp); phi(b0)=1/3 handles p=0. This is an explicit choice satisfying the nonvanishing-derivative condition, not an assumption about an unspecified generic bias.

For p>=1 form the centered difference

    H_(p,h)(x) = sum_(r=0)^p (-1)^(p-r) binom(p,r)
                 phi(b0+(r-p/2)hx) / [h^p phi^(p)(b0)].

This uses p+1 evaluations, not 2^p evaluations. Their absolute stencil weights sum to 2^p before the h^-p and derivative normalization factors.

The integral identity for centered differences gives

    ||H_(p,h)-x^p||_infinity
      <= h^2 p sup_(t real)|phi^(p+2)(t)| / [24 |phi^(p)(b0)|].

Indeed the difference equals the average of x^p phi^(p)(b0+hx sum t_i)/phi^(p)(b0) over independent t_i uniform on [-1/2,1/2]. The linear Taylor term averages to zero and E[(sum t_i)^2]=p/12. Tanh is bounded on any fixed strip strictly inside |Im z|<pi/2. Cauchy's derivative estimate therefore bounds the right side by h^2 exp(O(p log(p+2))).

For a fixed analytic class, choose a Chebyshev polynomial approximant P of degree d=O(log(1/epsilon)) with uniform error epsilon/2. When converted to monomials, P(x)=sum_(p=0)^d a_p x^p has sum|a_p| <= exp(O(d)). Consequently replacing its monomials by H_(p,h) incurs error at most h^2 exp(O(d log(d+2))). Choose h<=1/d sufficiently small so that this is at most epsilon/2, with

    log(1/h) = O(log(1/epsilon) + d log(d+2)).

All stencils share slopes in {hj/2: -d<=j<=d}; merging equal features leaves at most 2d+1 neurons. Hidden slopes and biases are O(1). The total absolute readout weight is bounded by

    A <= exp(O(d)) (2/h)^d,

so log A = O(d log(1/epsilon) + d^2 log(d+2)). Hence

    B_max <= exp(O(log^2(1/epsilon) log log(1/epsilon))).

With bounded hidden parameters, accurately evaluated tanh, rounded parameters, and ordinary summation, forward rounding error is conservatively bounded by u poly(d)(1+A). Sufficient mantissa precision is therefore O(log^2(1/epsilon) log log(1/epsilon)). This covers evaluation of the constructed network, not generating the Chebyshev coefficients from target data. It is an upper bound for this explicit instantiation, not a necessary precision bound for all Mhaskar-style implementations.

## References

1. H. N. Mhaskar. Neural Networks for Optimal Approximation of Smooth and Analytic Functions. Neural Computation 8(1), 164–177 (1996). DOI: https://doi.org/10.1162/neco.1996.8.1.164 . Original preprint: https://citeseerx.ist.psu.edu/document?doi=694ad455c119c0d07036792b80abbf5488a9a4ca&repid=rep1&type=pdf .
2. Tim De Ryck, Samuel Lanthaler, Siddhartha Mishra. On the approximation of functions by tanh neural networks. Neural Networks 143, 732–750 (2021). DOI: https://doi.org/10.1016/j.neunet.2021.08.015 . Author PDF: https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2021/2021-14_rev1.pdf .
3. J. A. A. Opschoor, Ch. Schwab, C. Xenophontos. Neural networks for singular perturbations. Numerische Mathematik 157, 1897–1936 (2025). DOI: https://doi.org/10.1007/s00211-025-01491-6 . The relevant general analytic-function result is Proposition 7.12, not only the PDE-specific main result.
4. Weinan E, Qingcan Wang. Exponential convergence of the deep neural network approximation for analytic functions. Science China Mathematics 61(10), 1733–1740 (2018). DOI: https://doi.org/10.1007/s11425-018-9387-x . This is a deep ReLU comparison for related work, not included in the tanh table above.
