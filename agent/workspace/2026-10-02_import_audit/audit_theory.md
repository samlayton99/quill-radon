# Radon-QUILL for known functions: theory audit

Agent-side document (preliminary). Read-only audit subagent, Oct 2, 2026; four cheap checks marked **[audit check]**. Status labels: **proven** (complete proof readable), **derived** (plausible derivation, not checked to proof standard), **empirical** (measured, no theorem), **conjectured**. Paths relative to `agent/imported/precisionMLPs/` unless absolute; the two `~/Downloads` notes are copied to `papers/related_notes/`.

## Summary
1. **The construction works for known functions with nothing fitted.** Each direction carries a filtered Radon profile, $\widehat q_v(\omega)=C_d|\omega|^{d-1}\widehat f(\omega v)$, $C_d=|S^{d-1}|/(2(2\pi)^{d-1})$; directions from a positive sphere rule; each profile encoded with the paper-1 corrected-halo formula. Every readout is explicit: no least squares, no SVD.
2. **Best number (corrected construction):** 3-D anisotropic Gaussian, $M=1600$, $N=129$, 244,800 neurons: value $6.40\times10^{-16}$, gradient $1.49\times10^{-15}$, Laplacian $7.87\times10^{-15}$, readout $\ell_1$ 1.90. $(M,N,\lambda)$ picked by sweep against the known target, then checked on fresh points.
3. **Biggest flag: 4-D/5-D results use the superseded long plain halo** (201 centers on $[-4,4]$ for a unit-ball target; 51 inside the band). Neuron counts (6.59M, 26.9M, 66.7M) are ~4x what the corrected construction needs. Rerun or label.
4. **Proven:** the 1-D leg (paper-1 appendix theorem); Fourier-slice representation and general-$d$ Gaussian ${}_1F_1$ profile (re-derived); error split angular + profile + rounding with no factor $M$ for positive rules; 2-D exact angular error (Jacobi-Anger); exact polynomial reproduction by Gegenbauer sphere frames; first-order snapping certificate (far too weak at $10^{-15}$).
5. **Missing theorem: angular error for $d\ge3$.** A spherical-design sketch exists only outside the repo (`review_compositional_ridge_QI.md` §2.2).
6. **New observation [audit check]:** angular error depends on the rule's exactness degree $t$, nearly independent of $d$ (4-D and 5-D within 1.6x at matched $t$). The curse of dimensionality sits in $M(t,d)$; product rules use ~$(d-1)!$-order more directions than the known lower bound (19x at $d=5$, $t=47$).
7. **Least-squares laws must not be transferred to the direct construction** (two-floor max law and its unproven $\sqrt2$ bracket; $e_N\sim N^{-10}$; $\alpha<1/2$; polynomial-floor direction count). Direct construction error is exactly a sum.
8. **Overclaims to avoid:** "no null space"; "1-D leg is dimension-free" (the $|\omega|^{d-1}$ filter sharpens profiles ~$\sqrt d$); "$d=256$ with 804 neurons" (4-ridge target); "arbitrary functions" (needs closed-form complex evaluations of $q_v$).

## 1. The construction

### 1.1 Notation
$x\in\mathbb R^d$; evaluation on ball radius $\rho$ (usually 1). $v\in S^{d-1}$; $\nu$ uniform probability on the sphere. $\widehat f(\xi)=\int f e^{-i\xi\cdot x}$. $M$ directions, weights $w_m>0$, $\sum w_m=1$. $N$ interior centers; $R=\lceil\sqrt N\rceil$ halo per side; $H=N+2R$; $B=MH$. Spacing $h$, slope $\gamma=\lambda/h$, $\lambda\approx0.25$. Complex shift $A=\pi/(2\gamma)$ (code calls it `d`).

### 1.2 Representation (proven)
$$\widehat q_v(\omega)=C_d|\omega|^{d-1}\widehat f(\omega v),\qquad f(x)=\int_{S^{d-1}} q_v(v\cdot x)\,d\nu(v).$$
Proof: Fourier inversion in polar coordinates, $d\xi=|\omega|^{d-1}d\omega\,d\sigma$, fold antipodes. [audit check] $C_3=1/(2\pi)$, so in 3-D $q_v=-\tfrac1{2\pi}\partial_t^2Rf$ and $f=\tfrac1{4\pi}\int_{S^2}q_v$ (matches expI04 `radon/report.md` eq. 1). Odd $d$: local derivative filter; even $d$: nonlocal, algebraic tails.

General-$d$ Gaussian (proven; re-derived): for $f=\exp(-(x-\mu)^\top B(x-\mu))$, $s_v=v^\top B^{-1}v$,
$$q_v(t)=\det(B)^{-1/2}s_v^{-d/2}\,{}_1F_1\!\big(\tfrac d2;\tfrac12;-(t-v\cdot\mu)^2/s_v\big).$$
Source `results/checkpoint_F_applications/expF19_radon_direct_pde/scaling/report.md`; consistent with the 3-D form in `quill_sweep.py` by Kummer. Profiles unique for $f$ on all of $\mathbb R^d$ with integrable spectrum; not on a bounded domain (§1.9).

### 1.3 Angular rules used
| Rule | Where | $M$ | Exactness/weights | Status |
|---|---|---|---|---|
| Equispaced lines on $[0,\pi)$ | 2-D | $M$ | trig degree $<2M$, equal | exact |
| Gauss-Legendre height x trapezoid azimuth, folded | 3-D (`scaling.directions`, `quill_sweep.py`, I04) | $n_z^2$ | degree $2n_z-1$, positive | standard |
| Recursive Gauss-Jacobi product, folded | `scaling_structured.sphere`, `ridge_frame_dimension_study.sphere_rule` | $n^{d-1}$ | degree $2n-1$, positive | standard; `tests/test_quill_sphere_rule.py` |
| Sobol on sphere | $d=4..64$ | $2^k$ | none, equal | empirical, no rate |
| Metric-adapted Sobol | `scaling_adapted.py` | $2^k$ | uses known $B$ | control only |
| Signed Stroud/Smolyak deg <= 3 | E04 | $d^2$, $O(d^3)$ | signed | classical |
| Angular trig interpolation | 2-D composition (H05) | 32 | signed | $3.4\times10^{-16}$; no $d\ge3$ version |
| Primitive lattice directions | finite Fourier states (`quill_ns_sweep.py`, `nonlinear_pde.py`) | # primitive dirs | none | exact for trig polynomials |
| Nearest-direction snapping | checkpoint H | any | first order | failed ($6.98\times10^{-4}$) |

Smooth angular spectra -> positive quadrature; spectra on lines -> include those directions or interpolate.

### 1.4 What carries over from 1-D per direction
Band $[-\rho,\rho]$ (cube: $[-\rho\|v\|_1,\rho\|v\|_1]$, no saving at fixed $N$). Centers $c_j=L+jh$, $j=-R..N-1+R$, $h=(U-L)/(N-1)$. **Trap:** `quill_boundary.encode` counts cells; wrappers pass `n_cells=N-1, halo=ceil(sqrt N)`. $\lambda$ value optima 0.40/0.30/0.25/0.25 at $N=33/65/129/257$; Laplacian optima 0.35/0.25/0.25/0.18. Halo $R=\lceil\sqrt N\rceil$ with finite-contour rational correction (no extra neurons).

### 1.5 Readouts in closed form
Source `papers/paper1_quill/appendix_source_v7/sections/7_appendix/01_Construction/09_26_sl.tex`; implemented in `experiments/expF19_radon_direct_pde/quill_boundary.py`.
$$a^{\rm base}_{mj}=\frac{h w_m}{2}\rho_m(c_j),\qquad\rho_m(c)=\frac{\operatorname{Im}q_{v_m}(c+iA)}{A},\qquad \widehat{\rho_m}(\omega)=\frac{i\sinh(A\omega)}{A}C_d|\omega|^{d-1}\widehat f(\omega v_m).$$
In 3-D the leading term is $-\frac{hw_m}{4\pi}(Rf)'''(v_m,c_j)$ (analogue of paper 1's readout $\propto f'$). Leading derivative samples alone fail: 29-36% (I04), 18.2% 2-D / 22.7% 3-D (F19).

Boundary correction: per endpoint $e$, $m=\lfloor(R+1)/2\rfloor$ contour moments $g_\ell=\mu_\ell-\nu_\ell$,
$$\mu_\ell=\frac1A\int_0^A\operatorname{Re}\{[q(e+iy)-q(e)]e^{\pm2i\ell\gamma y}\}dy,\quad \nu_\ell=2(-1)^\ell\int_0^\sigma\frac{\operatorname{Im}\{\rho(e+iy)e^{\pm2i\ell\gamma y}\}}{1+e^{2\pi y/h}}dy,$$
converted by explicit partial fractions ($\zeta=e^{-2\lambda}$, $r_i=\zeta^{i-1/2}$, $p_k=\prod_{j\le k}(1-\zeta^j)$):
$$c_i=(-1)^i\frac{\zeta^{[i(i+1)-1]/2}}{p_{i-1}p_{m-i}}\sum_{\ell=1}^m e_{\ell-1}(r_1,\dots,\widehat{r_i},\dots,r_m)g_\ell,$$
add $\pm c_i/2$ to the $i$-th outermost halo readout. Bias anchors each profile at $L$. Network: $\widehat f(x)=b+\sum_{m,j}a_{mj}\tanh(\gamma(v_m\cdot x-c_j))$.

Analyticity requirement (derived): each $q_v$ analytic and bounded on a tube of half-width $\delta\ge8A=4\pi h/\lambda$; equivalently $|\omega|^{d-1}\widehat f(\omega v)e^{\delta|\omega|}$ integrable uniformly in $v$. Entire profiles fine; otherwise the caller must supply `analytic_delta`.

### 1.6 Error decomposition (proven, elementary)
$f-\widehat f=E_{\rm ang}+E_{\rm prof}+E_{\rm fp}$, $E_{\rm ang}=f-\sum_m w_mq_{v_m}(v_m\cdot x)$. Positive rule: $\|E_{\rm prof}\|_\infty\le\max_v\|q_v-\mathcal Q_Nq_v\|_\infty$ (no factor $M$); signed rule factor $\sum|w_m|$. At $(1600,129)$: angular $6.5\times10^{-16}$, conversion $4.2\times10^{-16}$.

### 1.7 Angular error
1. Snapping certificate (proven, first order): $\|F-F_V\|_{L^\infty(B_r)}\le r\int\|\xi\|\theta(\hat\xi,V)\,d|\mu|$. Polynomial in $1/\varepsilon$; 2-D $kr=7.5$, $M=12$: bound 0.98 vs truth $1.1\times10^{-10}$.
2. 2-D exact (proven): equispaced lines, error of mode $J_\ell(k\rho)e^{i\ell\phi}$ is $\sum_{q\ne0}J_{\ell+qM}(kr)$; flat until $M\approx e\,kr/2$, then super-exponential ("direction cliff").
3. Spherical-design certificate "T1" (derived sketch, outside repo: `review_compositional_ridge_QI.md` §2.2, `depth_theory_response.md` §3.2): angular density a spherical polynomial of degree <= $L$ up to frequency $k$, $t$-design with $t\ge L+n$, $n+1\ge kr$ -> Bessel tail $\propto(kr/2)^{n+1}/\Gamma(n+1+d/2)$, so $M\asymp(L+\tfrac e2kr+O(\log1/\varepsilon))^{d-1}$. Gaps: positive rules exact to degree $t$; non-polynomial angular densities; spectral tail $k(\varepsilon)$. **The theorem paper 2 needs; not written with full hypotheses or checked.**
4. Empirical [audit check], anisotropic Gaussian angular error by $t$:

| $t$ | 3-D | 4-D | 5-D |
|---|---|---|---|
| 15 | 2.1e-3 | 7.3e-4 | 4.6e-4 |
| 23 | 5.9e-5 | 1.2e-5 | 9.8e-6 |
| 31 | 1.8e-6 | 2.7e-7 | 1.8e-7 |
| 47 | 4.2e-10 | 6.8e-11 | 6.4e-11 |
| 63 | 3.1e-13 | 5.6e-15 | -- |

Isotropic Gaussian floors at $t=31$ in both 4-D and 5-D; T1-style estimate predicts $t\approx28$ ($5\times10^{-11}$ at $t=23$ vs measured $3.8\times10^{-11}$/$1.5\times10^{-11}$). Anisotropic needs ~25 more degrees (the $L$ term). 3-D differs in rule, evaluation set, rotation.
5. QMC (empirical): 0.06-2.3% for $d=4..32$ at $M=16384$; no rate.

### 1.8 Scalar leg
Proven (paper 1): $\|q-\mathcal Qq\|_\infty\le\frac{128\pi B_q}{\lambda\delta}[e^{-\kappa W}+e^{-\pi^2/\lambda}/W]$; boundary part $\le16M_he^{-\lambda R^2/4}$. Constants ~12 orders loose at $N=129$ (bound $3\times10^{-4}$ vs measured $10^{-16}$). Measured boundary-limited decay ~$e^{-0.25N}$ [audit check]. For Radon profiles $B_q\le C_d\sup_v\int|\omega|^{d-1}|\widehat f(\omega v)|e^{\delta|\omega|}d\omega$ (derived, unwritten). [audit check] the $|\omega|^{d-1}$ filter peaks the isotropic-Gaussian profile spectrum at $|\omega|\approx\sqrt{2b(d-1)}$: effective bandwidth ~$\sqrt d$; profile TV 3.1 -> 18.0 from $d=2$ to $64$; so $N$ should grow ~$\sqrt d$. Conversion floor creeps with $d$: ~$4\times10^{-16}$ (3-D), $1$-$1.4\times10^{-15}$ (4-D), $3$-$4.6\times10^{-15}$ (5-D); cause not isolated. Derivatives empirical only (3-D Laplacian $7.9\times10^{-15}$ relative); best $\lambda$ differs for derivatives; no derivative theorem.

### 1.9 Non-uniqueness ("gauge")
Classical (citation to verify): non-parallel ridge sum vanishing on an open set forces polynomial profiles of degree <= $M-2$; counting $D_k=\max(0,M-\binom{d+k-1}{k})$ (derived only in the Sep 29 chat). Algebraic example (I04 `radon/report.md` §5): six directions, coefficients move 7.07x their norm with $f$ unchanged; invariant $\sum_m\alpha_mv_mv_m^\top$. Numerical: 99.9913% of constructed-vs-fitted coefficient difference lies in the discarded singular subspace (H05 `forward_check/README.md`). The canonical Radon readout is one representative; least squares/training pick another.

### 1.10 Allocating $M$ vs $N$
(a) Least squares (checkpoint H, empirical): max law within 21% (2-D), 46/48 cells within 10% (3-D), 0.97-1.00 (4-D); $\sqrt2$ bracket unproven; 2-D $e_M\sim e^{-aM^q}$, $q=0.9$-$2.1$, but $e_N\sim N^{-8.7..-12.9}$ (contradicts 1-D theory); $M^*\propto B^{0.27-0.44}$; direction count $\max(\binom{p+d-1}{d-1},c(kr)^{d-1})$, $p\approx12$ (H04) -- least-squares floor only.

(b) Direct construction (F19 `quill_review`, empirical): error exactly angular + conversion; max-rule ratios over 27 held-out cells value 1.000-1.026, gradient 1.000-1.048, Laplacian 0.969-1.155. Fixed-budget U-shapes: ~52k neurons $(64,811)\to10^{-3}$, $(576,89)\to2.5\times10^{-10}$, $(1024,49)\to1.1\times10^{-5}$; ~208k $(1600,129)\to1.9\times10^{-15}$. Optimized validated ladder:

| Neurons | Value error |
|---|---|
| 6,480 | 3.8e-5 |
| 18,000 | 2.3e-8 |
| 46,080 | 2.9e-10 |
| 84,992 | 2.1e-13 |
| 244,800 | 6.4e-16 |

(c) Allocation model (derived, conditional): angular $e^{-at}$, $t\propto M^{1/(d-1)}$; profile $e^{-bN}$; balance $t\propto N$, $M\propto B^{(d-1)/d}$, error $\approx e^{-cB^{1/d}}$. Natural law $t\approx N$ plus $\sqrt d$ sharpening; measured 3-D optimum $N/t\approx1$-$1.6$. Gaussian tails make both legs super-exponential, so a stretched form is more honest.

### 1.11 Cost
Construction: profile evaluations $O(MH\,C_{\rm prof})$ complex; contour moments $O(Mmq)$, $m\approx\sqrt N/2$, $q=\max(96,8mk+32)$ ($O(MN)$ in practice); partial fractions $O(m^3)$ shared; directions $O(Md)$ (+$O(d^3+Md^2)$ Gaussian). No linear solve. Memory $O(Md+MH)$. Inference $O(E(Md+MH))$. **Construction costs about one inference pass: Sam's "bounded by evaluation" holds when profiles are available in closed form.** Measured 5-D 66.7M neurons in 39.5 s; 521 MiB shared vs 3.56 GiB untied. Curse lives in $M(t,d)$: product rules $M=n^{d-1}$, $t=2n-1$; Delsarte-Goethals-Seidel lower bound ~$(t/2)^{d-1}/(d-1)!$; [audit check] $d=5$, $t=47$: 331,776 vs 17,550 (19x). Structured exceptions: Stroud $O(d^2)/O(d^3)$ for low-degree polynomials; sparse ridges free in $d$. Cubes: enclosing radius $\sqrt d$.

## 2. Claim ledger
| # | Claim | Status | Source | Note |
|---|---|---|---|---|
| 1 | Fourier-slice identity with $C_d$ | proven (re-derived) | I04 `radon/report.md`; F19 `scaling/report.md` | |
| 2 | ${}_1F_1$ Gaussian profile, general $d$ | proven (re-derived) | F19 `scaling/report.md` | |
| 3 | Complex shift removes sech^2 smoothing | proven (continuum) | I04; 09_26_sl.tex | samples alone 18-36% error |
| 4 | Corrected finite-contour halo | proven 1-D; matches 60-digit reference within $2.5\times10^{-16}$ | 09_26_sl.tex; `quill_boundary.py`; `boundary_validation.json` | naive halo $1.14\times10^{-4}$ |
| 5 | 3-D anisotropic Gaussian at floor | empirical | `quill_review/allocation_optimized_selected.json` | $6.40\times10^{-16}$; oracle-swept, validated |
| 6 | Naive $\sqrt N$ halo fails | empirical | same | $3.96\times10^{-5}$ |
| 7 | 4-D floor; 5-D angular-limited | empirical, **old long halo** | `scaling/structured*.json` | 4-D $8.15\times10^{-15}$ (6.59M); 5-D iso $2.6\times10^{-15}$; 5-D aniso $6.43\times10^{-11}$ |
| 8 | Error split, no $M$ factor | proven | §1.6 | |
| 9 | Snapping certificate | proven, too weak | `ridge_quadrature_theory.md` | |
| 10 | 2-D Jacobi-Anger cliff | proven | same | |
| 11 | T1 spherical-design certificate | derived sketch, outside repo | `review_compositional_ridge_QI.md` | |
| 12 | Angular error a function of $t$ | empirical [audit check] | §1.7 | |
| 13 | Polynomial-floor direction count | empirical, LS only | H04 | |
| 14 | Two-floor max law | empirical | H05/H06; `quill_review` | $\sqrt2$ bracket unproven |
| 15 | $e_N\sim N^{-10}$ | empirical; contradicts theory | H05 | unexplained |
| 16 | $\exp(-cB^{1/d})$ allocation | derived, conditional | catalogue K12 | never fit |
| 17 | Gegenbauer frames exact for degree $p$ | proven (classical) | `ridge_frame_dimension_study.py` | deg 4: $1.9$-$4.4\times10^{-15}$; deg 6 misses $10^{-14}$ |
| 18 | Angular interpolation fixes snapping on line spectra | empirical, 2-D | H05 `forward_check` | $3.37\times10^{-16}$; $4.10\times10^{-15}$ as tanh |
| 19 | Known sparse ridges free in $d$ | trivial | `scaling/sparse.json` | |
| 20 | Coefficient gauge | proven by example | I04 §5 | |
| 21 | Fitted coefficients ~ Radon prediction | empirical, interior only | H05 `radon_prediction/README.md` | 0.03-3.8% interior; 12-58% collar |
| 22 | Construction ~ one inference pass | derived from code | §1.11 | |
| 23 | Derivative accuracy | empirical only | `quill_review` | |

## 3. Contradictions, superseded versions, overclaims
1. 4-D/5-D results long-halo, ~4x inflated; `methods/analytical_construction.md` reports them beside corrected 3-D without saying so.
2. Least-squares laws blurred into the direct construction in `analytical_construction.md` (`ns_report.md`, `scaling/report.md` keep the distinction).
3. `ridge_quadrature_theory.md` Step 1 "no null space" holds only on all of space.
4. "1-D leg dimension-free" ignores $|\omega|^{d-1}$ sharpening.
5. "$\sqrt2$ bracket" labeled a theorem; unproven.
6. H05 $e_N$ power law unexplained (suspects: projected data density, uncorrected 25% collar, rcond); direct construction is exponential in $N$.
7. High-$d$ overclaims: $d=256$ is 4 ridges; QMC not precision; NS 207k-neuron result encodes an external Fourier-Galerkin state; "arbitrarily large functions in higher $d$" only for analytically known profiles.
8. "Known function" scope is narrow: needs $q_v$ at complex arguments (Gaussian mixtures, finite Fourier series, polynomials, known ridge sums, radial with closed-form transforms). H05 Abel profiles were real-valued and needed scalar LS (A10).
9. Superseded: A13/A14/A19 by A20; `practical_implementation.tex` by 09_26_sl.tex; snapping by quadrature / exact directions / interpolation.
10. Proven boundary bound ~12 orders loose at $N=129$; not predictive.

## 4. Open theoretical questions for paper 2
1. Full T1 for $d\ge3$ (positive rules, non-polynomial angular densities, spectral tail $k(\varepsilon)$); predict $t(\varepsilon)$ for both Gaussians.
2. Combined Radon-QUILL theorem: explicit $B_q(d)$, analytic tube from decay of $\widehat f$, $|\omega|^{d-1}$ bandwidth shift, balanced $t\approx N$, cost $B(\varepsilon,d)$.
3. Derivative error bounds ($\nabla$, $\Delta$) and derivative-optimal $\lambda$ -- the PDE half depends on this.
4. Better sphere cubature (designs, Lebedev, positive sparse rules) to recover the $(d-1)!$ factor; does error stay a function of $t$ alone?
5. Halo and $\lambda$ at small $N$ when many directions dominate the budget.
6. Floating-point floor vs $d$ in terms of readout $\ell_1$ and $MH$.
7. Functions known only on a bounded domain (PDE solutions): extension theory (Fourier-extension frames) or domain-adapted frames. Gegenbauer frame exact on the ball but ill-conditioned on the box (degree 12: condition 1.1 disk vs $3.9\times10^3$ inscribed square). Cubes add $\sqrt d$ to the radius.
8. Getting profiles from samples of $f$ without a global solve.
9. Line + smooth spectra in $d\ge3$: spherical angular interpolation; "background quadrature plus exact atoms" as a theorem.
10. The gauge: null space of the finite tanh dictionary; canonical readout via a Radon-domain norm (Parhi-Nowak)?
11. Explain or dismiss H05's $e_N\sim N^{-10}$.
12. Cost law with structure: when do active subspaces, quadratic forms, sparse ridges, low-degree polynomials beat $(k\rho)^{d-1}$ (`depth_theory_response.md` §3.4).

## 5. Key theory files (all copied)
- `papers/paper1_quill/appendix_source_v7/.../01_Construction/09_26_sl.tex` -- 1-D theorem + finite-contour correction (CORE).
- `experiments/expF19_radon_direct_pde/quill_boundary.py` -- corrected encoder (CORE; `validate_reference` loads `experiments/expD06_fixed_center_scales/construction_reference.py`, copied).
- `experiments/expF19_radon_direct_pde/{quill_sweep,scaling,scaling_structured,ridge_frame_dimension_study}.py` + `tests/test_quill_sphere_rule.py` (CORE).
- `results/.../expF19_radon_direct_pde/quill_review/boundary_method.md` + allocation/boundary JSON (CORE); `scaling/report.md` + `structured*.json` (CORE/SUPPORT; long-halo flag).
- `results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/radon/report.md` + `experiments/expI04_codex_geometry_unification/radon.py` (CORE/SUPPORT; origin of the training-free construction).
- `docs/ridge_quadrature_theory.md` (CORE with §3 corrections).
- `papers/related_notes/review_compositional_ridge_QI.md` §2, `papers/related_notes/depth_theory_response.md` §3 (CORE; were outside the repo).
- `results/checkpoint_H_highdim/expH05_direction_cliff_2d/spoke_profiles/radon_prediction/forward_check/README.md` (CORE).
- `docs/radon_catalogue_construction_audit.md` (CORE reference).
- SUPPORT: `results/checkpoint_I_depth_theory/compositional_qi_theory.md` §3, §11-12; H05/H06/H04/H01/H02 writeups; `docs/highdim_open_questions.md`; `docs/radon_results_library/**`; `ridge_pinn_theory.md`; lambda-rule docs.
