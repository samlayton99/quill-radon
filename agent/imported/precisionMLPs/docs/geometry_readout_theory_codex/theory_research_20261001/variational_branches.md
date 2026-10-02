# Structured readouts: ordered sigmoid geometry, scale transport, and the limits of variational explanations

Working theory note, October 1, 2026. This pass is analytic: no model training, readout-refitting experiment, or derivative-reconstruction experiment was performed. The results below concern exact mathematical models; finite-precision cutoff effects are stated separately. The principal common-width, transport, and variational results have received an independent mathematical audit.

## 1. Setup and what is being claimed

Let μ be a finite positive fitting measure on a bounded interval, with total mass μ₀>0. It may be a finite weighted sample measure with positive weights. Assume its support has at least N+1 distinct points. Fix N distinct real centers c₁<⋯<c_N, independent of γ, and a common inverse width γ>0. Fit

\[
f_\theta(x)=b+\sum_{j=1}^Nv_j\tanh(\gamma(x-c_j))
\]

by unregularized least squares in L²(μ), with an independently optimized output bias b. Define the centered feature

\[
\widetilde\phi_j(x)=\tanh(\gamma(x-c_j))-
\frac1{\mu_0}\int\tanh(\gamma(y-c_j))\,d\mu(y)
\]

and its N×N Gram matrix G_ij=∫φ̃_iφ̃_j dμ. The conclusions will be:

1. G is **strictly totally positive**: every minor with its row and column indices in increasing order is positive. This is stronger than positive definiteness.
2. G⁻¹ has an exact alternating sign pattern. Consequently deleting or fixing one readout produces an exactly alternating response, for arbitrary nonuniform centers and arbitrary targets.
3. As all kernels broaden together, γ→0, feature singular values have orders γ,γ³,…,γ^(2N−1). This predicts conditioning directly from geometry, without computing a singular decomposition.
4. As γ→∞ with fixed centers, the Gram matrix tends to a Brownian-bridge covariance matrix. Its inverse is tridiagonal. The solved readouts become half-differences of target averages on adjacent center intervals.

These are claims about a common-width family and a specified fitting rule. They are not universal statements about every mixed-width Adam solution.

## 2. A direct Cauchy-kernel proof of total positivity

Write

\[
t=e^{2\gamma x}>0,\qquad p_j=e^{2\gamma c_j}>0,
\qquad H_j(x)=\frac{1+\tanh(\gamma(x-c_j))}{2}=\frac{t}{t+p_j}.
\]

Include the constant feature as H₀(x)=1=t/(t+0), so p₀=0 precedes the positive p_j. For increasing sample locations x₁<⋯<x_m, let A be the collocation matrix of [1,H₁,…,H_N].

The Cauchy determinant formula gives, for increasing positive t_i and increasing nonnegative p_j,

\[
\det\left[\frac1{t_i+p_j}\right]_{i,j=1}^r
=\frac{\prod_{i<k}(t_k-t_i)\prod_{j<\ell}(p_\ell-p_j)}
{\prod_{i,j}(t_i+p_j)}>0.
\]

Multiplying each row by its positive t_i preserves this sign. Every square submatrix of A is therefore strictly totally positive, including submatrices using the constant column p₀=0.

For a discrete fitting measure, let W be its positive diagonal weight matrix and B=AᵀWA. The Cauchy–Binet identity says that any r×r minor satisfies

\[
\det B_{I,J}
=\sum_{K:\,|K|=r}\left(\prod_{k\in K}w_k\right)
\det A_{K,I}\det A_{K,J}.
\]

Every term is positive. There are enough distinct sample points for all required minors, so B is strictly totally positive. For a continuous measure the same argument is the integral Cauchy–Binet identity, with products of determinants integrated over increasing sample tuples. The support assumption makes these integrals positive.

Optimizing the free bias removes the leading constant block. The centered-sigmoid Gram matrix is its Schur complement,

\[
S=B_{1:N,1:N}-B_{1:N,0}B_{00}^{-1}B_{0,1:N}.
\]

For any equally sized ordered index sets I,J⊂{1,…,N}, the block determinant identity gives

\[
\det S_{I,J}=
\frac{\det B_{\{0\}\cup I,\{0\}\cup J}}{B_{00}}>0.
\]

The denominator is B₀₀, not B₀₀ raised to the size of the minor: it is a single scalar leading block in the bordered determinant. Centered tanh features are twice centered sigmoid features, so G=4S. Positive scalar multiplication preserves strict total positivity. This proves the claim for actual tanh **function-value least squares with a free bias**, not merely for derivative-kernel fitting.

The matrix closure facts used here are standard; a primary numerical treatment is [Koev, *Accurate Computations with Totally Nonnegative Matrices* (2007)](https://doi.org/10.1137/04061903X). The displayed Cauchy–Binet and bordered-determinant calculations give the needed special case directly.

## 3. An exact explanation of alternating compensation

Cofactor expansion of the inverse gives

\[
(G^{-1})_{ij}=(-1)^{i+j}\,
\frac{\det G_{\text{row }j\text{ deleted},\,\text{column }i\text{ deleted}}}{\det G}.
\]

Every determinant in this expression is positive. Thus

\[
\boxed{\operatorname{sgn}(G^{-1})_{ij}=(-1)^{i+j}.}
\]

Let v be the original least-squares readout, and require readout j to take a prescribed value t while reoptimizing all other readouts and the free bias. Put d=t−v_j. The loss in coefficient increments is a constant plus δvᵀGδv/2 because the original residual is orthogonal to every feature. Minimizing under δv_j=d gives

\[
\boxed{\delta v_i=d\frac{(G^{-1})_{ij}}{(G^{-1})_{jj}}.}
\]

One derivation is to introduce a multiplier α for δv_j=d. Stationarity gives Gδv=αe_j, hence δv=αG⁻¹e_j; the j constraint determines α=d/(G⁻¹)_jj.

Deleting neuron j means t=0 and d=−v_j:

\[
\boxed{\delta v_i=-v_j\frac{(G^{-1})_{ij}}{(G^{-1})_{jj}}.}
\]

For a positive deleted coefficient, immediate neighbors increase, second neighbors decrease, third neighbors increase, and so on. The result does not require uniform spacing, an oscillatory target, or a perfect original fit. It is an exact property of the ordered feature geometry and the fitting rule.

This explains why alternating readout bands can appear when a node is removed, even for a nonoscillatory target. It does **not** by itself prove that correction magnitudes decay monotonically, that the even/odd envelopes are smooth, or that every observed band has this origin. Those stronger statements require additional geometric estimates.

The theorem also applies directly to the workbench operation “fix this height and solve everything else.” There the sign of d, rather than the sign of the deleted original weight, determines the response.

## 4. Broad common kernels: an exact conditioning hierarchy

Keep the centers and fitting measure fixed and let γ→0. Let I be an ordered subset of r centers. For r+1 fitting points x_{k₀}<⋯<x_{k_r}, consider the square collocation matrix A_{K,{0}∪I}. Canceling the p₀=0 denominator factors in the Cauchy formula gives the exact expression

\[
\det A_{K,\{0\}\cup I}
=\frac{\Delta(t_K)\,\Delta(p_I)\prod_{j\in I}p_j}
{\prod_{k\in K}\prod_{j\in I}(t_k+p_j)},
\]

where Δ is the ordinary Vandermonde product of pairwise increasing differences.

For fixed x and c,

\[
t_k=1+2\gamma x_k+O(\gamma^2),\qquad
p_j=1+2\gamma c_j+O(\gamma^2).
\]

There are r(r+1)/2 differences in Δ(t_K) and r(r−1)/2 differences in Δ(p_I). Therefore

\[
\Delta(t_K)\Delta(p_I)
=(2\gamma)^{r^2}\Delta(x_K)\Delta(c_I)(1+o(1)).
\]

The product of the p_j tends to one. There are r(r+1) denominator factors, each tending to two. Consequently,

\[
\boxed{\det A_{K,\{0\}\cup I}
=2^{-r}\gamma^{r^2}\Delta(x_K)\Delta(c_I)(1+o(1)).}
\]

Let M_r be the (r+1)×(r+1) moment matrix

\[
(M_r)_{ab}=\int x^{a+b}\,d\mu(x),\qquad a,b=0,\ldots,r.
\]

Applying Cauchy–Binet to the polynomial Vandermonde matrix shows that

\[
\det M_r=\sum_{K:\,|K|=r+1}\left(\prod_{k\in K}w_k\right)\Delta(x_K)^2
\]

in the discrete case, with the corresponding ordered-tuple integral in the continuous case. Inserting the previous asymptotic in the augmented feature Gram determinant gives

\[
\det B_{\{0\}\cup I,\{0\}\cup I}
=4^{-r}\gamma^{2r^2}\Delta(c_I)^2\det M_r\,(1+o(1)).
\]

Residualizing the bias divides this determinant by μ₀. Passing from sigmoid to tanh multiplies the r×r Gram determinant by 4^r. These factors cancel, yielding

\[
\boxed{\det G_{I,I}
=\gamma^{2r^2}\Delta(c_I)^2\frac{\det M_r}{\mu_0}\,(1+o(1)).}
\tag{1}
\]

All constants in (1) are positive under the distinct-center and support assumptions. The bounded interval ensures the expansions can be passed through the integrals.

Let λ₁≥⋯≥λ_N>0 be the eigenvalues of G. The r-th elementary symmetric function e_r of these eigenvalues is the sum of all r×r principal minors. Equation (1) therefore implies

\[
e_r=\Theta(\gamma^{2r^2}).
\]

If P_r=λ₁⋯λ_r, the largest product of r eigenvalues, then

\[
P_r\le e_r\le {N\choose r}P_r.
\]

Hence P_r=Θ(γ^{2r²}), and taking the ratio with P_{r−1} gives

\[
\boxed{\lambda_r(G)=\Theta(\gamma^{4r-2}),\qquad r=1,\ldots,N.}
\]

The centered feature operator has singular values equal to the square roots of these eigenvalues, so

\[
\boxed{\sigma_r=\Theta(\gamma^{2r-1})\;:\quad
\gamma,\gamma^3,\gamma^5,\ldots,\gamma^{2N-1}.}
\]

In particular,

\[
\kappa_2(G)=\Theta(\gamma^{-4(N-1)}),\qquad
\kappa_2(\text{centered features})=\Theta(\gamma^{-2(N-1)}).
\]

This is a geometry-derived prediction, not a numerical singular-value observation. It explains why broad similar tanh neurons develop a hierarchy of increasingly weak directions: after the bias, the first visible direction is approximately linear, while successive independent directions emerge at successively higher odd powers of γ. Large compensating coefficients can excite those weak directions.

The result is a fixed-N, fixed-center asymptotic. It does not directly describe a coupled QI limit in which N grows and γh is held fixed. Center clustering worsens the constants through the Vandermonde factors; equal γ does not imply good conditioning.

## 5. Sharp common kernels: a Brownian-bridge geometry

Now let γ→∞ while holding the centers fixed. Assume μ places no atom exactly at a center, and every interval between adjacent centers and the two boundaries has positive μ-mass. Normalize μ to a probability measure for this section. Put

\[
F(c)=\mu(({-\infty},c]),\quad s_i=F(c_i),\quad
s_0=0,\ s_{N+1}=1,\quad d_i=s_{i+1}-s_i>0.
\]

The sigmoid H_i(x) converges to 1{x>c_i}. Dominated convergence therefore gives its centered Gram limit

\[
K_{ij}=\operatorname{Cov}(1\{X>c_i\},1\{X>c_j\})
=\min(s_i,s_j)-s_is_j.
\]

The tanh Gram tends to 4K. For a nonnormalized fitting measure it tends to 4μ₀K. K is the covariance kernel of a Brownian bridge evaluated at the cumulative-mass positions s_i.

Its inverse Q is exactly tridiagonal:

\[
Q_{ii}=\frac1{d_{i-1}}+\frac1{d_i},\qquad
Q_{i,i+1}=Q_{i+1,i}=-\frac1{d_i},\qquad
Q_{ij}=0\ (|i-j|>1).
\]

To verify this without assuming the Brownian-bridge formula, extend a column of K by K_{0j}=K_{N+1,j}=0. As a function of s_i, K_ij is piecewise linear, with slope 1−s_j before i=j and slope −s_j after it. Thus

\[
\frac{K_{ij}-K_{i-1,j}}{d_{i-1}}
-\frac{K_{i+1,j}-K_{ij}}{d_i}
=\begin{cases}1,&i=j,\\0,&i\ne j.\end{cases}
\]

This is exactly QK=I. Equivalently,

\[
z^TQz=\sum_{i=0}^{N}\frac{(z_{i+1}-z_i)^2}{d_i},
\qquad z_0=z_{N+1}=0.
\]

The inverse geometry has become a local finite-difference operator. For finite γ its checkerboard entries are strictly nonzero; in this sharp limit the non-neighbor entries vanish.

### 5.1 The exact readout formula in this limit

Partition the fitting interval at the centers. Let I_i be its i-th resulting cell, with mass d_i, and let

\[
m_i=\frac1{d_i}\int_{I_i}f(x)\,d\mu(x),\qquad i=0,\ldots,N.
\]

The sharp-step model spans all functions constant on these N+1 cells. Since indicators of disjoint cells are orthogonal in L²(μ), its least-squares fit takes value m_i on cell i. Moving across center c_j changes a tanh step by 2v_j, so

\[
\boxed{v_j=\frac{m_j-m_{j-1}}2.}
\]

This is a direct coefficient formula from target averages. No matrix solve is needed. For constant fitting density, let the left and right cell lengths at c_j be h_L and h_R. Taylor expansion of their averages gives

\[
v_j=\frac{h_L+h_R}{4}f'(c_j)
+\frac{h_R^2-h_L^2}{12}f''(c_j)
+O(\max(h_L,h_R)^3).
\]

For equal lengths h and a sufficiently smooth target, symmetry gives

\[
v_j=\frac h2f'(c_j)+\frac{h^3}{24}f'''(c_j)+O(h^5).
\]

This establishes the derivative-reading rule in a concrete limit and gives the first nonuniform-spacing correction. For a nonconstant fitting density, the leading factor uses the difference of the two conditional cell centroids instead of simply (h_L+h_R)/2.

### 5.2 Exact deletion response in the sharp limit

For an interior deleted neuron j, substituting the tridiagonal inverse into the earlier deletion formula yields

\[
\boxed{\delta v_{j-1}=v_j\frac{d_j}{d_{j-1}+d_j},\qquad
\delta v_{j+1}=v_j\frac{d_{j-1}}{d_{j-1}+d_j},\qquad
\delta v_i=0\quad(|i-j|>1).}
\]

With equal fitting masses on both adjacent intervals, each neighbor receives half the deleted coefficient. The same formula follows by replacing the two adjacent cell means with their mass-weighted merged mean. At an endpoint there is only one neighboring neuron and the bias accounts for the remaining baseline change.

Thus sharp tanh geometry has a local coefficient response. Smoothing the steps couples farther centers and produces the full alternating compensation pattern proved above. The theorem does not claim a monotone dependence of each response magnitude on γ.

## 6. Scope boundaries and counterexamples

### Mixed widths can reverse the alternating sign law

Consider three ordered centers −δ,0,+δ on [−1,1] with a symmetric positive fitting density. Give the outer neurons width parameter γ=1 and the middle neuron γ=ε. Center every feature and divide the middle feature by ε; positive feature rescaling does not change inverse sign patterns.

As δ,ε→0, the outer centered features both tend to u(x)=tanh x and the middle one tends to x. The cofactor determining the (1,3) inverse entry tends to

\[
\langle u,x\rangle^2-\|u\|^2\|x\|^2<0,
\]

by strict Cauchy–Schwarz, because tanh x is not proportional to x on an interval. For sufficiently small positive δ,ε the same cofactor remains negative. The features remain linearly independent: the two outer tanh functions have pole lines at distinct real parts ±δ, and the middle feature has a different pole line at zero, so a nontrivial finite linear combination cannot vanish identically. The Gram determinant is therefore positive. Hence (G⁻¹)₁₃<0, whereas the common-width checkerboard theorem would require it to be positive.

This is an analytic counterexample, not a numerical experiment. Even a plausible ordering by center does not rescue the sign theorem for arbitrary widths. Grouping a mixed-width network into common-width subfamilies also does not automatically recover it: conditioning on the other families changes the effective Gram matrix.

### Ridge and truncation change the theorem

Adding readout ridge changes G to G+λI. Since G has positive entries, for large λ

\[
(G+\lambda I)^{-1}=\lambda^{-1}I-\lambda^{-2}G+O(\lambda^{-3}),
\]

so all off-diagonal entries become negative, including the distance-two entries that a checkerboard inverse would make positive. A numerical pseudoinverse or truncated SVD also need not preserve the inverse sign law. The exact theorem therefore must not be silently applied to a saved cutoff-dependent solution.

Neither total positivity nor the exact sign pattern is a conditioning guarantee. Section 4 shows that this very same family can become arbitrarily ill-conditioned.

## 7. What the weak modes look like, not just how small they are

The preceding determinant calculation also predicts the coefficient profiles and the corresponding function profiles. This is more informative than saying that the matrix is ill-conditioned.

### 7.1 Exact oscillation count at every common width

A strictly totally positive square matrix is an oscillatory matrix. The Gantmacher–Krein theorem states that its eigenvalues are positive and simple and that its r-th eigenvector, with eigenvalues ordered from largest to smallest, has exactly r−1 sign changes. Zeros are handled by the standard lower/upper sign-variation counts, which coincide for these eigenvectors. In particular, the weakest coefficient direction alternates at every consecutive center. See the original monograph [Gantmacher and Krein, *Oscillation Matrices and Kernels and Small Vibrations of Mechanical Systems*](https://books.google.com/books/about/Oscillation_Matrices_and_Kernels_and_Sma.html?id=bI28m--C82wC) and [Pinkus, Chapter 5, Theorem 5.3](https://doi.org/10.1017/CBO9780511691713.006) for the theorem and proofs.

Applied to our common-width centered tanh Gram matrix, this gives a geometry-imposed hierarchy of coefficient oscillations. It holds for every γ>0 and arbitrary distinct ordered centers under the fitting-measure assumptions. Target oscillation is not required for the geometry to possess oscillatory weak directions. Whether a particular target excites a direction is a separate question.

### 7.2 Broad-limit coefficient modes are orthogonal polynomials on the centers

Repeat the calculation leading to (1), using center sets I and J for the two determinants in Cauchy–Binet. For any ordered r-element I,J,

\[
\boxed{\det G_{I,J}
=\frac{\det M_r}{\mu_0}\gamma^{2r^2}
\Delta(c_I)\Delta(c_J)(1+o(1)).}
\tag{2}
\]

The r-th compound matrix is the matrix whose entries are these r×r minors, indexed by I,J. Equation (2) says that after division by γ^(2r²) it converges to a positive multiple of the rank-one matrix w_rw_rᵀ, where

\[
(w_r)_I=\Delta(c_I).
\]

For a symmetric matrix with orthonormal eigenvectors q₁,…,q_N, the leading eigenvector of its r-th compound matrix is q₁∧⋯∧q_r, with eigenvalue λ₁⋯λ_r. The symbol ∧ means the alternating product: its coordinates are the minors of the matrix [q₁,…,q_r]. Since the limiting compound matrix has a simple nonzero leading eigenvalue, its normalized leading eigenvector converges to w_r/∥w_r∥, up to sign.

Now form the N×r Vandermonde matrix with columns 1,c,…,c^(r−1). Its minors are exactly Δ(c_I). Thus w_r represents the subspace spanned by those center-polynomial columns. We obtain

\[
\operatorname{span}\{q_1,\ldots,q_r\}
\longrightarrow
\operatorname{span}\{1,c,\ldots,c^{r-1}\}.
\]

Apply this for r and r−1. The unit vector q_r is orthogonal to the preceding r−1 vectors, so its limit is the unique normalized degree-(r−1) polynomial evaluated at the centers and orthogonal to all lower degrees under the counting measure ∑_jδ_{c_j}. If π^c_k is the monic polynomial of degree k orthogonal for that measure, and h_c,k=∑_jπ^c_k(c_j)², then, up to sign,

\[
\boxed{q_r(j)\longrightarrow
\frac{\pi^c_{r-1}(c_j)}{\sqrt{h_{c,r-1}}}.}
\]

For irregular centers these are orthogonal polynomials for the actual center locations; they are not necessarily sampled classical Legendre polynomials.

### 7.3 The matching function modes and the exact singular-value constants

Let P₀ remove the μ-mean of a function. The augmented determinant asymptotic in Section 4, multiplied by 2^r to pass from sigmoid columns to tanh columns, gives an exterior-product identity in L²(μ):

\[
\widetilde\phi_{i_1}\wedge\cdots\wedge\widetilde\phi_{i_r}
=\gamma^{r^2}\Delta(c_I)
 (P_0x)\wedge\cdots\wedge(P_0x^r)
+o(\gamma^{r^2}).
\tag{3}
\]

For detail, wedge both sides with the constant function 1 and evaluate the resulting alternating functions on r+1 sample locations. Their values are the augmented determinants already calculated. Centering leaves those determinants unchanged because subtracting a constant-column multiple does not change a determinant. The uniform determinant remainder on the bounded interval gives convergence in the exterior Hilbert-space norm. Wedging with 1 is injective on exterior products of functions orthogonal to 1, so the centered identity (3) follows.

Consequently, the span of the first r left singular functions tends to the span of P₀x,…,P₀x^r. Orthogonality then makes the r-th left singular function tend, up to sign, to the normalized μ-orthogonal polynomial of degree r. Write that monic polynomial as π^x_r and its squared norm as h_x,r=∫π^x_r(x)²dμ(x). Then

\[
\boxed{u_r(x)\longrightarrow\frac{\pi^x_r(x)}{\sqrt{h_{x,r}}}.}
\]

There is also an exact leading constant for the singular values. Let H_k be the center moment matrix with entries ∑_jc_j^(a+b), a,b=0,…,k, and set det H_−1=1. Summing (1) over I and using the polynomial Cauchy–Binet identity gives

\[
e_r(G)\sim\frac{\det M_r}{\mu_0}\det H_{r-1}\,\gamma^{2r^2}.
\]

The orders already proved imply λ_{r+1}/λ_r=Θ(γ⁴)→0. Therefore every eigenvalue product in e_r other than λ₁⋯λ_r is a vanishing fraction of that leading product. Hence e_r/(λ₁⋯λ_r)→1. Taking consecutive ratios and using the Gram–Schmidt determinant identities h_x,r=det M_r/det M_{r−1} and h_c,r−1=det H_{r−1}/det H_{r−2} gives

\[
\boxed{\lambda_r(G)\sim h_{x,r}h_{c,r-1}\gamma^{4r-2},\qquad
\sigma_r\sim\sqrt{h_{x,r}h_{c,r-1}}\,\gamma^{2r-1}.}
\tag{4}
\]

For r=1 this reads λ₁∼N∥P₀x∥²γ², as follows directly from the nearly identical linear feature columns.

The content is specific: broad equal-width geometry pairs a polynomial of degree r in physical space with a polynomial of degree r−1 in center/readout space, at a known geometric attenuation γ^(2r−1). Geometry determines the two polynomial measures and the attenuation; the target supplies its projections onto the physical-space polynomials.

For example, a cutoff-based readout is a sum of terms ⟨f,u_r⟩q_r/σ_r over retained modes. If the highest retained mode has a target projection large enough relative to the others, its tiny σ_r makes that oscillatory center-polynomial profile dominate the visible coefficients. This is a conditional explanation, not a claim that the weakest retained mode always dominates. Symmetry or a small target projection can suppress it. Mixed-width geometry requires a separate analysis.

## 8. Broad tanh networks approach polynomial spaces

The mode result has a complementary exact algebraic explanation and a coefficient formula requiring no target least-squares solve.

### 8.1 Rational form and a stable basis for the flat limit

Continue to use t=e^(2γx), p_j=e^(2γc_j), and put

\[
Q(t)=\prod_{j=1}^N(t+p_j),\qquad u_\gamma(x)=\frac{t(x)-1}{2\gamma}.
\]

Since tanh(γ(x−c_j))=1−2p_j/(t+p_j), the network space is exactly

\[
\left\{P(t(x))/Q(t(x)):\deg P\le N\right\}.
\]

Distinct p_j allow ordinary partial fractions, proving both inclusions. This is an exact fixed-pole rational-function space in the single variable t; unequal γ values do not share this common rational variable in general.

Define B₀(x)=1 and, for k=1,…,N,

\[
B_k(x)=\frac{Q(1)}{Q(t(x))}\,u_\gamma(x)^k.
\]

These form a basis. To see independence, multiply a linear relation by Q(t), then evaluate the numerator at t=1. Since u_γ=0 there and Q(1)>0, the coefficient of B₀ is zero. The remaining distinct powers of u_γ are independent polynomials. On a bounded x interval, u_γ(x)→x uniformly and Q(t(x))/Q(1)→1 uniformly. Therefore B_k→x^k uniformly.

The original neurons become nearly dependent, but their function space approaches the full degree-N polynomial space. This reconciles poor coefficient conditioning with nontrivial global approximation power.

### 8.2 An explicit polynomial-to-readout construction

For a prescribed polynomial p(x)=a₀+∑_{k=1}^Na_kx^k, define

\[
f_\gamma(x)=a_0+
\frac{Q(1)}{Q(t(x))}\,[p(u_\gamma(x))-a_0].
\]

This is a tanh network in the fixed geometry and converges uniformly to p. Constants are reproduced exactly. Its readouts follow by taking the residue at the pole t=−p_j:

\[
\boxed{v_j=
-\frac{Q(1)\left[p\!\left(\frac{-p_j-1}{2\gamma}\right)-a_0\right]}
{2p_jQ'(-p_j)}.}
\tag{5}
\]

Indeed, the rational part has residue equal to its numerator divided by Q′(−p_j), while v_j tanh(γ(x−c_j)), viewed in t, has residue −2p_jv_j. Matching them gives (5). The rational part tends to d_∞=Q(1)a_N/(2γ)^N as t→∞; hence the output bias is b=a₀+d_∞−∑_jv_j. Formula (5) is a geometry-only conversion from polynomial coordinates to network coefficients. It is a constructive approximation, not generally the optimal least-squares readout.

If a_N≠0, p_j→1, Q(1)→2^N, and

\[
Q'(-p_j)=\prod_{\ell\ne j}(p_\ell-p_j)
\sim(-1)^{N-1}(2\gamma)^{N-1}
\prod_{\ell\ne j}(c_j-c_\ell).
\]

The highest-degree numerator term is a_N(−1/γ)^N. Substitution gives

\[
\boxed{v_j\sim a_N\gamma^{-(2N-1)}
\frac1{\prod_{\ell\ne j}(c_j-c_\ell)}.}
\tag{6}
\]

The last factor is a barycentric interpolation weight. With uniformly spaced centers c_j=c₀+jh, j=0,…,N−1, it is

\[
\frac{(-1)^{N-1-j}}{h^{N-1}j!(N-1-j)!},
\]

an alternating binomial envelope up to a common factor. The highest center-polynomial mode in Section 7 is proportional to this vector: Lagrange interpolation gives ∑_jc_j^kw_j=0 for k=0,…,N−2.

### 8.3 The optimal least-squares limit and the target information it encodes

For fixed target f∈L²(μ), the Gram matrix and right-hand side in the stable basis B₀,…,B_N converge to those of 1,x,…,x^N. The limiting moment matrix is positive definite. Thus the stable-basis least-squares coordinates converge to those of the degree-N polynomial L² projection p_N of f. In particular, the fitted network functions converge uniformly to p_N on the bounded fitting interval.

If a_N, the highest coefficient of p_N, is nonzero, the same residue calculation proves (6) for the actual optimal readouts. Terms of smaller degree carry lower powers of γ^−1, while the stable-basis top coefficient converges to a_N. If a_N=0, this leading statement vanishes; lower-degree terms and width-dependent corrections can cancel. One must not infer a universal next exponent from the target's apparent degree. A constant target, for example, has exactly zero readouts because of the free bias.

For uniform fitting measure on [−1,1], this coefficient has a derivative interpretation. Let P_N be the usual Legendre polynomial, whose leading coefficient is L_N=(2N)!/[2^N(N!)²] and whose squared norm is 2/(2N+1). Then

\[
a_N=L_N\frac{2N+1}{2}\int_{-1}^1f(x)P_N(x)\,dx.
\]

Rodrigues' formula P_N=(2^NN!)^−1D^N(x²−1)^N and N integrations by parts give, for sufficiently smooth f,

\[
\int_{-1}^1fP_N
=\frac1{2^NN!}\int_{-1}^1f^{(N)}(x)(1-x^2)^N\,dx.
\]

All boundary terms vanish because (x²−1)^N and its first N−1 derivatives vanish at ±1. Therefore

\[
\boxed{a_N=
\frac{(2N+1)(2N)!}{2^{2N+1}(N!)^3}
\int_{-1}^1f^{(N)}(x)(1-x^2)^N\,dx.}
\tag{7}
\]

For f=x^N the integral evaluates to the reciprocal of the displayed prefactor, checking a_N=1. Normalizing uniform measure to a probability measure does not change the projection or this formula.

This gives two explicit derivative-reading regimes for the same tanh family. In the sharp localized regime, adjacent readouts measure local first-derivative information through differences of cell averages. In the very broad fixed-N regime, a generic dominant readout pattern measures a global weighted N-th derivative, multiplied by a known alternating geometry profile. A neuron’s nominal derivative order alone does not determine which target information dominates a badly conditioned readout plot.

The general phenomenon of smooth flat bases approaching polynomial interpolation is established in radial-basis theory; see [Driscoll and Fornberg, *Interpolation in the limit of increasingly flat radial basis functions* (2002)](https://doi.org/10.1016/S0898-1221(01)00295-4), [author-hosted paper](https://www.colorado.edu/amath/sites/default/files/attached-files/rbflimit.pdf). The rational tanh proof and formulas above establish the particular statement used here rather than assuming that an RBF theorem automatically covers tanh.

## 9. An exact positive transport between widths

There is also a direct relation between different-width families that does not require least squares or a Gram inverse. Define the unit-mass logistic kernel

\[
\kappa_\gamma(x)=\frac\gamma2\operatorname{sech}^2(\gamma x),\qquad
\widehat\kappa_\gamma(\omega)=
\frac{\pi\omega/(2\gamma)}{\sinh(\pi\omega/(2\gamma))},
\]

using the Fourier convention f̂(ω)=∫e^(−iωx)f(x)dx. For 0<γ<Γ, put θ=πγ/Γ∈(0,π) and

\[
\boxed{\nu_{\gamma\to\Gamma}(x)=
\frac{\Gamma}{\pi}
\frac{\sin\theta}{\cosh(2\gamma x)+\cos\theta}.}
\tag{8}
\]

This is strictly positive, even, and integrable. Its Fourier transform is

\[
\widehat\nu_{\gamma\to\Gamma}(\omega)=
\frac{\Gamma}{\gamma}
\frac{\sinh(\pi\omega/(2\Gamma))}
{\sinh(\pi\omega/(2\gamma))}
=\frac{\widehat\kappa_\gamma(\omega)}
{\widehat\kappa_\Gamma(\omega)}.
\tag{9}
\]

The continuous value at ω=0 is one, so ν has unit mass. Consequently

\[
\boxed{\kappa_\gamma=\nu_{\gamma\to\Gamma}*\kappa_\Gamma,\qquad
\tanh(\gamma x)=\int_\mathbb R\nu_{\gamma\to\Gamma}(u)
\tanh(\Gamma(x-u))\,du.}
\tag{10}
\]

The second identity follows from the first by differentiation; both sides are odd, which fixes the integration constant. Thus a broad tanh neuron is exactly a positive continuum of translated narrower tanh neurons. Multiplying by v preserves its signed total readout mass v. This is a concrete width-to-center transport law.

### 9.1 Verification of the Fourier transform and positivity

The needed integral can be derived by residues. For 0<θ<π and k>0, integrate e^(ikz)/(cosh z+cosθ) over a rectangle of height 2πi. The poles inside are i(π−θ) and i(π+θ), whose residue sum is

\[
\frac{2e^{-\pi k}\sinh(\theta k)}{i\sin\theta}.
\]

The top horizontal edge contributes −e^(−2πk) times the bottom edge, and the two vertical contributions vanish as the rectangle widens. The residue theorem therefore gives

\[
\int_\mathbb R\frac{e^{iku}}{\cosh u+\cos\theta}\,du
=\frac{2\pi}{\sin\theta}\frac{\sinh(\theta k)}{\sinh(\pi k)}.
\]

Evenness handles negative k and continuity handles k=0. Substitute u=2γx and θ=πγ/Γ to obtain (9).

There is an independent positivity argument. Euler's product for sinh gives

\[
\widehat\kappa_\gamma(\omega)=
\prod_{n\ge1}\left(1+\frac{\omega^2}{4\gamma^2n^2}\right)^{-1}.
\]

Writing r=γ/Γ, each factor in the ratio κ̂_γ/κ̂_Γ is

\[
\frac{1+\omega^2/(4\Gamma^2n^2)}{1+\omega^2/(4\gamma^2n^2)}
=r^2+(1-r^2)\left(1+\frac{\omega^2}{4\gamma^2n^2}\right)^{-1}.
\]

This is the characteristic function of a mixture of a point mass at zero and a centered Laplace distribution. Their variances sum to a finite number, so the infinite convolution defines a probability measure. This proves that the Fourier ratio is a positive probability transport independently of the closed-form inversion. The underlying hyperbolic selfdecomposability is classical; see Proposition 1 and Eq. (12) of [Jurek and Yor, *Selfdecomposable Laws Associated with Hyperbolic Functions* (2004)](https://arxiv.org/pdf/1009.3542).

### 9.2 Interpretation and restrictions

These transports compose exactly:

\[
\nu_{\gamma\to\Gamma}*\nu_{\Gamma\to\Omega}
=\nu_{\gamma\to\Omega}\qquad(\gamma<\Gamma<\Omega),
\]

because the Fourier ratios telescope. For Γ=2γ, (8) reduces to ν(x)=2γ sech(2γx)/π. As Γ→∞ it tends to κ_γ. As Γ decreases to γ it tends weakly to a point mass.

The reverse map has Fourier multiplier

\[
\widehat\nu^{-1}(\omega)\sim
\frac\gamma\Gamma\exp\!\left[
\frac{\pi|\omega|}{2}\left(\frac1\gamma-\frac1\Gamma\right)\right].
\]

Thus broadening can be represented by positive spreading in center space, while undoing that spreading amplifies fine-scale information exponentially. This is a precise asymmetry behind stable width refinement versus unstable coefficient compensation.

The identity uses all real centers and a continuum of replacement neurons. On a prescribed finite or lattice center set, sampling ν gives only an approximation; quadrature, tails, and aliasing have to be controlled. In particular, (10) does not assert that finite least-squares readouts are obtained by simply sampling a positive packet. Their projection back onto the allowed centers can add signed compensations.

### 9.3 A common-scale object for a mixed-width representation

For a finite network with γ_j≤Γ, transport every neuron to the common width Γ and define the finite signed center measure

\[
\rho_\Gamma=\sum_jv_j\nu_{\gamma_j\to\Gamma}(\,\cdot-c_j),
\]

interpreting ν_{Γ→Γ} as a point mass at zero. Then

\[
f'(x)=2(\kappa_\Gamma*\rho_\Gamma)(x),\qquad
\|\rho_\Gamma\|_{\rm TV}\le\sum_j|v_j|.
\]

If two such networks represent the same function globally up to a constant, their common-scale measures are identical. Indeed, their difference η satisfies κ_Γ*η=0; Fourier transformation gives κ̂_Γη̂=0. Since κ̂_Γ has no real zero, η̂=0, and uniqueness of Fourier transforms of finite measures gives η=0. The measure is therefore independent of the particular finite network representation once its global function and Γ are fixed. Scale composition supplies compatible coordinates for different choices of Γ.

This is an exact equivalence statement with restrictive data requirements. Agreement at finitely many training points does not imply agreement of these measures, and approximate function agreement can coexist with large measure differences because deconvolution is unstable. It does not by itself predict a finite geometry's least-squares coefficients.

There is also an impossibility result in the reverse direction. A sharper kernel κ_γ with γ>Γ cannot equal κ_Γ*η for any finite signed measure η: the required Fourier transform η̂=κ̂_γ/κ̂_Γ grows exponentially, whereas a finite measure has |η̂(ω)|≤∥η∥_TV. Thus the positive width transport expresses a genuinely one-way nesting of finite-total-variation spaces. Approximation in the reverse direction may be possible on a bounded fitting interval, but it can require diverging signed coefficient mass.

## 10. What a variational explanation can and cannot establish

It is tempting to explain smooth learned readout branches as the support of a dual variational object. There is a rigorous object here, but its assumptions matter and its simplest form does not explain generic mixed-width learned branches.

### 10.1 Finite-network stationarity

Let φ_{c,γ}(x)=tanh(γ(x−c)), let r=f−f_θ be the target residual, and consider

\[
\mathcal L=\tfrac12\|r\|_\mu^2+\tfrac\lambda2\sum_jv_j^2,
\qquad Q(c,\gamma)=\langle r,\phi_{c,\gamma}\rangle_\mu.
\]

For free bias, centers, widths, and readouts, differentiating the loss gives the necessary stationarity conditions

\[
\langle r,1\rangle_\mu=0,\qquad
Q(c_j,\gamma_j)=\lambda v_j,\qquad
v_j\partial_cQ(c_j,\gamma_j)=0,\qquad
v_j\partial_\gamma Q(c_j,\gamma_j)=0.
\tag{11}
\]

For example, ∂L/∂c_j=−v_j⟨r,∂_cφ_j⟩=−v_j∂_cQ, holding the residual fixed when defining the displayed derivative of Q. The other derivatives follow identically.

At an active neuron with v_j≠0, its geometry lies at a critical point of this residual correlation field. For ridge-penalized readouts, its height is Q/λ. With λ=0 and exact fit, r=0 and all conditions are vacuous. At v_j=0 the geometry conditions are also vacuous. Neither case supports an informative universal curve law.

Even when λ>0, generic isolated critical points of a smooth two-variable field do not form a continuous branch. A branch would require a critical ridge, a symmetry, a constraint, or an approximate rather than exact stationarity interpretation. Moreover Q depends on the residual and hence on the whole fitted solution; simply renaming it a lens does not create a target-independent predictor.

If μ is Lebesgue measure on [a,b], set R(x)=∫_a^xr(t)dt. The free-bias equation implies R(a)=R(b)=0, so integration by parts gives

\[
Q(c,\gamma)=-\int_a^bR(x)\gamma\operatorname{sech}^2(\gamma(x-c))\,dx.
\]

The dual field is therefore a scale-dependent smoothing of the residual primitive. This is an exact interpretation, but it concerns a residual field rather than the target derivative directly.

### 10.2 Continuum total-variation regularization

Allow a finite signed parameter measure ρ and model

\[
f_\rho(x)=b+\int\phi_\theta(x)\,d\rho(\theta),
\qquad
\min_{b,\rho}\tfrac12\|f-f_\rho\|_\mu^2+
\lambda\int w(\theta)\,d|\rho|(\theta),
\]

where w>0. Suppose a primal optimum and a dual certificate exist. A compact parameter domain and continuous bounded features provide routine finite-data existence conditions; unrestricted-center statements below are conditional on attainment. Introducing a function q for the fitted-output constraint yields the dual

\[
\max_q\;\langle f,q\rangle_\mu-\tfrac12\|q\|_\mu^2,
\quad\langle q,1\rangle_\mu=0,
\quad |\langle q,\phi_\theta\rangle_\mu|\le\lambda w(\theta).
\tag{12}
\]

To derive the inequality, the coefficient-measure terms in the Lagrangian are λ∫w d|ρ|−∫⟨q,φ_θ⟩dρ. Their infimum is zero exactly when the displayed pointwise bound holds; otherwise arbitrary signed mass drives the infimum to −∞. Minimizing the output-error term produces the quadratic dual objective, and the free bias forces mean-zero q. At optimum q=r.

Complementarity says that for |ρ|-almost every active θ,

\[
\frac{\langle r,\phi_\theta\rangle_\mu}{\lambda w(\theta)}
=\frac{d\rho}{d|\rho|}(\theta)\in\{-1,+1\}.
\]

The support therefore sits where the normalized dual field saturates its allowed magnitude. At an interior smooth support point its geometry gradient vanishes. This is a genuine support-geometry theorem for this specified variational model. Variation-norm formulations and representer theorems are developed, under their activation and function-space assumptions, in [Bach (2017)](https://jmlr.org/papers/v18/14-546.html) and [Parhi and Nowak (2021)](https://jmlr.org/papers/v22/20-583.html). The tanh dictionary dual above follows directly from the displayed calculation; it is not obtained by assuming tanh is positively homogeneous.

### 10.3 A counterprediction: unweighted variation norm removes broad widths

Take w=1, all real centers, and widths 0<γ≤Γ. Under the preceding primal-dual assumptions with λ>0, the exact transport gives

\[
Q(c,\gamma)=\int\nu_{\gamma\to\Gamma}(u)Q(c+u,\Gamma)\,du.
\tag{13}
\]

The free-bias condition and bounded tanh imply Q(c,Γ)→0 as c→±∞ by dominated convergence. Dual feasibility gives |Q(c,Γ)|≤λ everywhere. For any γ<Γ, ν is strictly positive on the whole real line. Therefore

\[
|Q(c,\gamma)|\le\int\nu(u)|Q(c+u,\Gamma)|du<\lambda.
\]

The inequality is strict because the tails of Q are strictly below λ on sets of positive ν-mass. Such a broad atom cannot saturate the dual bound and hence cannot carry nonzero optimal mass. Every active width must equal the sharpest permitted width Γ.

There is also a primal interpretation: replacing each broad neuron by its positive Γ-width packet leaves its function unchanged and cannot increase total variation. The strict dual argument is stronger: under the stated assumptions, an optimal representation cannot retain nonzero broad mass.

This is a useful falsification, not a proposed model of the observed network. Unweighted continuum readout variation norm with unrestricted centers predicts collapse to the sharpest width, not persistent mixed-width branches. Finite atom budgets, bounded center domains, width-dependent costs, or different optimization selection rules can prevent this collapse. Standard weight decay on every parameter is not automatically this variation norm for tanh. We have no basis for treating an arbitrary saved Adam or VarPro solution as a minimizer of it.

At zero training error, a separate minimum-norm interpolation problem can still have nonzero Lagrange multipliers as a certificate; that should not be confused with the zero residual of an unregularized exact fit. A claimed implicit-bias model must specify which problem and which certificate it means.

## 11. What this establishes for the present research question

The strongest results in this note are structural predictions for a controlled but substantial class: fixed distinct centers, common-width tanh features, specified fitting measure, a free bias, and ordinary full-rank least squares.

- Ordered Cauchy geometry makes deletion or forced-height compensation alternate exactly, independently of target oscillation and center uniformity.
- Broadening the family exposes a polynomial hierarchy with explicit singular-value powers and constants. The target's physical-space polynomial components are encoded as corresponding center-polynomial patterns; the dominant flat-limit term has an explicit global derivative interpretation and barycentric readouts.
- Sharpening the family produces a local Brownian-bridge inverse and exact adjacent-cell-average coefficients. Their leading local derivative rule is a consequence, with a calculable nonuniform-spacing correction.
- Between widths, a positive continuum transport gives an exact geometry-only redistribution rule. Its inverse exposes why sharpening information can demand large signed compensation.

These are mechanisms and formulas, not a universal explanation of a mixed-width learned network. Mixed widths, finite atom restrictions, and cutoff-dependent fitting can change the relevant geometry. A defensible next step is to treat common-width families as analytically understood components and derive their interactions or perturbations with explicit error bounds. The variational support route supplies necessary conditions only after a selection rule is justified; its simplest unweighted form demonstrably predicts the wrong width diversity.

Independent audit: a second agent checked the Cauchy/Schur proof, deletion signs, determinant constants, broad-limit exponents and exact constants, left/right polynomial flags, sharp-step inverse, interval-average expansion, mixed-width/ridge counterexamples, rational-basis and residue formulas, optimal-LS flat limit, Rodrigues prefactor, positive transport constants and proofs, finite-network stationarity, TV dual, and strict width-collapse conclusion. No substantive mathematical correction was required. The common-scale uniqueness and reverse finite-measure impossibility in Section 9.3 follow directly from the stated nonvanishing Fourier transform and boundedness of finite-measure transforms.

## 12. Follow-up checks: actual heterogeneous neurons and controlled edits

### 12.1 Exact finite tanh identifiability is stronger than practical stability

Distinct real tanh atoms with positive γ are linearly independent together with a constant. To prove this, suppose a finite linear combination vanishes on an open real interval. Real analyticity and then meromorphic continuation make it vanish wherever defined in the complex plane. Select the largest γ appearing with nonzero coefficient. The pole c+iπ/(2γ) of each such atom cannot coincide with a pole of a smaller-γ atom, whose nearest poles lie farther from the real axis. Other atoms with the same γ have different centers and hence different pole real parts. Its residue v/γ must therefore vanish. Remove the maximal-γ group and repeat. Finally the constant also vanishes.

Consequently two finite real-tanh networks representing the same function on an open interval have the same distinct atoms and readouts, up to permutation and merging identical atoms. There is no exact distinct-atom finite counterexample. Common-scale transport supplies nonunique allocation in a larger class that includes continuum center measures; it does not erase this finite-network identifiability theorem.

Finite training samples are different. For every γ>0,

\[
f_\gamma(x)=\frac{\tanh(\gamma x)}{\tanh\gamma}
\]

matches the two samples f(−1)=−1 and f(1)=1, although its single neuron's geometry and readout vary. On a fixed interval f_γ→x with O(γ²) error as γ→0, while its readout diverges like 1/γ. Thus even smooth, simple functions can have dramatically different nearly equivalent encodings.

Center collision gives an equally explicit cancellation mechanism. At fixed γ,

\[
g_\epsilon(x)=\frac{
\tanh(\gamma(x+\epsilon/2))-
\tanh(\gamma(x-\epsilon/2))}{\epsilon}
\longrightarrow\gamma\operatorname{sech}^2(\gamma x)
\]

with O(ε²) uniform error on bounded intervals, while its two distinct readouts are ±1/ε. A readout bound alone does not guarantee stable allocation either: replacing 1/ε by a fixed B makes the function tend to zero while both coefficient magnitudes remain B. Reversing their signs changes the coefficient vector by a fixed amount but changes the function by only O(Bγε).

A conditional stability theorem is available. Fix N and a compact parameter class with bounded bias and readouts, γ bounded above and away from zero, bounded centers, a positive separation between every distinct pair (c_j,γ_j), and |v_j| bounded away from zero. Measure function differences in L² of a positive-density interval measure. Modulo neuron permutation, the map from parameters to functions is injective and has a Lipschitz inverse on this compact class, with a class-dependent constant.

Here is the additional argument beyond compactness. At a maximal-γ atom, the double-pole coefficient in a real parameter variation is

\[
\frac{v_j}{\gamma_j}
\left(\delta c_j-\frac{i\pi}{2\gamma_j^2}\delta\gamma_j\right).
\]

If the first-order function variation vanishes, this complex coefficient forces both real variations δc_j and δγ_j to zero. The remaining simple pole forces δv_j=0. Recursing over γ groups and then the bias proves injectivity of the parameter derivative. Its smallest gain has a positive minimum on the compact regular class; a uniformly bounded second derivative gives a uniform local lower-Lipschitz estimate by Taylor expansion. For parameter pairs outside that local neighborhood, compactness and exact identifiability give a positive minimum separation of their images. Combining the two bounds proves the claim.

This is an existence theorem for a potentially extremely poor stability constant, not a promise of numerically useful recovery. Collision, vanishing readouts, γ→0, unbounded saturation geometries, or unbounded signed coefficients remove its assumptions. Bounded-readout regularization prevents some blowups but does not by itself enforce separation or nonvanishing active atoms. Finite sample measures need a separate sampling-identifiability condition.

Identifiability alone also gives no universal smooth branch law. On any fixed independent geometry, an arbitrary prescribed readout vector becomes the unique exact least-squares readout if its synthesized function is chosen as the target. A useful coefficient-profile prediction must therefore specify target structure, additional selection assumptions, or a controlled geometric regime such as the flat and sharp limits proved above.

### 12.2 Several simultaneous center edits in the common-width flat limit

For two fixed ordered center configurations with the same N, target and fitting measure, let γ,γ′ both tend to zero. Assume the degree-N polynomial projection coefficient a_N is nonzero. Equation (6) gives directly

\[
\boxed{\frac{v'_j}{v_j}\sim
\left(\frac\gamma{\gamma'}\right)^{2N-1}
\prod_{\ell\ne j}\frac{c_j-c_\ell}{c'_j-c'_\ell}.}
\]

The common target scalar a_N cancels: no new target solve is needed. The asymptotic is uniform for centers in compact sets with a fixed positive minimum gap. In differential form the leading logarithmic sensitivity is

\[
\boxed{d\log|v_j|=-(2N-1)d\log\gamma
-\sum_{\ell\ne j}\frac{dc_j-dc_\ell}{c_j-c_\ell}.}
\]

The differential describes the leading asymptotic term; it is not an exact finite-γ identity. Analytic stable-basis expansions give O(γ) errors in center derivatives and O(1) in the γ derivative of log|v_j| on such regular compact configurations. These formulas do not cover changing N, crossing a collision, a zero a_N, cutoff/ridge fitting, or independent heterogeneous-width edits.

### 12.3 Strongly overlapping tents can have exact periodic null modes

Let h>0, m≥2 an integer, and

\[
T_j(x)=\max\!\left(1-\frac{|x-jh|}{mh},0\right).
\]

For each residue class r modulo m, centers (r+km)h have spacing mh, equal to the tent support radius. Their tents form the ordinary hat partition of unity:

\[
\sum_{j\equiv r\pmod m}T_j(x)=1.
\]

Hence all coefficients 1/m represent the constant 1. Any periodic coefficient sequence a_j=α_{j mod m} with ∑_rα_r=0 lies in the exact nullspace. For m=2, deleting an even-indexed tent is compatible with the exact alternative coefficients v_even=0 and v_odd=1. Relative to the original coefficients 1/2, the change is −(−1)^j/2. The target is constant; the coefficient alternation is entirely geometric.

To make this a finite-vector example, choose L divisible by m and periodize the tents on a circle of length Lh, with j=0,…,L−1. The same partition identities hold. The nullspace has exactly m−1 dimensions. Indeed, the Fourier transform of an unperiodized tent is mh·sinc²(ωmh/2), where sinc z=sin z/z. The coefficient DFT mode indexed by k contributes at frequencies congruent to k modulo L. All those tent Fourier factors vanish exactly for k=qL/m, q=1,…,m−1; for every other k, the principal frequency already has a nonzero factor. Thus these residue-class directions account for the entire nullspace.

This example is different from radius-h tents, whose ordinary infinite-grid Gram symbol is h(2/3+cosθ/3), bounded below by h/3. More overlap can create exact missing coefficient frequencies, not merely worsen an already positive condition number. Open finite boundaries modify the identities, although sufficient halo can preserve the exact partitions on a fitted interior interval. The finite periodic statement avoids issues with non-square-summable constant sequences.
