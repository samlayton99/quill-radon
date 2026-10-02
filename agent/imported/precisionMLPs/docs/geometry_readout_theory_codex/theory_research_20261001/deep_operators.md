# Fixed deep geometry as restricted kernels and differential sources

Theory notes, October 1, 2026. No training experiments. The statements below separate exact identities, useful structured theorems, and proposed models. The point is to extend the QI derivative interpretation without pretending that every deep representation is a scalar coordinate warp.

## 1. The object a fixed hidden representation actually determines

Let input X have probability measure μ and let a fixed hidden representation be \(h:X\to\mathbb R^d\). The output layer approximates the target by \(F(h(x))\), where F belongs to an ambient ridge dictionary or another specified function family. The hidden geometry determines three things, not only a metric:

1. The image \(h(X)\), including folds and identifications of different inputs.
2. The pushed-forward measure \(\nu=h_\#\mu\), which specifies where approximation error matters.
3. The ambient features restricted to that image, such as \(\sigma(w\cdot h(x)+b)\).

Pullback \(P_h:F\mapsto F\circ h\) is an isometry from \(L^2(\nu)\) into \(L^2(\mu)\), since the defining property of pushforward gives

\[
\int_X|F(h(x))|^2d\mu(x)=\int_{h(X)}|F(z)|^2d\nu(z).
\]

Let \(\bar f(z)=\mathbb E[f(X)\mid h(X)=z]\). Orthogonality of conditional expectation gives the exact risk decomposition

\[
\boxed{\|f-F\circ h\|_{L^2(\mu)}^2
=\mathbb E[\operatorname{Var}(f(X)\mid h(X))]
+\|\bar f-F\|_{L^2(\nu)}^2.}
\tag{1}
\]

The cross term vanishes because \(f-\bar f\circ h\) has zero conditional mean. This separates irreversible loss of target information from approximation within the representation. Conditioning of the chosen readout coordinates is a third, separate issue.

For example, on symmetric \([-1,1]\), \(h(x)=x^2\) preserves even targets but cannot represent \(f(x)=x\): its conditional mean is zero and every output function of h has squared error at least \(\mathbb E[X^2]=1/3\). No change of output activation or readout solver repairs that loss. A fold helps only for targets whose values are compatible on the folded fibers.

For a scalar smooth map with finitely many regular inverse branches, conditional averaging has the explicit density weights

\[
\bar f(t)=
\frac{\sum_{x:h(x)=t}f(x)\rho(x)/|h'(x)|}
{\sum_{x:h(x)=t}\rho(x)/|h'(x)|},
\]

where \(d\mu=\rho(x)dx\). Thus a hidden map changes both coordinates and fitting weights. For an injective manifold immersion, the corresponding density relative to induced manifold volume includes the Jacobian \(\sqrt{\det(Dh^TDh)}\). Geometry without its observation measure is incomplete.

An ambient function norm induces a restriction norm

\[
\|f\|_{h}=\inf_{F:F\circ h=f}\|F\|_{\rm ambient}.
\tag{2}
\]

This is a quotient over extensions away from the hidden image. Different ambient ridge or Radon representations can agree exactly on the observed manifold. Recovering an ambient Radon transform from data on that manifold requires an extension convention; it is not intrinsic to the observed target alone.

## 2. Scalar monotone charts: an exact Green-operator result

Let \(g:I\to J\) be an increasing \(C^r\) diffeomorphism with \(g'>0\). Define the derivative in latent coordinates by

\[
D_g=\frac1{g'(x)}\frac d{dx}.
\]

If \(F(t)=f(g^{-1}(t))\), repeated differentiation gives \(D_g^rf(x)=F^{(r)}(g(x))\). That chain-rule identity is only the starting point. Define the measure-valued source operator

\[
\boxed{L_g=g'D_g^r=\frac d{dx}\left(\frac1{g'}\frac d{dx}\right)^{r-1}.}
\tag{3}
\]

For \(x_j=g^{-1}(t_j)\), consider a pulled-back truncated power

\[
G_{g,j}(x)=\frac{(g(x)-t_j)_+^{r-1}}{(r-1)!}.
\]

Its rth latent derivative is \(\delta(g(x)-t_j)\). Since \(g'>0\), multiplying by g′ converts this into \(\delta(x-x_j)\). Therefore

\[
\boxed{L_gG_{g,j}=\delta_{x_j},\qquad
\ker L_g=\operatorname{span}\{1,g,\ldots,g^{r-1}\}.}
\tag{4}
\]

The nullspace statement follows by transforming \(L_gf=0\) into \(F^{(r)}=0\). Thus neurons can be Green functions of a geometry-dependent differential operator, with readouts equal to source strengths. For r=2,

\[
L_gf=\frac d{dx}\left(\frac{f'}{g'}\right).
\]

For r=1, however, \(L_g=d/dx\) is independent of g. A monotone warp of an ideal Heaviside is just a Heaviside at the inverse-mapped knot. In the tanh/QI case the warp changes smoothing, spacing, and latent derivative density, not the underlying first-derivative measure df. This distinction is necessary.

For a smooth activation with \(K=\sigma^{(r)}\),

\[
L_g\sigma(\gamma(g(x)-t_j))
=g'(x)\gamma^rK(\gamma(g(x)-t_j)).
\tag{5}
\]

This is a transported bump instead of an ideal point source. With \(M_K=\int K\ne0\), uniform latent spacing h, and the same localization/low-alias assumptions as ordinary QI,

\[
v_j\approx\frac{h}{M_K\gamma^{r-1}}D_g^rf(x_j).
\tag{6}
\]

For tanh this is \(h f'(x_j)/(2g'(x_j))\). Local physical spacing is approximately \(h/g'(x_j)\), and local physical slope parameter approximately \(\gamma g'(x_j)\); their product remains \(\gamma h\). This supplies a coordinate-invariant meaning to the local overlap parameter in the slowly varying regime.

Function-L2 fitting in x becomes fitting in t with weight \(1/g'(g^{-1}(t))\), or \(\rho(g^{-1}(t))/g'(g^{-1}(t))\) for an input density ρ. The unweighted latent inverse filter is not automatically the physical least-squares solution.

### Exact coefficient prediction for warped ReLU interpolation

Given knots \(x_0<\cdots<x_N\), let \(t_j=g(x_j)\), \(y_j=f(x_j)\), and

\[
s_j=\frac{y_{j+1}-y_j}{t_{j+1}-t_j}.
\]

The interpolant linear in g on each interval is exactly

\[
f_g(x)=y_0+s_0(g(x)-t_0)
+\sum_{j=1}^{N-1}(s_j-s_{j-1})(g(x)-t_j)_+.
\tag{7}
\]

Hence the readout at a knot is the jump in latent secant slope, and \(L_gf_g=\sum_j(s_j-s_{j-1})\delta_{x_j}\). This predicts all coefficients from target values and the chart without a least-squares solve. It is interpolation, not an assertion of least-squares optimality.

If \(\|F''\|_\infty\le B\) and the largest latent interval is h, the interpolation remainder gives

\[
\|f-f_g\|_\infty\le Bh^2/8.
\tag{8}
\]

The physical geometry is good for this target when its latent curvature is small, its chart can be represented inexpensively, and the chart/encoder remains stable. A low outer error alone does not establish a cheap overall representation.

## 3. Variational energies and knot-placement predictions

The scalar chart supports precise intrinsic penalties:

\[
R_g(f)=\|L_gf\|_{\mathcal M(I)}=\|F^{(r)}\|_{\mathcal M(J)},
\qquad
E_g(f)=\int_I|D_g^rf|^2g'dx=\int_J|F^{(r)}|^2dt.
\tag{9}
\]

The measure equality uses the one-to-one pushforward under g. For a finite Green expansion with distinct knots, \(R_g(f)=\sum_j|v_j|\). At a fixed g, a data-fit objective plus \(\lambda R_g(f)\) is exactly a latent-coordinate generalized-TV spline problem. Its sparse representer results transfer by changing variables. This supplies a principled adaptive-knot model, but does not establish that unregularized Adam implicitly minimizes this functional.

The relevant established foundations are [Unser, Fageot, and Ward's generalized-TV representer theorem](https://arxiv.org/abs/1603.01427), [Unser's activation-spline theorem](https://www.jmlr.org/papers/v20/18-418.html), and [Parhi and Nowak's ridge-spline theorem](https://jmlr.org/papers/v22/20-583.html). These identify spline solutions under specified variational objectives; they are not universal descriptions of every fitted network.

### A direct adaptive-density calculation

For ordinary piecewise-linear interpolation, assume f is smooth and knot intervals are small. On an interval of length Δ, approximately constant curvature gives

\[
\int_{\rm cell}|f-I f|^2dx
\sim\frac{f''(x)^2\Delta^5}{120}.
\]

Indeed, the leading error is \(f''(x)t(t-\Delta)/2\), and \(\int_0^\Delta t^2(t-\Delta)^2dt=\Delta^5/30\). If ρ(x) is the local number of intervals per unit length, \(\Delta\approx1/\rho\), so total squared error is asymptotically

\[
\frac1{120}\int |f''(x)|^2\rho(x)^{-4}dx,
\qquad\int\rho=N.
\]

A Lagrange multiplier gives

\[
\boxed{\rho_*(x)\propto|f''(x)|^{2/5},\qquad
E_{m interp}^2\sim\frac{\left(\int |f''|^{2/5}\right)^5}{120N^4}.}
\tag{10}
\]

With weighted physical error, replace \(|f''|^2\) by the weighted quantity. Under uniform error rather than L2, equidistributing the leading local error gives \(\rho\propto\sqrt{|f''|}\). These are objective-dependent asymptotic laws for interpolation, not universal empirical laws of trained center density. Smooth-kernel overlap and least-squares coupling can change finite-width behavior.

### Why optimizing the chart needs a real cost

A monotone target can be made linear in latent coordinates by choosing g proportional to f. That puts the whole problem into the chart; it does not explain efficient representation. In addition, rescaling g changes derivative penalties. Endpoint normalization and an inner-map complexity cost are essential.

A specific well-posed surrogate illustrates what can be asked. On [0,1], write \(g'=e^q\), impose \(m\le e^q\le M\), \(\int_0^1e^qdx=1\), and \(\eta>0\). For \(f\in H^2\), minimize

\[
\mathcal J(q)=\int_0^1e^{-3q}(f''-q'f')^2dx
+\eta\int_0^1|q'|^2dx.
\tag{11}
\]

The first term is exactly \(E_g(f)\) for r=2; the second charges for rapid changes of chart scale. On this constrained class the direct method yields a minimizer: a minimizing sequence is bounded in H1, converges uniformly along a subsequence in one dimension, and its derivatives converge weakly. The constraints pass to the limit, while the quadratic integrand is lower semicontinuous in the weak derivative after its coefficients converge. Feasibility requires \(m\le1\le M\).

Away from active bounds, let \(u=f''-q'f'\). The formal Euler–Lagrange equation is

\[
-3e^{-3q}u^2+\lambda e^q
+2\frac d{dx}(e^{-3q}f'u)-2\eta q''=0.
\tag{12}
\]

This is a concrete operator-design problem, not a claim that this surrogate is the correct network objective. The chart penalty is not identical to parameter count. A neural representation theorem would also need to bound the cost of approximating the minimizing chart.

Monotone charts cannot reduce the number or order of target extrema: \(f'=F'(g)g'\) with \(g'>0\). They redistribute resolution but cannot erase oscillation count. Nonmonotone charts permit reuse through folds, subject to the information-loss constraint in (1).

## 4. Curved hidden embeddings produce more than a scalar chart

Let \(h(s)\in\mathbb R^d\) be a regular hidden curve parametrized by arclength. Write \(\tau=h'\), \(\kappa=h''\), and the preactivation \(a(s)=w\cdot h(s)+b\). A smooth ridge satisfies

\[
\frac{d^2}{ds^2}\sigma(a(s))
=\sigma''(a)(w\cdot\tau)^2+\sigma'(a)w\cdot\kappa.
\tag{13}
\]

For ReLU and finitely many simple crossings \(a(s_i)=0\), this becomes the distribution identity

\[
\boxed{\frac{d^2}{ds^2}(a(s))_+
=\sum_i|a'(s_i)|\delta_{s_i}+H(a(s))a''(s).}
\tag{14}
\]

The first term follows from \(\delta(a)(a')^2=\sum_i|a'(s_i)|\delta_{s_i}\). The second is distributed curvature on active arcs. A single neuron can generate many effective centers, whose amplitudes are tied by the common w and output weight. A curve tangency can eliminate a kink: if \(a(s)=(s-s_0)^2\), ReLU(a)=a is smooth even though a vanishes at s0.

There is generally no common second-order scalar correction that removes this curvature term. If \(D_s^2-q(s)D_s\) removes it for every ridge direction w, then

\[
w\cdot(h''-qh')=0\quad\text{for every such }w.
\]

If these directions span the ambient affine hull, \(h''=qh'\). Under arclength parametrization, \(h''\perp h'\), so q=0 and h is straight. In a general parametrization the condition still says that the image lies along a line. Thus a curved embedding cannot be reduced to a shared scalar warped-ReLU operator merely by reparametrizing its curve.

Including a zeroth-order correction does not repair this statement for the full family with arbitrary biases. An operator \(D^2+pD+r\) annihilating every affine preactivation must annihilate the constant function, forcing r=0. Restricting the allowed biases can change that conclusion; the bias-free circle example below is a concrete exception with a correspondingly restricted function space.

### A quantitative local QI regime

Near a transverse crossing, Taylor expansion gives

\[
|a(s)-a'(s_0)(s-s_0)|\le\tfrac12B|s-s_0|^2,
\quad B=\sup|a''|.
\]

Over one activation width \(|s-s_0|\sim1/|a'(s_0)|\), the dimensionless departure from an affine preactivation is controlled by

\[
\eta=\frac{B}{|a'(s_0)|^2}.
\tag{15}
\]

For a Lipschitz activation, the feature-value discrepancy is bounded by the preceding Taylor error times its Lipschitz constant. Bounds for the differentiated kernel additionally use bounded first and second derivatives of the activation. If η is small, a local translated kernel plus a controlled correction is justified. If the tangent projection vanishes, no such estimate is available.

This directly connects to the tested finite-defect theory: choose affine/tangent reference features, project each departure into an in-span transport B and an orthogonal leakage W, and use the same solved-coefficient update. The curvature estimate predicts when W should be small before a complete new fit. It does not imply that a collection of many crossings has independent coefficients: their source strengths remain tied by the parent neurons.

## 5. A higher-order geometry operator can still exist

Suppose the independent functions \(u_0=1,u_1=h_1,\ldots,u_{m-1}\) are sufficiently smooth and have nonvanishing Wronskian on an interval. Define the monic differential operator

\[
\boxed{L_hf=\frac{W(u_0,\ldots,u_{m-1},f)}{W(u_0,\ldots,u_{m-1})}.}
\tag{16}
\]

It has order m and annihilates every affine preactivation \(w\cdot h+b\). This follows because its leading term is \(f^{(m)}\), and a duplicate determinant column vanishes. For \(\phi=(w\cdot h+b)_+\), L_hφ therefore vanishes away from the crossings. At each simple crossing it is a combination of δ and its derivatives, determined by the preactivation's derivative jumps. For q≥1,

\[
[\phi^{(q)}]_{s_i}=\operatorname{sign}(a'(s_i))a^{(q)}(s_i),
\qquad [\phi]_{s_i}=0.
\]

Consequently the highest possible source derivative is \(\delta^{(m-2)}\). The output readout scales a constrained collection of crossing sources and derivative sources. This is a genuine operator selected by hidden geometry, but its usefulness requires that its order and coefficients be manageable. An order comparable to a large hidden width with an ill-conditioned Wronskian is another exact reformulation, not a compact explanatory model. Zeros of the Wronskian require local intervals or a different basis, and no global nonsingular operator has been established in general.

### Explicit curved example: a hidden circle

For \(h(s)=(\cos s,\sin s)\), the span \(\{1,\cos s,\sin s\}\) is annihilated by

\[
L_h=D^3+D.
\]

Let \(a=w_1\cos s+w_2\sin s+b\), with \(|b|<\|w\|\). At a zero of a, \(a''=b\) and \(|a'|=\sqrt{\|w\|^2-b^2}\). Thus

\[
\boxed{(D^3+D)(a)_+
=\sum_i\left[\operatorname{sign}(a'(s_i))b\,\delta_{s_i}
+|a'(s_i)|\delta'_{s_i}\right].}
\tag{17}
\]

The two crossings on a period carry tied source strengths. This cannot be described as independent ordinary curvature knots, yet it has a precise Green-source representation. The cost of constructing the circle embedding itself has not been set to zero; this is a fixed-geometry example, not a network-size advantage claim.

If every ridge bias is instead fixed to zero, the affine-preactivation space shrinks to span{cos s,sin s}. The simpler operator \(D^2+1\) then satisfies

\[
(D^2+1)(w\cdot h)_+
=\sum_i|w\cdot h'(s_i)|\delta_{s_i}.
\]

Each neuron produces two equal antipodal source masses. The simplified source form has a real representational restriction. Since \((a)_+-(-a)_+=a\), every such single-hidden-layer readout obeys

\[
f(s)-f(s+\pi)=\sum_jv_j w_j\cdot h(s).
\]

Its odd antipodal component contains only the first sine/cosine harmonics. It cannot represent any higher odd harmonic, including cos(3s), regardless of width. Under uniform circle L², all its available features are orthogonal to cos(3s), so the best relative error for that target is exactly one. The output bias does not repair this obstruction.

The same identity \(\sigma(a)-\sigma(-a)=a\) holds for standard GELU, softplus, and SiLU, so the same fixed-circle, bias-free, single-hidden-layer obstruction applies to them. Bias-free tanh has a different restriction: every feature is antipodally odd, leaving no nonconstant even harmonics. These are activation-and-architecture statements. Adding biases or changing the hidden representation can change the allowed source ties and remove the obstruction. In particular, they must not be extended without proof to deeper networks whose intermediate maps have changed the geometry.

For a polynomial embedding \((x,x^2,\ldots,x^d)\), \(L_h=D^{d+1}\). ReLU of a polynomial preactivation is piecewise polynomial of degree d with at most d simple crossing knots, and constrained source jets at those knots.

There is an important variational limitation: δ′ and its higher derivatives are not finite Radon measures. Therefore \(\|L_hf\|_{\mathcal M}\) is generally infinite for these curved ReLU examples. Ordinary scalar generalized-TV representer theorems do not apply unchanged. One needs a vector of source measures whose derivatives form L_hf, or an atomic norm retaining the allowed crossing/jet ties. The scalar monotone-chart result in Section 2 is the clean measure-valued case.

## 6. Depth as a factorization of effective source geometry

A one-dimensional finite ReLU network of any depth is a continuous piecewise-linear function. Its ordinary second derivative is always a finite signed sum of point masses, regardless of depth. Both shallow and deep finite networks belong to this broad function class, but depth changes the subset attainable under a fixed parameter budget by changing how economically a large knot pattern and its tied coefficients can be generated.

For a fixed last hidden layer \(h_1,\ldots,h_d\), let \(x_i\) be the union of its breakpoints. Define

\[
C_{ij}=[h_j']_{x_i}.
\]

Then a final readout \(f=\sum_jv_jh_j+b\) has

\[
f''=\sum_i(Cv)_i\delta_{x_i}.
\tag{18}
\]

This is an exact geometry-only derivative decoder. The matrix C includes inherited inner-layer knots and new crossings. A deep architecture can factor a very large effective shallow source pattern into a small compositional computation. An arbitrary target curvature pattern need not lie in the resulting source range; the architecture gains efficiency through structure and reuse, not by freely specifying every effective source.

### Exact oscillation example with all inner cost counted

On [0,1], define the tent map

\[
T(x)=2(x)_+-4(x-1/2)_++2(x-1)_+.
\]

Its k-fold composition has \(2^k\) affine pieces and is represented by O(k) ReLU units with O(k) depth. The entire hidden computation is counted. A single-hidden-layer ReLU network with n units has at most n+1 affine pieces along this interval. To approximate \(T^{\circ k}\) uniformly with error less than 1/2, its values must alternate above and below 1/2 at the alternating peak/trough grid points. It therefore requires at least \(2^k\) threshold crossings and hence \(n+1\ge2^k\).

This is an explicit exponential source-pattern compression by depth. It is not evidence that such oscillatory functions are robustly recoverable from a few samples: their Lipschitz constants and sampling demands also grow. [Telgarsky's depth-separation work](https://proceedings.mlr.press/v49/telgarsky16.html) develops substantially broader separation results using this type of compositional oscillation.

### A precision advantage for multiplication and product structure

The dyadic piecewise-linear interpolant of \(x^2\) is

\[
S_m(x)=x-\sum_{k=1}^m4^{-k}T^{\circ k}(x),\qquad
\|S_m-x^2\|_{\infty,[0,1]}\le4^{-m}/4.
\tag{19}
\]

At each refinement, the new tent correction supplies the midpoint discrepancy; induction proves that S_m is the interpolant on the mesh \(2^{-m}\). The interpolation error bound for a quadratic gives the stated constant. A constant-width network carries the current tent iterate and the accumulated sum, so all O(m) hidden units are counted.

Use

\[
xy=2\left[\left(\frac{x+y}{2}\right)^2
-\left(\frac x2\right)^2-\left(\frac y2\right)^2\right]
\]

to obtain a multiplication module on \([0,1]^2\) with error at most \(6\cdot4^{-m}/4\), using O(m) units and depth. Hence its cost is \(O(\log(1/\epsilon))\). Clip its output to [0,1], which cannot increase its error relative to the true product.

A binary tree of these clipped modules approximates \(\prod_{j=1}^d x_j\). At one internal node, the inherited errors add because \(|uv-\tilde u\tilde v|\le|u-\tilde u|+|v-\tilde v|\) on [0,1]. Using module error \(\delta=\epsilon/(d-1)\) gives final error at most ε, with

\[
\boxed{\text{units}=O(d\log(d/\epsilon)),\qquad
\text{depth}=O(\log d\,\log(d/\epsilon)).}
\tag{20}
\]

By contrast, any single-hidden-layer ReLU approximation of xy restricts on x=y=t to an n-knot piecewise-affine approximation of \(t^2\). One affine interval has length at least \(1/(n+1)\). The best affine uniform approximation to a quadratic on length Δ has error \(\Delta^2/8\), so

\[
\boxed{n\ge(8\epsilon)^{-1/2}-1.}
\tag{21}
\]

This is a precision advantage with no uncharged inner map. It does not prove an exponential-in-d lower bound for shallow product approximation. [Yarotsky's approximation paper](https://arxiv.org/abs/1610.01145) supplies the established square/multiplication construction and much broader ReLU approximation bounds; the elementary lower bound here is stated only for a single hidden ReLU layer.

The activation restriction matters. A smooth activation with \(\sigma''(b)\ne0\) can approximate \(t^2\) using a fixed-size centered second-difference stencil around b as its scale tends to zero. The readouts then grow like the inverse square scale. Such a shallow smooth network defeats the ReLU neuron-count lower bound if unlimited coefficient magnitude and numerical precision are free. This is the same cancellation phenomenon seen in the clustered tanh experiments. Representation quality must therefore count conditioning, readout norm, or precision in addition to neuron count and approximation error.

### General composition accounting

If \(f=F\circ h\), h is approximated within δ, F is Lipschitz with constant L on the relevant neighborhood, and the outer approximation error is ε, then

\[
\|f-\widehat F\circ\widehat h\|_\infty\le L\delta+\epsilon.
\]

For several layers, telescoping gives the sum of each layer's error multiplied by subsequent Lipschitz constants. The complexity is the sum of all layer approximation costs, with accuracies chosen to meet this propagated error budget. A favorable compositional theory requires small-dimensional, reusable constituent maps; an arbitrary hidden choice h=f proves nothing about economical representation.

## 7. Higher-dimensional manifold restriction and the Radon connection

For an embedded manifold M and a restricted affine function \(a(z)=w\cdot z+b\), the intrinsic Laplacian obeys

\[
\Delta_M\sigma(a)=\sigma''(a)|\nabla_Ma|^2+\sigma'(a)\Delta_Ma.
\tag{22}
\]

The second term contains the embedding's mean-curvature vector through \(\Delta_Ma=w\cdot\Delta_Mh\). Thus the one-dimensional curvature correction is not an accident. Hyperplane/ridge sources restricted to curved data manifolds acquire intrinsic geometry terms, and tangencies can change the shape or topology of their intersections.

Ambient ridge-spline theory uses directional offset derivatives in Radon coordinates rather than a single coordinate derivative. [Parhi and Nowak's deep variational framework](https://arxiv.org/abs/2105.03361) builds compositions of such spaces under explicit norm penalties. It establishes representer statements for specified regularized objectives, including structural features of their realizations. That is a relevant rigorous precedent for compositional source geometry, not a theorem that every unregularized deep solution is optimal under those norms.

Approximation on a known low-dimensional manifold also has constructive precedents, for example [Shaham, Cloninger, and Coifman](https://arxiv.org/abs/1509.07385). Their constructions count geometric localization and approximation components. This supports requiring chart cost and curvature dependence explicitly; intrinsic dimension alone is not a complete complexity measure.

## 8. What would make this a useful unified theory rather than an identity

The strongest common statement is that a fixed hidden representation transforms a target approximation problem into a measured restricted-dictionary problem. In favorable regimes that dictionary has an explicit source operator:

- Uniform QI gives translated, smoothed derivative sources with a common inverse filter.
- Monotone scalar charts give transported sources, explicit latent spline coefficients, and a specified variational norm.
- Curved hidden embeddings give multiple crossings, tied source strengths, and curvature corrections; structured embeddings can admit a low-order annihilating operator.
- Deep ReLU networks factor large effective curvature-source patterns through repeated compositions.

Several concrete tests follow without a full fit: check whether the target is compatible on hidden fibers; measure transverse slopes and curvature ratios; identify a small stable operator annihilating the affine hidden span; inspect the effective source-range constraints; and account for chart cost and coefficient amplification. The existing transport/leakage framework quantifies departures from an explicitly tractable source dictionary.

There are also precise obstructions. A scalar chart cannot flatten every curved ridge family, invert a lossy hidden map, reduce oscillation count while remaining monotone, or supply an inexpensive representation merely because h=f exists. A large ill-conditioned Wronskian operator is not explanatory. A sparse source pattern under a scalar measure norm need not describe curved ReLU jet sources. An activation without the derivative-kernel assumptions may still work through another basis, such as Fourier features. None of these fixed-solution statements proves that a particular training algorithm discovers the favorable representation.

The plausible research program is to classify **economical and stable source factorizations** of targets, then characterize which architectures and activations realize them. This retains the QI observation as a transparent case while allowing depth, curvature, folds, and conditioning to have distinct mathematical roles.

## 9. Feature geometry and information geometry: an exact bridge with explicit choices

There is a precise common object connecting a feature representation, a readout cost, a kernel, and statistical distinguishability. None of these is determined by the feature image alone. Specify a feature map \(\phi:X\to\mathbb R^m\), an input probability measure μ, and a positive-definite readout metric M. The represented function and readout cost are

\[
f_v(x)=\phi(x)^Tv,\qquad \|v\|_M^2=v^TMv.
\]

Write \(J_\phi(x)\in\mathbb R^{m\times d}\) for the feature Jacobian. Three associated objects have different roles:

\[
\boxed{
k(x,y)=\phi(x)^TM^{-1}\phi(y),\qquad
G=\int\phi(x)\phi(x)^T\,d\mu(x),\qquad
g(x)=J_\phi(x)^TM^{-1}J_\phi(x).
}
\tag{23}
\]

The kernel records global inner products between whitened features. The Gram matrix G measures function size in coefficient coordinates: \(\|f_v\|_{L^2(\mu)}^2=v^TGv\). The pullback metric g records local input displacement in whitened feature space. It is positive semidefinite, and is a genuine metric only where the whitened feature map has injective derivative. Where derivatives exist,

\[
g(x)=\left.\partial_x\partial_y k(x,y)\right|_{y=x}.
\]

The minimum readout cost for a represented function is \(\inf_{v:f_v=f}v^TMv\). This is the finite-feature reproducing-kernel norm after quotienting feature dependencies. Restricting the kernel to an input subset similarly induces the minimum extension norm, the classical restriction principle in [Aronszajn's original RKHS paper](https://web.mit.edu/9.520/www/spring06/Papers/aron.pdf). It does not require that a particular trained solution minimizes that norm.

### Gaussian observations distinguish two different Fisher matrices

Assume the explicitly specified statistical model

\[
Y\mid x,v\sim\mathcal N(f_v(x),\sigma^2),
\]

with known constant variance. Differentiating the log likelihood with respect to the readout gives \(\nabla_v\log p=(Y-f_v)\phi/\sigma^2\). Averaging its outer product over Y and then x gives the readout-parameter Fisher information

\[
\boxed{I_v=G/\sigma^2.}
\tag{24}
\]

For two readouts v and v+δv, the expected Gaussian KL divergence is exactly

\[
\mathbb E_x\operatorname{KL}(p_v(\cdot\mid x)\|p_{v+\delta v}(\cdot\mid x))
=\frac{\delta v^TG\delta v}{2\sigma^2}.
\]

Therefore the generalized eigenvalues of (G,M), divided by σ², quantify distinguishable output change per unit readout cost. If

\[
B=M^{-1/2}GM^{-1/2},
\]

has a small eigenvalue, the corresponding readout direction has small function effect despite appreciable cost. This is simultaneously a frame-conditioning statement and a statistical-identifiability statement under this observation model.

Differentiating the likelihood with respect to the input instead gives

\[
\boxed{I_x(v)=\sigma^{-2}J_\phi^Tvv^TJ_\phi.}
\tag{25}
\]

For a fixed scalar output this matrix has rank at most one. It is not generally the full feature pullback metric g. To obtain g from Fisher information, an additional ensemble assumption is needed. If readouts have zero mean and covariance \(\mathbb E[vv^T]=M^{-1}\), then

\[
\boxed{\sigma^2\,\mathbb E_v I_x(v)=g(x).}
\tag{26}
\]

Thus g measures the average input distinguishability of that stated family of readouts. A covariance choice is part of the model; it cannot be inferred solely from a single learned scalar function. Independent multiple outputs contribute the sum of their individual Fisher matrices and can have higher rank. These are elementary likelihood calculations; the broader role of a statistical metric in parameter optimization is developed in [Amari's natural-gradient paper](https://doi.org/10.1162/089976698300017746).

### Which coordinate changes preserve this object?

For an invertible feature-coordinate change S, set

\[
\phi'=S\phi,\qquad v'=S^{-T}v,\qquad M'=SMS^T.
\tag{27}
\]

Then \(\phi'^Tv'=\phi^Tv\) and \(v'^TM'v'=v^TMv\). Direct substitution gives k′=k and g′=g, while G′=SGSᵀ and the generalized spectrum of (G,M) is preserved. The readout Fisher transforms as the same quadratic form in the new parameter coordinates.

Keeping M=I after a nonorthogonal feature transformation changes the readout prior or penalty. It usually changes k and g, even if the unregularized function span is unchanged. The appropriate equivalence class therefore contains the dictionary **and its coefficient metric**, not just the coordinates of the feature image. For unconstrained, full-rank ordinary least squares, a basis change preserves the fitted function span; it need not preserve norm-based selection, finite-cutoff solutions, or optimization rates.

### The local metric cannot specify the whole lens

Consider two feature maps on the circle, each with M=I:

\[
\phi_1(x)=(\cos x,\sin x),\qquad
\phi_2(x)=\left(\frac{\sqrt3}{2},\frac12\cos2x,\frac12\sin2x\right).
\]

Both have k(x,x)=1 and g(x)=1. Nevertheless,

\[
k_1(x,y)=\cos(x-y),\qquad
k_2(x,y)=\frac34+\frac14\cos2(x-y).
\]

The second map identifies x with x+π and cannot encode a target distinguishing those points. The first can. The local metric and feature variance agree exactly, while global information retention and function spaces differ. An adequate geometry must retain global relations or fibers, such as the full restricted kernel/dictionary and input measure. Metric volume or intrinsic dimension alone cannot provide the desired theory of generalization.

## 10. Conditional equivalence of activation lenses and fixed-feature convergence

These statements identify a precise regime where changing activation can have only a controlled effect. They do not assert that all activations, arbitrary deep architectures, or optimizers are equivalent.

### A resolved-band comparison

Suppose two uniform translated dictionaries are compared in ordinary Lebesgue L² on the same resolved frequency band, with the same coefficient domain and a specified treatment of bias, DC, and aliases. Their function spectra may have the derivative-source form

\[
W_a(\omega)=\frac{\widehat K_a(\omega)}{(i\omega)^r},\qquad a\in\{1,2\},
\]

on nonzero frequencies. Assume the two synthesis maps obey \(A_2=L_qA_1\) on this band, where L_q is the Fourier multiplier with \(q=W_2/W_1\). If

\[
0<\alpha\le|q(\omega)|\le\beta<\infty,
\]

Parseval gives, for every admissible coefficient vector v,

\[
\alpha^2\|A_1v\|^2\le\|A_2v\|^2\le\beta^2\|A_1v\|^2,
\qquad
\boxed{\alpha^2G_1\preceq G_2\preceq\beta^2G_1.}
\tag{28}
\]

The generalized frame eigenvalues under the same M are correspondingly comparable. If each dictionary spans the whole observed band, both can encode every target in that band, and the spectral ratio explicitly converts their coefficient encodings. This explains a class of activation changes that preserve representation and conditioning to controlled factors.

The hypotheses have content. Spectral zeros break the lower bound. Very small nonzero values make α small and can amplify coefficients. Aliases must either be retained in the comparison or be controlled by a justified cutoff/regularization model. With mixed widths there is generally no single scalar multiplier q acting on every neuron. Even when A₂=L_qA₁ holds, different proper subspaces need not yield the same target projection: L_q may transform the range. The theorem guarantees the displayed norm comparison, not a universal statement about equal approximation floors.

### The associated quadratic optimization theorem

For fixed features, consider the loss \(\mathcal L(v)=\frac12\|Av-f\|^2\) and the specified preconditioned gradient flow

\[
\dot v=-M^{-1}(Gv-A^*f).
\]

Let v_* be any least-squares solution, and define z=M^(1/2)(v−v_*). The normal equations cancel the target term, giving exactly

\[
\dot z=-Bz,\qquad B=M^{-1/2}GM^{-1/2}.
\tag{29}
\]

A component in an eigenvector of B with eigenvalue λ decays as exp(−λt). Positive modes converge and null components stay fixed. Starting at zero selects the minimum-M-norm least-squares solution. The excess squared function error in a mode decays as exp(−2λt). Discrete gradient descent with this same preconditioner converges on the positive modes when \(0<\eta<2/\lambda_{\max}(B)\).

Approximation floor and convergence speed are consequently distinct. A dictionary can reach a relatively high floor quickly because its retained positive eigenvalues are favorable. Another can have a much lower floor but require long times to learn weak directions. Equations (28)–(29) make this distinction quantitative for the fixed-feature quadratic model. They are not an analysis of Adam, moving geometry, nonquadratic loss, or the complete deep-learning problem.

## 11. When the Green-source and kernel descriptions are exactly the same object

The source-operator and kernel descriptions can be joined by an exact construction, rather than by analogy. Fix boundary conditions removing the nullspace of a linear operator L. For an operator statement, take L to be a closed densely defined bijection from its domain in input L² to source L², with bounded inverse T=L⁻¹. The ordinary derivative operators with the initial boundary conditions below satisfy these assumptions. Let w be a positive bounded source weight with bounded reciprocal. Write \(G_L(x,u)\) for the Green kernel, so

\[
f(x)=\int G_L(x,u)\rho(u)\,du,\qquad Lf=\rho.
\]

Give this function the quadratic source norm

\[
\|f\|_L^2=\int w(u)|Lf(u)|^2du.
\]

If Green evaluation vectors lie in the weighted source space, evaluation at x is a bounded linear functional by Cauchy–Schwarz. Its reproducing kernel is therefore

\[
\boxed{k_L(x,y)=\int\frac{G_L(x,u)G_L(y,u)}{w(u)}du.}
\tag{30}
\]

To verify reproduction directly, the source of \(k_L(\cdot,y)\) is \(G_L(y,u)/w(u)\). Hence its inner product with f is \(\int w\rho G_L(y,u)/w\,du=f(y)\). As an integral operator in ordinary input L²,

\[
K=T\,w^{-1}T^*,\qquad K^{-1}=L^*wL
\tag{31}
\]

on the domain \(\{f\in D(L):wLf\in D(L^*)\}\), with adjoint boundary conditions included. For example, if f=Kq then \(Lf=w^{-1}T^*q\), so \(L^*wLf=L^*T^*q=q\). Conversely, on the displayed domain, \(K L^*wLf=T w^{-1}T^*L^*wLf=TLf=f\). The adjoint equalities follow from T=L⁻¹. This does not mean a compact smoothing K has a bounded inverse on all L². Its inverse is generally unbounded, and the displayed source norm identifies the admissible domain.

For a monotone scalar chart, set the first r latent derivatives at the left boundary to zero and use

\[
G_g(x,u)=\frac{(g(x)-g(u))_+^{r-1}}{(r-1)!},\qquad
L_g=g'D_g^r.
\]

For r=1, the truncated power is the Heaviside step. The chart energy from Section 3 satisfies

\[
E_g(f)=\int|D_g^rf|^2g'\,dx
=\int\frac{|L_gf|^2}{g'}dx,
\]

so w=1/g′. Substituting in (30), then changing variable t=g(u), gives

\[
\boxed{
k_g(x,y)=\int g'(u)G_g(x,u)G_g(y,u)du
=k_0(g(x),g(y)).
}
\tag{32}
\]

Here k₀ is the ordinary integrated-Brownian-motion/spline kernel on the latent interval with the same boundary convention. Thus the learned scalar chart literally pulls back a known source norm and its kernel. A free polynomial nullspace can be retained separately with the usual spline side constraints; it must not be silently treated as part of a strictly positive norm.

The boundary conditions can be made fully explicit. Assume g is smooth enough for the derivatives below and g′ is bounded above and away from zero. The source-domain conditions are \(D_g^kf(a)=0\), k=0,…,r−1. Integration by parts after t=g(x) gives the formal adjoint \(L_g^*=(-1)^rL_g\), with the corresponding right-end domain \(D_g^kz(b)=0\), k=0,…,r−1. Substituting \(z=(1/g')L_gf=D_g^rf\) therefore gives

\[
\boxed{K_g^{-1}=(-1)^r g'D_g^{2r},}
\tag{34}
\]

with left conditions \(D_g^kf(a)=0\) and right natural conditions \(D_g^{r+k}f(b)=0\), k=0,…,r−1. This is the inverse relative to ordinary dx integration. If the kernel integral operator uses observation measure \(d\mu=\rho\,dx\) with density bounded above and away from zero, its inverse is \(\rho^{-1}(-1)^r g'D_g^{2r}\) on the corresponding domain. The RKHS norm is unchanged; the observation-space operator and spectrum are changed.

For r=1, the source operator is simply D_x at every monotone chart, but the source weight is 1/g′. The precision operator is consequently \(-D_x(g'^{-1}D_x)\), which does change with the chart. This illustrates why source operator and source metric must be kept together. Also, the r=1 unsmoothed Brownian kernel has no finite differentiable feature metric of the type in (23); its diagonal is not twice differentiable. The Fisher/pullback derivative identity requires the regularity assumptions stated there, or a smooth source kernel.

Finite neurons approximate the integral in (30) only when the quadrature and readout metric are matched. If weights q_j approximate integration in u, use features \(\sqrt{q_j/w(u_j)}G_L(x,u_j)\) with Euclidean readout cost. Equivalently, raw Green features require diagonal coefficient metric \(M_{jj}=w(u_j)/q_j\). Arbitrary raw Euclidean readout penalties at nonuniform centers define a different norm. This is the same density factor that appears in the branch calculations.

### Stationary smoothing gives an explicit spectral source cost

On the line or a periodic domain with the zero/polynomial modes treated separately, suppose a smoothed source representation obeys

\[
D^rf=K_\gamma*\rho.
\]

Where \(\widehat K_\gamma\ne0\), the source is

\[
\widehat\rho(\omega)
=\frac{(i\omega)^r}{\widehat K_\gamma(\omega)}\widehat f(\omega).
\]

A unit quadratic source cost is exactly

\[
\boxed{
\|\rho\|_2^2=\frac1{2\pi}\int
\frac{|\omega|^{2r}}{|\widehat K_\gamma(\omega)|^2}
|\widehat f(\omega)|^2d\omega.
}
\tag{33}
\]

Consequently, choosing an activation's normalized r-th derivative kernel chooses a spectral source penalty for this continuum encoding convention. For ReLU, r=2 and K=δ, this is the ordinary curvature energy. For tanh, r=1 and \(\widehat K_\gamma=z/\sinh z\), \(z=\pi\omega/(2\gamma)\), broadening makes fine-frequency source cost exponentially large. A zero of \(\widehat K\) excludes the corresponding mode in this stationary model. Near-zeros make that mode expensive or unstable; they do not by themselves prove inability of a finite mixed-width or nonstationary network to represent it.

This gives one exact joint function/source/information object under a stated norm convention. Its kernel determines global feature inner products, its inverse determines the differential or nonlocal source cost, and its generalized Gram spectrum determines observable readout directions. Learned finite solutions need not minimize that continuum norm, so applying this model to a saved solution requires evidence for its selection rule and for the discretization regime.

## 12. Optimizing a scalar chart can change which source norm is induced

There is a simple exact example of chart selection, distinct from selecting knots to minimize interpolation error. Let f be absolutely continuous on I=[a,b], and let g map I increasingly to an interval of length one. Write q=g′>0, so \(\int_Iq=1\). The first-order chart energy is

\[
E_g(f)=\int_I\frac{|f'(x)|^2}{q(x)}dx.
\]

By Cauchy–Schwarz,

\[
\left(\int_I|f'|dx\right)^2
=\left(\int_I\frac{|f'|}{\sqrt q}\sqrt q\,dx\right)^2
\le E_g(f)\int_Iq\,dx=E_g(f).
\]

Equality requires q proportional to |f′|. If \(V=\int_I|f'|>0\), the formal optimum is q=|f′|/V. When this is not strictly positive, the admissible sequence

\[
q_\epsilon(x)=\frac{|f'(x)|+\epsilon}{V+\epsilon|I|}
\]

has unit mass and its energies tend to V². If further smoothness is required of g, suitable positive smooth approximations give the same infimum under ordinary approximation hypotheses. Thus

\[
\boxed{\inf_{g'>0,\;g(b)-g(a)=1}E_g(f)
=\left(\operatorname{TV}(f)\right)^2.}
\tag{35}
\]

Here total variation means \(\int|f'|\) for the absolutely continuous target; no claim about discontinuous target attainment is needed. The constant-target case has zero energy for every admissible chart. At the positive formal optimum, the latent derivative \(D_gf=f'/q\) has constant magnitude V, with the original sign changes retained. The chart redistributes sampling density while it cannot remove the target's reversals.

If one imposes 0<m≤q≤M and m|I|<1<M|I|, the first-order optimality condition is

\[
q_*(x)=\operatorname{clip}\left(\frac{|f'(x)|}{\sqrt\lambda},m,M\right),
\tag{36}
\]

with λ chosen to enforce unit mass, whenever a positive λ exists; in particular this holds when |f′|>0 almost everywhere. The zero-slope case requires care. If all nonzero-slope regions already saturate at M while total allocated mass is still below one, the multiplier is λ=0 and the remaining mass may be distributed over the zero-slope region within the bounds. It is not represented by a positive-λ clipping formula. The KKT condition \(-|f'|^2/q^2+\lambda\) plus the interval normal cone covers both cases.

This is a genuine connection between an adaptive quadratic geometry and a variation penalty. It does not identify the geometry selected by Adam or ordinary readout least squares. The chart itself is unpenalized here, so constructing q can carry the full complexity of |f′|. Moreover, this criterion allocates density proportional to |f′|; it differs from the \(|f''|^{2/5}\) density for asymptotic linear-interpolation L² error in Section 3. Each follows from its own explicitly stated objective.

The problem becomes even more degenerate for higher-order source energies if the latent polynomial nullspace is free. For r≥2 and a strictly monotone smooth f, choose g to be its affine rescaling. Then f is affine in g and \(D_g^rf=0\). The entire target has been moved into the uncharged chart/nullspace rather than represented economically. A plausible learned-geometry theory must therefore count the complexity, distortion, or architectural cost of g. The chart-penalized variational problem in Section 3 is one possible controlled model, not an established implicit objective of training.

## 13. What this contributes beyond a general kernel reformulation

Every finite dictionary can be assigned a Gram matrix, and many operators can be defined after the fact by inverting that matrix. That alone does not explain a learned solution. The useful restriction is that the source operator, transport law, or inverse-filter approximation be obtained from the activation and geometry before fitting a new target, with quantitative domain and stability conditions.

The present structured cases provide several such predictions:

- For ReLU or a warped Green dictionary, target samples directly give slope jumps in physical or latent coordinates. No new full least-squares fit is needed for the interpolation construction.
- For common-width tanh, the separate exact Cauchy analysis predicts deletion signs, broad-limit center-polynomial profiles, and narrow-limit cell-average coefficients. Those statements are more specific than a kernel definition.
- For slowly varying source families, the inverse-filter expansion predicts coefficients using derivatives of the target and geometry fields, with a remainder controlled only on a resolved band. Small perturbations can be coupled by the previously derived finite-rank transport/leakage system.
- The positive tanh width transport constructs a complete equivalent continuum representation with no target solve. The inverse filter predicts when undoing that redistribution is unstable.
- A scalar chart with a specified source cost gives a concrete pulled-back kernel, precision operator, and density criterion. A curved hidden dictionary can create multiple tied sources and curvature terms, which rules out pretending it is the same scalar chart.

The common object is therefore a **measured source dictionary with an encoding cost and an architectural construction**. Its invariant kernel is valuable when the cost is quadratic, but the labeled atoms remain necessary to predict individual neuron readouts, sparsity, or finite-width costs. A total-variation cost gives an atomic norm rather than the same RKHS. A finite atom budget may select representations that differ sharply from either continuum norm optimum.

Parameter meanings also depend on activation. For ReLU and γ>0, \(\operatorname{ReLU}(\gamma(x-c))=\gamma\operatorname{ReLU}(x-c)\). Changing γ at a fixed center rescales the readout coordinate; it does not change a kernel width or the represented span. Any penalty, initialization, or learning-rate effect of that rescaling must be analyzed as a coordinate/cost change. The smooth-width theory cannot be transferred literally to ReLU's γ parameter.

Nor is a common local differential inverse available for all activations. A smooth stationary activation typically requires the nonlocal multiplier \((i\omega)^r/\widehat K(\omega)\); Gaussian or tanh smoothing makes that inverse grow exponentially at high frequency. Fixed-frequency sine translates span only two dimensions, and fixed-width exponential translates span only one, although varying their frequencies or widths can create useful dictionaries. Polynomial activations at fixed shallow depth have finite-dimensional spans even as neuron count grows. These are explicit failures of particular activation/geometry combinations, not universal judgments about the named activation under every architecture.

This framework plausibly unifies a substantial set of fixed learned solutions. Its explanatory content comes from the structural predictions above. It still requires a justified encoding-selection convention, finite-versus-continuum control, cost for hidden maps, and evidence that an actual trained solution lies in an analyzed regime. It does not currently explain arbitrary high-dimensional feature learning, every activation's success, or the algorithmic discovery of favorable geometry.

## 14. The center-space Green kernel and input-space Green kernel are dual objects

The sharp-step example makes it possible to identify both sides of the synthesis map exactly and to distinguish observation density from neuron density. Let μ be a probability measure on [a,b] with continuous CDF F, and let

\[
\phi_c(x)=H(x-c)-(1-F(c))
\]

be the unit step after subtracting its μ-mean. Its center-space Gram kernel is

\[
\boxed{\langle\phi_c,\phi_d\rangle_\mu
=\min(F(c),F(d))-F(c)F(d).}
\tag{37}
\]

For c≤d, the uncentered product has expectation 1−F(d). Subtracting the product of the means gives F(c)(1−F(d)), proving (37). It is the Brownian-bridge kernel in data-CDF coordinates. Centered sharp tanh equals twice this step, so its Gram kernel is four times (37).

Now prescribe a measure ν of available centers and a positive readout cost w. The corresponding **input-space** kernel is instead

\[
k(x,y)=\int\frac{\phi_c(x)\phi_c(y)}{w(c)}\,d\nu(c).
\tag{38}
\]

Before centering the features, this equals the cumulative center measure/cost evaluated at min(x,y): \(\int_{c\le\min(x,y)}w(c)^{-1}d\nu(c)\), a Brownian-motion kernel after that cumulative coordinate change. Optimizing a free bias applies the μ-mean-removal projection to both kernel arguments. The result is a demeaned Brownian-motion kernel in input space, not generally a Brownian bridge in x. The bridge in (37) and the kernel in (38) live in different spaces.

Their nonzero spectra agree when the coefficient measures/costs are treated consistently. In a finite dictionary, define \(\widetilde A=AM^{-1/2}\); then \(\widetilde A^*\widetilde A=M^{-1/2}GM^{-1/2}\) and \(\widetilde A\widetilde A^*=AM^{-1}A^*\). For any eigenvector of the former with eigenvalue λ>0, \(\widetilde A v/\sqrt\lambda\) is an eigenfunction of the latter with the same eigenvalue. The same compact-operator argument applies to the continuum model when its hypotheses hold. The eigenfunctions themselves are different: one is a readout profile over centers; the other is a represented function over inputs.

For μ=ν=uniform measure on [0,1] and w=1, the complete correspondence is elementary:

\[
v_k(c)=\sqrt2\sin(k\pi c),\qquad
Av_k(x)=-\frac{\sqrt2}{k\pi}\cos(k\pi x),\qquad k\ge1.
\tag{39}
\]

Indeed, integrate \([H(x-c)-(1-c)]\sin(k\pi c)\) over c: the first term is \((1-\cos(k\pi x))/(k\pi)\), and the second is \(1/(k\pi)\). Both displayed sine and cosine families are orthonormal in their respective spaces. Therefore the singular values are exactly 1/(kπ), and the squared values 1/(kπ)² are the eigenvalues on both sides. The coefficient Gram has Dirichlet sine modes, while the centered input kernel has mean-zero Neumann cosine modes. Differentiating the represented function recovers its coefficient source.

This example does more than identify a generic matrix factorization: it explicitly pairs center/readout geometry with target-function geometry and gives the attenuation of every mode. Smoothing the step modifies that pairing by the activation transfer function; varying the observation or center measure changes different parts of the pair. It is therefore incorrect to identify data-CDF spacing, neuron-density spacing, and a learned chart without specifying which side of the synthesis map they describe.

## 15. A target-specific test of whether a lens is useful

A global condition number alone does not determine whether a geometry is useful for a given target. Let A synthesize into \(L^2(\mu)\), use coefficient cost \(\|v\|_M\), and suppose \(\|Av-f\|\le\epsilon\). For any unit probe q in the output Hilbert space,

\[
|\langle f,q\rangle|
\le|\langle Av,q\rangle|+\epsilon
=|\langle v,A^*q\rangle|+\epsilon
\le\|v\|_M\,\|A^*q\|_{M^{-1}}+\epsilon.
\]

Therefore

\[
\boxed{\|v\|_M\ge
\frac{(|\langle f,q\rangle|-\epsilon)_+}
{\|A^*q\|_{M^{-1}}}.}
\tag{40}
\]

If the denominator is zero and the numerator positive, the claimed ε-accuracy is impossible. Otherwise this lower bound quantifies the coefficient cost of encoding the target's component along q. No target fit is required to use it: a probe with an analytically known feature response suffices. Choosing a Fourier probe in a stationary family or a known broad-limit polynomial mode makes the denominator explicit from the corresponding theory.

Weak feature directions are harmless to an exact target with no component in them, although noise or a different target can excite them. Conversely, a nonnegligible target component in a nearly invisible direction forces a large readout cost. This separates approximation floor, target-relevant conditioning, and the smallest eigenvalue of the entire dictionary.

For example, in a known singular pair \(\widetilde A q_r=\sigma_r u_r\), where \(\widetilde A=AM^{-1/2}\), the denominator for probe u_r is σ_r. Equation (40) then gives the same target-projection divided by attenuation seen in the explicit broad-tanh and sharp-step formulas. If the target projection vanishes, the tiny σ_r alone does not force large coefficients.

A coefficient budget R also gives the error lower bound

\[
\inf_{\|v\|_M\le R}\|Av-f\|
\ge\sup_{\|q\|=1}
\bigl(|\langle f,q\rangle|-R\|A^*q\|_{M^{-1}}\bigr)_+.
\]

Large coefficients can expose sensitivity: an operator perturbation obeying \(\|\delta A\,M^{-1/2}\|\le\eta\) changes the represented function by at most \(\eta\|v\|_M\). This is a worst-case operator bound; it does not say every geometry perturbation or floating-point implementation attains that error.

An activation engineered to condition its represented modes well may consequently approach its floor quickly while the floor remains high because relevant target directions are missing or suppressed. A broader dictionary may have a lower floor yet require much larger coefficients in the target-relevant directions. There is no target-independent best activation or geometry without a specified target class, coefficient normalization, approximation criterion, and complexity cost.
