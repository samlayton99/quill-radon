# Geometry and solved readouts

Research checkpoint, September 29, 2026. Codex version.

This note develops a theory of the readouts obtained by least squares when a one-layer network's geometry is fixed, and of how those readouts change when the geometry changes. It contains the intermediate algebra needed to reconstruct the results. The immediate research aims are to predict several controlled center and width changes without refitting the full model, and to predict which activation and geometry combinations support accurate, stable approximation.

The central statement is that solved readouts are linear measurements of the target through geometry-dependent dual functions. With suitable independently fitted polynomial terms, they are exactly measurements of a derivative of the target. Derivative samples are a useful local approximation to those measurements. Inverting feature overlap produces the corrections, including alternating readouts and nonlocal compensation.

This is a fixed-geometry approximation theory with perturbation results. It does not establish a general account of representation learning, deep networks, Adam, or generalization. Its foundations are classical projection, spline, sampling, and perturbation theory. The research opportunity is to obtain simple, quantitative predictions for interpretable classes of neural-network geometries.

For the measured results with figures and a shorter explanation of what was tested, read the [illustrated results report](../../results/checkpoint_G_interactive/geometry_reader_codex/single_gamma_theory_20260929/checkpoint_validation/results_report.md). The present document contains the full derivations.

The subsequent [fixed-solution geometry investigation](../../results/checkpoint_G_interactive/geometry_reader_codex/geometry_object_20260929/report.md) develops the concrete tanh pole/rational representation, an explicit coordinate change and interpolation encoder, a polynomial-kernel deletion formula, and measured decompositions of saved learned solutions. It also distinguishes numerical redundancy from exact independence and tests direct moment decoding of broad-kernel compensation. No new training was involved.

The September 30 [implemented geometry-lens report](../../results/checkpoint_G_interactive/geometry_reader_codex/lens_construction_20260930/report.md) advances the finite-change program below: explicit reference encoders plus small interacting corrections predict up to twelve simultaneous center/width edits for tents and periodic tanh/GELU primitives. It supplies full proofs, reusable encoders, independent validations, and figures. A separate learned-geometry branch tests sample-to-readout transfer and decodes original broad-neuron readouts into derivative coordinates, with quantitative tail-error predictions and stated limitations.

## Reading map

1. [The model and conventions](#1-the-model-and-conventions)
2. [Least squares and dual measuring functions](#2-least-squares-and-dual-measuring-functions)
3. [Exact derivative measurement identities](#3-exact-derivative-measurement-identities)
4. [Activation derivatives and readout normalization](#4-activation-derivatives-and-readout-normalization)
5. [Uniform lattices and inverse overlap](#5-uniform-lattices-and-inverse-overlap)
6. [Replacing one neuron](#6-replacing-one-neuron)
7. [Several changed neurons and perturbation expansions](#7-several-changed-neurons-and-perturbation-expansions)
8. [Exact tent examples](#8-exact-tent-examples)
9. [An explicit prediction for four simultaneous changes](#9-an-explicit-prediction-for-four-simultaneous-changes)
10. [Strongly overlapping tents and exact redundancy](#10-strongly-overlapping-tents-and-exact-redundancy)
11. [ReLU coefficients as curvature](#11-relu-coefficients-as-curvature)
12. [Spectral criteria for activation selection](#12-spectral-criteria-for-activation-selection)
13. [Coefficient branches and optimization](#13-coefficient-branches-and-optimization)
14. [Evidence and its limits](#14-evidence-and-its-limits)
15. [Research directions and standards for a prediction](#15-research-directions-and-standards-for-a-prediction)
16. [References and reproducibility](#16-references-and-reproducibility)

## 1 The model and conventions

Fix a one-dimensional fitting domain \(\Omega\subseteq\mathbb R\) and a scalar target \(f:\Omega\to\mathbb R\). There are \(n\) neurons, with readout vector \(v=(v_1,\ldots,v_n)^T\in\mathbb R^n\). Each geometry parameter \(\theta_j=(c_j,\gamma_j)\in\mathbb R\times(0,\infty)\) specifies a center and an inverse scale, and each feature \(\phi_j(\,\cdot\,;\theta_j):\Omega\to\mathbb R\) is a scalar function. For an activation \(\sigma:\mathbb R\to\mathbb R\), the usual raw feature is \(\phi_j(x;\theta_j)=\sigma(\gamma_j(x-c_j))\); other normalizations are stated explicitly. Let \(p:\Omega\to\mathbb R\) be an independently fitted baseline function. The approximation is

\[
\widehat f(x)=p(x)+\sum_{j=1}^{n}v_j\phi_j(x;\theta_j),
\qquad \theta_j=(c_j,\gamma_j).
\tag{1.1}
\]

Each scalar \(v_j\) multiplies the **specified raw feature**. The baseline \(p\) belongs to a chosen finite-dimensional function space: usually the constants for tanh, or the affine functions for a curvature interpretation of GELU and ReLU. Features need not have equal widths or regular centers.

For a uniform grid the spacing is the scalar \(h>0\), and the dimensionless overlap parameter is \(\lambda=\gamma h\). If there are \(N\) interior centers including both endpoints of \([-1,1]\), then \(h=2/(N-1)\). Halo centers are additional: they count toward the total neuron count \(n\), but do not change that definition of \(N\). Some older experiments use \(N\) for the number of intervals and hence use \(h=2/N\). The formulas in this note use \(h\) to avoid conflating these conventions.

A tent's half-width will be denoted \(\rho h\), with scalar width ratio \(\rho>0\). Its inverse-width ratio relative to the baseline inverse width \(\gamma_0=1/h\) is \(\gamma/\gamma_0=1/\rho\). A raw tent has peak one. A unit-area tent has a different readout normalization.

Write \(\mathcal H\) for the observation Hilbert space: the space containing the target and model outputs, with the inner product used to measure fitting error. Three settings occur and must remain distinct:

* **Finite-interval continuous least squares:** \(\Omega=[a,b]\), \(\mathcal H=L^2([a,b])\), and \(\langle g_1,g_2\rangle_{\mathcal H}=\int_a^b g_1(x)g_2(x)\,dx\). Targets and features are square-integrable functions.
* **Sampled least squares:** at \(M\) points \(x_i\) where the functions are defined, observations are vectors in \(\mathbb R^M\). With positive weights \(w_i\), use \(\langle u,z\rangle_w=\sum_{i=1}^M w_i u_i z_i\), or equivalently multiply every sample by \(\sqrt{w_i}\) and use the Euclidean inner product. Below, design matrices absorb these weights when present.
* **Infinite or periodic translated kernels:** the coefficient space can change to \(\ell^2(\mathbb Z)\), while outputs belong to \(L^2(\mathbb R)\), or both spaces can be periodic. Each transition specifies the spaces and boundary conventions.

The same finite-dimensional coefficient algebra works in the first two settings, but sampling or truncating frequencies changes the observation space and fitting norm. A matrix identity in a sampled problem does not automatically assert equality with the continuous integral problem. Unless a subscript says otherwise, a function norm uses \(\mathcal H\), a finite coefficient norm is Euclidean, and an operator norm is induced by its specified input and output norms. Differentiating a model to interpret its readouts does **not** replace function-value fitting by derivative fitting. Those losses generally produce different coefficients.

Unless specified otherwise, exact coefficient identities assume linearly independent features and ordinary unregularized least squares. When features are redundant, the represented projection can be unique while its coefficients are not. Minimum norm, ridge, a singular-value cutoff, and initialization-dependent training can choose different coefficients. This choice is part of the model being predicted.

## 2 Least squares and dual measuring functions

Suppose the baseline has \(n_b\) fitted basis functions, and let \(d=n+n_b\) be the total number of features. For the projection algebra, include those baseline functions in the feature list \((\phi_1,\ldots,\phi_d)\) and use \(v\in\mathbb R^d\) for the complete coefficient vector. Thus \(v\) now includes baseline coefficients as well as the neuron readouts from (1.1); a “neuron coefficient” refers to an entry corresponding to a nonlinear feature, not a baseline entry.

Each \(\phi_i\) is an element of \(\mathcal H\). Define the synthesis operator \(A:\mathbb R^d\to\mathcal H\), which maps a coefficient vector to its predicted output, by

\[
A v=\sum_{i=1}^d v_i\phi_i.
\]

In continuous fitting, the “columns” of \(A\) are functions: \(Av\in L^2([a,b])\). In a sampled problem, \(A\in\mathbb R^{M\times d}\) is a design matrix with entries \(A_{ij}=\sqrt{w_i}\phi_j(x_i)\), and the target is represented by the weighted sample vector \(f_w=(\sqrt{w_i}f(x_i))_{i=1}^M\). The abstract symbol \(f\in\mathcal H\) in the algebra below means that target vector when the sampled setting is used.

The adjoint \(A^*:\mathcal H\to\mathbb R^d\) reverses the direction of the map, taking feature correlations rather than synthesizing an output. A star denotes the adjoint for the chosen inner products: transpose in real Euclidean coordinates and conjugate transpose in complex Fourier coordinates. It satisfies

\[
(A^*f)_i=\langle\phi_i,f\rangle.
\]

For real features, the loss is a scalar-valued function \(J:\mathbb R^d\to\mathbb R\):

\[
J(v)=\tfrac12\|f-Av\|^2.
\]

For an arbitrary coefficient direction \(z\in\mathbb R^d\) and scalar increment \(t\in\mathbb R\), expand

\[
J(v+tz)=\tfrac12\|f-Av-tAz\|^2
=J(v)-t\langle Az,f-Av\rangle+\tfrac12t^2\|Az\|^2.
\]

The derivative at zero vanishes for every \(z\) exactly when

\[
A^*(f-Av)=0.
\]

Define the Gram matrix \(G\in\mathbb R^{d\times d}\) and target-correlation vector \(b\in\mathbb R^d\) by

\[
G=A^*A,\qquad b=A^*f,
\qquad G_{ij}=\langle\phi_i,\phi_j\rangle.
\]

Then

\[
Gv=b.
\tag{2.1}
\]

If the features are independent, \(z^*Gz=\|Az\|^2>0\) for \(z\ne0\), so \(G\) is invertible and

\[
v=G^{-1}b.
\tag{2.2}
\]

This is an analytic identity, not a recommendation to form normal equations numerically. Stable QR or SVD solves are preferable for ill-conditioned numerical references.

For a fixed coefficient index \(j\), define the dual feature \(\psi_j\in\mathcal H\) as a linear combination of the original features. In continuous notation,

\[
\psi_j(x)=\sum_i(G^{-1})_{ji}\phi_i(x).
\tag{2.3}
\]

Substituting \(b_i=\langle\phi_i,f\rangle\) into (2.2) gives

\[
v_j=\sum_i(G^{-1})_{ji}\langle\phi_i,f\rangle
=\langle\psi_j,f\rangle.
\tag{2.4}
\]

Moreover, writing \(\delta_{jk}=1\) when \(j=k\) and \(0\) otherwise,

\[
\langle\psi_j,\phi_k\rangle
=\sum_i(G^{-1})_{ji}G_{ik}
=\delta_{jk}.
\tag{2.5}
\]

Thus \(\psi_j\) extracts the coefficient of feature \(j\) and rejects the other features. It depends on all columns through \(G^{-1}\). In contrast, \(\phi_j\) is the feature that contributes to the output. These analysis and synthesis functions usually have different shapes.

## 3 Exact derivative measurement identities

### 3.1 One derivative with a free constant

Use \(\mathcal H=L^2([a,b])\), and include the constant function as a separate feature. The map \(f\mapsto v_j=\langle\psi_j,f\rangle_{\mathcal H}\) is a scalar linear functional on this function space. For a neuron coefficient \(j\), (2.5) implies

\[
\int_a^b\psi_j(x)\,dx=0.
\]

Define the scalar measuring function \(D_{j,1}:[a,b]\to\mathbb R\) by

\[
D_{j,1}(x)=-\int_a^x\psi_j(t)\,dt.
\]

Then \(D_{j,1}'=-\psi_j\), \(D_{j,1}(a)=0\), and the zero integral gives \(D_{j,1}(b)=0\). For an absolutely continuous target with integrable products,

\[
\begin{aligned}
v_j
&=\int_a^b f\psi_j
=-\int_a^b fD_{j,1}'\\
&=-[fD_{j,1}]_a^b+\int_a^b f'D_{j,1}
=\int_a^b f'(x)D_{j,1}(x)\,dx.
\end{aligned}
\tag{3.1}
\]

This identity holds for any independent feature family with a free constant. It does not, by itself, identify tanh or prove that \(D_{j,1}\) is localized.

### 3.2 Two derivatives with a free affine term

Include both \(1\) and \(x\) as independent baseline features. For a neuron coefficient,

\[
\int_a^b\psi_j=0,\qquad\int_a^b x\psi_j(x)\,dx=0.
\tag{3.2}
\]

Define the scalar measuring function \(D_{j,2}:[a,b]\to\mathbb R\) by

\[
D_{j,2}(x)=\int_a^x(x-t)\psi_j(t)\,dt.
\]

Differentiation gives

\[
D_{j,2}'(x)=\int_a^x\psi_j(t)\,dt,
\qquad D_{j,2}''(x)=\psi_j(x).
\]

At \(a\), both \(D_{j,2}\) and \(D_{j,2}'\) vanish. At \(b\),

\[
D_{j,2}'(b)=\int_a^b\psi_j=0,
\]

and

\[
D_{j,2}(b)=b\int_a^b\psi_j(t)\,dt-\int_a^b t\psi_j(t)\,dt=0.
\]

For \(f\in C^2[a,b]\), integrate twice:

\[
\begin{aligned}
v_j
&=\int_a^b fD_{j,2}''\\
&=[fD_{j,2}']_a^b-\int_a^b f'D_{j,2}'\\
&=[fD_{j,2}'-f'D_{j,2}]_a^b+\int_a^b f''D_{j,2}\\
&=\int_a^b f''(x)D_{j,2}(x)\,dx.
\end{aligned}
\tag{3.3}
\]

A bias alone does not ensure (3.2). The target \(f(x)=x\) has zero second derivative, but a bias-only model may need nonzero neuron coefficients to represent it. The missing linear moment then leaves a boundary or affine contribution.

### 3.3 General order and the meaning of polynomial annihilation

Fix an integer derivative order \(m\ge1\), and let \(\mathcal P_{m-1}\subset L^2([a,b])\) be the space of polynomials of degree at most \(m-1\). Suppose the neuron coefficient functional \(f\mapsto\langle\psi_j,f\rangle\) vanishes on all of \(\mathcal P_{m-1}\):

\[
\int_a^b x^\ell\psi_j(x)\,dx=0,
\qquad 0\le\ell<m.
\tag{3.4}
\]

An independently fitted polynomial baseline spanning \(\mathcal P_{m-1}\), with all columns independent, supplies these conditions. For \(f\in C^m([a,b])\), Taylor's formula with integral remainder is

\[
f(x)=\sum_{\ell=0}^{m-1}\frac{f^{(\ell)}(a)}{\ell!}(x-a)^\ell
+\int_a^x\frac{(x-t)^{m-1}}{(m-1)!}f^{(m)}(t)\,dt.
\]

Multiply by \(\psi_j(x)\) and integrate. The polynomial sum vanishes by (3.4). Reversing the order of integration over \(a\le t\le x\le b\) defines a scalar measuring function \(D_{j,m}:[a,b]\to\mathbb R\) and yields

\[
v_j=\int_a^b f^{(m)}(t)
\underbrace{\left[\int_t^b\frac{(x-t)^{m-1}}{(m-1)!}\psi_j(x)\,dx\right]}_{D_{j,m}(t)}dt.
\tag{3.5}
\]

This proves the general derivative-measurement identity. For \(m=1,2\), the right-hand primitive in (3.5) equals the left-hand primitive above because the polynomial moments vanish. The derivative order natural for an activation is the order at which its forward feature becomes localized; the analysis identity can hold at other orders when extra polynomial moments vanish.

### 3.4 The sampled identity

Use \(M\) samples and \(d\) features, including the baseline. The unweighted design matrix is \(\Phi\in\mathbb R^{M\times d}\), with entries \(\Phi_{ik}=\phi_k(x_i)\), and the unweighted target vector is \(f_{\rm samples}=(f(x_i))_{i=1}^M\in\mathbb R^M\). Define the positive diagonal weight matrix \(W=\operatorname{diag}(w_1,\ldots,w_M)\in\mathbb R^{M\times M}\) and weighted design \(A_w=W^{1/2}\Phi\in\mathbb R^{M\times d}\).

Assume full column rank and a constant column. Let \(e_j\in\mathbb R^d\) be the coordinate vector selecting neuron coefficient \(j\). The Moore–Penrose inverse \(A_w^\dagger:\mathbb R^M\to\mathbb R^d\) is a \(d\times M\) matrix; here \(A_w^\dagger A_w=I_d\), where \(I_d\) is the identity on coefficient space. Define the row \(\ell\in\mathbb R^{1\times M}\) acting on **unweighted target samples** by

\[
\ell=e_j^*A_w^\dagger W^{1/2},\qquad v_j=\ell f_{\rm samples}.
\]

Let \(e_{\rm const}\in\mathbb R^d\) select the constant column and \(\mathbf1\in\mathbb R^M\) be the all-one sample vector. Then \(W^{1/2}\mathbf1=A_we_{\rm const}\). Therefore

\[
\ell\mathbf1=e_j^*A_w^\dagger A_we_{\rm const}
=e_j^*e_{\rm const}=0.
\]

A row of \(A_w^\dagger\) alone does not generally have zero unweighted sum; the factor \(W^{1/2}\) is necessary.

For ordered sample locations \(x_1<\cdots<x_M\), write

\[
f(x_i)=f(x_1)+\sum_{k=1}^{i-1}[f(x_{k+1})-f(x_k)].
\]

Substitution and reordering a finite sum give

\[
v_j=\sum_{k=1}^{M-1}\left(\sum_{i=k+1}^M\ell_i\right)
[f(x_{k+1})-f(x_k)].
\tag{3.6}
\]

Thus the exact finite-sample statement concerns weighted target increments. It does not require derivative observations.

### 3.5 Where these identities need qualification

For a sampled design \(A\in\mathbb R^{M\times d}\), let \(A_\tau^\dagger\in\mathbb R^{d\times M}\) denote its inverse with singular values below a specified cutoff \(\tau\) discarded. Then \(A_\tau^\dagger A\) does not generally equal the coefficient-space identity \(I_d\). In a redundant minimum-norm model, a constant or a linear function may be represented partly by neurons. In both cases the polynomial-annihilation conditions must be checked rather than assumed. One can explicitly project the target and neuron features onto the orthogonal complement of the baseline subspace in \(\mathcal H\) before solving the nonlinear coefficients, enforcing invariance to that baseline. Otherwise retain the polynomial-response terms in Taylor's formula.

The exact derivative representation and a local rule \(v_j\approx C f^{(m)}(c_j)\), with scalar normalization \(C\), have different strength. The latter needs localization and normalization of the function \(D_{j,m}\), together with a sufficiently slowly varying target derivative.

## 4 Activation derivatives and readout normalization

### 4.1 A general mass calculation

Let \(\sigma:\mathbb R\to\mathbb R\) be an activation and \(m\ge0\) an integer derivative order. Assume \(K=\sigma^{(m)}\in L^1(\mathbb R)\) is an integrable scalar kernel with nonzero mass; \(K\) here is a function, not a matrix. In this section, sums over \(j\) and the vector \(v\in\mathbb R^n\) concern only the \(n\) neuron readouts of (1.1); the baseline is treated separately. For \(m\ge1\), assume the baseline \(p\) is a polynomial of degree below \(m\), so it disappears after differentiation. For \(m=0\), use no baseline or apply the formulas to \(\widehat f-p\) and the corresponding baseline-subtracted target. Define the scalar mass

\[
M_K=\int_{\mathbb R}K(t)\,dt.
\]

For the raw feature \(\sigma(\gamma(x-c))\), the chain rule gives

\[
\frac{d^m}{dx^m}\sigma(\gamma(x-c))=\gamma^m K(\gamma(x-c)).
\]

Changing variables \(t=\gamma(x-c)\), \(dx=dt/\gamma\), shows that its derivative feature has mass \(\gamma^{m-1}M_K\). For each \(\gamma>0\), define the unit-mass function \(\kappa_\gamma\in L^1(\mathbb R)\) by

\[
\kappa_\gamma(x)=\frac{\gamma}{M_K}K(\gamma x).
\]

Then

\[
\widehat f^{(m)}(x)=\sum_j
\underbrace{v_j\gamma_j^{m-1}M_K}_{\text{derivative mass coefficient}}
\kappa_{\gamma_j}(x-c_j).
\tag{4.1}
\]

For a sufficiently smooth scalar function \(q\), a resolved uniform grid of unit-mass kernels can give the quadrature approximation \(q(x)\approx\sum_j hq(c_j)\kappa_\gamma(x-c_j)\). Comparing this approximation for \(q=f^{(m)}\) with (4.1) motivates

\[
v_j\approx\frac{h}{M_K\gamma_j^{m-1}}f^{(m)}(c_j).
\tag{4.2}
\]

This is a quadrature/localization approximation, not an exact least-squares identity. Broad kernels require deconvolution; a discrete lattice adds aliases; nonuniform geometry changes the dual functions. Replacing a common gamma by a local gamma in (4.2) does not account for those collective changes.

For a raw bump \(m=0\), (4.2) reads \(v_j\approx h\gamma f(c_j)/M_K\). For a bump already normalized to unit area, the coefficient instead approximates \(hf(c_j)\). These are the same prediction in different feature coordinates.

### 4.2 Tanh and sigmoid

For \(\sigma(z)=\tanh z\), \(K(z)=\operatorname{sech}^2z\), and

\[
M_K=\tanh(+\infty)-\tanh(-\infty)=2.
\]

Thus

\[
\widehat f'=\sum_jv_j\gamma_j\operatorname{sech}^2(\gamma_j(x-c_j)),
\qquad v_j\approx\frac h2 f'(c_j).
\tag{4.3}
\]

For the logistic sigmoid, \(\sigma(z)=[1+\tanh(z/2)]/2\). With a free constant, sigmoid at gamma spans exactly the same neuron space as tanh at gamma/2. If \(v^{\rm sig}\) and \(v^{\tanh}\) describe the same function under this matched geometry, then \(v^{\tanh}=v^{\rm sig}/2\), with the extra constants absorbed by the output bias. An activation rule must respect this equivalence.

### 4.3 GELU

Let \(\varphi:\mathbb R\to\mathbb R\) be the standard Gaussian density and \(\Phi:\mathbb R\to[0,1]\) its cumulative distribution function. This scalar \(\Phi\) is distinct from the design matrix in Section 3.4. Exact GELU is \(g(z)=z\Phi(z)\), with \(\Phi'=\varphi\) and \(\varphi'(z)=-z\varphi(z)\). Therefore

\[
g'(z)=\Phi(z)+z\varphi(z),
\]

\[
g''(z)=\varphi(z)+\varphi(z)+z\varphi'(z)
=(2-z^2)\varphi(z).
\tag{4.4}
\]

Its mass is \(g'(+\infty)-g'(-\infty)=1\). It is signed: negative outside \(|z|>\sqrt2\). A positive kernel is not required for the projection theory.

For raw GELU features,

\[
\widehat f''=\sum_jv_j\gamma_j^2K(\gamma_j(x-c_j)),
\qquad \gamma_jv_j\approx hf''(c_j).
\tag{4.5}
\]

Dividing each raw activation by gamma moves this gamma factor from the coefficient into the feature normalization. Normalized GELU \(g(\gamma z)/\gamma=z\Phi(\gamma z)\) tends pointwise to \(z_+\) as gamma grows, connecting its smooth curvature kernel to the ReLU slope jump.

### 4.4 Coefficient norm scaling in the local regime

At fixed \(\lambda=\gamma h\), substitute \(\gamma=\lambda/h\) into (4.2):

\[
v_j\approx\frac{h^m\lambda^{1-m}}{M_K}f^{(m)}(c_j).
\]

If \(h\sum_j|f^{(m)}(c_j)|^2\approx\|f^{(m)}\|_{L^2}^2\), then

\[
\begin{aligned}
\|v\|_2^2
&\approx \frac{h^{2m}\lambda^{2-2m}}{M_K^2}
\sum_j|f^{(m)}(c_j)|^2\\
&\approx\frac{h^{2m-1}\lambda^{2-2m}}{M_K^2}
\|f^{(m)}\|_{L^2}^2.
\end{aligned}
\]

Hence

\[
\|v\|_2\approx\frac{h^{m-1/2}\lambda^{1-m}}{|M_K|}
\|f^{(m)}\|_{L^2}.
\tag{4.6}
\]

The approximation presumes a resolved smooth derivative and modest inverse-filter correction. If the target has substantial energy where the kernel spectrum is tiny, this law can fail even though a good function approximation remains possible with large compensating coefficients.

## 5 Uniform lattices and inverse overlap

### 5.1 The convolution equation

Here the finite coefficient space is replaced by \(\ell^2(\mathbb Z)=\{v=(v_j)_{j\in\mathbb Z}:\sum_j|v_j|^2<\infty\}\), and outputs belong to \(\mathcal H=L^2(\mathbb R)\). Use a real localized kernel \(K\in L^1(\mathbb R)\cap L^2(\mathbb R)\), spacing \(h>0\), and features \(\phi_j(x)=K(x-jh)\) for every integer \(j\). The synthesis map is now \(A:\ell^2(\mathbb Z)\to L^2(\mathbb R)\), \(Av=\sum_jv_j\phi_j\), with the series interpreted in \(L^2\). Assume this map is bounded above and below; the lower bound excludes redundant or unstable lattices.

For a target \(f\in L^2(\mathbb R)\), the correlation sequence is \(b=A^*f\in\ell^2(\mathbb Z)\). The Gram operator \(G=A^*A:\ell^2(\mathbb Z)\to\ell^2(\mathbb Z)\) has an infinite array of entries, rather than being a finite design matrix.

Translation of the integration variable gives

\[
G_{jk}=\int K(x-jh)K(x-kh)\,dx=g_{j-k},
\]

Here \(g=(g_n)_{n\in\mathbb Z}\) is the scalar overlap sequence, \(g_n=\int_{\mathbb R}K(x)K(x-nh)\,dx\). The normal equation is therefore a sequence-convolution equation:

\[
\sum_k g_{j-k}v_k=b_j.
\tag{5.1}
\]

For a finitely supported sequence \(z\), define its Fourier series \(\widetilde z(\theta)=\sum_jz_je^{-ij\theta}\), with dimensionless frequency \(\theta\in[-\pi,\pi)\). This extends by limits to a map from \(\ell^2(\mathbb Z)\) into complex-valued \(L^2([-\pi,\pi))\); it is not a pointwise-convergent sum for every \(\ell^2\) sequence. Under absolute summability, or first for finite sums and then by the corresponding operator limits, expanding the double sum gives

\[
\sum_j\sum_k g_{j-k}v_ke^{-ij\theta}
=\sum_k v_ke^{-ik\theta}\sum_n g_ne^{-in\theta}
=\widetilde g(\theta)\widetilde v(\theta).
\]

Therefore

\[
\widetilde v(\theta)=\frac{\widetilde b(\theta)}{\widetilde g(\theta)}.
\tag{5.2}
\]

Uniformity makes coefficient recovery the same inverse filter at every center. It does not make that filter the identity.

### 5.2 The Gram spectrum and stability

For angular frequency \(\omega\in\mathbb R\), the continuous Fourier transform is the scalar-valued function \(\widehat K(\omega)=\int_{\mathbb R}K(x)e^{-i\omega x}dx\). This is different from the sequence Fourier series \(\widetilde z\). Parseval applied to the correlation gives

\[
g_n=\frac1{2\pi}\int_{\mathbb R}|\widehat K(\omega)|^2e^{i\omega nh}\,d\omega.
\]

Partition the frequency axis using \(\omega=(\theta+2\pi\ell)/h\), with \(-\pi\le\theta<\pi\), and use \(e^{i2\pi\ell n}=1\):

\[
g_n=\frac1{2\pi}\int_{-\pi}^{\pi}
\underbrace{\left[\frac1h\sum_{\ell\in\mathbb Z}
\left|\widehat K\!\left(\frac{\theta+2\pi\ell}{h}\right)\right|^2\right]}_{\widetilde g(\theta)}e^{in\theta}\,d\theta.
\tag{5.3}
\]

Comparing Fourier coefficients proves the expression for the nonnegative scalar function \(\widetilde g:[-\pi,\pi)\to[0,\infty)\), called the Gram symbol. In Fourier coordinates, \(G\) acts by multiplication by this function. Under sufficient decay, the sums converge absolutely; more general formulations use almost-everywhere identities.

For a finite coefficient sequence, and then by density for \(\ell^2\),

\[
\|Av\|_{L^2}^2=v^*Gv
=\frac1{2\pi}\int_{-\pi}^{\pi}\widetilde g(\theta)|\widetilde v(\theta)|^2d\theta.
\tag{5.4}
\]

Let \(a=\operatorname{ess\,inf}\widetilde g\) and \(b=\operatorname{ess\,sup}\widetilde g\) denote the sharp scalar spectral bounds, distinct from interval endpoints used earlier. If \(0<a\le b<\infty\), sequence Parseval yields

\[
a\|v\|_2^2\le\|Av\|_{L^2}^2\le b\|v\|_2^2.
\tag{5.5}
\]

Consequently the Gram condition number is \(b/a\), and the synthesis condition number is \(\sqrt{b/a}\). A small \(\widetilde g\) permits a large coefficient oscillation with little output energy. A zero at even an isolated frequency can destroy a uniform lower bound, although a nonzero square-summable null vector need not exist on the infinite line. A matching periodic grid can turn that frequency into an exact finite-dimensional null direction.

### 5.3 Why derivative samples need an inverse-filter correction

For tanh define the unit-mass function \(\kappa_\gamma\in L^1(\mathbb R)\cap L^2(\mathbb R)\) by \(\kappa_\gamma(x)=(\gamma/2)\operatorname{sech}^2(\gamma x)\). To distinguish a sequence from a function, write \(V:\mathbb R\to\mathbb R\) for a smooth coefficient envelope satisfying \(V(jh)=v_j\). In a continuum approximation to a dense-enough grid,

\[
\widehat f'(x)=2\sum_jv_j\kappa_\gamma(x-jh)
\approx\frac2h(\kappa_\gamma*V)(x),
\]

Here the star between functions denotes convolution, \((\kappa_\gamma*V)(x)=\int_{\mathbb R}\kappa_\gamma(x-c)V(c)\,dc\); a superscript star on an operator still denotes its adjoint. Matching the derivative gives the formal inverse-filter relation

\[
V\approx\frac h2\kappa_\gamma^{-1}*f'.
\tag{5.6}
\]

The notation \(\kappa_\gamma^{-1}*\) means the Fourier multiplier \(1/\widehat\kappa_\gamma(\omega)\) on inputs for which this inverse is defined, not pointwise division by \(\kappa_\gamma(x)\). It need not be a bounded inverse on all of \(L^2(\mathbb R)\). This approximation neglects lattice aliases and boundary effects; it is not an exact formula for finite-interval function least squares.

The normalized second moment can be computed from \(\operatorname{sech}^2t=4\sum_{n\ge1}(-1)^{n-1}ne^{-2nt}\) for \(t>0\):

\[
\int_{\mathbb R}\frac{t^2}{2}\operatorname{sech}^2t\,dt
=\int_0^\infty t^2\operatorname{sech}^2t\,dt
=\sum_{n\ge1}\frac{(-1)^{n-1}}{n^2}
=\frac{\pi^2}{12}.
\]

Here \(\int_0^\infty t^2e^{-2nt}dt=2/(2n)^3\); the integrated series is absolutely summable. Scaling gives moment \(\pi^2/(12\gamma^2)\). For a sufficiently smooth scalar function \(q\), Taylor expansion inside convolution therefore gives

\[
\kappa_\gamma*q=q+\frac{\pi^2}{24\gamma^2}q''+\text{higher derivative terms}.
\]

To invert to this order, set the scalar \(\varepsilon=\pi^2/(24\gamma^2)\) and substitute \(q-\varepsilon q''\) into the operator \(I+\varepsilon\partial_x^2\), where \(I\) is the identity on functions and \(\partial_x^2\) is their second derivative. The terms of order \(\varepsilon\) cancel. Hence

\[
V\approx\frac h2\left[f'-\frac{\pi^2}{24\gamma^2}f'''+\cdots\right].
\tag{5.7}
\]

The derivative-sampling rule is the first term of a filtered expansion. Smoothness and low frequencies relative to gamma are required; an irregular lattice cannot be handled merely by inserting each local gamma into this expression.

## 6 Replacing one neuron

### 6.1 Exact elimination of unchanged features

Return to a finite dictionary with \(d\) features in the observation space \(\mathcal H\) of Section 1. Let \(B:\mathbb R^{d-1}\to\mathcal H\) synthesize all unchanged features, including any unchanged baseline, and let \(\phi\in\mathcal H\) be the proposed new feature. The unchanged readouts are \(\alpha\in\mathbb R^{d-1}\), the edited readout is a scalar \(\beta\in\mathbb R\), and the new output is \(B\alpha+\beta\phi\in\mathcal H\).

Assume \(B\) has independent columns. Then \(B^\dagger=(B^*B)^{-1}B^*:\mathcal H\to\mathbb R^{d-1}\) extracts its least-squares coefficients. The orthogonal projector onto its output space is \(P=B(B^*B)^{-1}B^*=BB^\dagger:\mathcal H\to\mathcal H\). Thus \(I-P\), with \(I\) the identity on \(\mathcal H\), removes everything the unchanged features can represent. In a sampled problem, \(B\), \(B^\dagger\), and \(P\) have dimensions \(M\times(d-1)\), \((d-1)\times M\), and \(M\times M\), respectively.

For a fixed \(\beta\), solving the unchanged coefficients gives

\[
\alpha(\beta)=B^\dagger(f-\beta\phi).
\]

Substituting into the residual gives

\[
\begin{aligned}
f-B\alpha(\beta)-\beta\phi
&=f-P(f-\beta\phi)-\beta\phi\\
&=(I-P)f-\beta(I-P)\phi.
\end{aligned}
\]

Define two residual functions, or residual sample vectors in the sampled setting: \(r=(I-P)f\in\mathcal H\) and \(u=(I-P)\phi\in\mathcal H\). The remaining objective is a scalar quadratic in \(\beta\):

\[
\|r-\beta u\|^2=\|r\|^2-2\beta\langle u,r\rangle+\beta^2\|u\|^2.
\]

Its derivative is \(-2\langle u,r\rangle+2\beta\|u\|^2\). If \(u\ne0\),

\[
\boxed{\beta=\frac{\langle u,r\rangle}{\|u\|^2},\qquad
\alpha=B^\dagger f-\beta B^\dagger\phi.}
\tag{6.1}
\]

If \(u=0\), the added feature already lies in the unchanged span. Its coefficient is not identifiable without a convention. Taking a limit in a formula that divides by \(\|u\|^2\) is not legitimate at that point.

### 6.2 The geometry-only response theorem

Let \(\phi_0\in\mathcal H\) be the original selected feature, with scalar readout \(\beta_0\) and unchanged-feature readouts \(\alpha_0\in\mathbb R^{d-1}\). Suppose that old model fits exactly:

\[
f=B\alpha_0+\beta_0\phi_0.
\]

Define \(u_0=(I-P)\phi_0\in\mathcal H\). Applying \(I-P\) gives \(r=\beta_0u_0\), so equation (6.1) gives

\[
\beta=\beta_0\frac{\langle u,u_0\rangle}{\|u\|^2}.
\tag{6.2}
\]

Also \(B^\dagger f=\alpha_0+\beta_0B^\dagger\phi_0\), so

\[
\alpha-\alpha_0=\beta_0B^\dagger\phi_0-\beta B^\dagger\phi.
\tag{6.3}
\]

Every coefficient change is the old coefficient \(\beta_0\) times a response determined only by the geometry change. Translation symmetry makes this response shift with the defect on a uniform infinite lattice.

If the old least-squares fit instead leaves a residual \(e=f-B\alpha_0-\beta_0\phi_0\in\mathcal H\), then \(B^*e=0\) and

\[
r=\beta_0u_0+e,
\qquad
\beta=\beta_0\frac{\langle u,u_0\rangle}{\|u\|^2}
+\frac{\langle u,e\rangle}{\|u\|^2}.
\tag{6.4}
\]

Cauchy–Schwarz bounds the extra term by \(\|e\|/\|u\|\). Thus small output error need not imply a small coefficient correction when the proposed feature is almost redundant.

### 6.3 Limits depend on normalization and on reoptimization

On a bounded interval, \(\tanh(\gamma(x-c))\to\operatorname{sign}(x-c)\) as gamma grows. This is a step, not a deleted feature. As gamma tends to zero,

\[
\tanh(\gamma(x-c))=\gamma(x-c)+O(\gamma^3),
\]

so a readout growing like \(1/\gamma\) retains a linear contribution. Pointwise convergence at fixed readout does not determine the optimized limit.

For a scalar kernel \(\kappa\in L^1(\mathbb R)\cap L^2(\mathbb R)\) with \(\int_{\mathbb R}\kappa=1\), define \(\kappa_\gamma(x)=\gamma\kappa(\gamma x)\). Its whole-line \(L^2\) norm and inner product satisfy

\[
\|\kappa_\gamma\|_2^2=\gamma\|\kappa\|_2^2,
\qquad \langle\kappa_\gamma(\cdot-c),r\rangle\to r(c)
\]

for a continuous residual and a suitable approximate identity. With a fixed finite smooth background whose projection contributes only bounded terms, (6.1) then gives \(\beta=O(\gamma^{-1})\), and the optimized bump's output norm is \(O(\gamma^{-1/2})\). This is a legitimate deletion-like limit in continuous bump fitting. A finite sample exactly at the center or an infinite changing background can change the function limit. A nonzero column rescaling alone changes the coefficient's scaling but not its span or exact unregularized projection. Normalization can affect the fitted function when penalties, coefficient bounds, cutoffs, or floating-point effects are present.

A second useful counterexample uses the scalar tent function \(T(t)=(1-|t|)_+\), where \(z_+=\max(z,0)\), on an interval fully inside its scaled support. There \(T(\gamma(x-c))=1-\gamma|x-c|\). With a free constant, the two subspaces of functions on that interval obey

\[
\operatorname{span}\{1,T(\gamma(x-c))\}
=\operatorname{span}\{1,|x-c|\}
\]

for every sufficiently small positive gamma. Indeed, \(|x-c|=[1-T(\gamma(x-c))]/\gamma\). Although the tent approaches a constant pointwise, an inverse-gamma readout and a cancelling bias preserve an absolute-value contribution. For GELU, expanding \(\Phi(z)=1/2+\varphi(0)z+O(z^3)\) gives \(g(z)=z/2+\varphi(0)z^2+O(z^4)\). After removing the affine term, a readout of order \(1/\gamma^2\) can similarly retain a quadratic contribution on a bounded interval.

## 7 Several changed neurons and perturbation expansions

### 7.1 Exact reduction and what it does not explain by itself

Partition a finite dictionary of \(d\) features into \(k\) edited and \(d-k\) unchanged features. Define synthesis maps \(\Phi:\mathbb R^k\to\mathcal H\) for the edited features and \(B:\mathbb R^{d-k}\to\mathcal H\) for the unchanged ones; their coefficient vectors are \(\beta\in\mathbb R^k\) and \(\alpha\in\mathbb R^{d-k}\). Here \(\Phi\) is a feature map, not the scalar Gaussian cumulative distribution function. Let \(P_B=BB^\dagger:\mathcal H\to\mathcal H\) be the unchanged-space projector.

Define the residualized feature map \(U:\mathbb R^k\to\mathcal H\) and target residual \(r\in\mathcal H\) by

\[
U=(I-P_B)\Phi,\qquad r=(I-P_B)f,
\qquad \min_{\beta\in\mathbb R^k}\|r-U\beta\|_{\mathcal H}^2.
\]

Eliminating \(\alpha\) leaves the displayed \(k\)-variable problem. Its residualized overlap matrix \(H=U^*U\in\mathbb R^{k\times k}\) and correlation vector \(U^*r\in\mathbb R^k\) are finite even when outputs are functions. If the columns of \(U\) are independent, differentiation gives

\[
H\beta=U^*r,\qquad H=U^*U,
\qquad \alpha=B^\dagger(f-\Phi\beta).
\tag{7.1}
\]

Writing \(u_a\in\mathcal H\) for column \(a\) of \(U\), the scalar entry \(H_{ab}=\langle u_a,u_b\rangle\) measures how two changed neurons overlap **after the unchanged network has absorbed what it can**. This is why nearby changes cannot generally be treated independently.

Let \(\Phi_0:\mathbb R^k\to\mathcal H\) contain the original versions of the edited features and set \(U_0=(I-P_B)\Phi_0:\mathbb R^k\to\mathcal H\). If \(f=B\alpha_0+\Phi_0\beta_0\), with \(\alpha_0\in\mathbb R^{d-k}\) and \(\beta_0\in\mathbb R^k\), then \(r=U_0\beta_0\), and

\[
\beta=H^{-1}U^*U_0\beta_0.
\tag{7.2}
\]

This is an exact geometry-dependent response matrix. Computing every term anew from a dense factorization is only a reformulation of least squares. It becomes an explanatory prediction when the entries are known analytically, follow a reusable distance law, or admit a controlled approximation independent of a fresh full solve.

### 7.2 First-order coefficient sensitivity

Let \(t\in\mathbb R\) parameterize a smooth path of geometries, and let \(A(t):\mathbb R^d\to\mathcal H\) be its synthesis map, differentiable in operator norm with full column rank throughout. The target \(f\in\mathcal H\) is fixed. At each \(t\), the optimal coefficients \(v(t)\in\mathbb R^d\) leave a residual \(e(t)=f-A(t)v(t)\in\mathcal H\). Write \(G(t)=A(t)^*A(t)\in\mathbb R^{d\times d}\); dots mean derivatives with respect to this geometry-path parameter, not training time. Suppressing the argument \(t\), differentiate \(A^*e=0\):

\[
\dot A^*e+A^*\dot e=0,
\qquad \dot e=-\dot Av-A\dot v.
\]

Substitution gives

\[
\dot A^*e-A^*\dot Av-G\dot v=0,
\]

and hence

\[
\boxed{\dot v=G^{-1}(\dot A^*e-A^*\dot Av).}
\tag{7.3}
\]

At an exact fit \(e=0\),

\[
\dot v=-A^\dagger\dot Av.
\tag{7.4}
\]

If only a finite set \(J\) of neurons changes,

\[
\dot Av=\sum_{j\in J}v_j
\left(\dot c_j\partial_c\phi_j+\dot\gamma_j\partial_\gamma\phi_j\right).
\tag{7.5}
\]

The linearity of (7.4) proves first-order superposition of changes. It does not prove superposition for finite perturbations. For tanh,

\[
\partial_c\phi_j=-\gamma_j\operatorname{sech}^2(\gamma_j(x-c_j)),
\qquad
\partial_\gamma\phi_j=(x-c_j)\operatorname{sech}^2(\gamma_j(x-c_j)).
\]

### 7.3 A finite-change expansion with a remainder bound

Let \(A:\mathbb R^d\to\mathcal H\) be the original synthesis map and \(E:\mathbb R^d\to\mathcal H\) its finite feature change. The new map is \(A+E\), and \(E\) has zero columns at unchanged indices. In sampled coordinates, \(A\) and \(E\) are \(M\times d\) matrices. The original coefficients \(v\in\mathbb R^d\) leave residual \(e=f-Av\in\mathcal H\); write \(v+\delta v\) for the new coefficients, with \(\delta v\in\mathbb R^d\). The original Gram matrix is \(G=A^*A\in\mathbb R^{d\times d}\), and its change is the \(d\times d\) matrix

\[
\Delta G=A^*E+E^*A+E^*E.
\]

The new normal equation is

\[
(G+\Delta G)(v+\delta v)=A^*f+E^*f.
\]

Subtract \(Gv=A^*f\) and use \(f=Av+e\):

\[
\begin{aligned}
(G+\Delta G)\delta v
&=E^*f-\Delta Gv\\
&=E^*e-A^*Ev-E^*Ev.
\end{aligned}
\tag{7.6}
\]

Define a coefficient-space perturbation matrix \(H\in\mathbb R^{d\times d}\) and a forcing vector \(z\in\mathbb R^d\) by

\[
H=G^{-1}\Delta G,
\qquad z=G^{-1}(E^*e-A^*Ev-E^*Ev).
\]

Here \(H\) is the full \(d\times d\) perturbation matrix, not the \(k\times k\) residualized Gram matrix of Section 7.1. The identity \(I\) in \((I+H)\delta v=z\) acts on \(\mathbb R^d\). For an induced coefficient-space norm, if the scalar \(\eta=\|H\|<1\), the geometric series converges:

\[
\delta v=\sum_{r=0}^{\infty}(-H)^r z.
\tag{7.7}
\]

For a nonnegative integer truncation order \(p\), multiplying \(\sum_{r=0}^p(-H)^r\) by \(I+H\) gives \(I-(-H)^{p+1}\); the remaining term tends to zero. Truncation after \(p\) gives

\[
\left\|\delta v-\sum_{r=0}^{p}(-H)^rz\right\|
\le\sum_{r=p+1}^{\infty}\eta^r\|z\|
=\frac{\eta^{p+1}}{1-\eta}\|z\|.
\tag{7.8}
\]

This is a rigorous conditional approximation. Applying it with a newly computed dense \(G^{-1}\) and no structure would not meet our research goal. On a lattice with an explicit inverse kernel, it instead organizes local changes and their repeated interactions.

For a path with an operator-norm expansion \(A(t)=A+tE_1+t^2E_2+o(t^2)\) at an exact fit, the maps \(E_1,E_2:\mathbb R^d\to\mathcal H\) are the first two feature-change coefficients. Define the \(d\times d\) first Gram-change coefficient

\[
G_1=A^*E_1+E_1^*A.
\]

For coefficient vectors \(v_1,v_2\in\mathbb R^d\), substitute \(v(t)=v+t v_1+t^2v_2+o(t^2)\) into (7.6) and match powers:

\[
Gv_1=-A^*E_1v,
\]

\[
Gv_2=-A^*E_2v-E_1^*E_1v-G_1v_1.
\tag{7.9}
\]

These terms identify where pairwise interactions first enter. For moving tents, classical second feature derivatives are distributions; the overlap matrices, rather than an unjustified twice differentiable feature map in \(L^2\), should be used to justify a second-order expansion.

### 7.4 Exact multiple deletion from a known inverse overlap

Suppose \(f=Av^0\) exactly, with \(v^0\in\mathbb R^d\), and deletion requires \(v_j=0\) on an ordered index set \(J\) of size \(k\). Let \(E_J:\mathbb R^k\to\mathbb R^d\) insert a vector at those indices, with zeros elsewhere; its transpose \(E_J^*\) selects the same coordinates. Write \(v_J^0=E_J^*v^0\in\mathbb R^k\), \(Q=G^{-1}\in\mathbb R^{d\times d}\), and \(Q_{JJ}=E_J^*QE_J\in\mathbb R^{k\times k}\). With \(v=v^0+\delta\), \(\delta\in\mathbb R^d\), the problem becomes

\[
\min_\delta\tfrac12\delta^*G\delta
\quad\text{subject to}\quad E_J^*\delta=-v_J^0.
\]

With Lagrange multiplier \(\mu\in\mathbb R^k\), stationarity in coefficient space is \(G\delta+E_J\mu=0\). Thus \(\delta=-QE_J\mu\). Enforcing the constraint gives

\[
E_J^*QE_J\mu=v_J^0.
\]

Consequently

\[
\boxed{\delta=-QE_J(Q_{JJ})^{-1}v_J^0.}
\tag{7.10}
\]

If \(Q_{jk}\) is an explicit function of center separation, a four-deletion prediction needs only that distance law and four coupled amplitudes. Equation (7.10) alone is still an identity; the closed form for \(Q\) in the next section supplies its explanatory content.

## 8 Exact tent examples

### 8.1 The baseline and the fitting problem

Let \(T:\mathbb R\to\mathbb R\) be the peak-one tent \(T(t)=(1-|t|)_+\), where \(z_+=\max(z,0)\). At spacing \(h>0\), use the functions \(\phi_j(x)=T(x/h-j)\in L^2(\mathbb R)\), indexed by all \(j\in\mathbb Z\). Their synthesis map \(A:\ell^2(\mathbb Z)\to L^2(\mathbb R)\) is as in Section 5. On a cell \(x/h\in[j,j+1]\), only two tents are nonzero, with values \(1-(x/h-j)\) and \(x/h-j\). Their sum is one. Therefore

\[
\sum_{j\in\mathbb Z}T(x/h-j)=1.
\tag{8.1}
\]

The constant target \(1\) is not in \(L^2(\mathbb R)\), and the all-one coefficient sequence \(\mathbf1\) is not in \(\ell^2(\mathbb Z)\). The sum in (8.1) is nevertheless defined pointwise because only finitely many tents are nonzero at each \(x\). We fit a finite geometric defect in the affine coefficient space \(\mathbf1+\ell^2(\mathbb Z)\): write \(v_j=1+\delta_j\), with \(\delta\in\ell^2(\mathbb Z)\), and minimize the \(L^2(\mathbb R)\) norm of the **residual from the constant**. The residual belongs to \(L^2\) even though target and output separately need not: the unchanged baseline is exact and only finitely many features change. There is no output bias; a free constant would trivialize this target.

### 8.2 Compute the overlap and its inverse

Set \(h=1\) first. The Gram operator \(G=A^*A:\ell^2(\mathbb Z)\to\ell^2(\mathbb Z)\) measures \(L^2(\mathbb R)\) overlaps of the tents. Direct integration gives

\[
\langle T,T\rangle=2\int_0^1(1-x)^2dx=\frac23,
\]

\[
\langle T,T(\cdot-1)\rangle=\int_0^1x(1-x)dx=\frac16.
\]

Other nonidentical translates have disjoint interiors unless their indices differ by one. Consequently

\[
(Gz)_j=\tfrac16z_{j-1}+\tfrac23z_j+\tfrac16z_{j+1},
\quad \widetilde g(\theta)=\tfrac23+\tfrac13\cos\theta.
\tag{8.2}
\]

The spectrum lies in \([1/3,1]\), so the Gram condition number is three.

Let \(e_0\in\ell^2(\mathbb Z)\) be the sequence equal to one at index zero and zero elsewhere. To find the inverse column \(q=G^{-1}e_0\in\ell^2(\mathbb Z)\), solve \(Gq=e_0\). Away from zero the recurrence is \(q_{n-1}+4q_n+q_{n+1}=0\). A geometric sequence with scalar ratio \(r\) requires

\[
r^2+4r+1=0,
\qquad r=-2\pm\sqrt3.
\]

Only \(r=\sqrt3-2\) has modulus below one. Throughout the tent-response formulas, \(r\) denotes this scalar decay ratio, not the residual function of Section 6. Symmetry and decay give \(q_n=C r^{|n|}\), with scalar amplitude \(C\). At index zero,

\[
\frac23C+\frac13Cr=1
\quad\Longrightarrow\quad
C=\frac3{2+r}=\sqrt3.
\]

Thus

\[
\boxed{(G^{-1})_{jk}=q_{j-k},\qquad q_n=\sqrt3(\sqrt3-2)^{|n|}.}
\tag{8.3}
\]

This is the reusable inverse-overlap response. At spacing \(h\), all Gram integrals acquire a factor \(h\), so \(G_h^{-1}=G_1^{-1}/h\). Dimensionless perturbation formulas below are unchanged because their overlap forces acquire the compensating factor \(h\).

### 8.3 Delete one tent

Set \(v_0=0\), or \(\delta_0=-1\). Minimizing the residual energy gives stationarity at every unchanged index:

\[
\delta_{k-1}+4\delta_k+\delta_{k+1}=0,
\qquad k\ne0.
\]

The decaying solution satisfying \(\delta_0=-1\) is

\[
\boxed{v_k=1-r^{|k|},\qquad r=\sqrt3-2.}
\tag{8.4}
\]

The first four surviving readouts are \(1.267949,0.928203,1.019238,0.994845\). This alternation is entirely a property of inverse overlap. The target is constant, and the Gram condition number is three.

Its error can also be calculated without fitting. Since \(G_h\delta\) vanishes off index zero,

\[
\|A\delta\|^2=\delta^*G_h\delta
=\delta_0(G_h\delta)_0.
\]

At zero, \((G_h\delta)_0=h[-2/3-r/3]=-h/\sqrt3\). Multiplication by \(\delta_0=-1\) gives

\[
\min\|1-\widehat f\|_{L^2(\mathbb R)}^2=\frac h{\sqrt3}.
\tag{8.5}
\]

All surviving standard tents vanish at the deleted center, so exact recovery is impossible. The correction reduces integrated error but cannot fill that point continuously.

### 8.4 Narrow one tent by a finite amount

Replace the center tent by the function \(\phi_\rho(x)=T(x/\rho)\in L^2(\mathbb R)\), with \(0<\rho\le1\), keeping \(h=1\). Let \(B:\ell^2(\mathbb Z\setminus\{0\})\to L^2(\mathbb R)\) synthesize the remaining tents, and let \(P_B=BB^\dagger\) be the orthogonal projector onto their closed output space. Then \(B^\dagger\phi_\rho\) is a square-summable sequence of projection coefficients, while \(u_\rho=(I-P_B)\phi_\rho\) is an \(L^2\) function. Define the positive scalar \(a=2-\sqrt3=-r\). The changed tent has squared norm \(2\rho/3\). Its overlap with either adjacent standard tent is

\[
\int_0^\rho(1-x/\rho)x\,dx
=\left[x^2/2-x^3/(3\rho)\right]_0^\rho
=\rho^2/6.
\tag{8.6}
\]

With the center column removed, the background splits into two semi-infinite tridiagonal blocks. The projection coefficients of the changed tent on the positive block satisfy the homogeneous recurrence for \(k\ge2\). Hence they have form \(z_k=C r^{k-1}\). At \(k=1\),

\[
\tfrac23C+\tfrac16Cr=\rho^2/6
\quad\Longrightarrow\quad
C=\frac{\rho^2}{4+r}=a\rho^2.
\]

The negative side is its reflection. Thus \((B^\dagger\phi_\rho)_k=a\rho^2r^{|k|-1}\) for \(k\ne0\).

Since projection is orthogonal,

\[
\begin{aligned}
\|u_\rho\|^2
&=\|\phi_\rho\|^2-\|P_B\phi_\rho\|^2\\
&=\frac{2\rho}{3}
-2(a\rho^2)(\rho^2/6)
=\frac{2\rho-a\rho^4}{3}.
\end{aligned}
\tag{8.7}
\]

Let \(r_{\rm deletion}=1-\widehat f_{\rm deletion}\in L^2(\mathbb R)\) be the deletion residual from Section 8.3. The deletion fit has readout \(1+a\) at each adjacent center. The changed feature overlaps only those two background tents. Its inner product with that residual is consequently

\[
\langle\phi_\rho,r_{\rm deletion}\rangle
=\int\phi_\rho-2(1+a)(\rho^2/6)
=\rho-\frac{(1+a)\rho^2}{3}.
\tag{8.8}
\]

Here \(\int\phi_\rho=\rho\), the area of a height-one tent of base \(2\rho\). The residual is orthogonal to the background, so \(\langle u_\rho,r_{\rm deletion}\rangle\) has the same value. Dividing (8.8) by (8.7) gives the selected readout

\[
\boxed{\beta(\rho)=\frac{3-(1+a)\rho}{2-a\rho^3}.}
\tag{8.9}
\]

The background coefficients are the deletion coefficients minus \(\beta\) times the projection coefficients:

\[
\begin{aligned}
v_k
&=1-r^{|k|}-\beta a\rho^2r^{|k|-1}\\
&=1+(\beta\rho^2-1)r^{|k|},\qquad k\ne0,
\end{aligned}
\tag{8.10}
\]

where the second line uses \(a=-r\). At \(\rho=1\) all coefficients equal one. At \(\rho=1/2\), the changed readout is \(1.203162\), and the next readouts are \(1.187353,0.949799,1.013451\).

Define \(E(\rho)\ge0\) to be the minimized **squared** \(L^2\) error, a scalar function of width ratio. It is the deletion error minus the squared scalar projection gain:

\[
E(\rho)=h\left[\frac1{\sqrt3}
-\frac{\rho[3-(1+a)\rho]^2}{3(2-a\rho^3)}\right].
\tag{8.11}
\]

As \(\rho\downarrow0\), the height coefficient tends to \(3/2\), but its area and \(L^2\) output norm vanish. The remaining coefficients converge to the deletion pattern. For a unit-area feature, the readout is instead \(\rho h\beta\), which tends to zero. This explicitly separates readout normalization from the represented function.

### 8.5 Widening can have an exact local response

The identity

\[
T(t/2)=\tfrac12T(t+1)+T(t)+\tfrac12T(t-1)
\tag{8.12}
\]

can be verified by matching both sides at integer knots \(-2,-1,0,1,2\); both sides are linear between those knots and vanish outside \([-2,2]\). Replacing the center tent by the left-hand side therefore preserves the constant exactly with selected coefficient one, adjacent coefficients \(1/2\), and all others one.

For an integer \(m\ge1\), the same nodal argument gives

\[
T(t/m)=\sum_{|k|<m}(1-|k|/m)T(t-k).
\tag{8.13}
\]

An exact constant follows with selected weight one, weights \(|k|/m\) for \(0<|k|<m\), and weights one farther away. For noninteger \(\rho\), the changed tent has nonzero slope jumps at off-grid endpoints \(\pm\rho\). Unchanged standard tents have no jumps there. A constant would force its coefficient to zero, returning to the impossible deleted-center fit. This exact-recovery restriction is specific to this tent geometry, not a general width theorem for smooth kernels.

## 9 An explicit prediction for four simultaneous changes

### 9.1 Derive the two universal response sequences

Continue with the spacing-one lattice and synthesis map \(A:\ell^2(\mathbb Z)\to L^2(\mathbb R)\). For scalar displacement \(d\in\mathbb R\) and fractional half-width change \(\eta>-1\), replace the tent at zero by the function

\[
\phi_{d,\eta}(x)=T\!\left(\frac{x-d}{1+\eta}\right),
\]

Both parameters are dimensionless because spacing is one. At the unperturbed geometry, the parameter derivatives are functions \(a,b\in L^1(\mathbb R)\cap L^2(\mathbb R)\), defined almost everywhere by

\[
a(x)=\partial_d\phi_{0,0}(x)=\operatorname{sgn}(x)\mathbf1_{|x|<1},
\qquad
b(x)=\partial_\eta\phi_{0,0}(x)=|x|\mathbf1_{|x|<1}.
\tag{9.1}
\]

For the center derivative, oddness gives \(\langle T,a\rangle=0\). Its right and left neighbor overlaps are

\[
\langle T(\cdot-1),a\rangle=\int_0^1x\,dx=\tfrac12,
\qquad
\langle T(\cdot+1),a\rangle=-\tfrac12.
\]

For width,

\[
\langle T,b\rangle=2\int_0^1x(1-x)dx=\tfrac13,
\qquad
\langle T(\cdot\mp1),b\rangle=\int_0^1x^2dx=\tfrac13.
\]

All other overlaps vanish. At an exact fit, (7.4) says to negate these overlap forces and apply the inverse Gram operator, which convolves sequences with \(q_n=\sqrt3r^{|n|}\). Define the center and width response sequences \(C=(C_n)_{n\in\mathbb Z}\) and \(W=(W_n)_{n\in\mathbb Z}\), both in \(\ell^1(\mathbb Z)\cap\ell^2(\mathbb Z)\), by

\[
C_n=-\tfrac12(q_{n-1}-q_{n+1}),
\qquad
W_n=-\tfrac13(q_{n-1}+q_n+q_{n+1}).
\tag{9.2}
\]

To simplify \(C_n\), take \(n\ge1\), factor \(\sqrt3r^{n-1}\), and use \(r^2+4r+1=0\). This gives \(C_n=3r^n\). Oddness gives the negative side and \(C_0=0\). For \(W\), the inverse recurrence says \(q_{n-1}+4q_n+q_{n+1}=6\delta_{n0}\). Subtract \(3q_n\) from the left and divide by \(-3\):

\[
\boxed{C_n=3\operatorname{sgn}(n)r^{|n|},\qquad
W_n=\sqrt3r^{|n|}-2\delta_{n0},\qquad r=\sqrt3-2.}
\tag{9.3}
\]

For a finite index set \(J\subset\mathbb Z\), assign scalars \(d_j,\eta_j\) to each edited site. The predicted readout sequence lies in \(\mathbf1+\ell^2(\mathbb Z)\) and is

\[
\boxed{v_k^{\rm pred}=1+\sum_{j\in J}[d_jC_{k-j}+\eta_jW_{k-j}].}
\tag{9.4}
\]

No fitting or matrix factorization enters this prediction. Each change has a known symmetry, sign pattern, and geometric decay rate. A center shift has an odd response, a width change has an even response, and both decay by \(|r|\approx0.268\) per additional site.

If gamma changes by a scalar fraction \(g>-1\), the new width ratio is \(1/(1+g)\), so \(\eta=-g+O(g^2)\). The first-order gamma response is therefore the sequence \(-W\). For an exactly represented nonconstant target, multiply each site's contribution in (9.4) by its baseline coefficient \(v_j^0\). An imperfect initial fit adds the residual term in (7.3).

### 9.2 Why the coefficient remainder is second order

Moving a tent kink creates an \(L^2\) feature remainder of order \(|d|^{3/2}\); one must not justify (9.4) by pretending that tents have bounded ordinary second derivatives. The overlap averages supply a stronger coefficient result.

All sums over edited sites in this subsection are over the finite set \(J\). Put \(\tau_j=|d_j|+|\eta_j|\), and define the scalar aggregate sizes \(s=(\sum_{j\in J}\tau_j^2)^{1/2}\) and \(t=\sum_{j\in J}\tau_j\). Assume all half-width ratios are at least \(1/2\). Define the translated derivative functions \(a_j(x)=a(x-j)\) and \(b_j(x)=b(x-j)\). The feature remainder at site \(j\) is the function \(R_j\in L^1(\mathbb R)\cap L^2(\mathbb R)\):

\[
R_j(x)=\phi_{d_j,\eta_j}(x-j)-T(x-j)-d_ja_j(x)-\eta_jb_j(x).
\]

The following single-feature estimates use coordinates relative to its original center; translation leaves the norms unchanged. For a real location \(y\), let \(\delta_y\) denote the unit point-mass measure, defined by \(\int g\,d\delta_y=g(y)\) for continuous test functions \(g\). This is distinct from the Kronecker delta between sequence indices. At a fixed scalar width \(\rho>0\), write \(\phi_\rho(x)=T(x/\rho)\). Its distributional second spatial derivative is the finite signed measure

\[
\partial_x^2\phi_\rho=\frac1\rho(\delta_{-\rho}-2\delta_0+\delta_\rho),
\]

whose total variation is \(4/\rho\). For the width derivative,

\[
\partial_\rho\phi_\rho=\frac{|x|}{\rho^2}\mathbf1_{|x|<\rho}.
\]

Differentiating its amplitude and its moving endpoints gives the signed measure

\[
\partial_\rho^2\phi_\rho
=-\frac{2|x|}{\rho^3}\mathbf1_{|x|<\rho}\,dx
+\frac1\rho(\delta_{-\rho}+\delta_\rho).
\]

Its total variation is

\[
\frac2{\rho^3}\int_{-\rho}^{\rho}|x|dx+\frac2\rho
=\frac{2\rho^2}{\rho^3}+\frac2\rho=\frac4\rho.
\]

Integrating the spatial second derivative twice in a translation increment gives an \(L^1\) remainder at most \((4/\rho)d^2/2\le4d^2\). Integrating the width second derivative along the path from width one gives a width remainder at most \(4\eta^2\), because the path stays at \(\rho\ge1/2\). The center derivative at width \(\rho\) is \(\rho^{-1}\operatorname{sgn}(x)\mathbf1_{|x|<\rho}\). Directly integrating its difference from the width-one version bounds its \(L^1\) change by \(4|\eta|\). Multiplication by \(|d|\) bounds the mixed term. Therefore

\[
\|R_j\|_1\le4d_j^2+4\eta_j^2+4|d_j\eta_j|\le4\tau_j^2.
\tag{9.5}
\]

The baseline tents are nonnegative and sum to one. The adjoint \(A^*:L^2(\mathbb R)\to\ell^2(\mathbb Z)\) turns the sum of remainder functions into a correlation sequence. In the following estimate, the norms of that sequence are \(\ell^2\) and \(\ell^1\) norms, while the integral is a function \(L^1\) norm:

\[
\begin{aligned}
\left\|A^*\sum_jR_j\right\|_2
&\le\left\|A^*\sum_jR_j\right\|_1\\
&\le\int\left|\sum_jR_j(x)\right|\sum_kT(x-k)\,dx
\le4s^2.
\end{aligned}
\tag{9.6}
\]

Let \(E:\ell^2(\mathbb Z)\to L^2(\mathbb R)\) be the finite-column feature change, so the new synthesis map is \(A+E\). Its column \(E_j\) is the edited-minus-original feature at site \(j\), and is zero at unchanged sites. Define the output perturbation \(e_\Delta=\sum_{j\in J}E_j\in L^2(\mathbb R)\), abbreviated \(E\mathbf1\), and its linear approximation \(D=\sum_{j\in J}(d_ja_j+\eta_jb_j)\in L^2(\mathbb R)\). Although \(\mathbf1\notin\ell^2\), this particular \(E\mathbf1\) is well defined by the finite sum. The Gram change \(\Delta G=A^*E+E^*A+E^*E\) acts on \(\ell^2(\mathbb Z)\).

Along widths at least \(1/2\), the center derivative has \(L^2\) norm \(\sqrt{2/\rho}\le2\), and the width derivative has norm \(\sqrt{2/(3\rho)}\le2\). Integrating those derivatives gives \(\|E_j\|_2\le2\tau_j\). Cauchy–Schwarz over changed columns and the triangle inequality yield

\[
\|E\|\le2s,\qquad\|e_\Delta\|_2\le2t.
\]

Since \(\|A\|=1\),

\[
\|\Delta G\|\le2\|A\|\|E\|+\|E\|^2\le4s+4s^2.
\]

For any \(z\in\ell^2\),

\[
\|(A+E)z\|\ge\|Az\|-\|Ez\|
\ge(1/\sqrt3-2s)\|z\|.
\]

Thus if \(2s<1/\sqrt3\), the changed Gram inverse has norm at most \((1/\sqrt3-2s)^{-2}\).

Let \(\delta\in\ell^2(\mathbb Z)\) be the exact coefficient change and \(\delta_1=v^{\rm pred}-\mathbf1\in\ell^2(\mathbb Z)\) its first-order prediction. Their equations are

\[
(G+\Delta G)\delta=-(A+E)^*e_\Delta,
\qquad G\delta_1=-A^*D.
\]

Subtract and collect terms:

\[
(G+\Delta G)(\delta-\delta_1)
=-A^*(e_\Delta-D)-E^*e_\Delta-\Delta G\delta_1.
\]

Applying the bounds already proved gives

\[
\boxed{\|\delta-\delta_1\|_2
\le\frac{4s^2+4st+(4s+4s^2)\|\delta_1\|_2}
{(1/\sqrt3-2s)^2}.}
\tag{9.7}
\]

Finally, \(\|A^\dagger\|=\sqrt3\), while \(\|a\|_2=\sqrt2\) and \(\|b\|_2=\sqrt{2/3}\), so

\[
\|\delta_1\|_2=\|A^\dagger D\|_2\le\sqrt6\,t.
\]

For a fixed number \(k\) of changed neurons, \(t\le\sqrt{k}s\). The numerator of (9.7) is therefore \(O(s^2)\), while its denominator stays positive for sufficiently small changes. This proves a second-order coefficient remainder. The explicit bound is conservative; its value is certification of the order and a stated validity condition, not a sharp estimate for every example.

### 9.3 A controlled four-neuron check

The new validation uses the following fixed perturbation direction in grid units:

| Original center | Center displacement \(d\) | Half-width change \(\eta\) |
|---:|---:|---:|
| \(-3\) | \(0.04\) | \(0.03\) |
| \(-1\) | \(-0.03\) | \(0.05\) |
| \(1\) | \(0.02\) | \(-0.04\) |
| \(3\) | \(-0.04\) | \(-0.02\) |

All eight changes are multiplied by a common scale \(\varepsilon\). Prediction (9.4) is evaluated before an independent least-squares reference. No coefficient of that reference is fed into the prediction.

The reference uses 61 baseline tents centered at integers \(-30\) through \(30\), with target equal to their all-one sum: a constant on the interior and tapered on the two outer cells. It integrates over the full support and splits quadrature at every original and changed corner. Three-point Gaussian quadrature exactly integrates the piecewise-quadratic products in exact arithmetic. The finite boundary is far from the defects; the infinite-lattice formula is the predictor being checked, not an asserted exact finite-boundary identity.

| Perturbation scale | Coefficient error \(\|v^{\rm pred}-v^{\rm LS}\|_2\) | Error relative to the actual coefficient change | Maximum absolute coefficient error |
|---:|---:|---:|---:|
| 1 | 0.0085672 | 7.49% | 0.0060828 |
| 1/2 | 0.0022035 | 3.92% | 0.0015518 |
| 1/4 | 0.0005596 | 2.01% | 0.0003923 |
| 1/8 | 0.0001411 | 1.02% | 0.0000986 |

The absolute error drops approximately fourfold when perturbations are halved, as the proved order predicts. The relative error is measured against the **change**, not against the much larger all-one baseline. Increasing quadrature order from three to five changes reference coefficients by less than \(5.0\times10^{-15}\).

![Analytic four-defect coefficient prediction and error scaling](../../results/checkpoint_G_interactive/geometry_reader_codex/single_gamma_theory_20260929/checkpoint_validation/multiple_tent_prediction.png)

This is a small, successful theoretical prediction for a controlled activation and geometry. It does not establish equally accurate finite-amplitude predictions for four arbitrary smooth-kernel defects. That extension is a research direction.

## 10 Strongly overlapping tents and exact redundancy

### 10.1 Eight interleaved constant representations

Switch to a circle of length \(L=8M\), with integer \(M\ge4\), observation space \(\mathcal H=L^2(\mathbb R/L\mathbb Z)\), and no output bias. There are \(8M\) distinct integer centers \(j=0,\ldots,8M-1\); coefficients live in \(\mathbb R^{8M}\), with indices interpreted modulo \(8M\). Each feature is the periodization of \(T((x-j)/8)\), so \(A:\mathbb R^{8M}\to\mathcal H\) has a finite-dimensional coefficient domain. The constant target belongs to this periodic \(\mathcal H\). For each residue \(r\in\{0,\ldots,7\}\), substituting \(u=(x-r)/8\) into (8.1) gives

\[
\sum_{j\equiv r\ ({\rm mod}\ 8)}T((x-j)/8)=1,
\tag{10.1}
\]

with periodic images understood. Each of eight subgrids independently represents the constant.

The nullspace \(\ker A\subset\mathbb R^{8M}\) has exactly seven independent directions. To see that there are no others, differentiate the synthesized piecewise-linear function twice as a distribution on the circle; point masses below are interpreted periodically. Since

\[
\frac{d^2}{dx^2}T((x-j)/8)
=\frac18[\delta_{j-8}-2\delta_j+\delta_{j+8}],
\]

zero output requires \(v_{j-8}-2v_j+v_{j+8}=0\). On each finite periodic residue class, this recurrence forces the sequence to be constant. A vector with residue-class values \(a_0,\ldots,a_7\) synthesizes the constant \(\sum_r a_r\) by (10.1). It is a null vector precisely when that sum is zero, a seven-dimensional condition.

An exact constant therefore has a class-value vector \((a_0,\ldots,a_7)\in\mathbb R^8\), with \(\sum_r a_r=1\), determining all \(8M\) readouts. Its squared Euclidean coefficient norm is \(M\sum_r a_r^2\). Minimizing it gives \(a_r=1/8\) for all classes, either by a Lagrange multiplier or by Cauchy–Schwarz.

### 10.2 Delete one neuron

Removing center zero requires its coefficient to be zero. The recurrence above forces the entire residue class zero to have value zero in an exact constant representation. The remaining seven values must sum to one. Their norm is minimized when each equals \(1/7\). Thus the exact minimum-norm solution is

\[
v_j=\begin{cases}0,&j\equiv0\pmod8,\\1/7,&\text{otherwise}.\end{cases}
\tag{10.2}
\]

This is a global compensation pattern in an exactly redundant periodic space. It is not the square-summable defect problem of Section 8: a nonzero periodic coefficient change on the infinite line is not in \(\ell^2\).

### 10.3 Halve or double the selected width

Halving the selected half-width from eight to four introduces slope jumps at \(\pm4\). To prove that its coefficient must vanish in an exact constant fit, consider the old span in second-derivative coordinates. The sum of second-derivative coefficients in each residue class modulo eight is zero, because the old difference operator acts within each class and telescopes. The new width-four tent has coefficient \(-2/4\) at residue zero and a total \(2/4\) at residue four. These class sums are nonzero. It cannot belong to the old span. Since the unchanged old neurons already represent the constant exactly by (10.2), any nonzero coefficient on this distinct component would spoil exactness. Its optimal coefficient is zero and the minimum-norm background is (10.2).

Doubling the selected half-width gives

\[
T(x/16)=T(x/8)+\tfrac12T((x-8)/8)+\tfrac12T((x+8)/8).
\tag{10.3}
\]

Let its coefficient be the scalar \(\beta\in\mathbb R\). In the equivalent old representation, the residue-zero class must be constant with value \(\beta\), because the old coefficient at center zero has been replaced by the new tent. The actual coefficients at \(\pm8\) are \(\beta/2\), while the other surviving sites in that class have value \(\beta\). By symmetry, the other seven classes take a common scalar value \(a=(1-\beta)/7\).

The squared Euclidean norm in the new feature coordinates is the scalar objective \(J:\mathbb R\to\mathbb R\):

\[
J(\beta)=(M-\tfrac32)\beta^2+7M a^2
=(M-\tfrac32)\beta^2+\frac M7(1-\beta)^2.
\]

Differentiating gives

\[
2(M-\tfrac32)\beta-\frac{2M}{7}(1-\beta)=0,
\]

so

\[
\boxed{\beta=\frac{M}{8M-10.5}.}
\tag{10.4}
\]

At \(L=128\), \(M=16\), this is \(0.1361702\). Unlike the deletion result, it changes with period length because the minimum-norm objective counts a different number of repeated cells.

### 10.4 What the nearby-width control establishes

Half-width \(8.25\) removes the special exact residue-class identity in the tested periodic grids. The baseline constant itself then has a small ripple. At period 128, relative errors were approximately \(0.0011855\) for the baseline and \(0.0012333\) after deletion. The repeated zero/one-seventh pattern becomes modulated. These are finite periodic observations; not all perturbed coefficients were converged under period doubling. Broad overlap makes compensation possible, but the exact pattern depends on spectral zeros, grid alignment, and the coefficient convention.

## 11 ReLU coefficients as curvature

### 11.1 The exact representation

Return to a finite interval \([a,b]\). For a center \(c\in(a,b)\), the hinge \(r_c:[a,b]\to\mathbb R\), \(r_c(x)=(x-c)_+\), belongs to \(L^2([a,b])\) and has derivative \(\mathbf1_{x>c}\) almost everywhere. Its second derivative is the point-mass measure \(\delta_c\), not an \(L^2\) function. For \(n\) hinges, let \(w=(w_1,\ldots,w_n)\in\mathbb R^n\) be their effective readouts and \(b_0,b_1\in\mathbb R\) the affine baseline coefficients. Then

\[
\widehat f(x)=b_0+b_1x+\sum_j w_jr_{c_j}(x)
\quad\Longrightarrow\quad
\widehat f''=\sum_jw_j\delta_{c_j}.
\tag{11.1}
\]

Each \(w_j\) is exactly a jump in slope. It is a discrete curvature mass, not the network's pointwise second derivative. Between knots the pointwise second derivative is zero.

For a target \(f\in C^2([a,b])\), the fundamental theorem of calculus gives

\[
f'(x)=f'(a)+\int_a^x f''(c)dc.
\]

Integrate from \(a\) to \(x\), then reverse the order over \(a\le c\le t\le x\):

\[
\begin{aligned}
f(x)&=f(a)+f'(a)(x-a)+\int_a^x\int_a^t f''(c)dc\,dt\\
&=f(a)+f'(a)(x-a)+\int_a^x(x-c)f''(c)dc\\
&=f(a)+f'(a)(x-a)+\int_a^b(x-c)_+f''(c)dc.
\end{aligned}
\tag{11.2}
\]

Replacing the last integral by a uniform quadrature sum motivates \(w_j\approx hf''(c_j)\).

### 11.2 Exact finite-knot readouts

Choose consecutive knots \(a=x_0<\cdots<x_q=b\), with integer \(q\ge2\). Their fitted nodal values form a vector \(s=(s_0,\ldots,s_q)\in\mathbb R^{q+1}\), where \(s_j=\widehat f(x_j)\). These values determine a continuous piecewise-linear function. Its interior slope jumps form a vector \(w\in\mathbb R^{q-1}\), with hinge centers \(c_j=x_j\), \(1\le j<q\). For uniform spacing \(h\), the slopes adjacent to an interior knot are \((s_j-s_{j-1})/h\) and \((s_{j+1}-s_j)/h\). Their difference is

\[
\boxed{w_j=\frac{s_{j+1}-2s_j+s_{j-1}}h.}
\tag{11.3}
\]

For positive nonuniform gaps \(h_L=x_j-x_{j-1}\) and \(h_R=x_{j+1}-x_j\), replace it by

\[
w_j=\frac{s_{j+1}-s_j}{h_R}-\frac{s_j-s_{j-1}}{h_L}.
\tag{11.4}
\]

These formulas use fitted values, not necessarily target samples. Substituting target samples gives interpolation, generally a different optimization problem from function-value least squares.

For positive gamma,

\[
\operatorname{ReLU}(\gamma(x-c))=\gamma(x-c)_+.
\]

Therefore \(w_j=\gamma_jv_j\). Changing gamma while fixing the center does not change the span. In an identifiable unregularized fit, its readout changes inversely and all effective slope jumps remain unchanged. A penalty on raw readouts can break that invariance.

### 11.3 Prove the quadratic example and its nonzero error

Fit \(f(x)=x^2\) in \(L^2([-1,1])\) using the finite-dimensional subspace \(S_h\) of continuous piecewise-linear functions on uniform knots with spacing \(h\), including both endpoints. On a cell starting at \(a\), write \(x=a+t\), \(0\le t\le h\). The linear interpolant of \(x^2\) exceeds \(x^2\) by \(t(h-t)\), as direct expansion shows:

\[
[a^2+(2a+h)t]-(a+t)^2=ht-t^2.
\]

Propose fitted nodal values \(s_j=x_j^2-h^2/6\). Their spline residual, fitted minus target, is

\[
e(t)=t(h-t)-h^2/6.
\]

Its integral on a cell is \(h^3/6-h^3/6=0\). Also \(e(t)=e(h-t)\), so \(\int_0^h te(t)dt=(h/2)\int_0^he(t)dt=0\). Every nodal hat function is linear on each adjacent cell, so the residual is orthogonal to every hat. These are exactly the normal equations; hence the proposed spline is the unique least-squares fit.

Its coefficient at an interior knot is

\[
w_j=\frac{(x_j+h)^2-2x_j^2+(x_j-h)^2}{h}=2h.
\tag{11.5}
\]

Nevertheless it is not an exact quadratic. Scale \(t=hu\):

\[
\int_0^h e(t)^2dt
=h^5\int_0^1[u(1-u)-1/6]^2du.
\]

Expanding the square gives \(u^4-2u^3+\tfrac43u^2-\tfrac13u+\tfrac1{36}\). Its integral is

\[
\frac15-\frac12+\frac49-\frac16+\frac1{36}=\frac1{180}.
\]

There are \(2/h\) cells, so the squared error is \(h^4/90\). The target squared norm is \(\int_{-1}^1x^4dx=2/5\). Therefore

\[
\boxed{\frac{\|f-\widehat f\|_2}{\|f\|_2}
=\sqrt{\frac{h^4/90}{2/5}}=\frac{h^2}{6}.}
\tag{11.6}
\]

This is an exact precision limit for this target at this fixed uniform knot spacing, independent of optimizer or positive gamma. Increasing the number of knots lowers it quadratically. It is not an absolute precision ceiling for ReLU or a statement about deep ReLU networks.

For any \(C^2\) target, the interpolation error on a cell is bounded by \(\|f''\|_\infty h^2/8\): the remainder is \(f''(\xi)(x-a)(x-a-h)/2\), and \(t(h-t)\le h^2/4\). Least squares cannot have larger integrated error than this admissible interpolant. This supplies an \(O(h^2)\) upper bound; the quadratic proves that the order cannot generally be improved for this spline space.

## 12 Spectral criteria for activation selection

### 12.1 Define what it means to work well

For the present theory, an activation and geometry work well for a target class when they provide:

1. enough representational freedom for that class;
2. small approximation error at the specified neuron count;
3. coefficient recovery and evaluation that are sufficiently stable at the working precision;
4. interpretable or predictable responses under the chosen geometry changes.

These are different criteria. Accurate functions can have unstable readouts. A stable low-order spline can need many knots for high precision. Small raw coefficient norm is not an invariant quality measure because multiplying every feature by a constant rescales all readouts inversely. Comparisons must specify normalization, fit measure, baseline terms, domain, halo, and solver convention.

### 12.2 Exact span obstructions

Let \(p\ge0\) be an integer degree and \(\mathcal P_p\) the \((p+1)\)-dimensional space of real polynomials of degree at most \(p\), restricted to the fitting interval. Here \(p\) is a degree, not the baseline function in (1.1). For a polynomial activation \(\sigma(z)=\sum_{r=0}^p a_rz^r\), with scalar coefficients \(a_r\in\mathbb R\), expand a feature:

\[
\sigma(\gamma(x-c))
=\sum_{r=0}^pa_r\gamma^r\sum_{q=0}^r\binom rq x^q(-c)^{r-q}.
\]

Every feature belongs to \(\mathcal P_p\). So does every one-layer linear combination, however many centers and gammas are used. Without a larger independent baseline, the output subspace has dimension at most \(p+1\). This is a genuine failure mode for approximating arbitrary targets with increasing width. It does not prevent excellent performance on low-degree polynomial targets or make a statement about growing depth.

For sine with a common fixed frequency,

\[
\sin(\gamma(x-c))=\cos(\gamma c)\sin(\gamma x)-\sin(\gamma c)\cos(\gamma x).
\tag{12.1}
\]

Translations explore a space of dimension at most two. More centers cannot represent new frequencies. Distinct or trainable frequencies remove this particular restriction; deeper sinusoidal networks also fall outside it. Successful periodic-activation networks therefore do not contradict (12.1). The SIREN paper is an empirical example of successful periodic activations in a different, trainable multilayer setting [R5].

### 12.3 Derive the alias constraint before choosing an activation

Use the dimensionless spatial coordinate \(u=x/h\), common inverse width \(\gamma>0\), and \(\lambda=\gamma h\). Let \(m\ge0\) be an integer derivative order and \(K=\sigma^{(m)}\in L^1(\mathbb R)\) the scalar kernel. Write \(q\in\mathbb R\) for frequency conjugate to \(u\), so Fourier transformation uses \(e^{-iqu}\). Differentiating the scalar feature \(\sigma(\lambda u)\) \(m\) times gives \(\lambda^mK(\lambda u)\). Its Fourier transform is

\[
\lambda^{m-1}\widehat K(q/\lambda).
\]

At a nonzero frequency \(q\), differentiating multiplies the original generator spectrum by \((iq)^m\). Define the scalar spectral profile \(W_m:\mathbb R\setminus\{0\}\to\mathbb C\) by

\[
W_m(q)=\frac{\lambda^{m-1}\widehat K(q/\lambda)}{(iq)^m}.
\tag{12.2}
\]

This does not assert that a raw tanh or ramp feature lies in whole-line \(L^2\). For a precise setting, take a circle of positive integer length \(P\) in grid units, with centers \(j=0,\ldots,P-1\), coefficient vectors in \(\mathbb C^P\), and observation space \(\mathcal H=L^2(\mathbb R/P\mathbb Z;\mathbb C)\). Complex coefficients make the Fourier calculation convenient; real functions are recovered by conjugate symmetry. Use the normalized inner product \(P^{-1}\int_0^P\overline{g_1(u)}g_2(u)\,du\), so the Fourier-coordinate norm is the sum of squared amplitudes.

Assume the periodic generator belongs to this \(\mathcal H\) and its alias sequences are square-summable; mere integrability of \(K\) is not sufficient for this at order zero. The listed kernels satisfy the needed decay. Periodize the kernel, remove its zero mode before integrating when \(m>0\), and define its periodic \(m\)-fold primitive by nonzero Fourier coefficients proportional to (12.2). Include a separately fitted scalar constant in addition to the \(P\) translate coefficients. Choose an integer \(\ell\) giving a nonzero target frequency \(\theta=2\pi\ell/P\in(-\pi,\pi)\). A finite interval with halos is a separate approximation to this translation-invariant setting.

A translate by integer \(j\) multiplies a frequency \(q\) by \(e^{-iqj}\). Frequencies \(\theta+2\pi k\) have the same factor because \(e^{-i2\pi kj}=1\). Therefore all coefficients can choose only one common amplitude for the family

\[
\{\theta+2\pi k:k\in\mathbb Z\}.
\]

Assume \(W_m(\theta)\ne0\). Let \(y\in\mathbb C\) be the output amplitude at the desired tone, and define the scalar alias ratios \(t_k\in\mathbb C\), indexed by \(k\in\mathbb Z\), by

\[
y t_k,\qquad
t_k=\frac{W_m(\theta+2\pi k)}{W_m(\theta)}
=\frac{\widehat K((\theta+2\pi k)/\lambda)}{\widehat K(\theta/\lambda)}
\left(\frac{\theta}{\theta+2\pi k}\right)^m.
\tag{12.3}
\]

Thus the amplitude at alias \(\theta+2\pi k\) is \(yt_k\), with \(t_0=1\). For target amplitude one at the desired frequency and zero at its aliases, define the nonnegative scalar \(S=\sum_{k\ne0}|t_k|^2\). Fourier orthogonality reduces normalized squared error to

\[
|1-y|^2+\sum_{k\ne0}|yt_k|^2
=|1-y|^2+S|y|^2,
\qquad S=\sum_{k\ne0}|t_k|^2.
\]

Completing the square, including complex amplitudes, gives

\[
|1-y|^2+S|y|^2
=(1+S)\left|y-\frac1{1+S}\right|^2+\frac S{1+S}.
\]

Hence

\[
\boxed{y_*=(1+S)^{-1},\qquad E_{\rm alias}=\sqrt{\frac S{1+S}}.}
\tag{12.4}
\]

This is an exact approximation result in the stated periodic space, not a finite-precision error bound. Real sine and cosine combine conjugate fibers and give the same relative error for a symmetric real generator.

For a general target, its amplitudes in this frequency class form a sequence \(F=(F_k)_{k\in\mathbb Z}\in\ell^2(\mathbb Z;\mathbb C)\). Define the generator sequence \(W=(W_m(\theta+2\pi k))_{k\in\mathbb Z}\in\ell^2(\mathbb Z;\mathbb C)\). The symbol \(W\) here is a sequence, not the sample-weight matrix of Section 3.4. This frequency class, also called a fiber, lives in sequence space, and the attainable amplitudes form the one-dimensional subspace \(\{cW:c\in\mathbb C\}\). Minimize \(\|F-cW\|_{\ell^2}^2\) over the scalar \(c\in\mathbb C\), and let \(E_{\rm fiber}^2\) be its minimum. Expanding and completing the square gives

\[
c_* =\frac{\sum_k\overline{W_k}F_k}{\sum_k|W_k|^2},
\qquad
E_{\rm fiber}^2=\sum_k|F_k|^2
-\frac{|\sum_k\overline{W_k}F_k|^2}{\sum_k|W_k|^2}.
\tag{12.5}
\]

If every component of \(W\) is zero, the formula with a denominator is replaced by \(c_*=0\) and \(E_{\rm fiber}^2=\sum_k|F_k|^2\): the generator provides no direction in that fiber.

Sum (12.5) over the nonzero periodic frequency classes. The zero residue class \(\{2\pi k\}\) needs separate treatment because the free constant adds an independent direction at DC: fit the target DC exactly with the baseline, then project the remaining components \(k\ne0\) onto the generator's corresponding nonzero components. Equivalently residualize the generator against the constant before applying the one-dimensional projection there. If the remaining generator vector is zero, all remaining target energy in that class is residual. On a circle the zero class cannot be dropped as a measure-zero set.

In an admissible whole-line \(L^2\) shift-invariant space, partition the Fourier axis into \([-\pi,\pi)+2\pi k\) and integrate the same nonzero-frequency expression over \(\theta\), with Parseval factor \(1/(2\pi)\); the single point \(\theta=0\) does not affect that integral. For a target with only the principal component in each fiber, the full denominator is \(|W_0|^2+\sum_{k\ne0}|W_k|^2\). Replacing it by \(|W_0|^2\) is a **principal-band-dominance approximation**, not an equality.

Two distinct spectral obstructions now have precise meanings. For a nonzero target frequency class, if the principal component \(W_0\) vanishes but some aliases do not, the corresponding pure tone is orthogonal to the fiber and has relative error one. For a zero eigenvalue of the **translate-only** Gram matrix on a periodic grid, all components of that generator fiber must vanish. An appended constant can create additional redundancy if it is already in the translated-feature span. On the infinite line an isolated zero needs the stability qualification in Section 5.2.

### 12.4 The bandwidth tradeoff

For low grid frequency \(|\theta|\), with a slowly varying principal kernel spectrum, dominant first aliases, and \(S\ll1\), (12.3)–(12.4) give the approximation

\[
E_{\rm alias}\approx\sqrt2\,
\frac{|\widehat K(2\pi/\lambda)|}{|\widehat K(0)|}
\left(\frac{|\theta|}{2\pi}\right)^m.
\tag{12.6}
\]

All three qualifications matter. Higher aliases contribute to the exact expression even in the limit \(\theta\to0\); dropping them requires their tails to be negligible. At fixed nonzero target frequency and decreasing \(h\), \(\theta=\omega h\) tends to zero. The integration factor distinguishes function approximation by tanh or GELU from directly fitting their derivative kernels.

For kernels with \(\widehat K(0)\ne0\), define two nonnegative scalar diagnostics of the dimensionless bandwidth \(\lambda>0\):

\[
\mathcal A_K(\lambda)=\frac{|\widehat K(2\pi/\lambda)|}{|\widehat K(0)|},
\qquad
\mathcal B_K(\lambda)=\frac{|\widehat K(\pi/\lambda)|}{|\widehat K(0)|}.
\]

The first probes the first grid aliases; the second probes a frequency relevant to alternating coefficients near the coefficient-grid edge. For common smooth kernels, decreasing lambda broadens kernels relative to spacing. It suppresses aliases but also weakens some coefficient directions. The exact stability object is the full Gram symbol (5.3), not a single transform value. \(\mathcal B_K\) is a useful proxy under the usual positive/unimodal-spectrum and tail-dominance assumptions, not a universal condition-number formula.

Setting \(\mathcal A_K(\lambda)\) near working precision gives a conservative activation-dependent bandwidth scale for sufficiently resolved targets. It does not determine the exact minimizing lambda: target frequency, halo, numerical rank, and arithmetic affect that location. The repository's hardened evidence supports the aliasing wall much more strongly than a universal numerical-floor model.

### 12.5 Kernel spectra and the algebra behind them

These transforms map integrable scalar functions on \(\mathbb R\) to complex-valued frequency functions, using \(\widehat K(\omega)=\int_{\mathbb R}K(x)e^{-i\omega x}dx\), \(\omega\in\mathbb R\).

For a Gaussian \(g(x)=e^{-x^2}\), differentiating its transform under the integral and using \(xg=-g'/2\) gives

\[
\widehat g'(\omega)=\int(-ix)g(x)e^{-i\omega x}dx
=\frac i2\int g'(x)e^{-i\omega x}dx
=-\frac\omega2\widehat g(\omega).
\]

The boundary term vanishes in the integration by parts. With \(\widehat g(0)=\sqrt\pi\), solving this scalar differential equation gives \(\widehat g(\omega)=\sqrt\pi e^{-\omega^2/4}\). Rescaling to the standard normal density gives \(\widehat\varphi=e^{-\omega^2/2}\).

For GELU's curvature kernel, \(\widehat{x^2\varphi}=-\partial_\omega^2\widehat\varphi=(1-\omega^2)e^{-\omega^2/2}\). Therefore

\[
\widehat{(2-x^2)\varphi}(\omega)
=2e^{-\omega^2/2}-(1-\omega^2)e^{-\omega^2/2}
=(1+\omega^2)e^{-\omega^2/2}>0.
\tag{12.7}
\]

Let \(B(z,w)\) denote Euler's scalar beta function of two complex arguments with positive real parts, and \(\Gamma(z)\) Euler's scalar gamma function; \(B\) in this paragraph is not a synthesis operator. For \(K(x)=\operatorname{sech}^2x\), substitute \(u=e^{2x}\): \(K=4u/(1+u)^2\), \(dx=du/(2u)\). Thus

\[
\widehat K(\omega)=2\int_0^\infty\frac{u^{-i\omega/2}}{(1+u)^2}du
=2B(1-i\omega/2,1+i\omega/2).
\]

Using the standard beta identity \(B(z,w)=\Gamma(z)\Gamma(w)/\Gamma(z+w)\), followed by \(\Gamma(1+iy)\Gamma(1-iy)=\pi y/\sinh(\pi y)\), yields

\[
\widehat K(\omega)=\frac{\pi\omega}{\sinh(\pi\omega/2)}.
\tag{12.8}
\]

The displayed gamma identity follows from \(\Gamma(1+iy)=iy\Gamma(iy)\) and Euler's reflection formula \(\Gamma(z)\Gamma(1-z)=\pi/\sin(\pi z)\). These beta/gamma integral identities are the standard special-function inputs to this transform calculation; no unshown network approximation is used.

The sigmoid derivative is \(\tfrac14\operatorname{sech}^2(x/2)\), so substitution \(x=2t\) gives the scalar transform \(H:\mathbb R\to\mathbb R\), \(H(\omega)=\pi\omega/\sinh(\pi\omega)\), extended continuously at zero.

For Swish \(s(x)=x\sigma(x)\), with \(\sigma\) the logistic sigmoid, \(s''=2\sigma'+x\sigma''\). With \(H=\widehat{\sigma'}\) as above, the derivative and multiplication rules give

\[
\widehat{s''}=2H+i\partial_\omega(i\omega H)=H-\omega H'.
\]

Differentiating the explicit \(H\) and canceling the \(\pi\omega/\sinh\) terms yields

\[
\widehat{s''}(\omega)=\frac{\pi^2\omega^2\cosh(\pi\omega)}{\sinh^2(\pi\omega)}.
\tag{12.9}
\]

All expressions at zero are interpreted by their continuous limits. The resulting normalized spectra are

| Activation | Kernel order | Kernel transform divided by its mass |
|---|---:|---|
| tanh or raw sech-squared kernel | 1 or 0 | \((\pi\omega/2)/\sinh(\pi\omega/2)\) |
| sigmoid | 1 | \(\pi\omega/\sinh(\pi\omega)\) |
| raw Gaussian \(e^{-x^2}\) | 0 | \(e^{-\omega^2/4}\) |
| GELU | 2 | \((1+\omega^2)e^{-\omega^2/2}\) |
| Swish | 2 | \(\pi^2\omega^2\cosh(\pi\omega)/\sinh^2(\pi\omega)\) |

The tent supplies a useful contrast. Define the scalar indicator function \(B\in L^1(\mathbb R)\cap L^2(\mathbb R)\) by \(B(x)=\mathbf1_{[-1/2,1/2]}(x)\). Its self-convolution is \(T\), because the overlap length of the two unit intervals is \((1-|x|)_+\). Direct integration gives \(\widehat B=2\sin(\omega/2)/\omega\), so

\[
\widehat T(\omega)=\left[\frac{\sin(\omega/2)}{\omega/2}\right]^2.
\tag{12.10}
\]

Its zeros at nonzero multiples of \(2\pi\) explain the exact partition of unity: the nonconstant Fourier coefficients of the sum of integer translates vanish. Broad integer-scaled tents also place zeros at particular coefficient-grid frequencies, giving the residue-class redundancy proved directly in Section 10. Their algebraic tails and exact zeros are different properties from Gaussian or tanh spectral decay.

### 12.6 What the criteria predict and what they do not

* **Resolved smooth targets:** rapidly decaying alias tails can support very accurate fits with moderate widths, provided the target band is not suppressed too strongly and boundaries are handled. Gaussian-type and tanh-type kernels satisfy this in suitable regimes; it is not an unconditional ranking between them.
* **Rough targets:** significant high-frequency content can demand strong deconvolution, large compensating coefficients, or more neurons. A failed local readout law is not equivalent to failed function approximation. A knot placed at an actual kink can make a piecewise-linear model particularly efficient.
* **Too little shape diversity:** low-degree polynomials or translations of one fixed-frequency sine cannot acquire arbitrary approximation power merely by adding centers. These are exact span restrictions.
* **Strong overlap:** output redundancy can help compensate missing neurons while making coefficients nonunique or sensitive. A smooth readout is not guaranteed, and coefficient stability is distinct from output accuracy.
* **Signed kernels:** spatial sign changes alone do not predict failure. GELU provides an explicit counterexample: its curvature kernel is signed while its Fourier transform is strictly positive.
* **Activations outside the localized-derivative framework:** failing the nonzero-mass localized-kernel assumptions means this particular QI argument does not apply. It does not prove that the activation is ineffective. Oscillatory activations with learned frequencies need a different geometry description.

The best supported question is therefore which **activation, target class, and allowed geometry** work well together. The current theory does not rank activations for unrestricted deep-learning tasks.

## 13 Coefficient branches and optimization

### 13.1 A shared coefficient field

Return to a real sampled problem with \(M\) observations and \(d\) features, including any baseline columns. Use the weighted design \(A\in\mathbb R^{M\times d}\), \(A_{ij}=\sqrt{w_i}\phi_j(x_i)\), and target \(y=(\sqrt{w_i}f(x_i))_{i=1}^M\in\mathbb R^M\); take \(w_i=1\) for unweighted fitting.

Let \(r_A=\operatorname{rank}A\). Its rank-\(r_A\) singular value decomposition is \(A=U\Sigma V^*\), with orthonormal-column matrices \(U\in\mathbb R^{M\times r_A}\), \(V\in\mathbb R^{d\times r_A}\), and positive diagonal \(\Sigma\in\mathbb R^{r_A\times r_A}\). Retain an index set \(R\) of size \(s\): then \(U_R\), \(V_R\), and \(\Sigma_R\) have dimensions \(M\times s\), \(d\times s\), and \(s\times s\). The truncated-SVD coefficient vector \(v_R\in\mathbb R^d\) is

\[
v_R=V_R\Sigma_R^{-1}U_R^*y.
\]

Define the observation-space vector \(q=U_R\Sigma_R^{-2}U_R^*y\in\mathbb R^M\). Its entries are signed measurement weights, distinct from the positive fitting weights \(w_i\) and from readout coefficients in \(\mathbb R^d\). Then

\[
A^*q=V\Sigma U^*U_R\Sigma_R^{-2}U_R^*y
=V_R\Sigma_R^{-1}U_R^*y=v_R.
\tag{13.1}
\]

For the scalar feature family \(\phi(x;c,\gamma)\), define a scalar field \(\mathcal V:\mathbb R\times(0,\infty)\to\mathbb R\) on center–inverse-width space. Equation (13.1) means that every retained-solve neuron coefficient is its value at that neuron's geometry:

\[
(v_R)_j=\mathcal V(c_j,\gamma_j),
\qquad
\mathcal V(c,\gamma)=\sum_{i=1}^M\sqrt{w_i}\,q_i\phi(x_i;c,\gamma),
\tag{13.2}
\]

The factor \(\sqrt{w_i}\) is required because \(q\) uses the weighted observation coordinates; it disappears in unweighted fitting. For a smooth activation, fixing gamma gives a smooth function of center. Groups of neurons with similar gammas can therefore sample related coefficient curves. Smooth does not imply slowly varying, and different gamma values can produce nearby coefficients. The vector \(q\) changes when the geometry, target, or cutoff changes; holding it fixed is not a perturbation prediction. Forming inverse squared singular values is also a poor numerical implementation when those values are tiny. Equation (13.1) is an explanatory identity.

The saved floor run supports an association between coefficient bands and gamma regimes, but not a classification theorem. Equal-gamma neurons can also split into even/odd branches through inverse-overlap compensation, as the tent response already demonstrates.

### 13.2 Why coefficient motion can be nearly invisible in the function

Let \(z_i\in\mathbb R^d\) be a unit right singular vector in coefficient space, \(u_i\in\mathbb R^M\) the corresponding unit left singular vector in observation space, and \(s_i>0\) its singular value. Then

\[
A z_i=s_i u_i.
\]

For a scalar \(t\in\mathbb R\), the coefficient displacement \(t z_i\in\mathbb R^d\) has Euclidean norm \(|t|\), whereas its observation displacement \(A(tz_i)\in\mathbb R^M\) has norm \(|t|s_i\). Small singular values permit large readout changes with small changes on the fitting samples. This can explain large oscillatory readouts without asserting that a specific screenshot's oscillation frequency has already been derived. Dense evaluation is still needed to ensure that a direction weak on fitting samples is also weak between them.

### 13.3 Fixed-geometry gradient flow

Keep \(A:\mathbb R^d\to\mathbb R^M\) and \(y\in\mathbb R^M\) fixed. The scalar loss \(J:\mathbb R^d\to\mathbb R\), \(J(v)=\tfrac12\|Av-y\|_2^2\), generates a coefficient trajectory \(v(t)\in\mathbb R^d\), with time \(t\ge0\), through Euclidean gradient flow:

\[
\dot v=-A^*(Av-y).
\]

Define the residual trajectory \(r(t)=y-Av(t)\in\mathbb R^M\). The matrix \(AA^*\in\mathbb R^{M\times M}\) acts in observation space, whereas \(A^*A\in\mathbb R^{d\times d}\) acts in coefficient space. Since geometry is fixed,

\[
\dot r=-A\dot v=-AA^*r.
\]

Take the scalar component \(r_i(t)=\langle u_i,r(t)\rangle\), using the observation-space singular vector above. Because \(AA^*u_i=s_i^2u_i\),

\[
\dot r_i=-s_i^2r_i,
\qquad r_i(t)=e^{-s_i^2t}r_i(0).
\tag{13.3}
\]

Directions with small singular values learn slowly. Components in \(\operatorname{ran}(A)^\perp\subset\mathbb R^M\) cannot be learned at this fixed geometry. With discrete gradient descent, scalar step size \(\alpha>0\), and integer iteration count \(k\ge0\), the corresponding factor is \((1-\alpha s_i^2)^k\), stable when \(0<\alpha<2/s_{\max}^2\), where \(s_{\max}\) is the largest singular value. Rescaling the loss by sample count rescales time or learning rate.

This exact calculation connects ill-conditioned readout directions to slow precision improvement. It is not an exact model of Adam or of training that moves centers and gammas. For variable projection, let \(\theta=(c_1,\gamma_1,\ldots,c_n,\gamma_n)\) denote all geometry parameters and \(A(\theta)\in\mathbb R^{M\times d}\) their sampled design. The projector \(P_{A(\theta)}=A(\theta)A(\theta)^\dagger:\mathbb R^M\to\mathbb R^M\) selects the attainable observation subspace. Variable projection eliminates the readouts and optimizes the scalar objective \(\|(I-P_{A(\theta)})y\|_2^2\), with \(I\) the identity on observation space; its constant-rank differentiation is classical [R2].

## 14 Evidence and its limits

### 14.1 Controlled examples from this investigation

| Claim | Evidence | Scope |
|---|---|---|
| Single-neuron projection formula | Periodic normalized smooth-bump predictions versus fresh full SVD solves; relative coefficient agreement at most \(5.8\times10^{-10}\) in tested cases | Direct bump fitting on a circle; broadest nonconverged case excluded |
| Exact narrow-tent formulas | Independent continuous quadrature matches coefficients within \(3.7\times10^{-12}\) | Constant target, no bias, standard tents, distant finite boundaries |
| Strong-overlap residue classes | Half-width-eight periodic coefficients match the exact formulas within about \(7.5\times10^{-14}\) | Minimum norm on periods divisible by eight; not the infinite square-summable perturbation problem |
| ReLU curvature readouts | Four function-value fits, independent nodal-hat solve, doubled quadrature, gamma-rescaling check | Explicit affine baseline, continuous fitting, 63 interior knots |
| Four simultaneous center/width changes | Explicit response sequences predict coefficients, with approximately quadratic error reduction under perturbation scaling | New bounded check in Section 9; no fitted response parameters |

The ReLU coefficient-curvature relative differences are approximately \(0.001584\) for sine, \(0.000303\) for Runge, and \(0.005173\) for mixed sine. The quadratic has \(w_j/h=2\) within about \(10^{-12}\) relative error, while its function error remains the nonzero value (11.6).

### 14.2 Saved trained geometries

The saved Xavier-initialized sine model from VarPro Adam followed by Gauss–Newton has 461 neurons and reproduces approximately \(6.6\times10^{-15}\) relative function error. After orienting tanh neurons to positive gamma, the descriptive band near readout \(-0.05\) has median gamma about \(0.584\); a near-zero band has median gamma about \(9.927\). The bands were selected after inspecting the plot. They demonstrate an association in one checkpoint, not a universal or causal law.

The solve uses a relative singular cutoff \(10^{-13}\), retaining numerical rank 81 of 462 columns. Raising that cutoff to \(10^{-10}\) changes rank and error materially. Consequently the pure derivative-only dual theorem must not be applied to these stored coefficients without checking how the intercept and discarded singular directions are treated.

Three saved ordinary-Adam controls are only 500-step runs. The nearly uniform QI-initialized case has saved error \(0.003396\), versus \(3.639\times10^{-11}\) after a fresh readout solve on the same geometry. This is evidence of an optimization gap in that example. The Xavier controls are not sufficiently converged to establish a broad gamma-branch claim for ordinary Adam.

### 14.3 Existing activation evidence

The following values were traced to stored data, not inferred from the discussion. Historical \(N\) in this subsection counts intervals, so \(h=2/N\).

For sine, the historical ReLU scaling study gives:

| Historical \(N\) | ReLU relative function error | Tanh relative function error |
|---:|---:|---:|
| 64 | \(1.44376\times10^{-3}\) | \(2.88736\times10^{-15}\) |
| 128 | \(3.59747\times10^{-4}\) | \(2.90938\times10^{-14}\) |
| 256 | \(8.98751\times10^{-5}\) | \(1.23236\times10^{-14}\) |
| 512 | \(2.25133\times10^{-5}\) | \(5.43740\times10^{-14}\) |
| 1024 | \(5.67340\times10^{-6}\) | \(4.56059\times10^{-14}\) |

The roughly fourfold ReLU improvement per doubling agrees with the spline order. This setup used a bias and halo neurons; left ReLU halos supply a slope but make individual halo coefficients redundant. It verifies output approximation behavior, whereas the newer explicit-affine test verifies identifiable interior slope-jump coefficients.

The six-activation bandwidth sweep contains 2,592 solves: six activations, four targets, 36 lambdas, and three widths. Its historically anchored Fourier scales and median argmins among cells whose minimum exceeds \(10^{-14}\) are:

| Activation | Anchored scale | Median measured argmin in hard cells |
|---|---:|---:|
| sech-squared | 0.2500 | 0.22209 |
| Gaussian | 0.5302 | 0.49423 |
| tanh | 0.2500 | 0.20841 |
| sigmoid | 0.5000 | 0.43688 |
| GELU | 0.7070 | 0.71557 |
| Swish | 0.4554 | 0.43688 |

These medians support an activation-dependent scale. Only 21 of 26 individual hard-cell argmins are within one grid step of the nearest predicted grid point. The hardened interpretation is an aliasing-wall prediction and conservative scale, not an exact argmin theorem. Using machine epsilon rather than the tanh-calibrated anchor gives slightly different scales.

A stronger test concerns the shape of the aliasing wall. For Runge, the stored median ratios of measured to predicted error are:

| Historical \(N\) | Tanh | GELU | Swish |
|---:|---:|---:|---:|
| 64 | 0.999975 | 1.000049 | 1.000047 |
| 128 | 0.999983 | 1.000109 | 1.000076 |
| 256 | 0.999995 | 1.000076 | 1.000042 |
| 512 | 1.000244 | 1.000471 | 1.000221 |

The comparison selects points right of the minimum with measured error above 30 times the cell floor and below \(10^{-3}\). These are selected wall points, not every point in each curve. They support the approximation mechanism, while the arithmetic-dominated side remains less understood.

An instructive correction concerns GELU on \(|x|^3\). At \(N=1024\), an earlier run using lambda 0.25 had an interior readout norm about 22.3243. With GELU-appropriate lambda 0.707, the norm is about 0.000595698, the local-law ratio is 0.995967, and function error is about \(2.46\times10^{-11}\). The earlier interpretation that GELU intrinsically failed on this target was withdrawn. Tanh at lambda 0.25 gives norm-law ratio 1.00307 and similar function error. The comparison demonstrates the importance of matched geometry; raw norms across activations cannot themselves define quality.

For rougher targets, the legacy solver using its default rank cutoff, without an explicit cutoff choice, can produce extreme coefficient norms that are partly cutoff artifacts. Do not turn those numbers into evidence about a unique mathematical coefficient sequence. Historical GELU/Swish runs also generally used a bias-only baseline; they do not directly test the exact affine-baseline dual theorem in Section 3.2.

### 14.4 Corrections preserved in this checkpoint

* Alternation does not prove target aliasing or numerical failure; standard tents provide a well-conditioned counterexample.
* Function-value and derivative-value least squares are not interchangeable.
* A derivative-measurement identity does not establish a local sampling rule.
* A higher-order identity requires annihilation of the corresponding lower-degree polynomials.
* The full spectral projection denominator contains all aliases. Principal-band deconvolution is an approximation.
* A good aliasing prediction does not by itself bound finite-precision error or locate the exact optimum.
* Raw feature rescaling changes readout norms without changing the span.
* Coefficient divergence can preserve a feature contribution in a pointwise-degenerate gamma limit.
* Infinite square-summable defect responses and periodic minimum-norm responses are different problems.
* Saved least-squares readouts and readouts actually found by Adam must be reported separately.

## 15 Research directions and standards for a prediction

### 15.1 Predict several center and gamma changes

The research deliverable should be a formula or a small family of reusable response functions that predicts coefficient changes before solving the perturbed full model. It should explain signs, amplitudes, decay, and interaction, and state when it ceases to be accurate.

The explicit tent predictor (9.4) already meets this standard locally. It uses two closed-form sequences and the chosen perturbations, with no response fitting. The four-neuron validation supplies a first controlled example; it is not evidence for arbitrary-size changes.

The next mathematical steps are:

1. **Finite-amplitude deletion interactions.** Use (7.10) and the tent Green function to predict four deletions. Because \(Q_{jk}=\sqrt3r^{|j-k|}\), the interaction strength and parity are explicit. A small system for four amplitudes is justified by this derived law; computing a new full SVD would provide only the reference.
2. **A fixed low-order interaction correction.** Use the same analytic inverse and elementary local tent overlap integrals in (7.7). Freeze a first- or second-correction predictor before comparing with the reference. Report its remainder criterion rather than increasing expansion order until every case is effectively solved.
3. **Smooth kernels with a uniform background.** Derive or approximate the inverse-overlap response from the kernel spectrum. State an error bound for truncating its spatial range or spectral approximation. A numerically tabulated response is useful only if it transfers across targets, sites, and lattice sizes and exposes a meaningful dimensionless law; a new full factorization for each geometry is not the intended result.
4. **Finite nonuniform classes.** Progress from isolated changes to slowly varying centers or widths, then a small number of interleaved width families. Analyze residualized pair interactions before assuming independent superposition.

For the smooth-kernel case, a natural response family depends on background \(\lambda\), displacement \(\Delta c/h\), gamma ratio \(\gamma_{\rm new}/\gamma_0\), and center separation in grid units. The baseline coefficients carry target information; a nonzero initial residual contributes the correction in (7.3).

A controlled comparison should hold the four changed indices and perturbation ratios fixed while scaling their magnitude. It should then vary separation to test interaction range. Score coefficient-change error, output error, and any error in predicted sign or branch structure. Use at least one target exactly representable by the baseline and one imperfectly fitted target, so the residual term is separately testable. Include a near-redundant case as a predicted breakdown regime rather than hiding it.

The proposed next sweep is not run in this checkpoint. Choosing its final activations, targets, and finite-amplitude regimes is an experiment-design decision, not a consequence of the algebra.

### 15.2 Predict activation performance

The useful question is not whether an activation universally works. It is whether the theory predicts behavior when **target class, geometry freedom, precision, and fitting loss are specified**.

A discriminating program would compare:

| Controlled comparison | Prediction before fitting | What would be learned |
|---|---|---|
| Smooth sine/quadratic; fixed uniform ReLU versus suitably scaled smooth activations | ReLU has second-order spline error; smooth kernels may attain much smaller approximation error at the same count | Approximation efficiency, separate from training |
| Exact kink with a knot at the kink; ReLU versus smooth kernels | ReLU can represent the piecewise-linear target exactly; smooth finite-width features generally require approximation or a limiting process | A target class where nonsmooth features are favorable |
| Shared-frequency sine versus multiple allowed frequencies | Centers alone leave rank at most two; added frequencies expand the span | A decisive geometry-dependent success/failure control |
| Low-degree polynomial activation versus a nonpolynomial target | Width cannot escape a fixed polynomial degree | A genuine representational obstruction |
| Matched tanh and sigmoid geometries | Same-span predictions and outputs coincide in exact arithmetic | A normalization and implementation control |
| Broad tents at integer and nearby noninteger width ratios | Exact residue-class redundancy at the integer ratio; altered compensation nearby | Whether the spectral-zero mechanism predicts observed branches |
| Tanh/GELU across smooth and rough targets at activation-appropriate overlap | Local readout laws work when target derivative content is resolved; inverse filtering matters otherwise | The range of validity of derivative sampling |

For a rigorous activation comparison, freeze the predicted quantity before solving: an exact rank, an error order, a spectral wall curve, an invariance, or a response decay rate. A claim such as "activation A should work better" without a target class and metric is too vague to falsify.

The existing ReLU scaling and six-activation wall data already support several predictions. New tests are still needed for the fixed-frequency sine and polynomial negative controls in this repository, and for a unified study of coefficient perturbations across activations. Universal approximation theorems concern existence with sufficient representational freedom; they do not give the finite-width, finite-precision, fixed-geometry predictions sought here [R6].

### 15.3 What would count as progress beyond an exact reformulation

An exact identity is the starting point. A useful result should add at least one of the following:

* a closed-form response with interpretable constants, as for the tents;
* a small, reusable family that predicts several new geometries without a full solve;
* a controlled approximation with a stated remainder and predicted breakdown;
* a distance or width scaling law that explains interaction and transfers across targets;
* an activation or geometry failure predicted before observing its fitted coefficients.

The independent numerical solve remains the reference for validation. It must not enter the predictor for the same perturbed case. In particular, plotting the output of a fresh pseudoinverse and calling it a theoretical prediction does not satisfy this program.

## 16 References and reproducibility

### Mathematical sources

* **R1:** Michael Unser, [Sampling—50 Years After Shannon](https://bigwww.epfl.ch/publications/unser0001/), 2000. Projection and sampling in shift-invariant spaces, including spline settings. The elementary projection and lattice calculations used here are derived in Sections 2, 5, and 12.
* **R2:** Gene Golub and Victor Pereyra, [The Differentiation of Pseudo-Inverses and Nonlinear Least Squares Problems Whose Variables Separate](https://epubs.siam.org/doi/10.1137/0710036), 1973. Separable least squares and constant-rank differentiation. Section 7 derives the full-column-rank formulas used in this checkpoint.
* **R3:** Dan Hendrycks and Kevin Gimpel, [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415). Definition of exact GELU. Its derivatives, normalization, and transform are calculated explicitly here.
* **R4:** Justin Sahs and collaborators, [Shallow Univariate ReLU Networks as Splines](https://arxiv.org/abs/2008.01772), 2020. The spline interpretation; Section 11 supplies the relevant identities and quadratic calculation.
* **R5:** Vincent Sitzmann and collaborators, [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661), 2020. Evidence that periodic activations can succeed with different frequency/architecture freedom; this does not contradict the common-fixed-frequency rank calculation.
* **R6:** Moshe Leshno, Vladimir Lin, Allan Pinkus, and Shimon Schocken, [Multilayer Feedforward Networks with a Nonpolynomial Activation Function Can Approximate Any Function](https://pinkus.net.technion.ac.il/files/2021/02/neural.pdf), 1993. Classical existence theory; the polynomial span obstruction needed here is proved directly in Section 12.2.

### Local derivations and measurements

* [Earlier single-gamma derivation and numerical conventions](../../results/checkpoint_G_interactive/geometry_reader_codex/single_gamma_theory_20260929/derivation.md).
* [Narrow-tent exact formulas and verification](../../results/checkpoint_G_interactive/geometry_reader_codex/single_gamma_theory_20260929/theory/tent_derivation.md).
* [Strong-overlap tent report and period controls](../../results/checkpoint_G_interactive/geometry_reader_codex/single_gamma_theory_20260929/theory/tent_overlap_notes.md).
* [ReLU coefficient experiment](../../results/checkpoint_G_interactive/geometry_reader_codex/single_gamma_theory_20260929/relu_test/report.md).
* [Saved trained-run analysis](../../results/checkpoint_G_interactive/geometry_reader_codex/single_gamma_theory_20260929/trained_runs/report.md), with checkpoint hashes in its adjacent `summary.json`.
* [Four-defect validation script](../../results/checkpoint_G_interactive/geometry_reader_codex/single_gamma_theory_20260929/checkpoint_validation/multiple_tent_prediction.py), [measurements](../../results/checkpoint_G_interactive/geometry_reader_codex/single_gamma_theory_20260929/checkpoint_validation/multiple_tent_prediction.json), and [figure](../../results/checkpoint_G_interactive/geometry_reader_codex/single_gamma_theory_20260929/checkpoint_validation/multiple_tent_prediction.pdf).
* [Historical ReLU scaling data](../../results/checkpoint_A_numerics/expA06_readout_structure/scaling_law/scaling_rows.json).
* [Six-activation sweep data](../../results/checkpoint_C_geometry/expC07_lambda_energy_rule/expC07_rows.json).
* [Hardened lambda-rule interpretation and withdrawn claims](../../results/checkpoint_C_geometry/expC07_lambda_energy_rule/lambda_rule/hardened_rule.md); [stored wall ratios](../../results/checkpoint_C_geometry/expC07_lambda_energy_rule/lambda_rule/data/c09_wall_ratios.json).
* [Original activation magnitude data](../../results/checkpoint_A_numerics/expA07_inner_norm_rule/expA07_rows.json), including the old GELU lambda-0.25 case, and [activation-specific smoothness data](../../results/checkpoint_A_numerics/expA07_inner_norm_rule/smoothness/expA07s_rows.json). The old [magnitude discussion](../theory_magnitude_rule.md) includes approximations that are qualified explicitly in this checkpoint.

The new four-defect check used NumPy/SciPy and Matplotlib from the repository's `pfloat` environment. Its script separates the explicit predictor from the numerical reference. No application code, saved checkpoints, training results, or older theory documents were modified for this checkpoint. Two independent read-only reviews checked the multi-defect mathematics and the activation evidence; outstanding scope limits are stated in the relevant sections.
