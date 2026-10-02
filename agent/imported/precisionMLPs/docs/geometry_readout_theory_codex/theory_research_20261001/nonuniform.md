# Nonuniform multibranch derivative encodings

Theory-only research pass, October 1, 2026. No training or new empirical claims.

There is a useful extension of the stationary branch model. It predicts a common smooth readout curve, a width-dependent third-derivative correction, and additional corrections from nonuniform neuron density. It also gives a target-independent compatibility test between branches. These predictions concern a specified minimum-norm or regularized encoding. A finite lattice can select a different encoding even when its aliasing error is exponentially small; that distinction is proved below.

## 1. Fix the normalization and the coefficient metric first

The target is a differentiable scalar function \(f:\mathbb R\to\mathbb R\); write \(d=f'\). Boundary-free calculations use the real line with sufficient decay, or periodized kernels on a circle with the zero-frequency convention specified below. A branch index \(s\in\{1,\ldots,S\}\) identifies neurons with smoothly varying widths and locally ordered centers. The unit-mass tanh derivative kernel is

\[
K_\gamma(t)=\frac\gamma2\operatorname{sech}^2(\gamma t),
\qquad \widehat K_\gamma(\xi)=k_\gamma(\xi)
=\frac{\pi\xi/(2\gamma)}{\sinh(\pi\xi/(2\gamma))}.
\tag{1}
\]

The Fourier convention is \(\widehat u(\xi)=\int u(x)e^{-i\xi x}\,dx\). At zero frequency, set \(k_\gamma(0)=1\).

On a circle of length \(L\), a periodized unit-mass kernel has no periodic primitive. The periodic tanh-like primitive differentiates to \(2(K_\gamma^{\rm per}-1/L)\). Whenever the formulas below use the unit-mass synthesis \(T\) on a circle, we impose zero total coefficient mass \(\sum_s\int\rho_s=0\), or \(\sum_{s,j}v_{sj}=0\) in the discrete model, and fit the target mean separately. The subtracted constants then cancel in the sum. An alternative convention allows arbitrary coefficient mass and explicitly uses \(P_0T\), where \(P_0\) removes the mean. That alternative has a different coefficient null direction and must retain the projections stated below. The nonzero-frequency alias counterexample is unaffected: its sinusoidal readout sequences already have zero sum.

For branch \(s\), the center density is \(n_s(c)>0\), its reciprocal spacing is \(h_s(c)=1/n_s(c)\), and its inverse width is \(\gamma_s(c)>0\). A continuous coefficient density \(\rho_s(c)\) synthesizes the derivative through

\[
(T_s\rho_s)(x)=\int K_{\gamma_s(c)}(x-c)\rho_s(c)\,dc,
\qquad d=\sum_sT_s\rho_s.
\tag{2}
\]

The discrete tanh readout at a center \(c_{sj}\) is approximately

\[
v_{sj}=\frac{h_s(c_{sj})}{2}\rho_s(c_{sj}).
\tag{3}
\]

The factor two comes from differentiating raw tanh. Equation (3) is a quadrature identification, not an assertion that every arbitrary discrete coefficient vector already samples a smooth field.

This normalization changes the appropriate continuum norm. A sum of raw squared readouts obeys

\[
\sum_jv_{sj}^2
\approx\int n_s(c)\frac{h_s(c)^2}{4}\rho_s(c)^2dc
=\frac14\int h_s(c)\rho_s(c)^2dc.
\tag{4}
\]

Thus minimum raw-readout norm corresponds to the weight \(w_s=h_s\), with the common factor \(1/4\) omitted. Equal unweighted L2 norms of the density fields are a different coefficient convention when branch densities differ. Common constants can be absorbed into the regularization parameter, but branch-dependent factors cannot.

More generally use a weighted norm \(\sum_s\int w_s\rho_s^2\), with \(w_s\) bounded above and below by positive constants. In formulas below write \(n_s=w_s^{-1}\); only in the raw-readout convention is this literally the physical center density.

## 2. The common potential and a branch test that does not reconstruct the target

Assume the minimum weighted-norm encoding is well posed and has a dual potential \(\psi\) in the stated output space. In infinite-dimensional L2 this requires more than existence of a primal representation: a sufficient explicit assumption is \(d\in\operatorname{ran}(\mathcal S)\), for the operator in (6). The finite resolved-band coercivity conditions below, or positive derivative-loss ridge, provide the needed inverse. Introduce that Lagrange multiplier \(\psi\). Varying

\[
\frac12\sum_s\int w_s\rho_s^2
-\left\langle\psi,\sum_sT_s\rho_s-d\right\rangle
\]

with respect to each \(\rho_s\) gives

\[
w_s\rho_s=T_s^*\psi,
\qquad
(T_s^*\psi)(c)=\int K_{\gamma_s(c)}(x-c)\psi(x)dx.
\tag{5}
\]

Consequently the common potential satisfies

\[
\mathcal S\psi=d,
\qquad
\mathcal S=\sum_s T_s n_s T_s^*.
\tag{6}
\]

For the raw-readout convention, (3) and (5) simplify to the particularly useful statement

\[
\boxed{2v_s(c)=T_s^*\psi(c).}
\tag{7}
\]

Every branch is a differently smoothed observation of one common potential. Density and width determine how that potential relates to the target. The potential is generally not simply \(f'\).

The dual-potential assumption is substantive. For the diagonal operator \(T=\operatorname{diag}(1/n)\) on \(\ell^2\), the target \(d_n=1/n^2\) has a finite-norm exact coefficient vector \(\rho_n=1/n\), but \(TT^*\psi=d\) would require \(\psi_n=1\notin\ell^2\). A generalized potential can sometimes be used in a larger space, but its domain must be specified. We do not infer an L2 potential from primal solvability alone.

If the widths are constant along each branch, \(T_s^*\) is convolution with \(K_s\). Convolution commutes, so

\[
\boxed{K_s*(w_t\rho_t)=K_t*(w_s\rho_s).}
\tag{8}
\]

This remains exact for spatially varying weights/densities. In the raw-readout convention it becomes \(K_s*v_t=K_t*v_s\). It is a test of the branch allocation independent of the target: filter one branch using the other branch's kernel and compare. It is not a plot of the network derivative.

The condition is necessary for canonical allocation, not for every valid representation. For example, with two stationary branches the exchange

\[
\rho_1\mapsto\rho_1+K_2*q,
\qquad
\rho_2\mapsto\rho_2-K_1*q
\]

preserves the output for arbitrary admissible \(q\), but generally violates (8). A rough \(q\) can destroy smooth branch structure even for a smooth target. Small output error alone does not control this exchange freedom.

## 3. A direct nonuniform coefficient prediction from kernel moments

The kernel in (1) is even. Its second and fourth Taylor coefficients are

\[
a_s(c)=\frac{1}{2!}\int t^2K_{\gamma_s(c)}(t)dt
=\frac{\pi^2}{24\gamma_s(c)^2},
\]
\[
b_s(c)=\frac{1}{4!}\int t^4K_{\gamma_s(c)}(t)dt
=\frac{7\pi^4}{5760\gamma_s(c)^4}
=\frac7{10}a_s(c)^2.
\tag{9}
\]

For \(\psi\in C_b^4\), Taylor's theorem and positivity of the kernel give the exact bound

\[
\left|T_s^*\psi(c)-\psi(c)-a_s(c)\psi''(c)\right|
\le b_s(c)\|\psi^{(4)}\|_\infty.
\tag{10}
\]

There are no derivatives of \(a_s\) in this analysis formula: the kernel width is held at the observation center \(c\).

Synthesis is different. Taking the adjoint of the second-order analysis formula gives

\[
T_s\rho_s=\rho_s+(a_s\rho_s)''+\text{fourth-order remainder}.
\tag{11}
\]

The derivatives must act on the width as well as on the coefficient field. One rigorous version of the remainder is weak: for a compactly supported smooth test function \(\eta\),

\[
\left|\langle\eta,T_s\rho_s-\rho_s-(a_s\rho_s)''\rangle\right|
\le \|\eta^{(4)}\|_\infty\int b_s(c)|\rho_s(c)|dc.
\tag{12}
\]

Define the total inverse coefficient weight and its width moment by

\[
N(c)=\sum_s n_s(c),\qquad A(c)=\sum_s n_s(c)a_s(c).
\tag{13}
\]

Using \(\rho_s=n_sT_s^*\psi\) in (11), and retaining terms through second width order, gives

\[
\begin{aligned}
\mathcal S\psi
&=\sum_s\left[n_s(\psi+a_s\psi'')+(a_sn_s\psi)''\right]
+O(\text{width}^4)\\
&=N\psi+A\psi''+(A\psi)''+O(\text{width}^4)\\
&=N\psi+2A\psi''+2A'\psi'+A''\psi+O(\text{width}^4).
\end{aligned}
\tag{14}
\]

This is a variable-coefficient local equation for the shared potential, rather than an unspecified feature-matrix inverse.

Let

\[
u(c)=\frac{f'(c)}{N(c)}.
\]

Solving (14) to the same order, by inserting \(u\) in its already second-order term, yields

\[
\psi_{\rm pred}
=u-\frac{A u''+(Au)''}{N}.
\tag{15}
\]

Using (7) now gives an explicit target-to-readout prediction:

\[
\boxed{
v_s^{\rm pred}(c)
=\frac12\left[
u(c)+a_s(c)u''(c)
-\frac{A(c)u''(c)+(A(c)u(c))''}{N(c)}
\right].
}
\tag{16}
\]

The input is \(f'\), its next two derivatives, the branch densities, and the branch widths with their derivatives. No target feature fit or target-dependent inverse enters this formula. It is a low-frequency/narrow-kernel expansion of a specified encoding, not a prediction for arbitrary unregularized irregular LS coefficients.

Two branches have the especially simple difference

\[
\boxed{v_s-v_t\approx\frac{a_s-a_t}{2}\left(\frac{f'}{N}\right)''.}
\tag{17}
\]

The common compensation term cancels. Smooth branches are therefore expected to separate by a geometry-scaled third-derivative curve, with density corrections when \(N\) varies. A width derivative need not appear individually in (17); it enters the common correction through \(A\). These statements are restricted to the canonical allocation and the resolved regime.

For a derivative-loss ridge convention \(\frac12\|T\rho-d\|^2+\frac\tau2\sum_s\int w_s\rho_s^2\) without an extra coefficient-domain constraint, replace the potential equation by \((\mathcal S+\tau I)\psi=d\), and use \(u=d/(N+\tau)\), with the denominator \(N+\tau\) in the last term of (15). If a separate zero-mass coefficient constraint is imposed, its multiplier must also be retained; regularized derivative fitting alone does not enforce that constraint exactly.

This is not the same regularizer as ordinary function-loss ridge. With the zero-total-mass coefficient convention on a circle, function-loss ridge gives \((\mathcal S-\tau\partial_c^2)\psi=d\); the potential may contain a constant, which serves as the mass-constraint multiplier. With arbitrary coefficient mass and periodic zero-mean primitives, the correct equation is instead \((P_0\mathcal SP_0-\tau\partial_c^2)\psi=d\) on the mean-zero output space. Variable widths and densities need not preserve means, so these projections cannot be dropped. When \(\tau\partial_c^2\) is treated perturbatively in the zero-mass convention, its second-order correction contains the additional term \(-\tau u''\) inside the numerator in (15); large regularization cannot be handled by this truncated correction. Section 6 retains regularization in the denominator, with the same domain/projection qualifications.

## 4. Two branches and a sinusoid: amplitudes, mean, and separation

Take \(f(x)=\sin(\omega x)\), so \(d(x)=\omega\cos(\omega x)\). For constant widths and densities define

\[
k_s=k_{\gamma_s}(\omega),\quad
Q=n_1k_1^2+n_2k_2^2,\quad
N=n_1+n_2,\quad \bar a=\frac{n_1a_1+n_2a_2}{N}.
\]

Exact continuum minimum raw-readout norm gives

\[
v_s(c)=\frac{\omega k_s}{2Q}\cos(\omega c).
\tag{18}
\]

Thus

\[
\frac{v_1+v_2}{2}
=\frac{\omega(k_1+k_2)}{4Q}\cos(\omega c),
\qquad
v_1-v_2=\frac{\omega(k_1-k_2)}{2Q}\cos(\omega c).
\tag{19}
\]

Where the cosine is nonzero, \(v_1/v_2=k_1/k_2\), independent of the density allocation. Both branches have the same phase. Opposite-phase single-frequency branches cannot be the minimum-norm encoding under these assumptions; they require exchange components, another metric, or another model.

The second-order expansion is

\[
v_s(c)\approx\frac\omega{2N}
\left[1+(2\bar a-a_s)\omega^2\right]\cos(\omega c).
\tag{20}
\]

At low frequency every individual neuron receives approximately \(f'/(2N)\), even if one branch is denser. The dense branch contributes more total derivative mass by having more neurons. This differs from an equal-L2-density convention, which would instead produce \(v_s\approx h_sf'/(2S)\).

For varying densities, put \(r(c)=1/N(c)\). Then

\[
u=\omega r\cos(\omega c),\qquad
u''=\omega(r''-\omega^2r)\cos(\omega c)-2\omega^2r'\sin(\omega c).
\tag{21}
\]

Consequently the two-branch separation becomes

\[
v_1-v_2\approx\frac{a_1-a_2}{2}
\left[\omega(r''-\omega^2r)\cos(\omega c)-2\omega^2r'\sin(\omega c)\right].
\tag{22}
\]

The predicted sine component is a density-gradient-induced phase shift. It is absent on a uniform lattice. The mean is obtained directly from (16) by replacing \(a_s\) with \((a_1+a_2)/2\); it also contains the common terms \(2A'u'\) and \(A''u\). Equations (16) and (22) continue to use local \(a_s(c)\) when the widths vary smoothly.

## 5. Quantitative validity: an explicit stationary bound and a nonuniform bound

### 5.1 Explicit stationary fourth-order bound

Assume the target derivative is bandlimited to \(|\xi|\le\Omega\). Set \(\beta=\Omega^2\max_s a_s\). The cosine Taylor remainder and (9) imply

\[
0\le k_s(\xi)-(1-a_s\xi^2)\le\tfrac7{10}a_s^2\xi^4.
\]

If \(\beta\le1/4\), then \(k_s\ge1-\beta\), so

\[
Q(\xi)=\sum_s n_sk_s(\xi)^2\ge N(1-\beta)^2\ge9N/16.
\]

Elementary expansion of the numerator and denominator gives the conservative bound

\[
\left|\frac{k_s(\xi)}{Q(\xi)}
-\frac1N[1+(2\bar a-a_s)\xi^2]\right|
\le\frac{15\beta^2}{N}.
\tag{23}
\]

One way to verify the constant is to write \(k_s=1-\alpha_s+e_s\), with \(0\le\alpha_s\le\beta\) and \(0\le e_s\le0.7\beta^2\). Then \(Q/N=1-2\bar\alpha+E\), with \(0\le E\le(2.4+0.49\beta^2)\beta^2\). Subtract the proposed ratio, bound the numerator by \([0.7+4+(2.4+0.49\beta^2)(1+2\beta)]\beta^2\), and divide by \((1-\beta)^2\). This is less than \(15\beta^2\) for \(\beta\le1/4\).

By Parseval, the raw-readout L2 error of the second-order predictor is at most \(15\beta^2\|d\|_2/(2N)\). This explicitly identifies the controlling quantity: kernel width relative to target variation scale, together with a lower resolved spectral bound. Mere smoothness of the geometry is insufficient if the target lies where every \(k_s\) is exponentially small.

### 5.2 A rigorous nonuniform resolved-band version

Here is a finite-band result that avoids making an unqualified strong-norm claim from the weak remainder (12). Work on a circle, with periodized kernels. Let \(H_\Omega\) be the span of the \(D\) Fourier modes with \(|\xi|\le\Omega\), and \(P\) its orthogonal projection. The exact fitting constraint is the resolved derivative constraint \(P\sum_sT_s\rho_s=d\), for \(d\in H_\Omega\). This explicitly leaves unresolved frequencies outside the observation space.

Suppose the total density is constant, \(\sum_sn_s(c)=N_0>0\), while individual densities and widths may vary. Let \(n_{s,\max}=\|n_s\|_\infty\), \(\beta_s=\Omega^2\|a_s\|_\infty\), and \(r_s=\sqrt D\,\Omega^4\|b_s\|_\infty\). On \(H_\Omega\),

\[
T_s^*=I+a_s\partial_c^2+R_s,
\qquad\|R_s\|_{H_\Omega\to L^2}\le r_s.
\tag{24}
\]

To prove the bound, apply the Fourier multiplier remainder in (10) to each Fourier basis vector, bound its output L2 norm by \(\Omega^4\|b_s\|_\infty\), and use the Hilbert–Schmidt bound over \(D\) columns. This is conservative but explicit and valid for variable widths.

The resolved frame operator satisfies

\[
P\mathcal SP=N_0I+L+E,\qquad
L=P[A\partial_c^2+\partial_c^2A]P,
\]
\[
\|L\|\le2\|A\|_\infty\Omega^2,
\qquad
\|E\|\le\sum_s n_{s,\max}
[\beta_s^2+2(1+\beta_s)r_s+r_s^2].
\tag{25}
\]

The second estimate follows by expanding \(\sum_s(T_s^*)^*n_sT_s^*\), retaining the identity and single-second-derivative terms, and bounding every remaining product. It does not require solving a feature matrix.

Allow derivative-loss ridge \(\tau\ge0\) and put \(R_0=N_0+\tau\), \(\ell=\|L\|/R_0\), \(e=\|E\|/R_0\). If \(\ell+e<1\), the exact potential exists and has inverse bound \(1/[R_0(1-\ell-e)]\). The explicit prediction

\[
\psi_{\rm pred}=d/R_0-Ld/R_0^2
\]

satisfies

\[
\boxed{\|\psi-\psi_{\rm pred}\|_2
\le\frac{e+\ell^2+e\ell}{R_0(1-\ell-e)}\|d\|_2.}
\tag{26}
\]

Proof: multiply the predicted potential by \(R_0I+L+E\). Its residual is \(Ed/R_0-L^2d/R_0^2-ELd/R_0^2\); bound it and apply the inverse bound. The right side is fourth width order when \(\beta_s=O(\delta^2)\), \(r_s=O(\delta^4)\).

Using the raw-readout predictor \(v_s^{\rm pred}=\tfrac12[d/R_0+a_s(d/R_0)''-Ld/R_0^2]\), its additional error beyond half the potential error is bounded by

\[
\frac12\left[(\beta_s+r_s)\|\psi-\psi_{\rm pred}\|_2
+\frac{r_s+(\beta_s+r_s)\ell}{R_0}\|d\|_2\right].
\tag{27}
\]

Together (26)–(27) are a concrete nonuniform theorem. Total density is constant, fitting is in a specified resolved band, and the smallness criterion is explicit. General varying \(N\) replaces the leading scalar \(N_0I\) by \(PNP\); the next bound retains the resulting projection terms.

### 5.3 A certified predictor for varying total density

Retain the same resolved space \(H_\Omega\), but now suppose \(N(c)\) varies and \(N_{\min}>0\). Let \(E_{\max}\) be the right-hand side of (25)'s remainder bound, and assume

\[
m=N_{\min}-2\|A\|_\infty\Omega^2-E_{\max}>0.
\tag{28a}
\]

Then the exact resolved frame is bounded below by \(m\). This follows from \(PNP\ge N_{\min}I\), the bound on \(L\), and the bound on \(E\). There is no unknown coefficient matrix in the condition.

Construct directly from the target and geometry

\[
u=d/N,\quad L_0=A\partial_c^2+\partial_c^2A,\quad
\psi_1=-N^{-1}L_0u,\quad \varphi=u+\psi_1.
\]

Multiplication by \(1/N\) can generate frequencies outside \(H_\Omega\). Keep their sizes explicitly:

\[
\eta_0=\|(I-P)\varphi\|_2,\qquad
\eta_2=\|\partial_c^2(I-P)\varphi\|_2.
\]

Assume the displayed derivatives are square integrable. Define the computable residual bound

\[
\mathcal R=
\|L_0\psi_1\|_2
+(N_{\max}+\Omega^2\|A\|_\infty)\eta_0
+\|A\|_\infty\eta_2
+E_{\max}\|\varphi\|_2.
\tag{28b}
\]

The exact resolved minimum-norm common potential satisfies

\[
\boxed{\|\psi-P\varphi\|_2\le\mathcal R/m.}
\tag{28c}
\]

**Proof.** The pointwise identity \((N+L_0)\varphi-d=L_0\psi_1\) follows from \(Nu=d\) and \(N\psi_1=-L_0u\). Substituting \(P\varphi\) instead of \(\varphi\) changes its projected residual by \(-P(N+L_0)(I-P)\varphi\). Its three pieces are bounded by \(N_{\max}\eta_0\), \(\|A\|_\infty\eta_2\), and \(\Omega^2\|A\|_\infty\eta_0\), using \(P\partial_c^2=\partial_c^2P\). The exact frame contributes at most \(E_{\max}\|\varphi\|_2\) beyond this second-order operator. Apply its lower bound \(m\) to the residual.

For the raw-readout prediction in (16), the corresponding explicit bound is

\[
\boxed{
\|v_s-v_s^{\rm pred}\|_2
\le\frac12\left[
(1+\beta_s+r_s)\frac{\mathcal R}{m}
+\eta_0+\|a_s\|_\infty(\|\psi_1''\|_2+\eta_2)
+r_s\|\varphi\|_2
\right].}
\tag{28d}
\]

To obtain this, expand \(T_s^*P\varphi=(I+a_s\partial_c^2+R_s)P\varphi\), subtract \(\varphi+a_su''\), and bound the omitted tail, the term \(a_s\psi_1''\), and the remainder. This proves a coefficient prediction rather than just a forward approximation statement.

If \(a_s=\delta^2\bar a_s\), densities and their required derivatives remain bounded independently of \(\delta\), and the spectral tails are negligible, every error term above is fourth width order: \(\psi_1=O(\delta^2)\), \(L_0\psi_1=O(\delta^4)\), \(r_s,E_{\max}=O(\delta^4)\). A target guard band and slowly varying geometry make the Fourier tails small. Their norms are retained rather than assumed zero, so the same certificate can detect when division by a rapidly varying density makes the local predictor unreliable. The certificate requires no target feature solve: it uses derivatives, products, and Fourier tails of known fields.

### 5.4 Why bandwidth or regularization is unavoidable

For stationary branches the canonical squared weighted norm is proportional to

\[
\int\frac{|\widehat d(\xi)|^2}{\sum_sn_sk_s(\xi)^2}\,d\xi.
\]

Finitely many fixed tanh widths make the denominator decay exponentially at large \(|\xi|\). Even a smooth target derivative with subexponential Fourier decay, such as \(\widehat d(\xi)=\exp[-(1+\xi^2)^{1/4}]\), gives an infinite norm. Thus the exact canonical encoder is not a bounded map on all smooth targets. A resolved band, suitable analytic target class, or regularization is part of the mathematical claim, not merely a numerical precaution.

Within a fixed resolved band, the stationary canonical branch has exactly the spectral support of the target derivative. If \(q_{\min}=\inf_{|\xi|\le\Omega}\sum_sn_sk_s(\xi)^2>0\), then

\[
\|\partial_c^r v_s\|_2
\le\frac{\Omega^r}{2\sqrt{n_s q_{\min}}}\|d\|_2.
\]

Here \(k_s^2\le q/n_s\) bounds the transfer, and differentiation multiplies the spectrum by \((i\xi)^r\). This is a genuine explanation of smooth canonical branches. It ceases to control arbitrary exchange components or an unresolved inverse.

## 6. Slowly varying geometry beyond the narrow-kernel regime

The moment formula requires kernels narrow relative to the resolved target. A different expansion allows broad kernels, provided geometry changes slowly and the chosen inverse remains stable.

Assume \(\gamma_s(c)=\Gamma_s(\epsilon c)\), \(n_s(c)=\mathcal N_s(\epsilon c)\), with smooth bounded profiles and positive uniform upper/lower bounds; their required derivatives are bounded independently of \(\epsilon\). Define \(k_s(c,\xi)=k_{\gamma_s(c)}(\xi)\) and \(q(c,\xi)=\sum_sn_s(c)k_s(c,\xi)^2\).

For a symbol \(a(c,\xi)\), left quantization means

\[
[\operatorname{Op}_L(a)u](c)
=\frac1{2\pi}\int e^{ic\xi}a(c,\xi)\widehat u(\xi)d\xi.
\]

It is a locally changing Fourier filter. Exactly, \(T_s^*=\operatorname{Op}_L(k_s)\). Synthesis uses the kernel width at the source center rather than the output location; conversion to a left symbol gives

\[
T_s=\operatorname{Op}_L(k_s-i\partial_c\partial_\xi k_s)+O(\epsilon^2).
\]

The composition rule \(a\# b=ab-i a_\xi b_c+O(\epsilon^2)\) then yields

\[
\mathcal S=\operatorname{Op}_L(q-\tfrac i2q_{c\xi})+O(\epsilon^2).
\tag{28}
\]

The first-order terms combine into exactly half the mixed derivative of \(q\). In symmetric/Weyl coordinates the first-order correction to the frame operator vanishes. This is a useful structural cancellation, not an arbitrary approximate inverse.

Let \(r=q+\tau b(\xi)\), where \(b=1\) for derivative-loss ridge and \(b=\xi^2\) for function-loss ridge. On a stable resolved regime, the inverse potential symbol through first geometry-variation order is

\[
B=\frac1r+\frac{i q_{c\xi}}{2r^2}
-\frac{i r_\xi q_c}{r^3}.
\tag{29}
\]

To check it, multiply \(r-iq_{c\xi}/2\) by \(1/r+B_1\), include the composition term \(-ir_\xi(1/r)_c\), and set the first-order sum to zero. This gives the two correction terms in (29).

The independent coefficient predictor is

\[
\boxed{
\rho_s^{\rm pred}=\operatorname{Op}_L(C_s)d,
\quad
C_s=n_s\left[
\frac{k_s}{r}
+\frac{i k_s q_{c\xi}}{2r^2}
+\frac{i q_c}{r^2}\left(k_{s,\xi}-\frac{k_s r_\xi}{r}\right)
\right].
}
\tag{30}
\]

For raw readouts multiply by \(h_s/2\), which cancels the leading \(n_s\). All terms are explicit functions of geometry, frequency, and a scalar target transform. There is no target-dependent feature factorization.

Standard symbol-calculus remainder estimates give an \(O(\epsilon^2)\) operator residual for fixed positive regularization and bounded profile seminorms. A useful statement retains the inverse scale: if the residual operator has norm at most \(C\epsilon^2\), and the exact potential operator is bounded below by \(m>0\), the potential error is at most \(C\epsilon^2\|d\|_2/m\); the coefficient error additionally multiplies by \(\|n_sT_s^*\|\). The constants depend on profile derivatives, width contrast, and the reciprocal resolved spectral bound. They are not uniform as regularization vanishes or all kernel spectra become tiny. On the unregularized full line, tanh smoothing has no bounded inverse on all L2; claiming a global uniform remainder there would be false.

The calculation uses the standard pseudodifferential composition framework developed in [Evans–Zworski's lectures](https://math.mit.edu/~vwg/semiclassicalEvansZworski.pdf). Formula (30) is its specialization to the current kernel and coefficient convention. Section 5 supplies an elementary fully quantified restricted bound without relying only on asymptotic notation.

## 7. Variable-width branch compatibility and proportional widths

Put \(u_s=w_s\rho_s\). Canonical encoding gives \(u_s=T_s^*\psi\), so

\[
T_s^*u_t-T_t^*u_s=[T_s^*,T_t^*]\psi.
\tag{31}
\]

The leading commutator symbol is

\[
-i[k_{s,\xi}k_{t,c}-k_{t,\xi}k_{s,c}].
\tag{32}
\]

Write \(k_s=H(\xi/\gamma_s(c))\), with \(H(z)=(\pi z/2)/\sinh(\pi z/2)\). Direct differentiation gives

\[
k_{s,\xi}k_{t,c}-k_{t,\xi}k_{s,c}
=\frac{\xi H'(\xi/\gamma_s)H'(\xi/\gamma_t)}{\gamma_s\gamma_t}
\partial_c\log\frac{\gamma_s}{\gamma_t}.
\tag{33}
\]

Thus proportional width fields, \(\gamma_s(c)=a_s^0\Gamma(c)\) with constants \(a_s^0\), have zero first-order commutator. Their branch compatibility survives to \(O(\epsilon^2)\), despite substantial slow variation in their common width. Independent width trends generate a first-order, explicitly predicted phase correction. This is an actionable structural test for smooth branches with related gamma curves.

A globally warped stationary family makes the idea exact. If \(z=g(x)\) is an increasing chart, define the transported analysis operator

\[
(R_s\psi)(c)=\int g'(x)K_{\gamma_s^0}(g(x)-g(c))\psi(x)dx.
\]

With \((U\psi)(z)=\psi(g^{-1}(z))\), it equals \(R_s=U^{-1}(K_s*)U\). Every pair \(R_s,R_t\) therefore commutes exactly. Ordinary variable-width symmetric bumps only approximate these transported kernels; curvature of \(g\) generates an asymmetry/drift term. For example, their analysis difference starts with \(-\pi^2g''\psi'/(24(\gamma_s^0)^2g'^3)\), obtained by Taylor expanding \(g^{-1}(g(c)+t)\). Exact chart transport should not be silently equated with merely changing gamma at each center.

## 8. Finite spacing: what aliases do to the coefficient selection

### 8.1 Exact alias organization

For clarity take equal aligned lattice spacing \(h\), constant widths, and branch fields bandlimited to \((-\pi/h,\pi/h)\). Poisson summation applied to \(\sum_jh\rho_s(jh)K_s(x-jh)\) gives, for a base frequency \(\xi\) and \(\xi_l=\xi+2\pi l/h\),

\[
\widehat d_h(\xi_l)=\sum_s k_s(\xi_l)\widehat\rho_s(\xi).
\tag{34}
\]

Offsets add known phase factors and do not remove the basic distinction. If the target has only the base component, function-value LS weights the alias derivative errors by \(1/|\xi_l|^2\), because a primitive divides by \(i\xi_l\). On every nonzero base frequency the coefficient quadratic has Gram matrix

\[
G_h(\xi)=\frac{k(\xi)k(\xi)^*}{\xi^2}+E_h(\xi),
\qquad
E_h=\sum_{l\ne0}\frac{k(\xi_l)k(\xi_l)^*}{\xi_l^2}.
\tag{35}
\]

Here \(k\) is the column vector of branch transfers. After whitening a nontrivial coefficient metric, the same formula holds with the correspondingly scaled vector. The first term has rank one. Tiny aliases act precisely on its otherwise free branch-exchange directions.

### 8.2 A counterexample: negligible aliases need not mean canonical coefficients

Take two branches with fixed \(\gamma_1<\gamma_2\), a fixed nonzero base sinusoidal frequency, and let \(h\to0\). A precise finite-energy setting is a circle of length \(2\pi\), periodized kernels, integer target frequency, and \(h=2\pi/M\) with integer \(M\to\infty\); then aliases are the modes \(\xi+lM\). Branch 1 is broader, so its alias transfers decay exponentially faster:

\[
\frac{k_1(\xi_l)}{k_2(\xi_l)}\to0
\quad\text{for the nearest nonzero aliases.}
\]

The continuum equal-norm encoding allocates nonzero coefficients to both branches. Exact unregularized finite-lattice function LS instead satisfies

\[
\widehat\rho_2(\xi)\to0,
\qquad
\widehat\rho_1(\xi)\to\widehat d(\xi)/k_1(\xi).
\tag{36}
\]

The reason can be proved with only two rows, without an SVD. A competitor using branch 1 alone matches the principal frequency exactly and has total loss \(R_h^2\), where \(R_h=O(h\max_{l=\pm1}k_1(\xi_l))\). The optimizer has no larger loss. Its principal-frequency residual is therefore \(O(R_h)\), so

\[
k_1(\xi)\rho_1+k_2(\xi)\rho_2=\widehat d+O(R_h).
\]

Its nearest alias residual is \(O(R_h/h)\). Substitute the preceding expression for \(\rho_1\) into that alias equation. The coefficient of \(\rho_2\) is

\[
k_2(\xi_l)-k_1(\xi_l)k_2(\xi)/k_1(\xi),
\]

which is asymptotic to \(k_2(\xi_l)\). The remaining terms are \(O(\max_{l=\pm1}k_1(\xi_l))\). Their ratio tends to zero exponentially, proving \(\rho_2\to0\); the principal row then proves the other limit. Polynomial prefactors and the fixed positive/negative alias asymmetry do not change the exponential conclusion.

Both coefficient selections produce an excellent function. The vanishing perturbation selects different coordinates along an asymptotically unidentifiable exchange direction. This is why a minimum-norm continuum branch formula cannot be claimed as a prediction of unregularized finite-lattice coefficients solely because function aliasing is tiny.

### 8.3 The precision/regularization window that restores canonical allocation

Let \(a_{h,\tau}\) solve the whitened alias problem with coefficient ridge \(\tau I\), and \(a_{0,\tau}\) the problem retaining only the principal frequency. Then

\[
a_{0,\tau}=\frac{k(\xi)\widehat d(\xi)}{\|k(\xi)\|^2+\tau\xi^2}.
\]

The resolvent identity gives

\[
a_{h,\tau}-a_{0,\tau}
=-(G_h+\tau I)^{-1}E_h a_{0,\tau},
\]

and, since \(G_h\) is positive semidefinite,

\[
\boxed{\|a_{h,\tau}-a_{0,\tau}\|
\le\frac{\|E_h\|}{\tau}\|a_{0,\tau}\|.}
\tag{37}
\]

The retained signal eigenvalue is \(\lambda_{\rm signal}=\|k(\xi)\|^2/\xi^2\). Define the following computable upper bound for the alias perturbation:

\[
\|E_h\|\le\lambda_{\rm alias}
:=\sum_{l\ne0}\frac{\|k(\xi_l)\|^2}{\xi_l^2}.
\]

For a desired small fractional tolerance \(\eta\), it is sufficient to choose

\[
\boxed{\lambda_{\rm alias}/\eta\le\tau\le\eta\lambda_{\rm signal}.}
\tag{38}
\]

This window exists whenever \(\lambda_{\rm alias}\le\eta^2\lambda_{\rm signal}\). The lower inequality bounds branch contamination in (37) by \(\eta\); the upper inequality bounds principal-frequency shrinkage by at most \(\eta\). This is a hard coefficient-selection criterion, not merely “small aliases.” A singular-value cutoff implements a related resolution choice but must be compared in squared-singular-value/Gram units before using this criterion.

## 9. Irregular center quadrature and the scope of the prediction

For ordered centers and trapezoidal weights \(h_j=(c_{j+1}-c_{j-1})/2\), let \(H=\sup_j(c_{j+1}-c_j)\). If \(F_x(c)=K_{\gamma(c)}(x-c)\rho(c)\) has integrable second derivative and suitable endpoint/decay behavior, the composite trapezoid error satisfies

\[
\left|\sum_jh_jF_x(c_j)-\int F_x(c)dc\right|
\le\frac{H^2}{8}\|\partial_c^2F_x\|_{L^1(dc)}.
\tag{39}
\]

To see this, use the Peano kernel \((c-a)(b-c)/2\) on each interval and its bound \((b-a)^2/8\), then sum. The derivative inside the bound includes geometry terms: writing \(r=x-c\),

\[
\partial_c^2K_{\gamma(c)}(r)
=K_{rr}-2\gamma'K_{r\gamma}+\gamma''K_\gamma+(\gamma')^2K_{\gamma\gamma}.
\]

Thus large gaps, abrupt width changes, or rapidly varying coefficient fields invalidate the dense-quadrature approximation in specific ways. The bound is conservative; on a stationary analytic lattice the exact alias formula is much sharper. Importantly, (39) bounds the synthesized function, not the stability of its individual coefficients. Sections 8.2–8.3 explain the additional inverse-stability requirement.

The resulting predictive class is now explicit: smooth branch densities and widths; a stated coefficient metric; target frequencies supported where the common transfer is resolved; and either an exact continuum minimum-norm convention or a discrete regularization/precision choice satisfying the alias window. Under these conditions the branch curves are filters of a common potential, with the local formulas (16)–(22), the broader-band correction (30), and compatibility conditions (8), (31)–(33). Rough exchange components, unresolved spectra, folds in the branch indexing, or uncontrolled lattice null directions are precise ways this explanation can cease to apply.

The relevant precedents are [nonuniform sampling in shift-invariant spaces](https://epubs.siam.org/doi/10.1137/S0036144501386986), [nonstationary Gabor frames with explicit duals in a structured case](https://pmc.ncbi.nlm.nih.gov/articles/PMC3257872/), and [localization of canonical dual frames](https://www.sbai.uniroma1.it/pubblicazioni/doc/preprints/abs/04-08-for-gro.pdf). These works supply stability and localization tools under frame assumptions. They do not automatically establish those assumptions for every learned neuron dictionary, nor do they select the metric or precision convention for its observed raw weights. The formulas here specialize that framework to the current multibranch tanh problem and expose those missing choices.

## 10. A different regime: broad mixed widths encode polynomial jets

The preceding sections assumed a dense, resolved kernel representation. There is another controlled regime in which the raw readouts have a different interpretation. Keep a finite number of centers fixed and send every inverse width to zero. The resulting functions are global polynomial combinations produced by cancellation. A small width edit can change which polynomial combination is available; it need not merely deform a local derivative-reading branch.

Work on a compact interval with a fixed finite positive fitting measure \(\mu\). Write \(\mu_0=\int d\mu>0\), and suppose its moment matrix is positive definite through every degree used below. An interval with positive density suffices. A free output bias is fitted independently. Thus the neuron features relevant to readout fitting are their centered versions in the Hilbert space \(L^2_0(\mu)\),

\[
\phi_{j,\varepsilon}(x)=\tanh\!\bigl(\varepsilon g_j(x-c_j)\bigr)
-\mu_0^{-1}\int\tanh\!\bigl(\varepsilon g_j(t-c_j)\bigr)d\mu(t),
\]

where \(g_j>0\) and \(c_j\) are fixed, while \(\varepsilon\downarrow0\). Let \(u_k(x)=x^k-\mu_0^{-1}\int t^k d\mu(t)\). The centered synthesis operator \(\Phi_\varepsilon:\mathbb R^N\to L^2_0(\mu)\) maps a readout vector \(v\) to \(\sum_jv_j\phi_{j,\varepsilon}\). Its singular values quantify how much cancellation is required in each independent function direction.

### 10.1 Three neurons: one width edit can change the rank scale

Taylor expansion, uniform on the fitting interval, gives

\[
\phi_{j,\varepsilon}
=\varepsilon g_j u_1
-\frac{\varepsilon^3g_j^3}{3}
\left(u_3-3c_ju_2+3c_j^2u_1\right)+O(\varepsilon^5).
\tag{40}
\]

Define the geometry matrix

\[
R=\begin{pmatrix}
g_1&g_2&g_3\\
g_1^3c_1&g_2^3c_2&g_3^3c_3\\
g_1^3&g_2^3&g_3^3
\end{pmatrix}.
\tag{41}
\]

If \(\det R\ne0\), the singular-value scales are

\[
\boxed{\sigma_1(\Phi_\varepsilon)=\Theta(\varepsilon),\qquad
\sigma_2(\Phi_\varepsilon)=\Theta(\varepsilon^3),\qquad
\sigma_3(\Phi_\varepsilon)=\Theta(\varepsilon^3).}
\tag{42}
\]

Here and below the constants can depend on the fixed geometry and fitting measure. To prove (42), first remove the leading rank-one map \(\varepsilon u_1g^T\), where \(g=(g_1,g_2,g_3)^T\). On its right and left orthogonal complements the first remaining map is

\[
\varepsilon^3P_{u_1^\perp}
\left[u_2(g^3c)^T-\tfrac13u_3(g^3)^T\right]P_{g^\perp}.
\]

Its rank is two exactly when the three rows in (41) are independent. The projected input functions \(u_2,u_3\) are independent by the moment assumption. Standard finite-dimensional block elimination, or equivalently the min-max characterization of singular values applied after bounded changes of basis, proves the stated scales. The \(u_1(g^3c^2)^T\) term changes the strong direction but adds no direction in its left complement.

There is also a determinant check. Put \(M_3=(\int x^{k+l}d\mu)_{k,l=0}^3\). The Gram matrix \(G_\varepsilon=\Phi_\varepsilon^*\Phi_\varepsilon\) satisfies

\[
\det G_\varepsilon
\sim\frac{(\det R)^2\det M_3}{9\mu_0}\,\varepsilon^{14}.
\tag{43}
\]

Indeed the determinant of the first three polynomial coefficient rows in (40) is \(-\varepsilon^7\det R/3+O(\varepsilon^9)\), and the Gram determinant of \(u_1,u_2,u_3\) is \(\det M_3/\mu_0\).

For a concrete single edit, let every \(g_j=g\), except \(g_k=g(1+\delta)\), with fixed \(\delta>-1\). Then

\[
\det R=g^7(1+\delta)(2\delta+\delta^2)
\det\begin{pmatrix}1&1&1\\c_1&c_2&c_3\\(e_k)_1&(e_k)_2&(e_k)_3\end{pmatrix}.
\tag{44}
\]

For distinct centers this is nonzero whenever \(\delta\ne0\). The common-width singular scales are \(\varepsilon,\varepsilon^3,\varepsilon^5\); a fixed nonzero relative width edit changes them to (42). This improves the weakest power of \(\varepsilon\), but the improvement is not uniform as \(\delta\to0\). It does not establish that any width edit at any scale improves conditioning.

### 10.2 Several edited widths and centers: the polynomial-degree flag

For each \(\ell\ge0\), form a geometry matrix with the following rows, interpreted componentwise over neurons:

\[
R_\ell=\begin{pmatrix}
g\\g^3c\\g^3\\g^5c\\g^5\\\vdots\\g^{2\ell+1}c\\g^{2\ell+1}
\end{pmatrix},
\qquad R_0=(g).
\tag{45}
\]

The sufficient rank conditions are

\[
\operatorname{rank}R_\ell=\min(N,2\ell+1)
\quad\text{at every level up to full rank.}
\tag{46}
\]

Under these conditions the singular-value powers are

\[
\varepsilon,\quad
\varepsilon^3,\varepsilon^3,\quad
\varepsilon^5,\varepsilon^5,\quad\ldots,
\tag{47}
\]

stopping after \(N\) entries. The proof is an induction on the Taylor order. After order \(2\ell-1\), the already resolved left space is the centered polynomials through degree \(2\ell-1\). In the next Taylor term, all lower-degree output components lie in that space and are removed by block elimination. The two new highest-degree coefficients are nonzero constant multiples of the rows \(g^{2\ell+1}c\) and \(g^{2\ell+1}\), multiplying \(x^{2\ell}\) and \(x^{2\ell+1}\). Condition (46) states that they add the two required independent right directions, or the final one if only one remains. Corrections used to eliminate earlier blocks alter higher-order terms but not this leading quotient map.

These rank conditions hold away from proper algebraic exceptional sets. For example, choose distinct positive \(g_j\) and \(c_j=g_j\). After dividing each column by \(g_j\), the rows in (45) are distinct monomial powers \(0,3,2,5,4,\ldots\). Their generalized Vandermonde matrices have full row or column rank on ordered positive nodes. This supplies one explicit full-rank geometry at each size; a nonzero minor cannot vanish identically as a polynomial in all geometries.

For odd \(N\), the limiting function space, including bias, is \(\mathcal P_N\). For even \(N\), the last available direction is generally a geometry-dependent combination of degrees \(N\) and \(N+1\), modulo lower degrees. The limiting space need not be \(\mathcal P_N\). Already two different-width neurons centered at zero have limiting nonconstant directions \(x,x^3\), not \(x,x^2\). This distinction is essential when predicting approximation quality from geometry.

### 10.3 A target-to-readout prediction after multiple geometry edits

Take odd \(N=2q+1\), impose (46), and keep the geometry fixed while \(\varepsilon\to0\). Write the odd Taylor coefficients of tanh as \(\tanh z=\sum_{m\ge0}t_{2m+1}z^{2m+1}\), where \(t_1=1,t_3=-1/3,t_5=2/15\). Define an \(N\)-by-\(N\) geometry matrix by

\[
(B_\varepsilon)_{kj}
=\sum_{\substack{0\le m\le q\\2m+1\ge k}}
t_{2m+1}\varepsilon^{2m+1}g_j^{2m+1}
\binom{2m+1}{k}(-c_j)^{2m+1-k},
\quad1\le k,j\le N.
\tag{48}
\]

It contains the coefficients of powers \(x^1,\ldots,x^N\) in the truncated neuron polynomials; centering only replaces those powers by \(u_k\). All of its entries are geometry moments. No target samples enter this matrix.

Compute once the centered polynomial projection of the target,

\[
P_{\mathcal P_N}f-\mu_0^{-1}\int f\,d\mu
=\sum_{k=1}^Np_ku_k.
\]

The vector \(p\) comes from the \(N\) target moments \(\langle f,u_k\rangle\) and the fixed polynomial Gram matrix \(\langle u_k,u_l\rangle\). For an explicitly known target these may be analytic integrals. They are properties of the target and fitting measure, reusable after geometry changes. The readout prediction is

\[
\boxed{v_{\rm pred}=B_\varepsilon^{-1}p.}
\tag{49}
\]

This small inverse converts polynomial coordinates into neuron coordinates. Its content is the explicit identification of which target coordinates are being encoded and the error control below, not the claim that matrix inversion itself is a new algorithm.

Here is the accuracy statement. Let \(U:\mathbb R^N\to L^2_0(\mu)\) map \(p\) to \(\sum p_ku_k\). Taylor's theorem and (47) imply

\[
\Phi_\varepsilon=UB_\varepsilon+E_\varepsilon,
\quad\|E_\varepsilon\|=O(\varepsilon^{N+2}),
\quad\|B_\varepsilon^{-1}\|=O(\varepsilon^{-N}).
\]

Consequently

\[
\Phi_\varepsilon B_\varepsilon^{-1}=U+O(\varepsilon^2)
\quad\text{as maps into }L^2_0(\mu).
\tag{50}
\]

Since \(U\) is fixed and injective, its Gram matrix has a positive lower eigenvalue. The least-squares coordinates in the normalized basis on the left of (50) therefore vary continuously, with an \(O(\varepsilon^2)\) perturbation. If \(v_{\rm LS}\) is the exact least-squares readout in the actual neuron basis,

\[
\boxed{\|B_\varepsilon v_{\rm LS}-p\|
\le C\varepsilon^2\|f\|_{L^2(\mu)},\qquad
\|v_{\rm LS}-v_{\rm pred}\|
\le C'\varepsilon^{2-N}\|f\|_{L^2(\mu)}.}
\tag{51}
\]

The constants are bounded on compact geometry sets satisfying the rank conditions with a quantitative margin. The second bound is a small relative error only when the target actually excites the leading \(\varepsilon^{-N}\) readout scale. It is the stable polynomial-coordinate statement, not an unconditional componentwise relative-error claim, that always follows. If the target is itself in \(\mathcal P_N\), the synthesized error from (49) is \(O(\varepsilon^2)\).

For three neurons (48) is particularly transparent:

\[
B_\varepsilon=\begin{pmatrix}
\varepsilon g_j-\varepsilon^3g_j^3c_j^2\\
\varepsilon^3g_j^3c_j\\
-\varepsilon^3g_j^3/3
\end{pmatrix}_{j=1}^3.
\tag{52}
\]

The quadratic and cubic target coordinates determine the leading \(\varepsilon^{-3}\) readouts, while the linear target coordinate enters at a weaker scale. A simultaneous change to all three centers and widths changes only this explicit geometry conversion, while the target moments remain fixed. Near a failed rank condition, or on a geometry path depending on \(\varepsilon\), the constants can diverge and this predictor must be replaced by the appropriate different polynomial-degree flag.

### 10.4 A small width edit can cancel the weakest mode instead

There is an explicit exception to a blanket “width diversity helps” claim. Fix \(a\ne0\), use outer centers \(-a,+a\) with inverse width \(\gamma\), and a center at zero with inverse width \(\Gamma\). Put

\[
O(x)=\frac{\tanh\gamma(x-a)+\tanh\gamma(x+a)}2,
\qquad s=\sech^2(\gamma a).
\]

The odd outer combination has

\[
O'(0)=\gamma s,\qquad
\frac{O'''(0)}{O'(0)}=\gamma^2(4-6s).
\]

Choose the middle width along the curve

\[
\boxed{\Gamma^2=\gamma^2(3s-2),}
\qquad s>2/3.
\tag{53}
\]

For small \(\gamma\), \(\Gamma/\gamma=1-\tfrac32a^2\gamma^2+O(\gamma^4)\): this is only an order-\(\gamma^2\) relative width edit. Nevertheless it makes the normalized outer and middle features agree through their first and third derivatives at zero. To see the next surviving term, define

\[
D(x)=\frac{O(x)}{O'(0)}-\frac{\tanh\Gamma x}{\Gamma}.
\]

Direct differentiation yields

\[
D'(0)=D'''(0)=0,\qquad
D^{(5)}(0)=-24\gamma^4(1-s)(2-s).
\tag{54}
\]

Indeed \(O^{(5)}(0)/O'(0)=\gamma^4(16-120s+120s^2)\), while the middle feature contributes \(16\Gamma^4\). Substituting (53) and subtracting gives (54). Taylor expansion on any fixed compact interval therefore gives

\[
D(x)=-\frac{a^2}{5}\gamma^6x^5+O(\gamma^8),
\quad
O(x)-\frac{O'(0)}{\Gamma}\tanh\Gamma x
=-\frac{a^2}{5}\gamma^7x^5+O(\gamma^9).
\tag{55}
\]

The remainder is uniform on that interval. The \(x^7\) coefficients cancel at their otherwise leading order as well; this follows either from the tanh series and \(\Gamma/\gamma=1+O(\gamma^2)\), or by expanding the two analytic expressions in \(\gamma\).

After removing the output bias, a bounded change of the three feature coordinates now gives: a strong odd feature \(\gamma x+O(\gamma^3)\); the outer even difference \(a\gamma^3u_2+O(\gamma^5)\), up to an irrelevant sign convention; and the weak combination (55), centered if the fitting measure is not symmetric. Their leading polynomials are independent. Hence the singular scales are

\[
\boxed{\gamma,\quad\gamma^3,\quad\gamma^7,}
\tag{56}
\]

and the limiting span including bias is

\[
\boxed{\operatorname{span}\{1,x,x^2,x^5\},}
\tag{57}
\]

with the weak singular function equal to \(x^5\) after orthogonalization against the lower-degree directions. The ordinary common-width three-neuron limit instead has singular scales \(\gamma,\gamma^3,\gamma^5\) and limiting space \(\mathcal P_3\).

This is a precise mechanism by which a tiny geometry edit can both worsen cancellation and change the target property encoded by the weak readout direction. It does not contradict (42)–(47), whose width ratios were fixed independently of the flatness parameter. It warns against extending the local derivative-reading picture across a geometry-dependent loss of rank without checking which polynomial directions survive.

There is an exact approximation-floor prediction. Take \(f(x)=x^3\) and uniform fitting measure on \([-1,1]\). Symmetry removes the constant and \(x^2\) directions from its projection onto (57). Write the remaining approximation as \(ax+bx^5\). Its two orthogonality equations, after dividing the integrals by two, are

\[
\frac a3+\frac b7=\frac15,\qquad
\frac a7+\frac b{11}=\frac19.
\]

They give \(a=7/30\), \(b=77/90\). Consequently

\[
\left\|x^3-\frac7{30}x-\frac{77}{90}x^5\right\|_{L^2(-1,1)}^2
=\frac{32}{14175},\qquad \|x^3\|_2^2=\frac27,
\]

and the exact neuron least-squares error along the tuned curve obeys

\[
\boxed{\lim_{\gamma\to0}
\frac{\|f-\widehat f_{\rm LS,\gamma}\|_2}{\|f\|_2}
=\frac4{45}\approx8.8889\%.}
\tag{57a}
\]

The normalized feature-basis convergence proved above implies convergence of the projection and hence this error limit. Along the unedited common-width geometry, the limiting space is \(\mathcal P_3\), so the same target's relative error tends to zero. This is a persistent approximation-floor change caused by a relative width edit that itself tends to zero. It concerns exact arithmetic and unrestricted readout coefficients; a finite cutoff or coefficient penalty changes the limiting space again.

### 10.5 What extends beyond tanh

The polynomial-coordinate argument uses analyticity and a quantitative rank condition, not a special inverse for tanh. If an activation has a convergent Taylor series \(\sigma(z)=\sum_{k\ge0}a_kz^k\) near the relevant flat preactivations, its degree-\(m\) geometry coefficients are

\[
J_{mj}(\varepsilon)=\sum_{k\ge m}a_k\varepsilon^kg_j^k
\binom{k}{m}(-c_j)^{k-m}.
\tag{58}
\]

Choose a finite truncation and a limiting polynomial space for which its coordinate matrix is invertible. If the discarded feature remainder has norm at most \(r_\varepsilon\), and the inverse coordinate matrix has norm at most \(b_\varepsilon\), the normalized-basis error is at most \(r_\varepsilon b_\varepsilon\). Whenever this tends to zero and the limiting polynomial Gram matrix has a positive lower bound, the proof of (50)–(51) applies with \(O(\varepsilon^2)\) replaced by \(O(r_\varepsilon b_\varepsilon)\). Missing Taylor coefficients, activation symmetries, and geometry-dependent cancellations determine the polynomial-degree flag. Analyticity alone does not establish a particular flag or a useful condition number.

This extension is separate from the positive-kernel smoothing interpretation. For example, standard GELU is \(\sigma(x)=x\Phi(x)\), where \(\Phi\) is the standard normal cumulative distribution function. Its normalized second-derivative kernel is

\[
\sigma''(x)=(2-x^2)\varphi(x),\qquad
\widehat{\sigma''}(\xi)=(1+\xi^2)e^{-\xi^2/2},
\]

with \(\varphi\) the standard normal density. Its integral is one but its second moment is \(2\mathbb E Z^2-\mathbb E Z^4=-1\). Thus its long-wave multiplier starts with \(1+\xi^2/2\), rather than the \(1-a\xi^2\) of a positive smoothing kernel. The geometry-to-target operator framework survives, but a claim that all activations merely blur the same derivative would be false even for this common activation.
