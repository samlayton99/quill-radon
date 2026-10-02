# Kernel-ratio compatibility audit

Date: 2026-09-14. The old proof is `QIs_workshop.pdf`; the **new proof** is `theorem_for_sam.pdf`. The new proof arrived after the historical reconstruction below was drafted.

## Current conclusion

**The new proof strengthens compatibility with the amplitude-matched kernel-ratio argument.** Its explicit complex-shift density deconvolves the normalized sech-squared kernel exactly on each Fourier mode. The old cardinal normalizer's desired-amplitude deficit is absent. Moreover, twice the old tanh output-ratio bound directly bounds the new proof's finite-contour replica term, not only an auxiliary infinite lattice. The factor two comes from anchoring the output at \(x_0=-1\).

The general kernel/derivative-order lattice identity remains valid. The new finite-interval construction and the exact finite-contour connection established here concern **tanh, derivative order one**. The new paper does not establish the analogous boundary theorem for arbitrary activations.

## 1. The new density matches the desired amplitude

The new proof defines, on p. 3, equation (A.2),

\[
d=\frac{\pi}{2\gamma},\qquad
K_\gamma(t)=\frac\gamma2\operatorname{sech}^2(\gamma t),\qquad
a_\gamma[f](z)=\frac{f(z+id)-f(z-id)}{2id}.
\]

For \(f(z)=b e^{i\omega z}\), let \(\theta=h\omega\), \(\lambda=\gamma h\), and

\[
H(\xi)=\frac{\pi\xi/2}{\sinh(\pi\xi/2)}.
\]

Direct substitution gives

\[
\boxed{a_\gamma[f](z)=\frac{i\omega}{H(\omega/\gamma)}b e^{i\omega z}.}
\]

Convolution by \(K_\gamma\) therefore multiplies this density by exactly the factor needed to recover \(f'\). Sampling that density on the lattice produces derivative replica ratios

\[
\alpha_m(\lambda,\theta)
=\frac{H((\theta+2\pi m)/\lambda)}{H(\theta/\lambda)},
\]

while integration changes the output coefficients to

\[
t_m=\frac{\theta}{\theta+2\pi m}\alpha_m
=\frac{\sinh(\pi\theta/(2\lambda))}
{\sinh(\pi(\theta+2\pi m)/(2\lambda))}.
\]

For nonzero modes, there is no \(1/(1+R_0)\) divisor. The new local density is different from the old cardinal coefficient map.

The new anchored network uses \(D_x(t)=\tfrac12[\tanh(\gamma(x-t))-\tanh(\gamma(x_0-t))]\). Its ideal full-lattice output consequently satisfies

\[
q_\infty(x)-f(x)
=b\sum_{m\ne0}t_m
\left[e^{i\omega x+i2\pi m(x+1)/h}-e^{-i\omega}\right].
\]

Writing \(R_1=\sum_{m\ne0}|t_m|\), this proves

\[
\|q_\infty-f\|_\infty\le2|b|R_1.
\]

The unanchored Fourier antiderivative admits the old \(|b|R_1\) bound. The anchoring convention explains the difference. Constants have zero density and are represented exactly by the bias.

## 2. The same bound controls the actual finite-contour pole term

The new error decomposition is, p. 10, equation (C.1),

\[
f-q^*=(\mathcal B_h^{\rm eff}f-q_{\rm bd})
+\mathcal P_h[F_x]-\mathcal H_h[F_x].
\]

Here the replica contribution is the finite meromorphic residue term \(\mathcal P_h\), not the old cardinal-normalization deficit. Set

\[
s=\frac{\pi|\theta|}{2\lambda},\qquad A=\frac{\pi^2}{\lambda},\qquad 0<|\theta|<\pi.
\]

The exact output-ratio sum has two equivalent representations:

\[
\begin{aligned}
R_1
&=\sinh(s)\sum_{m\ge1}
\big[\operatorname{csch}(mA-s)+\operatorname{csch}(mA+s)\big]\\
&=4\sinh(s)\sum_{\ell\ge0}
\frac{\cosh((2\ell+1)s)}{e^{(2\ell+1)A}-1}.
\end{aligned}
\]

To prove the equality, expand each csch as
\(\operatorname{csch}(v)=2\sum_{\ell\ge0}e^{-(2\ell+1)v}\)
and sum the resulting geometric series over \(m\). All terms are positive and \(s<A\), so the interchange is justified.

The finite contour contains poles at \(x\pm i(2\ell+1)d\) and \(x_0\pm i(2\ell+1)d\), \(0\le\ell<k\). The tanh residues have magnitude \(1/(2\gamma)\). On a Fourier mode, the two imaginary signs contribute the factor \(2\cosh((2\ell+1)s)\); the two real locations contribute another factor two. Equations (A.14) and (A.16), pp. 4–5, therefore give

\[
\boxed{\|\mathcal P_h[F_x(be^{i\omega\cdot})]\|_\infty
\le8|b|\sinh(s)\sum_{\ell=0}^{k-1}
\frac{\cosh((2\ell+1)s)}{e^{(2\ell+1)A}-1}
\le2|b|R_1.}
\]

Only the scalar positive sum is extended to infinity; this step does not evaluate the target outside the permitted local tube. For a finite Fourier sum \(P=\sum_jb_je^{i\omega_jx}\), linearity yields

\[
\boxed{\|\mathcal P_h[F_x(P)]\|_\infty
\le2\sum_j|b_j|R_1(\lambda,h\omega_j).}
\]

Thus any valid compact first-pair upper bound on \(R_1\), multiplied by two, controls the actual new pole term. It remains only one component of the complete representation error.

### General locally analytic targets

The new theorem applies to arbitrary locally analytic targets without global Fourier assumptions. Its pole estimate

\[
\|\mathcal P_h[F_x(f)]\|_\infty
\le\frac{16\pi B h}{\lambda\delta}\,\mathcal U(A)
\]

remains the certificate for that class. A Fourier mode/sum is an exact way to explain and refine the mechanism. Replacing an arbitrary target by a Fourier approximant requires care: approximation on the real interval alone does not control the complex evaluations in \(a_\gamma\) and the residues.

A precise extension is possible whenever the residual density is controlled on the used contour. If

\[
\varepsilon_a=\sup_{t\in J,\,|y|\le\sigma}
|a_\gamma[f-P](t+iy)|,
\]

then the same residue calculation gives the rigorous remainder

\[
\|\mathcal P_h[F_x(f)]\|_\infty
\le2\sum_j|b_j|R_1(\lambda,h\omega_j)
+\frac{4\pi\varepsilon_a}{\gamma}\,\mathcal S(A).
\]

This states the additional approximation requirement explicitly. It is unnecessary if the one-page section uses Fourier components for intuition and keeps the new analytic-class bound for its general guarantee.

## 3. What the width and bandwidth dependence mean

For \(|\theta|/\lambda\ll1\) and small \(\lambda\), the nearest pair gives

\[
R_1\sim\frac{2\pi|\theta|}{\lambda}e^{-\pi^2/\lambda}
=\frac{2\pi h|\omega|}{\lambda}e^{-\pi^2/\lambda}.
\]

The new contour estimate has the same \(h\lambda^{-1}e^{-\pi^2/\lambda}\) dependence, with the local derivative bound \(B/\delta\) replacing mode-specific frequency/amplitude information and conservative constants accounting for anchoring and all poles. Since \(N\le W\le2N\), it becomes the stated \(W^{-1}\lambda^{-1}e^{-\pi^2/\lambda}\) term.

The exponential \(e^{-\pi^2/\lambda}\) is therefore the same tanh-replica mechanism as before. The new proof makes the output integration factor and the width dependence visible in the certified theorem. A practical zero-frequency kernel ratio remains a useful bandwidth prior, but is not literally the new output error.

The new boundary correction changes the separate boundary contribution from a simple halo tail to a bound proportional to \(h e^{-\lambda R^2/4}\). This is why \(R\) proportional to \(\sqrt N\) suffices. It does not cancel the replica term. The precise selection rule should allocate a replica budget and choose the largest admissible \(\lambda\) satisfying that budget, while keeping the boundary and numerical terms controlled.

## 4. Targeted proof check

I independently checked the finite-rectangle residue signs in A.2, the midpoint contour signs and pole factors in A.3–A.4, the need to correct \(\mathcal B_h-\mathcal V_h\) together, the rational correction's realization in the existing halo neurons, the scaled partial-fraction formula and width-independent bound in B.3, and the \(N\)-to-\(W\) estimates and representation constants in C. **No material defect was found in those checked steps.** This is a focused check of the representation proof and compatibility bridge, not a review of all finite-data and arithmetic results.

For a numerical algebra check at \(\lambda=1\), \(N=32\), \(\omega=2\pi\), the frequency sum and pole-layer sum both give \(R_1=0.0001625588918089375\), agreeing to relative error \(1.7\times10^{-16}\). Direct summation of the anchored tanh lattice agrees with the derived mode formula within \(5.8\times10^{-16}\) at five core points. These checks support the identities; the derivations above supply the proof.

---

# Historical baseline: the old cardinal QI

Sources for this historical comparison: `QIs_workshop.pdf`, especially pp. 17–30, and `bundle-1/choosing_optimal_lambda/choosing_optimal_lambda_appendix.tex`. The new proof has since arrived; the leading section above contains the current compatibility conclusion.

## Conclusion for the intended subsection

The older compact kernel-ratio argument is consistent with the supplied QI proof **when its object is stated accurately**: it bounds unwanted frequencies after the retained output amplitudes have been matched. The prescribed cardinal QI has an additional amplitude deficit at each retained frequency. Integrating a derivative QI suppresses high-frequency replicas but does not correct that deficit. Thus the output-order factor can remain the centerpiece of an amplitude-matched theorem; it cannot silently replace the prescribed QI's alias factor.

The supplied appendix already makes this distinction correctly. The old PDF's Lemma 7 and Lemma 10 establish the same bridge, rather than invalidating it. No global Fourier-decay assumption on the original target is required for this connection: it applies to the auxiliary finite Fourier approximant used by Theorem 13.

## Exact lattice response

Use the old paper's conventions

\[
x_k=-1+kh,\quad h=2/N,\quad u=(x+1)/h,\quad \lambda=\gamma h,
\quad K_\gamma(x)=\gamma K(\gamma x).
\]

For a retained physical frequency \(\omega\), put \(\theta=h\omega\), \(|\theta|<\pi\), and define

\[
H_m=\widehat K((\theta+2\pi m)/\lambda),\qquad
\alpha_m=H_m/H_0,\qquad
R_r=\sum_{m\ne0}\left|\frac{\theta}{\theta+2\pi m}\right|^r\alpha_m.
\]

Fourier positivity K5 makes the ratios positive. In particular, \(R_0=\sum_{m\ne0}\alpha_m\). A constant normalization of the kernel, such as choosing \(\tfrac12\operatorname{sech}^2\) rather than \(\operatorname{sech}^2\), cancels from every ratio.

The paper's normalizer and cardinal transform are

\[
D_h(\omega)=\sum_m H_m,\qquad
\widehat L_h(\xi)=h\frac{\widehat K(\xi/\gamma)}{D_h(\xi)}.
\]

The second identity is Lemma 5, pp. 19–20, equations (46)–(50). To derive the full mode response directly, periodize \(v\mapsto e^{-i\omega v}L_h(v)\). Its Fourier-series coefficient at index \(m\) is

\[
\frac1h\widehat L_h(\omega+2\pi m/h)=
\frac{H_m}{\sum_n H_n}=:p_m.
\]

Restoring the shifted grid gives exactly Lemma 7, p. 21, equation (62):

\[
Q_\infty e^{i\omega x}
=e^{i\omega x}\sum_{m\in\mathbb Z}p_m e^{i2\pi m u}.
\]

Here

\[
p_0=\frac1{1+R_0},\qquad
p_m=\frac{\alpha_m}{1+R_0}\;(m\ne0),\qquad
\sum_m p_m=1.
\]

Consequently the desired amplitude is \(p_0\), rather than one, and

\[
Q_\infty e^{i\omega x}-e^{i\omega x}
=e^{i\omega x}\sum_{m\ne0}p_m(e^{i2\pi m u}-1).
\]

Thus

\[
\|Q_\infty e^{i\omega x}-e^{i\omega x}\|_\infty
\le\frac{2R_0}{1+R_0}.
\]

The paper defines its alias quantity by the desired-frequency deficit, not solely by added replicas:

\[
\boxed{A_{\rm alias}(\lambda,\epsilon)
=\sup_{|\omega|\le\omega_c}\frac{R_0(\lambda,h\omega)}{1+R_0(\lambda,h\omega)}.}
\]

This is equation (130), p. 28. Lemma 10, pp. 23–24, equations (91)–(98), proves the full mode defect is at most \(2A_{\rm alias}\). The factor two accounts for both the desired-amplitude loss and the shifted components. It is a bound, not an asserted equality for the pointwise or uniform error.

## Integrated output and the role of r

Suppose the final target has a nonzero mode \(f(x)=b e^{i\omega x}\) and QI is applied to \(f^{(r)}\). Write \(\nu_m=\omega+2\pi m/h\). The canonical Fourier antiderivative of the QI response is

\[
G_{\rm QI}(x)
=b e^{i\omega x}\left[
\frac1{1+R_0}
+\frac1{1+R_0}\sum_{m\ne0}
\left(\frac{\theta}{\theta+2\pi m}\right)^r
\alpha_m e^{i2\pi m u}\right].
\]

Its \(r\)-th derivative is exactly \(Q_\infty f^{(r)}\). The complex powers retain their signs; absolute values enter only in the bound. Subtracting the target gives

\[
\boxed{\|G_{\rm QI}-f\|_\infty
\le |b|\frac{R_0+R_r}{1+R_0}.}
\]

The \(R_0\) contribution is the retained-frequency deficit. It survives integration. Multiplying the derivative lattice coefficients for this mode by \(1+R_0\) instead produces a response with the desired amplitude exactly matched:

\[
G_{\rm match}(x)
=b e^{i\omega x}\left[1+\sum_{m\ne0}
\left(\frac{\theta}{\theta+2\pi m}\right)^r\alpha_m e^{i2\pi m u}\right],
\qquad
\|G_{\rm match}-f\|_\infty\le |b|R_r.
\]

For \(P=\sum_j b_j e^{i\omega_jx}\), superposition gives

\[
\|G_{\rm match}[P]-P\|_\infty\le\sum_j|b_j|R_r(\lambda,h\omega_j),
\]

and the analogous prescribed-QI bound is

\[
\|G_{\rm QI}[P]-P\|_\infty
\le\sum_j|b_j|\frac{R_0(\lambda,h\omega_j)+R_r(\lambda,h\omega_j)}{1+R_0(\lambda,h\omega_j)}.
\]

Constants in the final output for \(r\ge1\) belong in the separate bias/integration polynomial. In contrast, a constant *derivative target* is a genuine QI mode and may have a nonzero normalization/replica defect.

### Boundary conditions are an additional choice

The preceding formulas use canonical Fourier antiderivatives. Prescribing the first \(r\) target jets at \(a=-1\) adds the polynomial that removes the first \(r\) Taylor coefficients of \(G_{\rm QI}-f\) at \(a\). The simple \((R_0+R_r)/(1+R_0)\) bound does not automatically include that correction.

An exact boundary-matched formula is available. For \(s=x+1\), define

\[
J_r(\nu,s)=\frac{e^{i\nu s}-\sum_{\ell=0}^{r-1}(i\nu s)^\ell/\ell!}{(i\nu)^r}.
\]

Then the boundary-matched error is

\[
\widetilde f(x)-f(x)
=b e^{-i\omega}(i\omega)^r\left[(p_0-1)J_r(\omega,s)
+\sum_{m\ne0}p_mJ_r(\nu_m,s)\right].
\]

For arbitrary derivative error, the clean general bound remains

\[
\|\widetilde f-f\|_{\infty,[-1,1]}
\le\frac{2^r}{r!}\|Q f^{(r)}-f^{(r)}\|_{\infty,[-1,1]}.
\]

This follows from the integral remainder formula and applies directly to the finite QI theorem.

### Concrete counterexample to using Rr for prescribed QI

Take the tanh kernel, \(r=1\), \(\lambda=1\), \(N=32\), and \(\omega=2\pi\). Then \(\theta=\pi/8\), and direct evaluation gives

\[
R_0\simeq0.0025118432,\qquad R_1\simeq0.0001625589,
\qquad\frac{R_0}{1+R_0}\simeq0.0025055496.
\]

On \([-1,1]\), the Fourier coefficient at \(2\pi\) of the integrated QI error equals the retained-amplitude deficit. All shifted modes and any added bias are orthogonal to that coefficient. Hence the uniform error is at least \(0.0025055496\), over fifteen times \(R_1\), even after the allowed bias is chosen. This is a structural obstruction, independent of truncation or roundoff.

## Connection to the complete old theorem

For a finite Fourier approximation \(P_g=\sum_j\beta_j e^{i\omega_jx}\) to the kernel target \(g\), suppose its approximation error is at most \(\delta\) on the interval containing the required sample nodes. Keeping the actual spectral weights gives

\[
\|g-Q_{\rm fin}g\|_\infty
\le(1+C_Q)\delta
+2\sum_j|\beta_j|\frac{R_0(\lambda,h\omega_j)}{1+R_0(\lambda,h\omega_j)}
+B_g\left[C_H e^{-\sigma R}+C_S e^{-\sigma K_c}\right],
\quad B_g=\sum_j|\beta_j|.
\]

Theorem 14, pp. 28–29, obtains its displayed form by replacing the weighted alias expression by its band supremum and using Theorem 13's bound \(B_g\le C_{\rm spec}M_{\rm FE}(g)(1+\omega_c)^{1/2}\). This is the natural place to connect the compact ratio theorem to the four-term QI decomposition.

The current comparison above resolves these questions for the new proof.

## Limited issues already identified in the old draft

These notes are separate from the exact bridge above; they do not establish anything about the forthcoming new proof.

1. **Finite reindexing changes the operator.** Page 5, equations (6)–(7), and p. 30, equations (154)–(155), use the same output range \(-R\le m\le N+R\) after expanding a stencil. Reindexing the direct finite operator actually gives output indices \(-R-K_c\le m\le N+R+K_c\), with
   \[
   a[m]=\sum_{|j|\le K_c}c_j f(x_{m-j})\mathbf1_{\{-R\le m-j\le N+R\}}.
   \]
   Keeping the paper's shorter output range and unrestricted convolution defines a different operator. The finite theorem proves the direct operator of equation (22); the claimed MLP equality needs the correct reindexing or a separate comparison estimate.

2. **The old main text's prefactor and scaling claims are stronger than its appendix establishes.** Theorem 1, p. 6, calls the \(\lambda\)-dependent prefactors polynomial. For normalized sech squared,
   \[
   d_0(\sigma,\lambda)\le\widetilde D_\lambda(\pi)
   \sim\frac{2\pi^2}{\lambda}e^{-\pi^2/(2\lambda)}.
   \]
   The appendix's own \(\gamma C_c=\lambda/d_0\) therefore has exponential small-\(\lambda\) growth. Moreover, the appendix truncation rates are \(e^{-\sigma R}\) and \(e^{-\sigma K_c}\); replacing \(\sigma\) by a fixed multiple of \(\lambda\) requires an appropriate zero-free-strip estimate. Page 15's width argument holds \(\lambda\) away from zero while allocating no error budget to the nonvanishing fixed-\(\lambda\) alias floor. It does not establish an all-\(\varepsilon\) width law by itself.

3. **The Fourier-extension coefficient proof has a specific analytic-continuation gap.** Page 25, Proposition 2, claims that T1 supplies functions \(h_1,h_2\) analytic on \(B(\varrho_\star)\), \(\varrho_\star>E(T)\), through
   \(g_o(y)=\sin(\pi y/T)h_2(m_T(y))\).
   For the entire target \(g(y)=y\), T1 holds as stated, but the implied \(h_2(m_T(y))=y/\sin(\pi y/T)\) is singular at \(y=T\). The point \(m_T(T)=-(E(T)+E(T)^{-1})/2\) lies strictly inside that Bernstein ellipse. Thus the claimed analyticity step is false under stated T1. This identifies a gap in that proof; it is not, by itself, a disproof of the final coefficient bound or of alternative approximant constructions.

No edits to the paper or earlier bundles were made. Numerical values above were checked using the closed-form sech-squared transform and summing replicas \(-20\le m\le20\); omitted terms are negligible at the displayed precision.
