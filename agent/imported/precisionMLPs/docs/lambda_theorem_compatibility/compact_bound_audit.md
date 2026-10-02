# Compact alias bound: reconstruction and proof audit

This report reconstructs the bound requested at the cutoff of Sam's previous conversation and checks it against the newly supplied `theorem_for_sam.pdf`. The long bundle contains the requested first-pair formula; the short bundle replaced its main display by a full sum. Sam's cutoff asks to restore the compact formula, not to replace the weighted spectrum by an unqualified mean-frequency theorem.

## 1. The requested formula is valid with a stated tail hypothesis

Write (H(t)=|\widehat K(t)|), (0\le\theta<\pi), and

\[
\rho=\frac{H((4\pi-\theta)/\lambda)}{H((2\pi-\theta)/\lambda)},\qquad
\boxed{\mathcal R_{K,r}(\lambda,\theta)=
\frac{
\left(\frac{\theta}{2\pi-\theta}\right)^rH((2\pi-\theta)/\lambda)
+\left(\frac{\theta}{2\pi+\theta}\right)^rH((2\pi+\theta)/\lambda)}
{(1-\rho)\min\{H(0),H(\theta/\lambda)\}}.}
\]

For (r=0), both powers are one, including at zero. For (r>0), constants are supplied by the output bias and contribute zero.

The sufficient hypothesis is: (H) is positive and even at the relevant frequencies, (\log H) is concave on ([(2\pi-\theta)/\lambda,\infty)), and (\rho<1). This hypothesis was explicit in the long bundle; symmetry and decay alone do not imply it.

Define the exact amplitude sum

\[
S_{K,r}(\lambda,\theta)=
\sum_{m\ne0}\left|\frac{\theta}{\theta+2\pi m}\right|^r
\frac{H((\theta+2\pi m)/\lambda)}{H(\theta/\lambda)}.
\]

If (T_r) is its first-pair contribution, then

\[
T_r\le S_{K,r}\le\frac{T_r}{1-\rho}\le\mathcal R_{K,r},\qquad
0\le S_{K,r}-T_r\le\frac{\rho T_r}{1-\rho}.
\]

**Proof.** Set (d=2\pi/\lambda). Concavity makes
\(\log H(t+d)-\log H(t)\) nonincreasing. Every successive ratio in either alias sequence is consequently at most \(\rho\). The factors \((\theta/(2\pi m\pm\theta))^r\) also decrease with \(m\). Each sequence is bounded by its first term times the geometric sum \((1-\rho)^{-1}\). Replacing \(H(\theta/\lambda)\) by its minimum with \(H(0)\) enlarges the bound.

The minimum is conservative and optional for the proof. It equals (H(\theta/\lambda)) for tanh and Gaussian. GELU and Swish have a central spectral rise, so dropping the minimum changes the tested formula but yields a sharper valid bound with the exact denominator. For a tanh-only theorem, the minimum can be removed without changing its numerical value.

## 2. Why these ratios bound an output

For a finite lattice kernel sum in grid coordinates,

\[
q(u)=\sum_k a_kK(\lambda(u-k)),\quad
\widehat q(\theta)=\lambda^{-1}\widehat K(\theta/\lambda)
\sum_k a_ke^{-ik\theta}.
\]

The coefficient factor repeats every (2\pi). Thus its response at a shifted frequency relative to its desired response is the kernel-transform ratio. An infinite tone-weighted lattice has the exact expansion

\[
\sum_k e^{ik\theta}K(\lambda(u-k))
=\lambda^{-1}\sum_{m\in\mathbb Z}
\widehat K((\theta+2\pi m)/\lambda)e^{i(\theta+2\pi m)u}.
\]

To prove this, multiply the left side by (e^{-i\theta u}), observe that it is one-periodic, and compute its Fourier coefficients by shifting each unit integration interval. The shifted intervals tile the real line. Localization and summable transform values justify the exchanges.

Normalize the desired amplitude to one. If (K=\psi^{(r)}), integrating each nonzero component (r) times gives multiplier

\[
t_m=\left(\frac{\theta}{\theta+2\pi m}\right)^r
\frac{\widehat K((\theta+2\pi m)/\lambda)}{\widehat K(\theta/\lambda)}.
\]

For \(P(x)=\sum_j b_je^{i\omega_jx}\), \(\theta_j=h\omega_j\), the ideal response that reproduces every retained amplitude therefore satisfies

\[
\|F_P-P\|_\infty\le\sum_j|b_j|S_{K,r}(\lambda,|\theta_j|)
\le\sum_j|b_j|\mathcal R_{K,r}(\lambda,|\theta_j|).
\]

This is the triangle inequality for an explicitly constructed response. It is not an unconditional bound on a finite fitted network. If the integration constant is chosen to match (P(x_0)), the error is the preceding alias function minus its value at (x_0); its bound acquires a factor two.

## 3. The new coefficient density matches this model exactly for tanh

The new proof uses the normalized derivative kernel

\[
K_\gamma(t)=\frac\gamma2\operatorname{sech}^2(\gamma t),\qquad
H_*(\xi)=\frac{\pi\xi/2}{\sinh(\pi\xi/2)},\qquad
d=\frac\pi{2\gamma}.
\]

For a tone (f(x)=e^{i\omega x}), its density in (A.2) is exactly

\[
a_\gamma(x)=\frac{f(x+id)-f(x-id)}{2id}
=i\omega\frac{\sinh(\omega d)}{\omega d}f(x)
=\frac{i\omega}{H_*(\omega/\gamma)}f(x).
\]

Consequently, sampling this density against (K_\gamma) reproduces the desired derivative amplitude exactly; its replicas have the kernel ratios above. This directly agrees with the old amplitude-matched output calculation. The cardinal multiplier (p_0=1/(1+S_{K,0})), and its retained-amplitude deficit, belong to the old cardinal construction and are absent from this new tone calculation.

There is also a direct connection to the **finite-contour pole term** in the new proof, without identifying the finite network with an infinite lattice. Put \(c=\pi/(2\lambda)\), \(A=\pi^2/\lambda\), and \(\theta=|h\omega|\). For tanh,

\[
S_{K,1}(\lambda,\theta)
=\sinh(c\theta)\sum_{m\ge1}
\left[\operatorname{csch}(c(2\pi m-\theta))
+\operatorname{csch}(c(2\pi m+\theta))\right]
=4\sinh(c\theta)\sum_{\ell\ge0}
\frac{\cosh((2\ell+1)c\theta)}{e^{(2\ell+1)A}-1}.
\]

The second equality follows from \(\operatorname{csch}z=2\sum_{\ell\ge0}e^{-(2\ell+1)z}\) and summing over \(m\). All terms are nonnegative and converge for \(0\le\theta<\pi\).

At pole layer \(y=(2\ell+1)d\), the new integrand has residues at \(x\pm iy\) and \(x_0\pm iy\). Their density magnitudes for a unit tone sum to

\[
4\frac{\sinh(|\omega|d)}d\cosh(|\omega|(2\ell+1)d).
\]

Each residue has the additional factor \(1/(2\gamma)\); (A.14) contributes \(2\pi\), and (A.16) contributes at most \((e^{(2\ell+1)A}-1)^{-1}\). Because \(\gamma d=\pi/2\), the resulting bound per layer is

\[
8\frac{\sinh(c\theta)\cosh((2\ell+1)c\theta)}{e^{(2\ell+1)A}-1}.
\]

Summing the finitely many enclosed layers and enlarging the positive scalar sum to all layers proves the following usable statement:

> **Compact theorem compatible with the new proof.** For a finite Fourier target \(P=\sum_jb_je^{i\omega_jx}\) with \(|h\omega_j|<\pi\), the replica contribution \(E_{\rm rep,alias}:=\sup_{x\in[-1,1]}|\mathcal P_h[F_x]|\) of the new tanh construction satisfies
> \[
> E_{\rm rep,alias}\le2\sum_j|b_j|S_{K,1}(\lambda,|h\omega_j|)
> \le2\sum_j|b_j|\mathcal R_{K,1}(\lambda,|h\omega_j|).
> \]

For one absolute target frequency and total amplitude \(B_{\rm spec}\), this becomes the single compact display \(E_{\rm rep,alias}\le2B_{\rm spec}\mathcal R_{K,1}(\lambda,\theta)\). One may normalize error by \(2B_{\rm spec}\) to obtain precisely \(E_{\rm alias}\le\mathcal R_{K,1}\), provided that normalization is explicit. For mixtures retain the weighted sum, or use a supremum over the retained frequencies. This statement bounds the replica term; the new proof separately bounds the horizontal contour and corrected boundary contributions.

The new theorem's local analytic target class does **not** automatically supply a finite Fourier representation with controlled coefficient mass. Its existing pole bound is the rigorous theorem for that whole class; the spectral theorem is a refinement for specified spectral targets and an explanatory model otherwise. Approximating \(f\) only on the real interval does not control its values on the complex pole segments, so a Fourier approximation used to transfer this pole estimate needs appropriate complex-domain control.

## 4. Tail hypotheses for the four old experimental activations

The normalized transforms are correct:

| Activation | \(r\) | \(H(\xi)/H(0)\) |
|---|---:|---|
| tanh | 1 | \(t/\sinh t\), \(t=\pi\xi/2\) |
| exact GELU \(x\Phi(x)\) | 2 | \((1+\xi^2)e^{-\xi^2/2}\) |
| Swish \(x/(1+e^{-x})\) | 2 | \(t^2\cosh t/\sinh^2t\), \(t=\pi\xi\) |
| Gaussian \(e^{-x^2}\) | 0 | \(e^{-\xi^2/4}\) |

For GELU, differentiating twice gives \((2-x^2)\phi(x)\), whose transform is the displayed expression. For Swish, if \(L\) is the transform of the logistic density, then \(\widehat{\psi''}=L-\xi L'\); substituting \(L=t/\sinh t\) gives the table.

Tanh satisfies the tail hypothesis for every positive argument: \((\log(t/\sinh t))''=-1/t^2+\operatorname{csch}^2t<0\), and its transform decreases. Gaussian is immediate. GELU decreases and is log-concave for \(\xi>1\), since its first derivative of log transform is \(\xi(1-\xi^2)/(1+\xi^2)\) and its second is \(2(1-\xi^2)/(1+\xi^2)^2-1\). For Swish, the first and second log derivatives with respect to \(t\) are

\[
\frac2t+\tanh t-2\coth t<\frac2t-1,\qquad
-\frac2{t^2}+\operatorname{sech}^2t+2\operatorname{csch}^2t.
\]

The second is negative for \(t\ge2\): its positive part is at most \(12e^{-2t}/(1-e^{-2t})^2<2/t^2\), first checked at 2 and then preserved by monotonicity. Thus all four satisfy the hypothesis for the old tested range \(0<\lambda\le1.5\), \(0\le\theta<\pi\), whose smallest alias argument exceeds \(\pi/1.5>2\). These facts do not extend the new local tanh construction to other activations; they validate the older spectral bound and experiments.

## 5. Mean frequency is a practical approximation, not the theorem

For \(B_{\rm spec}=\sum_j|b_j|\) and \(\bar\theta=\sum_j|b_j||\theta_j|/B_{\rm spec}\), there is no general upper bound replacing \(\sum_j|b_j|S(\theta_j)\) by \(B_{\rm spec}S(\bar\theta)\). The rapidly increasing frequency response can make the latter much smaller.

A concrete tanh counterexample uses \(\lambda=.25\), \(r=1\), amplitudes .99 and .01, and positive grid frequencies .01 and 1. Their mean is .0199. Directly summing the convergent aliases gives

\[
.99S_{K,1}(.25,.01)+.01S_{K,1}(.25,1)=2.0525053\times10^{-14},\qquad
S_{K,1}(.25,.0199)=3.6170214\times10^{-18}.
\]

For the unanchored ideal response to \(.99\cos(.01u)+.01\cos u\), its error at \(u=0\) is \(2.0523236\times10^{-14}\), so this is an actual counterexample, not merely a gap between two bounds. The first-pair tail factors are negligible in this example, and the same failure applies to the compact formula. The mean-frequency rule can remain in the main text, clearly introduced as an approximation following the theorem.

## 6. What lambda = .25 does and does not mean

For tanh the basic kernel ratio is

\[
A_K(\lambda)=\frac{\pi^2/\lambda}{\sinh(\pi^2/\lambda)}
\sim\frac{2\pi^2}{\lambda}e^{-\pi^2/\lambda}.
\]

At \(\lambda=.25\), it is \(5.6510716341\times10^{-16}\), or 2.545 times binary64 machine epsilon \(2^{-52}\). The exact solution of \(A_K(\lambda)=2^{-52}\) is \(\lambda=0.2440764171\). Thus “approximately .25 gives a binary64-scale starting point” is correct; “.25 guarantees error below machine epsilon for every width and function” is not.

The old cardinal full-mode upper bound at zero frequency is approximately \(4A_K=2.26043\times10^{-15}\). The new construction has no cardinal normalization deficit. Its tanh output sum instead satisfies, at fixed \(\lambda\) as \(\theta\to0\),

\[
S_{K,1}(\lambda,\theta)
=\frac{|\theta|}{\pi}A_K(\lambda)(1+o(1))
\quad\text{when the first pair dominates}.
\]

Using \(\theta=h\omega\) and the small-\(\lambda\) form of \(A_K\), this is approximately \(2\pi h|\omega|\lambda^{-1}e^{-\pi^2/\lambda}\). The anchored/finite-pole bound has another factor two. This explains the new theorem's \(W^{-1}e^{-\pi^2/\lambda}/\lambda\) replica scale for fixed target derivative scale and \(h=\Theta(W^{-1})\). The correspondence is in both the exponential and the width dependence.

For the whole locally analytic class, the new theorem explicitly gives the certified replica budget

\[
E_{\rm alias}^{\rm new}(W,\lambda)
=\frac{128\pi B}{\lambda\delta W}e^{-\pi^2/\lambda}.
\]

Its \(B\) is a bound on the function in a complex neighborhood, not the spectral coefficient mass above. A theorem-based selection retains this prefactor and the admissibility constraints. The kernel-only and mean-frequency rules remain useful practical simplifications, rather than equalities with this certified quantity. Width here must distinguish \(N\) core cells from \(W=N+2R+1\) neurons; \(h\omega=2\omega/N\) remains exact.


There is a stronger statement for .25 using the new proof's unsimplified pole estimate. Let \(\mathcal U(A)=e^{-A}/[(1-e^{-A})(1-e^{-2A})]\), \(A=\pi^2/\lambda\). Since \(M_\sigma\le4B/\delta\) and admissibility includes \(\gamma\delta\ge4\pi\),

\[
E_{\rm alias}\le\frac{4\pi M_\sigma}{\gamma}\mathcal U(A)
\le4B\mathcal U(A).
\]

At \(\lambda=.25\), this is \(2.8628663341\times10^{-17}B\), or \(.128932\,B\,2^{-52}\). Thus .25 **does rigorously put the replica term below binary64 machine epsilon times the analytic class bound \(B\), for every admissible width and target in that class**. This stronger statement is compatible with the basic proxy being 2.545 epsilon: the proxy and the actual bounded output pole term are different quantities. It still does not certify the other error contributions or arbitrary widths and target scales.

The companion appendix uses the notation \(\mathcal R_{K,r}\) for twice the recovered historical compact formula, incorporating anchoring. That convention makes its new finite-contour theorem \(E_{\rm alias}\le\sum_j|b_j|\mathcal R_{K,1}\). The present report retains the historical convention in sections 1–5 so the change remains explicit.

## 7. Minimal changes needed in a one-page presentation

- Restore the compact first-pair display requested by Sam; move its geometric-tail proof into the appendix.
- For the new construction, connect it to the local density and replica residues, rather than the old cardinal multiplier.
- State whether \(E_{\rm alias}\) is a normalized spectral budget or the actual pole term. The latter carries \(2B_{\rm spec}\) for a tone and a weighted sum for a mixture.
- Keep ordinary weighted average frequency in the practical refinement. It cannot silently replace the theorem's actual frequencies.
- Keep .25 as a practical fp64 default with a resolution qualification; the theorem's complete error also depends on target size, analytic radius, width, boundary terms, and computation.
- With a strict constraint \(E<e_{\rm tol}\), use a supremum unless a discrete search is intended. With \(E\le e_{\rm tol}\) and a closed bounded admissible search interval, an attained maximum is appropriate.

The compact bound and the intended argument survive. The new tanh coefficient density actually removes the main conceptual obstacle in the old cardinal-to-output bridge. The necessary adjustments concern which error is bounded, the factor from anchoring, and the distinction between a weighted theorem and its representative-frequency approximation.
