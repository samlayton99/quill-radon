# Review of "From One-Dimensional QI to Compositional Ridge Networks"

**Scope.** Every statement in the writeup was checked line by line against (i) the workshop paper (Theorem 14 and Appendix C), the Section 3 rewrite and the implementation notes, (ii) the experimental record of checkpoints A, B, C, H and the first rows of I, and (iii) direct numerical verification in 30-digit arithmetic where a constant or an inequality could be tested. The record's notation is used throughout: $M$ directions, $N_1$ offsets per direction, $N_2$ outer offsets, $\mathfrak r$ the data radius about the origin $x_0$, $k$ the spectral scale, $\lambda=\gamma h$.

**Summary of findings.**

1. Section 1 (the 1-D block) is correct as an import, with two caveats on its hypotheses: the paper's own verification of the zero-free strip (K3) is vacuous at the operating $\lambda$, and the aliasing constant carries a prefactor the shorthand $e^{-\pi^2/\lambda}$ drops.
2. Theorem 1 (direction snapping) is correct and its proof is complete. It is, however, a first-order certificate: it makes $M$ polynomial in $1/\varepsilon$, while the record shows $\varepsilon$ enters only logarithmically. The reconciliation offered in Section 2.5 ("valid when $B_0$ and $\varepsilon_{\rm dir}$ are held fixed") is not the right one. A second certificate, an angular-quadrature bound built on the plane-wave expansion and spherical designs, gives the measured law $M\asymp(c\,k\mathfrak r+O(\log 1/\varepsilon))^{d-1}$ and explains the cliff in every dimension. It is stated and sketched below.
3. Section 2.3 is correct; the offsets count should be made explicit, and it comes out with the same "Nyquist plus logarithm" form as the angular count. The record's measured $N^{-10}$ offsets floor in 2-D contradicts this prediction and should be flagged as unexplained rather than absorbed.
4. Section 2.4 is honest. It undersells one point: every smooth target in the record is globally analytic, so its Fourier measure is available directly with explicit tails, and the tolerance then enters the direction count through an effective spectral scale $k(\varepsilon)$ that is $O(1)$ for bandlimited targets, $O(\sqrt{\log 1/\varepsilon})$ for Gaussian ones and $O(\log 1/\varepsilon)$ for pole-type ones. That ordering is what expH05 measured.
5. The compositional theorem (Section 3.1) is correct. It is pure error propagation; the approximation content lives entirely in the class hypothesis. Three things are missing that the record already has: the floating-point floor it implies, the direction-exactness requirement it implies, and the fact that it does not transfer to a least-squares solve (only the outer readout is a projection).
6. Sections 4 and 5 build the architecture from the Sprecher form. Its analytic-factor version is a much thinner class than Section 3's, essentially incompatible with how the Kolmogorov construction achieves universality, and it excludes the record's own composition target with smooth pieces. The theorem "we actually want" should be stated for the general block (learned directions, per-channel untied profiles), with the Sprecher form kept as the universality anchor and as the most rigid corner of a rank sweep. Two further points in Section 5: shared outer functions save no units under QI, and, if the Sprecher shift is commensurate with the mesh, all channels can read one extended feature bank with an index shift.

The rest of this note goes through the writeup in order.

---

## 1. The one-dimensional building block

The statements are correct: $\widehat C_h=h/D_h$, the rescaled normalizer depends on $(h,\gamma)$ only through $\lambda$, the derivative trick, the four-term theorem, and $N=O(\log 1/\varepsilon)$ for analytic targets above the floor. Four remarks on what the import silently assumes.

**1.1 Normalization.** The paper's proofs use $K_0=\tfrac12\mathrm{sech}^2$ (unit mass); the construction uses $K=\mathrm{sech}^2=2K_0$, whose cardinal coefficients are half those of $K_0$. Harmless, but the constants in Section 1 should say which convention they carry.

**1.2 The halo and stencil exponents.** Theorem 14 proves the halo and stencil terms as $C_He^{-\sigma R}$ and $C_Se^{-\sigma K_c}$, where $\sigma$ is the half-width of the strip on which the rescaled normalizer $\widetilde D_\lambda(\theta)=\lambda\sum_j K(\lambda j)e^{-ij\theta}$ is bounded away from zero, subject to $\sigma<a_K\lambda=2\lambda$. The rewrite's $e^{-c_3\lambda R}$ is a relabeling of the same quantity. Measured from the repo's cardinal coefficients (fp64 Toeplitz solve, $K_c=160$):

| $\lambda$ | $\lvert c_0\rvert$ at $N=64$ | decay rate of $\lvert c_j\rvert$ | $2\lambda$ |
|---|---|---|---|
| 0.30 | 338 | 0.319 | 0.60 |
| 0.25 | 6225 | 0.263 | 0.50 |

So $\sigma\approx1.05\lambda$ at the operating bandwidths, and the implementation's halo rule $e^{-2\lambda R}$ overstates the decay by a factor two in the exponent. This is invisible in practice only because the halo also grows with $N$. Note also that $c_j\propto h$ (from $\sum_jc_jK(\lambda j)=h/\lambda$), so "$\lvert c_0\rvert\approx338$" is an $N=64$ number.

**1.3 The zero-free strip is not verified by the paper's own proposition at the operating $\lambda$.** Appendix C.6.1 verifies K3 for $\mathrm{sech}^2$ through diagonal dominance, Proposition 5: K3 holds if

$$q(\sigma,\lambda)=\frac{8e^{-(2\lambda-\sigma)}}{1-e^{-(2\lambda-\sigma)}}<1 .$$

This requires $2\lambda-\sigma\gtrsim2.2$, i.e. $\lambda\gtrsim1.1$, where the aliasing term is already $10^{-4}$. At $\lambda=0.25$ one finds $q\approx13$ to $76$ for $\sigma\in[0.02,0.4]$. K3 on the real axis does hold, by Fourier positivity (K5): $\widetilde D_\lambda(\theta)=\sum_m\widehat K((\theta+2\pi m)/\lambda)>0$, with minimum $\widetilde D_\lambda(\pi)=2\widehat K(\pi/\lambda)\approx2.1\times10^{-7}$ at $\lambda=0.25$ (this ratio $\widetilde D(0)/\widetilde D(\pi)\approx4.7\times10^6$ is the Nyquist amplification of the magnitude rule). The strip version, which is what Lemmas 2, 4, 8, 9 need, is at present supported by the measured coefficient decay of 1.2, not by a proof. Since the writeup imports the theorem with "assumes a zero-free strip", it should say that at $\lambda=0.25$ this hypothesis is numerically verified, not established by the paper's sufficient condition.

**1.4 The aliasing constant and the $O(\log1/\varepsilon)$ count.** The normalized first alias of $\mathrm{sech}^2$ is

$$A(\lambda)=\frac{\pi^2/\lambda}{\sinh(\pi^2/\lambda)}\approx\frac{2\pi^2}{\lambda}e^{-\pi^2/\lambda},$$

with $A(0.25)=5.65\times10^{-16}$, $A(0.30)=3.4\times10^{-13}$, $A(1.5)=1.8\times10^{-2}$. The shorthand $e^{-\pi^2/\lambda}$ drops a prefactor of about 66 at these values; the repo's theory note quotes the shorthand values at 0.30 and 1.5 and should be corrected. For the count: "$N=O(\log1/\varepsilon)$" holds either at fixed $\lambda$ for $\varepsilon$ above the alias floor $A(\lambda)$ (at $\lambda=0.25$ that floor is fp64 epsilon, which is the design), or, as $\varepsilon\to0$, with $\lambda(\varepsilon)\asymp\pi^2/\log(1/\varepsilon)$ and prefactors growing polynomially in $1/\lambda$ (the paper's Appendix C.1). The sentence "once $\lambda,R,K_c$ are chosen so that the last three terms lie below the tolerance" hides that the aliasing term is not controllable by $N$, $R$ or $K_c$ at all.

---

## 2. The Radon/Fourier lift

### 2.1 The polar representation and the Radon profiles

Correct. Two remarks worth a sentence each.

The profile $p_v(t)=(2\pi)^{-d}\int_0^\infty\widehat F(\rho v)\rho^{d-1}e^{i\rho t}\,d\rho$ is complex, and $p_{-v}(t)=\overline{p_v(-t)}$ for real $F$; the real ridge attached to the unoriented line through $v$ is $2\,\mathrm{Re}\,p_v(v^\top z)$. This matters because Theorem 1 works with unoriented lines and oriented snapping.

More important: the Radon profile exists as a function of $t$ only when the radial slices $\rho\mapsto\widehat F(\rho v)$ are integrable, i.e. when $\widehat F$ is a reasonable density. For a general finite measure $\mu$ (a single plane wave, a ridge target) there is no such profile; the direction is a point mass. Theorem 1 is stated with $\mu$ and snapping precisely to avoid this, and the writeup should say so, because Sections 2.1 and 2.2 are otherwise read as the same object. This distinction becomes the whole story in 2.2 below: smooth angular spectra admit high-order angular quadrature, point masses in angle do not, and the two need different certificates.

### 2.2 Theorem 1: correct, and first order

The proof is complete. The checks: the nearest-oriented-member selection $\xi\mapsto u(\xi)$ is Borel (Voronoi cells with any fixed tie-break); $\lVert\xi-\xi_V\rVert=2\lVert\xi\rVert\sin(\theta/2)\le\lVert\xi\rVert\theta$; $\lvert e^{ia}-e^{ib}\rvert\le\lvert a-b\rvert$ for real $a,b$; the Cauchy-Schwarz step $\lvert(\xi-\xi_V)^\top z\rvert\le\lVert\xi-\xi_V\rVert\,\mathfrak r$; the pushforward bookkeeping $\sum_r\lVert\nu_r\rVert_{\rm TV}\le\lVert\mu\rVert_{\rm TV}$. If the tie-break is symmetric ($u(-\xi)=-u(\xi)$) and $\mu$ is Hermitian, $F_V$ is real. The bound holds on $B_{\mathfrak r}(x_0)$ and degrades linearly in $\lVert z\rVert$ outside it; that is the origin-centering the record measured (the data radius, not the domain, sets the direction count) and, together with tanh saturation, the record's $O(1)$ far field, which the writeup does not mention.

**H1 (the main quantitative hole).** The certificate is first order in $\Theta_V$, so Section 2.5's conclusion is $M\gtrsim(C_dk\mathfrak rB_0/\varepsilon_{\rm dir})^{d-1}$: polynomial in $1/\varepsilon_{\rm dir}$. At $\varepsilon_{\rm dir}=10^{-13}$ that is $10^{13(d-1)}$ times $(k\mathfrak r)^{d-1}$. The record's numbers are $M=12$ to $24$ in 2-D for $10^{-10}$ and the two-floor law with $e_M\sim e^{-aM^q}$, $q\in[0.9,2.1]$. The following exact computation shows the size of the gap. Take the radial target $F(z)=J_0(k\lvert z\rvert)$ in 2-D, whose spectral measure is the unit uniform measure on the circle of radius $k$ ($\lVert\mu\rVert=1$, $B_1=k$). For $M$ equispaced lines the snapping certificate gives $\mathfrak rB_1\Theta_V=k\mathfrak r\,\pi/(2M)$. The equal-weight $M$-line Radon quadrature is also an $M$-term ridge sum (profiles $\cos(ks)/M$), and by Jacobi-Anger its error on $\lvert z\rvert=\mathfrak r$ is exactly $\sum_{q\ne0}i^{2Mq}J_{2Mq}(k\mathfrak r)e^{i2Mq\phi}$:

| $M$ | snapping bound, $k\mathfrak r=7.5$ | exact ridge-sum error | snapping bound, $k\mathfrak r=2$ | exact ridge-sum error |
|---|---|---|---|---|
| 4 | 2.9 | $3.5\times10^{-1}$ | 0.79 | $4.4\times10^{-5}$ |
| 8 | 1.5 | $6.3\times10^{-5}$ | 0.39 | $9.0\times10^{-14}$ |
| 12 | 0.98 | $1.1\times10^{-10}$ | 0.26 | $3.1\times10^{-24}$ |
| 16 | 0.74 | $1.2\times10^{-17}$ | 0.20 | $7.4\times10^{-36}$ |
| 24 | 0.49 | $4.3\times10^{-34}$ | 0.13 | $1.6\times10^{-61}$ |

The exact error is flat until $2M\approx e\,k\mathfrak r/2$ and then collapses super-exponentially; the certificate never leaves $O(1)$. This is not a matter of constants: the certificate's $\varepsilon$-dependence is the wrong function. So the sentence in 2.5, "that shorthand is only valid when the amplitude scale $B_0$ and the desired angular error are being held fixed", is not the reconciliation. The shorthand $M\sim(k\mathfrak r)^{d-1}$ is valid because angular convergence is super-exponential past $k\mathfrak r$, so that $\varepsilon$ enters only through a logarithm. Theorem 1 cannot see this because snapping is a zeroth-order quadrature (nearest node, no cancellation).

**T1 (proposed tightening: the angular-quadrature certificate).** The mechanism above is the plane-wave expansion, and it holds in every dimension. For $x,z\in\mathbb R^d$ write $x=\rho v$, $z=\lvert z\rvert\hat z$; then

$$e^{i\rho\,v^\top z}=\sum_{\ell\ge0}a_\ell(\rho\lvert z\rvert)\,\Pi_\ell(v,\hat z),\qquad a_\ell(x)=(2\pi)^{d/2}\,i^\ell\,x^{1-d/2}J_{\ell+d/2-1}(x),$$

where $\Pi_\ell(v,\hat z)=\sum_mY_{\ell m}(v)\overline{Y_{\ell m}(\hat z)}$ is the reproducing kernel of the degree-$\ell$ spherical harmonics, with $\sup\lvert\Pi_\ell\rvert=N(d,\ell)/\lvert S^{d-1}\rvert$ and $N(d,\ell)\le C\ell^{d-2}$. From $J_\nu(x)\le(x/2)^\nu/\Gamma(\nu+1)$ for $x\ge0$,

$$\sup_v\big\lvert a_\ell(\rho\lvert z\rvert)\Pi_\ell(v,\hat z)\big\rvert\le C_d\,\ell^{d-2}\frac{(\rho\lvert z\rvert/2)^\ell}{\Gamma(\ell+d/2)} .$$

**Proposition (angular quadrature).** Let $F(z)=\int_{S^{d-1}}\int_0^k e^{i\rho v^\top z}\,m(\rho,v)\,\rho^{d-1}d\rho\,d\sigma(v)$, where for each $\rho$ the angular density $m(\rho,\cdot)$ is a spherical polynomial of degree at most $L$ (or is within $\tau(\rho)$ of one in the sup norm). Let $V=\{v_1,\dots,v_M\}$ be a spherical $t$-design on $S^{d-1}$ with $t\ge L+n$, and define the ridge profiles by the Radon slices, $p_r(s)=\frac{\lvert S^{d-1}\rvert}{M}\int_0^ke^{i\rho s}m(\rho,v_r)\rho^{d-1}d\rho$, and $F_V(z)=\sum_rp_r(v_r^\top z)$. Then for $\lVert z\rVert\le\mathfrak r$ and $n+1\ge k\mathfrak r$,

$$\lvert F(z)-F_V(z)\rvert\;\le\;2\lvert S^{d-1}\rvert\int_0^k\rho^{d-1}\lVert m(\rho,\cdot)\rVert_\infty\,d\rho\;\cdot\;C_d\,n^{d-2}\frac{(k\mathfrak r/2)^{n+1}}{\Gamma(n+1+d/2)}\;\;(+\,\text{a }\tau\text{ term}).$$

*Proof sketch.* Fix $\rho$ and $z$ and set $G(v)=e^{i\rho v^\top z}m(\rho,v)$. Let $E_n$ be the degree-$n$ truncation of the plane-wave expansion of $v\mapsto e^{i\rho v^\top z}$. Then $E_n\,m(\rho,\cdot)$ is a spherical polynomial of degree at most $n+L\le t$, so the design integrates it exactly. Both the integral and the equal-weight quadrature are bounded by $\lvert S^{d-1}\rvert$ times a supremum, so the quadrature error of $G$ is at most $2\lvert S^{d-1}\rvert\,\lVert m(\rho,\cdot)\rVert_\infty\,\lVert e^{i\rho\langle\cdot,z\rangle}-E_n\rVert_\infty$, and the tail is summed with the Bessel bound; the sum over $\ell>n$ is geometric once $n+1\ge\rho\lvert z\rvert$. Integrate in $\rho$. $\square$

Existence of the nodes is the Bondarenko-Radchenko-Viazovska theorem: for every $t$ there is a spherical $t$-design on $S^{d-1}$ with $M\le c_d\,t^{d-1}$ points (arXiv:1009.4407). Choosing $n\approx e\,k\mathfrak r/2+\log(1/\varepsilon)/\log\!\big(2n/(e\,k\mathfrak r)\big)$ gives

$$M\;\asymp\;\Big(L+\tfrac e2\,k\mathfrak r+O(\log1/\varepsilon)\Big)^{d-1},$$

which is the record's law $M\approx\max\!\big(\binom{p+d-1}{d-1},c(k\mathfrak r)^{d-1}\big)$ with the tolerance entering logarithmically, and it produces the cliff: flat while $n<e\,k\mathfrak r/2$, then super-exponential. In 2-D with equispaced lines the bound is exact (the Jacobi-Anger series above). The hypothesis that carries the weight is the angular smoothness $L$ of the spectral measure: a radial target has $L=0$; a target whose spectrum lies on a few lines has $L=\infty$ and gets nothing from quadrature. That is the correct division of labor for the two certificates, and it is the design the record converged on empirically: Theorem 1 (snapping) handles line components exactly once the lines are in $V$ (the atoms), T1 handles the angularly smooth remainder with an even background of $\asymp(k\mathfrak r)^{d-1}$ directions. Stated together: for $\mu=\mu_{\rm smooth}+\sum_s\mu_s$ with each $\mu_s$ supported on a line $\ell_s$,

$$\lVert F-F_V\rVert_{L^\infty(B_{\mathfrak r})}\le\underbrace{\text{(T1 bound on }\mu_{\rm smooth})}_{\text{even background}}+\underbrace{\mathfrak r\sum_sB_1(\mu_s)\,\theta(\ell_s,V)}_{\text{atoms; zero if }\ell_s\in V}.$$

Two consequences the writeup can then state as theorems rather than as "experiments show a cliff": (i) direction exactness for atoms is forced, since a tilt $\delta$ of an atom costs $k\mathfrak r\delta$ at first order (the record: $0.01^\circ$ on a 7.5-period ridge costs $3\times10^{-3}$); (ii) the number of background directions is set by $k\mathfrak r$ and the angular degree $L$, not by the tolerance. The polynomial floor $\binom{p+d-1}{d-1}$ is the separate fact that ridge monomials along a generic set of that many directions span all polynomials of degree $\le p$; it belongs in the same paragraph.

**H2 (what the bound is a bound on).** Theorem 1 and the T1 proposition bound a constructed approximant, not the fitted network. For the shallow model the transfer is a one-line lemma the writeup should include: the readout is linear, so the empirical least-squares fit is the orthogonal projection onto the span in the empirical $L^2$ norm, and its training residual is at most that of any member of the span, in particular of the certificate. Test error and $L^\infty$ error then need the sampling conditions of the record ($n\ge8$ to $16$ units, the collar, truncation at rcond). The same transfer does not hold for the composed block (Section 3 below).

### 2.3 Adding the 1-D error: correct; make the offsets count explicit

The argument is right: for $\mathrm{supp}\,\mu\subset\{\lVert\xi\rVert\le k\}$ every profile is entire of exponential type $k$ with $\lvert p_r(t+iy)\rvert\le e^{k\lvert y\rvert}\lVert\nu_r\rVert$, the 1-D bound is linear in the sup over the ellipse, and $\sum_r\lVert\nu_r\rVert\le\lVert\mu\rVert$ removes the factor $M$. The record measured exactly this ($e_N$ flat in $M$, identical in 2, 3, 4-D).

**T2.** Make $E_{\rm QI}(N_1;k,\mathfrak r)$ explicit. For a function of exponential type $k$ on a band of half-width $\mathfrak r$, the paper's resolution term with the ellipse parameter optimized behaves like the Taylor tail of $e^{ikt}$, i.e.

$$E_{\rm QI}\;\lesssim\;\Big(\frac{e\,k\mathfrak r}{2N_1}\Big)^{N_1}\quad\Longrightarrow\quad N_1\;\gtrsim\;\tfrac e2\,k\mathfrak r+\frac{\log(1/\varepsilon)}{\log\big(2N_1/(e\,k\mathfrak r)\big)},$$

the same "Nyquist plus logarithm" form as the angular count, and for the same reason (both are Bessel tails). This is worth stating because it says the two floors are structurally identical: each is flat until $k\mathfrak r$ and super-exponential after. The record's measured offsets floor in 2-D is a power law, $e_N\sim N^{-10}$ with a shoulder near $N=20$ (expH05). That contradicts the prediction and is unexplained; the writeup should say so explicitly rather than fold it into "high-frequency profiles need greater scalar resolution".

**Sum versus max.** The certificate gives a sum of the two errors; the record measures $e\approx\max(e_M,e_N)$ within 10 to 20%. Consistent, since sum $\le2\max$, but the bound is loose by that factor, and the max structure is what makes the two floors an allocation instrument.

**"No factor of $M$" is about accuracy, not cost.** The unit count is $MN_1$, and the least-squares system pays for $M$ through rows per column and through the redundancy that forces rcond $10^{-14}$ on wide dictionaries (expH06). The certificate is silent on both.

### 2.4 The compact-domain hypothesis: right, and it undersells the good case

The caution is correct: a function known only on $K$ has no canonical Fourier measure, and the 1-D Fourier-extension step (Theorem 13 of the paper) has not been proved in $d$ dimensions with a controlled total-variation bound. Two additions.

**T3 (the globally analytic case is free, with explicit $k(\varepsilon)$).** Every smooth target in the record is defined by a formula analytic on all of $\mathbb R^d$ with an integrable Fourier transform: Gaussian bumps, $\cos(\omega\pi\rho)$, $1/(1+\alpha^2\rho^2)$, $\exp(\sin\pi z_1\cos\pi z_2)$, products of Runge factors. For these $\mu=(2\pi)^{-d}\widehat F\,d\xi$ directly, $B_1$ is finite and explicit, and no extension is needed. What the hypothesis then costs is the truncation of $\mu$ to $\lVert\xi\rVert\le k$, whose error is the spectral tail, and that tail fixes an effective spectral scale $k(\varepsilon)$:

| target | Fourier tail | $k(\varepsilon)$ |
|---|---|---|
| bandlimited ($\cos(\omega\pi\rho)$ up to a window) | compact | $k$ fixed |
| Gaussian of width $\sigma$ | $e^{-\sigma^2\lVert\xi\rVert^2/2}$ | $\sigma^{-1}\sqrt{2\log(1/\varepsilon)}$ |
| Runge $1/(1+\alpha^2\rho^2)$, pole at distance $1/\alpha$ | $\propto(\lVert\xi\rVert/\alpha)^{d/2}K_{d/2}(\lVert\xi\rVert/\alpha)\sim e^{-\lVert\xi\rVert/\alpha}$ | $\alpha\log(1/\varepsilon)$ |

Inserted into T1, the direction count becomes $M\asymp(c\,\mathfrak r\,k(\varepsilon))^{d-1}$: independent of the tolerance for bandlimited targets, $(\log1/\varepsilon)^{(d-1)/2}$ for Gaussian ones, $(\log1/\varepsilon)^{d-1}$ for pole-type ones. That is the ordering expH05 measured (bump and slow waves cheapest, narrow Runge never reaching $10^{-10}$ within $M\le16$), and it is a sharper statement than "high accuracy needs more directions".

**The compact case.** The $d$-dimensional analogue of Theorem 13 is what Fourier-extension frames provide (Matthysen and Huybrechs 2018 for arbitrary 2-D domains embedded in a box; Adcock and Huybrechs 2019 for the frame theory): exponential rates for functions analytic on a neighbourhood of $K$, with constants depending on the geometry. What has to be extracted, and is not in the literature in the needed form, is the total-variation bound on the extension's coefficients; by the paper's own route the $\ell^2\to\ell^1$ passage over $\asymp\omega_c^d$ modes costs $(1+\omega_c)^{d/2}$ instead of $(1+\omega_c)^{1/2}$, i.e. a polynomial factor in $k$, exponential only in the sense of $C^d$. The writeup's "we should not yet claim" is the right status; the paragraph can say what the missing lemma is.

### 2.5 Why directions are expensive

The cap-area argument and $\Theta_M\asymp M^{-1/(d-1)}$ are correct. The boxed direction count should be replaced by the T1 form, for the reason in H1. The closing statement, that resolving angular degree $L$ needs $\asymp L^{d-1}$ degrees of freedom, is right and is exactly what T1 makes quantitative, with $L\approx\tfrac e2k\mathfrak r+O(\log1/\varepsilon)$. (Typo: "t{the radial problem is 1-D;}" in the box.)

---

## 3. The compositional idea and its error theorem

### 3.1 The theorem is correct

The propagation argument is right: $\lvert v^\top z-\hat v^\top z\rvert\le\mathfrak r\delta$, the two-term split of each profile error, the sum into $\eta_q$, the Lipschitz step through $g_q$. One inconsistency: $\eta_q$ is defined with the bias term $\lvert b_q-\hat b_q\rvert$, and the boxed inequality then says "when the biases are exact"; keep $\eta_q$ as defined and drop the qualifier. The exponential corollary is correct.

### 3.2 What the theorem is, and what it is not

It is error propagation through a composition. It has no approximation content: given $f$ in the class, it bounds the error of an approximant of the same shape. All the content is in the hypothesis that $f$ has a short factorization with quantitatively analytic pieces, which the writeup states honestly. Four things the record already forces and the writeup omits.

**H3 (the floating-point floor).** Roundoff in evaluating $\hat P_q$ is amplified by $L_q$. With unit roundoff $u$ and channel coefficients $C_q$, the evaluation error of $\hat P_q$ is of order $u\,\lVert C_q\rVert_1$ (cancellation inside the ridge sum), so the theorem's own structure gives the floor

$$\lVert f-\hat f\rVert_\infty\;\gtrsim\;u\Big(1+\sum_qL_q\,\lVert C_q\rVert_1\Big),$$

a compositional condition number invariant under rescaling a channel. It should appear next to the exponential corollary, because the exponential rates are only meaningful above it. For the fast-waves target ($g(s)=\cos(6\pi\sqrt s)$, $L\approx18\pi^2$ near the channel's lower end) the floor is already $\sim10^{-14}$.

**H4 (direction exactness).** The direction term $L_{qr}\mathfrak r\delta_{qr}$ with $L_{qr}\sim k$ says $\delta\lesssim\varepsilon/(k\mathfrak r)$: for $\varepsilon=10^{-13}$ and $k\mathfrak r\sim5$ that is $\delta\sim10^{-14}$, i.e. directions exact to fp64. This is a strong constraint on learnability and the record has it measured (projection pursuit alone leaves 20 to 80% of the residual; the Gauss-Newton polish is what reaches $10^{-13}$).

**H5 (the theorem does not transfer to a solve).** In the shallow model certificate-to-projection holds because the readout is linear (H2). In the composed block $C$ sits inside the level-2 tanh, so only the outer readout $\Psi$ is a projection; finding $C$ and $V$ is projection-pursuit regression with nonlinear indices. The theorem is therefore an existence statement, and the record's own phrasing ("the gap between the oracle and the learned arms is the optimization problem") is the honest reading. The writeup's final paragraph says "whether those factorizations can be learned"; the body should say where the transfer breaks.

**H6 (the range hypothesis is a construction requirement).** "$L_q$ on an interval containing both the exact and approximate channel ranges" is the hypothesis under which the outer QI approximant $\hat g_q$ is accurate, and $\hat g_q$ is accurate only on its own mesh band with its collar. Since the range of $\hat P_q$ moves with the level-1 coefficients, the level-2 mesh must be re-laid on the tracked range; otherwise the hypothesis fails silently and the bound is void. This is not an implementation detail; it is the layer-2 form of the $\lambda\to0$ violation of the paper.

**T4 (a provable dividend class).** The writeup says the theorem does not show that $Q$ and $S$ are small. One can state a class where they are, and where the comparison with Section 2 is a theorem. Any function of a quadratic form,

$$f(x)=g\big(x^\top Ax+b^\top x+c\big),\qquad A=\sum_{i=1}^{\mathrm{rank}A}\lambda_iu_iu_i^\top,$$

is one channel with $\mathrm{rank}(A)+1$ ridge profiles ($\lambda_it^2$ along $u_i$, $t$ along $b/\lVert b\rVert$; $\mathrm{rank}(A)$ profiles if the origin is moved to the center), each entire, so $N_1=O(\log1/\varepsilon)$ with a small constant, and one outer function on the channel's range, $N_2\gtrsim\tfrac e2k_g\,\mathrm{diam}(P(K))+O(\log1/\varepsilon)$ with $k_g$ the outer function's spectral scale. Total units $(\mathrm{rank}A+1)N_1+N_2$, linear in $d$, against the shallow count $(c\,k\mathfrak r)^{d-1}N_1$. Sums of $Q$ such terms cost $Q$ channels. This class contains every radial target of the suite (bumps, concentric waves, Runge spikes, the packet with two anchors) and the record's dividend table, and it is where the compositional claim is a theorem rather than a hypothesis. A control outside the class (the record's random-ridge sum, or any target whose spectral measure is angularly smooth with no short composition) is where the two-floor law and the direction tax remain, and the writeup should name one.

### 3.3 The mixed-derivative example

Correct as stated: $e^{s+t}$ is not $u(s)+v(t)$. Its scope should be stated: it shows that two directions are not enough for the shallow model, not that the shallow model fails; with $\asymp(k\mathfrak r)^{d-1}$ directions it approximates $\exp(p_1+p_2)$ by Section 2. The dividend is the count, not representability.

---

## 4. Kolmogorov-Arnold

### 4.1 The statements

Kolmogorov's form and Lorentz's single-outer-function form are stated correctly, as is the regularity picture (inner functions can be taken Lipschitz, Fridman 1967 and Morris 2020; they cannot be $C^1$, Vitushkin 1954, Vitushkin-Henkin 1967). Sprecher's 1965 form is $f=\sum_{q=0}^{2d}\Phi\big(\sum_p\lambda_p\phi(x_p+\eta q)+q\big)$, a single outer function with the constant $q$ added outside the sum, which is the writeup's $\Delta_q=q$. The writeup attributes the general-$\Delta_q$, single-$\Phi$ form to Braun's treatment of the corrected construction. I could not reach the Springer or Bonndoc pages to check the exact statement; my recollection of Braun and Griebel (Constr. Approx. 2009) is that the corrected constructive theorem has outer functions $\Phi_q$ indexed by $q$ and inner shifts $\psi(x_p+qa)$ with $a=[\gamma(\gamma-1)]^{-1}$, $\gamma\ge2d+2$, $m\ge2d$. This should be verified against the source before the paper cites it. Nothing downstream depends on it except Section 5's "shared outer function", which in the $\Phi_q$ form becomes per-channel outer functions, i.e. the untied outer map the record already uses.

### 4.2 The propagation bound

Correct, including the modulus-of-continuity version. Note that it assumes the universal constants $\alpha_p$, $a$, $\Delta_q$ exact; in a learned setting they carry errors of the Section 3.1 type.

### 4.3 The central hole: the analytic Sprecher class is too thin to be the theorem

The writeup's conditional theorem reads: if $f=\sum_q\Phi(\Delta_q+\sum_p\alpha_p\psi(x_p+qa))$ with $\psi$ and $\Phi$ quantitatively analytic, then $N_1,N_2=O(\log1/\varepsilon)$. The implication is correct. The hypothesis is the problem.

In this form every channel's inner map is the same additive function of the coordinates, $\sum_p\alpha_p\psi(\,\cdot\,+qa)$, up to a translation; the only direction information in the whole inner layer is the single weight vector $\alpha$. Universality in the Kolmogorov-Sprecher construction is achieved because $\psi$ is a Cantor-type staircase whose shifted copies make the channel map $x\mapsto(P_0,\dots,P_{2d})$ injective on the cube; that is exactly the property analyticity destroys. Two concrete consequences:

- With $\psi$ linear, $P_q(x)=\Delta_q+\alpha^\top x+qa\sum_p\alpha_p$: all channels are the same linear functional plus constants, and $\sum_q\Phi(P_q)$ is a single ridge function along $\alpha$.
- The Sprecher tie is the rank-1 tie $C_{rjk}=a_r\Phi_{jk}$ of the record with the further constraints $a=\alpha$ fixed across channels, one profile $\psi$ for all channels up to shift, and $V=I$. The record's corrections already show the rank-1 tie cannot represent its own composition target with smooth pieces (three ridges with two distinct profile shapes inside one channel); the Sprecher form is a proper sub-family, so the same target is outside it a fortiori.

So the class "targets with an analytic Sprecher factorization" is not known to contain any of the targets the program cares about, and there is no reason to expect it to. The conditional theorem should be stated for the general block of Section 3 (learned unit directions, per-channel profiles, untied channel tensor, per-channel outer functions), which is what the record's Version 2 does and what expI01 measures. The Sprecher form then plays two legitimate roles: the universality anchor (which the untied block inherits, and which it also inherits from the shallow ridge model it contains, so KAT is a second proof rather than a needed one), and the most rigid corner of the rank sweep $C_{rjk}=\sum_{s=1}^Sa_r^{(s)}\Phi^{(s)}_{jk}$ from $S=1$ to untied. The first expI01 row is consistent with this reading: rank 1 at $1.1\times10^{-6}$ and untied at $5.9\times10^{-7}$ on fast waves in $d=3$, both far from the oracle floor, single seed.

---

## 5. The bias-shifted multichannel MLP

The algebra is correct: $\hat\psi(x_p+qa)$ expands into units $\tanh(\gamma_1x_p+b_j+q\Delta_{\rm bias})$ with $b_j=-\gamma_1s_j$, $\Delta_{\rm bias}=\gamma_1a$; the channel coefficients are $\alpha_pc_j$, rank one in $(p,j)$; the outer layer is a 1-D QI per channel. Three points.

**T5 (one bank, index-shifted readouts).** If the Sprecher shift is commensurate with the level-1 mesh, $a=m_ah_1$ for an integer $m_a$, then $b_j+q\Delta_{\rm bias}=b_{j-qm_a}$ and every channel reads the same extended bank on $N_1+2d\,m_a$ cells: $h_{qpj}=\tanh(\gamma_1x_p+b_{j-qm_a})$, with the channel tensor Toeplitz in $(q,j)$, $C_{q,p,j'}=\alpha_pc_{j'+qm_a}$. The first layer then has $d(N_1+2dm_a)$ units instead of $(2d+1)dN_1$, and the picture "copies of the bank with different bias offsets" becomes "one bank, shifted readouts". With Sprecher's $a=[\gamma(\gamma-1)]^{-1}$ (for $d=3$, $\gamma=8$, $a=1/56$) the mesh spacing would have to be $h_1\le a$, i.e. $N_1\ge56$ per unit interval, which is within QI's normal range.

**H7 (shared outer functions save no units).** In the Sprecher construction the offsets $\Delta_q$ (Sprecher's "$+q$") separate the channel ranges into disjoint intervals; that is how one continuous $\Phi$ can act as $2d+1$ different functions. Under QI the level-2 mesh must then cover $\bigcup_q\mathrm{range}(P_q)$, a union of $Q$ disjoint intervals, so the outer unit count is $\sum_qN_2^{(q)}$ whether or not the coefficients are shared. Sharing ties coefficients across channels; it removes nothing. If instead the ranges overlap, one $\Phi$ cannot act differently on different channels and sharing is a genuine restriction. Either way "outer weight sharing" is a constraint, not an economy, and per-channel $g_q$ costs nothing extra. The parameter comparison $M_2$ versus $mM_2$ in the SQIN document (and implicitly here) counts coefficients on a common grid that the construction does not use.

**H8 (range tracking, again).** The outer expansion is written with fixed centers $\tilde s_\ell$. The channel ranges $\mathrm{range}(\hat P_q)$ depend on the learned $C$, so the centers must be re-laid on the tracked range with a collar (H6). In the ordinary-MLP reading this is the statement that the second layer's biases must follow the first layer's output scale.

**H9 (scope of the picture).** "Shared 1-D QI functions with channel-dependent bias shifts" describes the Sprecher corner, where directions are the coordinate axes. In the general block of Section 3 the first layer is directional rays $w_{rj}=\gamma_1v_r$, $b_{rj}=-\gamma_1(c_j+v_r^\top x_0)$, and the channel tensor is untied. The paper's architecture figure should show that object, with the Sprecher form as a special case.

---

## 6. The one-page summary and the central claim

The summary is consistent with the body and inherits its issues. Suggested changes to the central claim, in order of importance.

1. Replace "angular resolution becomes expensive because direction space has dimension $d-1$" with the quantitative statement: the shallow model needs $M\asymp(c\,\mathfrak r\,k(\varepsilon))^{d-1}$ directions, with $\varepsilon$ entering only through $k(\varepsilon)$ (fixed, $\sqrt{\log}$ or $\log$ depending on the target's spectral tail), so the cost is set by the data radius and the spectral scale, not by the tolerance. This follows from T1 and T3, and it is what the record measured.
2. Replace "Kolmogorov-Arnold proves that finite scalar-channel representations are universal, while the QI theory gives exponential scalar resolution when the factors are quantitatively analytic" with a version that names the class: universality holds with non-smooth factors and is not needed (the untied block contains the shallow ridge model); the exponential rate holds on the class of short quantitatively analytic factorizations, of which functions of a few quadratic forms and short ridge sums are a provable subclass with a linear-in-$d$ unit count (T4).
3. Add the floor and the transfer caveat (H3, H5): the rates hold above $u(1+\sum_qL_q\lVert C_q\rVert_1)$, and they bound the construction, with the fitted network inheriting them only where the readout is a projection.
4. Drop the confidence line. A paper carries theorems and their hypotheses, not a probability.

---

## 7. Agreement with the record

| claim in the writeup | record | agreement |
|---|---|---|
| direction error $\le\mathfrak rB_1\Theta_V$ (Thm 1) | cost grows with $k\mathfrak r$; radius law; atoms exact once in $V$ | qualitative yes; the rate is a cliff, not first order (H1, T1) |
| $M\gtrsim(C_dk\mathfrak rB_0/\varepsilon)^{d-1}$ | $M\approx\max(\binom{p+d-1}{d-1},c(k\mathfrak r)^{d-1})$, $\varepsilon$ logarithmic | no; T1 gives the measured form |
| offsets floor free of $M$ | $e_N$ flat in $M$, same in 2, 3, 4-D | yes |
| offsets floor exponential in $N_1$ | $e_N\sim N^{-10}$ in 2-D, shoulder near $N=20$ | no; unexplained, should be flagged |
| total error = direction + 1-D (sum) | $e\approx\max(e_M,e_N)$ within 10 to 20% | consistent up to the factor 2 |
| bound centered on $x_0$, radius $\mathfrak r$ | data radius, not domain, sets $M$; $O(1)$ far field | yes; far field not mentioned |
| direction error term $L\mathfrak r\delta$ | $0.01^\circ$ tilt costs $3\times10^{-3}$; GN polish to fp64 exactness | yes (H4 makes it explicit) |
| compositional dividend (conditional) | E2, $d=3$ fast waves: block $10^{-6}$ vs shallow $2.5\times10^{-2}$ at equal cost, one seed, not at floor | partial; T4 gives the provable subclass |
| Sprecher/rank-1 form as the architecture | rank-1 tie fails the composition target; untied is the record's primary object | no (4.3) |
| level-2 range as a hypothesis | range tracking required in the notebook; QI scale relation holds in the tracked coordinate | yes once stated (H6, H8) |
| fp64 floor | composed floor derived in the record, measured by E1 | missing (H3) |

---

## 8. Concrete edits, in order

1. Section 2.5: replace the first-order direction count with the T1 proposition and its consequence $M\asymp(L+\tfrac e2k\mathfrak r+O(\log1/\varepsilon))^{d-1}$; keep Theorem 1 for line components; state the combined certificate (background plus atoms). Fix the typo in the box.
2. Section 2.3: state $N_1\gtrsim\tfrac e2k\mathfrak r+O(\log1/\varepsilon)$ for exponential-type profiles (T2); flag the measured $N^{-10}$ as an open discrepancy; note sum versus max.
3. Section 2.4: add the globally analytic case with the $k(\varepsilon)$ table (T3); name the missing $d$-dimensional TV lemma.
4. Section 2.2: add the certificate-to-projection lemma for the shallow model and the far-field remark (H2).
5. Section 3.1: fix the $\eta_q$ / "biases exact" mismatch; add the floor (H3), the direction-exactness corollary (H4), the non-transfer to a solve (H5), the range hypothesis as a hypothesis (H6); add the quadratic-form class as a theorem (T4) and a named control outside it.
6. Section 4.3: state the conditional theorem for the general untied block; keep Sprecher as universality anchor and rank-sweep corner; verify the Braun-Griebel statement before citing it for a single outer function.
7. Section 5: add the commensurate-shift observation (T5), the shared-outer-function unit count (H7), range tracking (H8), and the general-block picture (H9).
8. Section 1: state the convention ($K$ vs $K_0$), that $\sigma\approx1.05\lambda$ measured, that K3 at $\lambda=0.25$ is numerically verified rather than covered by Proposition 5, and use $A(\lambda)=(\pi^2/\lambda)/\sinh(\pi^2/\lambda)$ for any quoted alias value.
9. Remove the confidence line.
