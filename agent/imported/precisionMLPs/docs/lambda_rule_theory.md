# The aliasing theorem behind the $\lambda$ rule

**Statements, proofs, and numerical verification.** Written 2026-09-07 for the paper. Every constant quoted was recomputed in 30-digit arithmetic, and every measured number comes from either the stored expC03 sweep or the reproducible script `experiments/expC07_lambda_energy_rule/fiber_check.py`.

**What this document establishes.** The rule "choose the largest $\lambda$ with $|\widehat K(2\pi/\lambda)|/|\widehat K(0)|\le\varepsilon$" is the shadow of an exact identity. For a network of lattice translates of a kernel, the best possible relative $L^2$ error on a target of grid frequency $\theta_0$ is a closed-form function $\Lambda_r(\theta_0;\lambda)^{1/2}$ of the kernel's Fourier transform (Theorem 2). To leading order it equals $\sqrt{2}\,A(\lambda)\,(\theta_0/2\pi)^r$ with $A(\lambda)=|\widehat K(2\pi/\lambda)|/|\widehat K(0)|$ and $r$ the kernel order of the activation. This is not a bound: it matches the measured error of least-squares fits on the aliasing wall to within 1 to 3 percent, in the periodic setting, in the repo's interval-plus-halo setting, and in sixteen cells of the stored expC03 sweep (Section 9). The paper's cardinal interpolant has a different, larger floor, $|1-M(\theta_0)|\approx 2A(\lambda)$, also confirmed to four digits (Theorem 3, Section 9). From these, the admissible set of bandwidths is an interval $(0,\lambda_{\max}]$ with $\lambda_{\max}$ given by the rule (Theorem 6), and the optimum sits at its right end under a stated monotonicity model of the finite-precision error (Theorem 7). Two consequences the record did not have: the exact floor carries a factor $(\theta_0/2\pi)^r$, so for tanh it improves like $1/N$ at fixed $\lambda$; and the paper's Proposition 3 constant $c_{\mathrm{alias}}(\epsilon)=\epsilon\pi^2$ is the band-edge value of a frequency-resolved quantity whose low-frequency value is $\pi^2$ (Remark 3.3).

---

## 0. Setting, conventions, hypotheses

**Fourier convention.** $\hat g(\xi)=\int_{\mathbb R} g(u)e^{-i\xi u}\,du$, inverse $g(u)=\frac{1}{2\pi}\int\hat g(\xi)e^{i\xi u}d\xi$, Plancherel $\|g\|_2^2=\frac1{2\pi}\|\hat g\|_2^2$.

**Grid units.** The network lives on $[-1,1]$ with centers $c_k=-1+kh$, $h=2/N$, and inner weights $\gamma$. Set $u=x/h$ and $\lambda=\gamma h$. A neuron is then $\psi(\lambda(u-k))$, $k\in\mathbb Z$: unit lattice spacing, one shape parameter. Physical frequency $\omega$ becomes grid frequency $\theta=\omega h$; the grid Nyquist frequency is $\theta=\pi$ and the first grid harmonic is $\theta=2\pi$.

**Kernel hypotheses.** $K:\mathbb R\to\mathbb R$ satisfies

- (K1) $K$ is even, continuous, and $|K(u)|\le C_1e^{-c_1|u|}$;
- (K2) $|\widehat K(\xi)|\le C_2e^{-c_2|\xi|}$;
- (K3) $\widehat K(0)=\int K\neq 0$.

By (K1), $\widehat K$ is real, even and continuous. (K1) and (K2) together make every Poisson summation below valid and every lattice sum absolutely convergent. All six kernels considered later satisfy (K1) to (K3); the entire ones satisfy (K2) with room to spare.

**Activation and kernel order.** An activation $\psi$ has kernel order $r\ge 0$ if $r$ is the smallest integer with $K:=\psi^{(r)}$ satisfying (K1) to (K3). Then tanh and sigmoid have $r=1$, gelu and swish have $r=2$, $\mathrm{sech}^2$ and $e^{-x^2}$ have $r=0$. Minimality is forced, not chosen: if $K$ satisfies (K3) then $\widehat{K'}(0)=0$, so $K'$ fails (K3) (Lemma 8(d)).

**Generator and fibers.** Fix $\lambda>0$ and put $\phi(u):=K(\lambda u)$, so $\hat\phi(\theta)=\lambda^{-1}\widehat K(\theta/\lambda)$. For $\theta\notin 2\pi\mathbb Z$ define the generator spectrum of order $r$

$$w_r(\theta):=\frac{\lambda^{r}\,\hat\phi(\theta)}{(i\theta)^r}=\frac{\lambda^{r-1}\,\widehat K(\theta/\lambda)}{(i\theta)^r},$$

the fiber vector $\mathrm W_\theta:=\big(w_r(\theta+2\pi m)\big)_{m\in\mathbb Z}\in\ell^2(\mathbb Z)$, and $\Sigma(\theta):=\|\mathrm W_\theta\|^2=\sum_m|w_r(\theta+2\pi m)|^2$. For $r=0$, $\Sigma=G_\lambda$ is the usual bracket product $\sum_m|\hat\phi(\theta+2\pi m)|^2$, bounded and continuous. For $r\ge 1$, $\Sigma(\theta)\to\infty$ as $\theta\to 0$ and is finite elsewhere.

**The function class.** Let $\mathcal P_{r-1}$ be polynomials of degree $<r$ ($\mathcal P_{-1}=\{0\}$). The network class is

$$\mathcal N_\lambda:=\Big\{\,g=p+\sum_{k\in F}a_k\,\psi(\lambda(\cdot-k)) : F\subset\mathbb Z\text{ finite},\ a_k\in\mathbb R,\ p\in\mathcal P_{r-1}\Big\},$$

and $g^{(r)}=\lambda^r\sum_k a_k\phi(\cdot-k)$ for $g\in\mathcal N_\lambda$. The analysis space is

$$W:=\big\{\,g\in L^2(\mathbb R):\ \hat g=c\,w_r\ \text{a.e. for some measurable }2\pi\text{-periodic } c\big\}.$$

**The two rule quantities.**

$$A(\lambda):=\frac{|\widehat K(2\pi/\lambda)|}{|\widehat K(0)|},\qquad B(\lambda):=\frac{|\widehat K(\pi/\lambda)|}{|\widehat K(0)|},$$

together with $\alpha_j(\lambda):=|\widehat K(2\pi j/\lambda)|/|\widehat K(0)|$ ($\alpha_1=A$) and $\beta_j(\lambda):=|\widehat K((2j-1)\pi/\lambda)|/|\widehat K(0)|$ ($\beta_1=B$), and $\mu:=\sup|\widehat K|/|\widehat K(0)|\ge 1$.

**Regime hypotheses**, used only to turn exact formulas into clean bounds:

- (M$_\lambda$) $|\widehat K|$ is non-increasing on $[\pi/\lambda,\infty)$.
- (U$_\lambda$) $|\widehat K|$ is unimodal on $[0,\infty)$ (non-decreasing up to a mode $\omega_1$, non-increasing after), with $\omega_1\le\pi/\lambda$ and $B(\lambda)\le 1$.

For the six kernels the mode is at $\omega_1=0$ ($\mathrm{sech}^2$, sigmoid's kernel, Gaussian), $\omega_1=1$ (gelu, $\mu=1.213$) and $\omega_1\approx 0.51$ (swish, $\mu=1.169$), and unimodality was checked numerically on $(0,10)$; so (M$_\lambda$) and (U$_\lambda$) hold for every $\lambda\le 1$ of interest.

---

## 1. The fiber decomposition

**Lemma 1 (network outputs lie in $W$).** If $g\in\mathcal N_\lambda$ and $g\in L^2(\mathbb R)$, then $g\in W$ with $c(\theta)=\hat a(\theta):=\sum_{k\in F}a_ke^{-ik\theta}$.

*Proof.* $g^{(r)}=\lambda^r\sum_{k\in F}a_k\phi(\cdot-k)$ is a finite sum of translates of an $L^1\cap L^2$ function, with Fourier transform $\lambda^r\hat a(\theta)\hat\phi(\theta)$ by the shift rule. As tempered distributions $\widehat{g^{(r)}}=(i\theta)^r\hat g$. Hence $(i\theta)^r\hat g=\lambda^r\hat a\hat\phi$ as distributions. Both sides are locally integrable functions ($\hat g\in L^2$), so the identity holds almost everywhere, and dividing by $(i\theta)^r$ on $\theta\ne 0$ gives $\hat g=\hat a\,w_r$ a.e. $\square$

For $r=0$ every $g\in\mathcal N_\lambda$ is in $L^2$. For $r\ge 1$ membership in $L^2$ constrains the polynomial part and the low-order moments of $a$ (for tanh: $\sum_k a_k=0$ and $p=0$); on a bounded interval this is a single degree of freedom supplied by the bias column. Only fibers $\theta\ne 0$ enter the computations below, and $\{0\}$ is a null set.

**Lemma 2 (fiber Parseval).** For $f\in L^2(\mathbb R)$ let $F_\theta:=(\hat f(\theta+2\pi m))_{m\in\mathbb Z}$. Then $F_\theta\in\ell^2$ for a.e. $\theta$, and $\|f\|_2^2=\frac1{2\pi}\int_{-\pi}^{\pi}\|F_\theta\|^2_{\ell^2}\,d\theta$. If $g\in W$ with $\hat g=c\,w_r$, then the fiber of $g$ is $c(\theta)\mathrm W_\theta$ and

$$\|f-g\|_2^2=\frac1{2\pi}\int_{-\pi}^{\pi}\big\|F_\theta-c(\theta)\mathrm W_\theta\big\|^2_{\ell^2}\,d\theta.$$

*Proof.* Plancherel and Tonelli over the cells $[-\pi,\pi)+2\pi m$ give the first identity. Since $c$ is $2\pi$-periodic, $\hat g(\theta+2\pi m)=c(\theta)w_r(\theta+2\pi m)$, which is the second. $\square$

**Lemma 2$'$ (the ripple identity).** Under (K1), (K2), for every $u\in\mathbb R$,

$$\sum_{k\in\mathbb Z}K(\lambda(u-k))=\frac{1}{\lambda}\sum_{m\in\mathbb Z}\widehat K\!\Big(\frac{2\pi m}{\lambda}\Big)e^{2\pi i m u}=\frac{\widehat K(0)}{\lambda}\Big(1+2\sum_{m\ge 1}\alpha_m(\lambda)\cos(2\pi m u)\Big).$$

*Proof.* Poisson summation for $u\mapsto\phi(u)$, using $\hat\phi(2\pi m)=\lambda^{-1}\widehat K(2\pi m/\lambda)$ and evenness. $\square$

So the sum of all the bumps, the network's attempt at the constant function with coefficients $a_k\equiv 1$, is a constant plus a ripple at the grid period whose relative peak-to-trough amplitude is $4A(\lambda)(1+O(\alpha_2/\alpha_1))$. Numerically, for $K=\mathrm{sech}^2$ at $\lambda=0.25$ the ripple is $2.26\times 10^{-15}$ and $4A(0.25)=2.26\times10^{-15}$. This is the mechanism in its most concrete form; the theorems below say that no other choice of coefficients does better than this order.

---

## 2. The projection theorem

**Theorem 1 (fiber projection theorem).** For every $f\in L^2(\mathbb R)$,

$$\operatorname{dist}_{L^2}(f,W)^2=\frac{1}{2\pi}\int_{-\pi}^{\pi}\Big(\|F_\theta\|^2-\frac{|\langle F_\theta,\mathrm W_\theta\rangle|^2}{\|\mathrm W_\theta\|^2}\Big)d\theta ,$$

and the infimum is attained by $g^*\in W$ with $\hat g^*=c^*w_r$, where $c^*(\theta):=\langle F_\theta,\mathrm W_\theta\rangle/\|\mathrm W_\theta\|^2$ (set to $0$ where $\|\mathrm W_\theta\|\in\{0,\infty\}$).

*Proof.* Lower bound. For $g\in W$, Lemma 2 gives $\|f-g\|^2=\frac1{2\pi}\int\|F_\theta-c(\theta)\mathrm W_\theta\|^2d\theta$. For each $\theta$, $\inf_{z\in\mathbb C}\|F_\theta-z\mathrm W_\theta\|^2$ is the squared distance in $\ell^2$ from $F_\theta$ to the line $\mathbb C\mathrm W_\theta$, which equals $\|F_\theta\|^2-|\langle F_\theta,\mathrm W_\theta\rangle|^2/\|\mathrm W_\theta\|^2$, attained at $z=c^*(\theta)$. Integrating gives $\|f-g\|^2\ge$ the right-hand side.

Attainment. $c^*$ is measurable and $2\pi$-periodic. Cauchy–Schwarz gives $\|c^*(\theta)\mathrm W_\theta\|=|\langle F_\theta,\mathrm W_\theta\rangle|/\|\mathrm W_\theta\|\le\|F_\theta\|$, so by Lemma 2 the function with transform $c^*w_r$ has $\frac1{2\pi}\int_{\mathbb R}|c^*w_r|^2=\frac1{2\pi}\int_{-\pi}^{\pi}\|c^*\mathrm W_\theta\|^2d\theta\le\|f\|^2<\infty$. Thus $g^*\in L^2$, $g^*\in W$, and $\|f-g^*\|^2$ equals the right-hand side. $\square$

*Remarks.* (i) This is the shift-invariant-space projection formula; for $r=0$ it is Theorem 2.20 of de Boor, DeVore and Ron, *Approximation from shift-invariant subspaces of $L_2(\mathbb R^d)$*, Trans. AMS 341 (1994), and the quantity $1-|\hat\phi|^2/G_\lambda$ is the least-squares "approximation kernel" of Blu and Unser (IEEE Trans. Signal Process. 47, 1999). The proof above is self-contained and covers $r\ge 1$. Verify the exact statements before citing. (ii) Finite networks approach $g^*$: for $r=0$, trigonometric polynomials are dense in $L^2(\mathbb T)$ and $\|c\,w_0\|^2\le\|G_\lambda\|_\infty\|c\|^2_{L^2(\mathbb T)}$; for $r\ge 1$ the same holds in the weighted space $L^2(\mathbb T,\Sigma)$ after factoring out the required vanishing at $\theta=0$. The lower-bound direction, which is what "irreducible" means, needs none of this: it holds for every $g\in\mathcal N_\lambda\cap L^2$ by Lemma 1.

---

## 3. Band-limited targets: the exact aliasing floor

Define the **aliasing kernel of order $r$**

$$\Lambda_r(\theta;\lambda):=1-\frac{|w_r(\theta)|^2}{\Sigma(\theta)}=\frac{\displaystyle\sum_{m\neq 0}\widehat K\Big(\frac{\theta+2\pi m}{\lambda}\Big)^2|\theta+2\pi m|^{-2r}}{\displaystyle\sum_{m\in\mathbb Z}\widehat K\Big(\frac{\theta+2\pi m}{\lambda}\Big)^2|\theta+2\pi m|^{-2r}}\qquad(\theta\notin 2\pi\mathbb Z).$$

**Theorem 2 (exact floor for band-limited targets).** Let $0<\theta_0<\pi$ and $f\in L^2(\mathbb R)$ with $\operatorname{supp}\hat f\subset[-\theta_0,\theta_0]$. Then

$$\operatorname{dist}(f,W)^2=\frac1{2\pi}\int_{-\theta_0}^{\theta_0}|\hat f(\theta)|^2\,\Lambda_r(\theta;\lambda)\,d\theta .$$

In particular $\min_{|\theta|\le\theta_0}\Lambda_r^{1/2}\le\operatorname{dist}(f,W)/\|f\|\le\max_{|\theta|\le\theta_0}\Lambda_r^{1/2}$, and the lower bound $\|f-g\|\ge\operatorname{dist}(f,W)$ holds for every $g\in\mathcal N_\lambda\cap L^2$.

*Proof.* For $|\theta|<\pi$ and $m\ne 0$, $|\theta+2\pi m|>\pi>\theta_0$, so $F_\theta=\hat f(\theta)e_0$. Then $\|F_\theta\|^2-|\langle F_\theta,\mathrm W_\theta\rangle|^2/\|\mathrm W_\theta\|^2=|\hat f(\theta)|^2\big(1-|w_r(\theta)|^2/\Sigma(\theta)\big)$; the factor $\lambda^{2r-2}$ common to numerator and denominator cancels. Apply Theorem 1 and Lemma 1. $\square$

**Corollary 2.1 (pure tone).** On the circle $\mathbb R/P\mathbb Z$ with $P\in\mathbb N$ lattice points, periodized generator $\phi^{\mathrm{per}}=\sum_j\phi(\cdot+jP)$ and $f(u)=e^{i\theta_0u}$, $\theta_0\in\frac{2\pi}{P}\mathbb Z\cap(-\pi,\pi)$, the same computation gives exactly

$$\frac{\operatorname{dist}(f,W^{\mathrm{per}})}{\|f\|}=\Lambda_r(\theta_0;\lambda)^{1/2},$$

because $\widehat{\phi^{\mathrm{per}}}$ at the frequencies $\frac{2\pi}{P}\mathbb Z$ coincides with $\hat\phi$ (Poisson) and the coefficient DFT is $P$-periodic, so the fibers are again $\{\theta_0+2\pi m\}$. This is the exactly finite-dimensional least-squares problem used in the numerical check of Section 9.

**Corollary 2.2 (two-ghost form; the low-frequency limit).** For $0<|\theta|\le\theta_0<\pi$ write

$$\rho_\pm(\theta):=\frac{\widehat K\big(\tfrac{2\pi\pm|\theta|}{\lambda}\big)^2}{\widehat K(\theta/\lambda)^2}\Big(\frac{|\theta|}{2\pi\pm|\theta|}\Big)^{2r},\qquad T(\theta):=\text{the same sum over }|m|\ge 2 .$$

Then $\Lambda_r=\dfrac{\rho_-+\rho_++T}{1+\rho_-+\rho_++T}$, hence

$$\Lambda_r(\theta;\lambda)^{1/2}=\sqrt{\rho_-(\theta)+\rho_+(\theta)}\,\big(1+O(\rho_-+\rho_++T)\big),\qquad \Lambda_r(\theta;\lambda)^{1/2}\xrightarrow[\theta\to 0]{}\sqrt{2}\,A(\lambda)\Big(\frac{|\theta|}{2\pi}\Big)^{r}\big(1+o(1)\big).$$

Under (M$_\lambda$), $T\le\rho_-\cdot 2\sum_{j\ge2}\big(\widehat K((2\pi j-\theta_0)/\lambda)/\widehat K((2\pi-\theta_0)/\lambda)\big)^2$, which for the six kernels and $\lambda\le 1$ is below $10^{-6}\rho_-$.

*Proof.* Algebra on the definition; the $m=\pm 1$ terms are $\rho_\pm$ after dividing through by the $m=0$ term. As $\theta\to 0$, $\widehat K(\theta/\lambda)\to\widehat K(0)$ and $\rho_\pm\to A^2(\theta/2\pi)^{2r}$. $\square$

**Reading of Corollary 2.2.** Two effects, both exact in the formula and both visible in data:

1. *Frequency resolution.* At grid frequency $\theta$ the dominant ghost sits at $2\pi-|\theta|$, closer than $2\pi$. For a pole-type kernel with $|\widehat K(\xi)|\sim e^{-a\xi}$ the effective exponent is $a(2\pi-2|\theta|)/\lambda$, which decreases from $2\pi a/\lambda$ at low frequency to $2a\epsilon\pi/\lambda$ at $|\theta|=(1-\epsilon)\pi$.
2. *Integration attenuation.* A network of order $r$ (tanh: $r=1$) reproduces the target by integrating a bump network $r$ times. The ghost lives at grid frequency $\approx 2\pi$ and the signal at $\theta$, so integration attenuates the ghost relative to the signal by $(|\theta|/2\pi)^r$. In physical units $\theta_0=\omega_0h=2\omega_0/N$: **for a tanh network the aliasing floor at fixed $\lambda$ falls like $1/N$**, and like $1/N^2$ for gelu. The bump network ($r=0$), and the paper's cardinal interpolant (Theorem 3), have no such attenuation.

**Corollary 2.3 (targets that are not band-limited).** Write $f=f_0+f_1$ with $\hat f_0=\hat f\,\mathbf 1_{[-\theta_0,\theta_0]}$. Then $\operatorname{dist}(f,W)\ge\operatorname{dist}(f_0,W)-\|f_1\|$, with $\operatorname{dist}(f_0,W)$ given by Theorem 2. If $|\hat f(\omega)|\le Ce^{-\rho|\omega|}$ in physical units (analytic in a strip of half-width $\rho$), then in grid units $|\hat f(\theta)|\le Ce^{-\rho|\theta|/h}$ and $\|f_1\|\le C'e^{-\rho\theta_0/h}=C'e^{-\rho\theta_0N/2}$. This is the paper's resolution term. Choosing $\theta_0=2\ln(1/\varepsilon)/(\rho N)$ makes $\|f_1\|\lesssim\varepsilon$, and $\theta_0\to 0$ as $N\to\infty$: **for large $N$ every analytic target is low-frequency in grid units**, and the $\theta\to 0$ form of Corollary 2.2 applies with this $\theta_0$.

**Remark 3.3 (the paper's Assumption K2 and Proposition 3).** The paper's mode-response function is $M_h(\omega)=\widehat K_\gamma(\omega)/D_h(\omega)$, which in grid units is $M(\theta)=\hat\phi(\theta)/\sum_m\hat\phi(\theta+2\pi m)$, and its aliasing assumption bounds $\sup_{|\theta|\le(1-\epsilon)\pi}|1-M(\theta)|$. Now

$$1-M(\theta)=\frac{\sum_{m\ne 0}\hat\phi(\theta+2\pi m)}{\sum_m\hat\phi(\theta+2\pi m)}\approx\frac{\widehat K\big(\tfrac{2\pi-|\theta|}{\lambda}\big)+\widehat K\big(\tfrac{2\pi+|\theta|}{\lambda}\big)}{\widehat K(\theta/\lambda)},$$

so for $K=\mathrm{sech}^2$ ($a=\pi/2$) the exponent at grid frequency $\theta$ is $\pi(\pi-|\theta|)/\lambda$: equal to $\epsilon\pi^2/\lambda$ at the band edge $|\theta|=(1-\epsilon)\pi$, which is exactly Proposition 3's $c_{\mathrm{alias}}(\epsilon)=\epsilon\pi^2$, and equal to $\pi^2/\lambda$ at low frequency, which is the rule's exponent. The paper's constant is therefore correct but is the worst case over the band; for a resolved target the relevant exponent is the full $\pi^2/\lambda$. Stated with $\epsilon\pi^2$, Theorem 1 of the paper predicts a viable $\lambda$ far smaller than the one that works; stated frequency-resolved, it predicts $0.25$.

---

## 4. Cardinal interpolation has a larger floor

The paper's construction is not a projection. It interpolates: $Qf=\sum_kf(k)L(\cdot-k)$ with the cardinal function $\hat L=M$. Assume Fourier positivity $\widehat K>0$ (the paper's K5), so $D(\theta):=\sum_m\hat\phi(\theta+2\pi m)>0$, $M:=\hat\phi/D$ is well defined, and $\sum_mM(\theta+2\pi m)=1$ (partition of unity; the paper's equation (3)).

**Theorem 3 (error of the cardinal quasi-interpolant).** Let $f\in L^2(\mathbb R)$ be continuous with $\operatorname{supp}\hat f\subset[-\theta_0,\theta_0]$, $\theta_0<\pi$. Then

$$\|f-Qf\|^2=\frac1{2\pi}\int_{-\theta_0}^{\theta_0}|\hat f(\theta)|^2E(\theta)\,d\theta,\qquad E(\theta):=\big(1-M(\theta)\big)^2+\sum_{m\ne 0}M(\theta+2\pi m)^2 .$$

Moreover $E\ge\Lambda_0$ pointwise, and as $\theta\to 0$: $1-M(\theta)\to 2A(\lambda)(1+O(\alpha_2/\alpha_1+A))$ and $M(\pm 2\pi)\to A(\lambda)(1+O(A))$, so $E(\theta)^{1/2}\to\sqrt{6}\,A(\lambda)(1+o(1))$.

*Proof.* A band-limited $L^2$ function is entire of exponential type with $\sum_k|f(k)|^2<\infty$, and Poisson summation gives $\sum_kf(k)e^{-ik\theta}=\sum_m\hat f(\theta+2\pi m)$, which equals $\hat f(\theta)$ for $|\theta|<\pi$ and is $2\pi$-periodic. Hence $\widehat{Qf}(\theta+2\pi m)=M(\theta+2\pi m)\hat f(\theta)$ and the fiber of $f-Qf$ is $\hat f(\theta)\big(e_0-(M(\theta+2\pi m))_m\big)$, whose squared norm is $|\hat f(\theta)|^2E(\theta)$. $E\ge\Lambda_0$ because $Qf\in W$ (for $r=0$) and Theorem 1 is the minimum over $W$. The limits follow from $1-M(0)=2\sum_{j\ge1}\hat\phi(2\pi j)/D(0)$ and $M(2\pi)=\hat\phi(2\pi)/D(0)$. $\square$

**What this means for the tanh construction.** The paper applies $Q$ to $f'$ and integrates once. The in-band defect $(1-M(\theta))\widehat{f'}(\theta)$ integrates to $(1-M(\theta))\hat f(\theta)$: it is *not* attenuated. Only the ghost terms gain the $(\theta/2\pi)$ factor. So at the network level:

- least-squares readout (projection): relative floor $\approx\sqrt 2\,A(\lambda)\,\theta_0/2\pi$ (Corollary 2.2, $r=1$);
- cardinal QI construction (interpolation): relative floor $\approx|1-M(\theta_0)|\approx 2A(\lambda)$.

The ratio is $2\sqrt2\pi/\theta_0$, about $90$ for $\sin 2\pi x$ at $N=128$. This is the mechanism behind expA02's "lstsq $\le$ QI in all 48 cells" and expC01/C02's "QI's optimum is a narrow band near $0.25$–$0.30$, lstsq's bottom is wide": both sit on walls governed by the same $e^{-\pi^2/\lambda}$, but the interpolant's wall is two orders of magnitude higher. Section 9 confirms both floors to four digits.

---

## 5. Riesz bounds and conditioning

Let $T_\lambda:\ell^2\to L^2$, $a\mapsto\sum_ka_k\phi(\cdot-k)$ be the synthesis operator of the bump space ($r=0$).

**Theorem 4 (Riesz bounds).** Under (K1) to (K3), for all $a\in\ell^2(\mathbb Z)$,

$$\|T_\lambda a\|_2^2=\frac1{2\pi}\int_{-\pi}^{\pi}|\hat a(\theta)|^2G_\lambda(\theta)\,d\theta,\qquad G_\lambda(\theta)=\lambda^{-2}\sum_m\widehat K\Big(\frac{\theta+2\pi m}{\lambda}\Big)^2 .$$

Under (U$_\lambda$) (which contains (M$_\lambda$)):

$$\lambda^{-2}\widehat K(0)^2B(\lambda)^2\ \le\ \inf_\theta G_\lambda\ \le\ G_\lambda(\pi)=2\lambda^{-2}\widehat K(0)^2\sum_{j\ge 1}\beta_j(\lambda)^2,\qquad \lambda^{-2}\widehat K(0)^2\ \le\ G_\lambda(0)\ \le\ \sup_\theta G_\lambda\ \le\ \lambda^{-2}\widehat K(0)^2\Big(\mu^2+2\sum_{j\ge1}\beta_j^2\Big),$$

so the condition number $\kappa(T_\lambda)=\sqrt{\sup G_\lambda/\inf G_\lambda}$ satisfies

$$\frac{1}{\sqrt2\,B(\lambda)\sqrt{1+\sum_{j\ge 2}\beta_j^2/B^2}}\ \le\ \kappa(T_\lambda)\ \le\ \frac{\sqrt{\mu^2+2\sum_{j\ge1}\beta_j^2}}{B(\lambda)} .$$

*Proof.* The identity is Lemma 2 with $f=0$ and Parseval on $\mathbb T$ ($\frac1{2\pi}\int|\hat a|^2=\|a\|^2$). Lower bound on $G_\lambda$: for $|\theta|\le\pi$ the $m=0$ term is $\lambda^{-2}\widehat K(\theta/\lambda)^2$ with $|\theta|/\lambda\le\pi/\lambda$; by unimodality with mode $\omega_1\le\pi/\lambda$, $|\widehat K(\xi)|\ge\min\{|\widehat K(0)|,|\widehat K(\pi/\lambda)|\}=|\widehat K(\pi/\lambda)|$ for $\xi\in[0,\pi/\lambda]$ (using $B\le1$). The value at $\theta=\pi$ is the displayed sum by symmetry. Upper bound: for $|\theta|\le\pi$ and $m\ne0$, $|\theta+2\pi m|\ge(2|m|-1)\pi\ge\pi$, so by (M$_\lambda$) the $m$-th term is at most $\lambda^{-2}\widehat K((2|m|-1)\pi/\lambda)^2$; the $m=0$ term is at most $\lambda^{-2}\sup|\widehat K|^2$. $\square$

**Coefficient amplification.** The fiber minimizer of Theorem 1 is $c^*(\theta)=\langle F_\theta,\mathrm W_\theta\rangle/\Sigma(\theta)$; for a band-limited target, $c^*(\theta)=\hat f(\theta)\overline{w_r(\theta)}/\Sigma(\theta)$ and $|c^*(\theta)|\le|\hat f(\theta)|/|w_r(\theta)|$, which for $r=0$ reads $\lambda|\hat f(\theta)|/|\widehat K(\theta/\lambda)|$. Representing content at grid frequency $\theta$ costs a factor $1/|\widehat K(\theta/\lambda)|$ in coefficient size, rising to $1/B$ at the band edge. This is the deconvolution formula of expA07 and the origin of the alternating, large coefficients at small $\lambda$.

**Proposition 5 ($B$ against $A$).** If $|\widehat K(\xi)|/|\widehat K(0)|=c\,\xi^{q}e^{-a\xi^{p}}$ for $\xi\ge\xi_1$ and $\pi/\lambda\ge\xi_1$, then

$$B(\lambda)=A(\lambda)^{2^{-p}}\cdot c^{\,1-2^{-p}}\,(\pi/\lambda)^{q(1-2^{-p})}\,2^{-q2^{-p}} .$$

*Proof.* Solve $A=c(2\pi/\lambda)^qe^{-a(2\pi/\lambda)^p}$ for $e^{-a(2\pi/\lambda)^p}$, take the $2^{-p}$ power to get $e^{-a(\pi/\lambda)^p}$, substitute into $B$. $\square$

For $\mathrm{sech}^2$ ($c=\pi$, $q=1$, $a=\pi/2$, $p=1$): $B=A^{1/2}\sqrt{\pi^2/2\lambda}$, giving $B(0.25)=1.06\times10^{-7}$ against $A(0.25)^{1/2}=2.4\times10^{-8}$. For Gaussian tails ($p=2$) $B=A^{1/4}$: $1.5\times10^{-4}$ for $e^{-x^2}$ and $1.1\times10^{-3}$ for gelu at their rule points. **At the rule point, aliasing is at $\varepsilon$ while the kernel-spectrum part of the conditioning is at $\varepsilon^{1/2}$ (pole-type) or $\varepsilon^{1/4}$ (entire).** The $10^{-7}$ figure matches the $10^{7}$ to $10^{9}$ condition numbers expA01 measured on the interior geometry; the $10^{19}$ values of expC04 come from saturated halo columns, which are outside this theory.

---

## 6. Admissibility and the selection principle

**Theorem 6 (the admissible bandwidths form an interval whose right end is the rule).** Fix $r$, $0<\theta_1\le\theta_0<\pi$, and $\lambda$ satisfying (M$_\lambda$). Let $f$ be band-limited to $[-\theta_0,\theta_0]$ and let $\eta^2\in(0,1]$ be the fraction of $\|f\|^2$ carried by $\theta_1\le|\theta|\le\theta_0$. Then for every $g\in\mathcal N_\lambda\cap L^2$ and every $e\in L^2$ with $\|e\|\le\delta$,

$$\frac{\|f-(g+e)\|}{\|f\|}\ \ge\ \frac{\eta}{\sqrt 2}\,\min\Big\{1,\ \frac{A(\lambda)}{\mu}\Big(\frac{\theta_1}{2\pi}\Big)^{r}\Big\}-\frac{\delta}{\|f\|} .$$

Consequently, if the computed output reaches relative error $\epsilon_{\mathrm{tot}}$, then

$$A(\lambda)\ \le\ \frac{\sqrt2\,\mu}{\eta}\Big(\frac{2\pi}{\theta_1}\Big)^{r}\Big(\epsilon_{\mathrm{tot}}+\frac{\delta}{\|f\|}\Big),$$

and since $\lambda\mapsto A(\lambda)$ is non-decreasing wherever (M$_\lambda$) holds, the set of bandwidths compatible with a given precision is an interval $(0,\lambda_{\max}]$.

*Proof.* By Lemma 1 and Theorem 2, $\|f-g\|^2\ge\frac1{2\pi}\int_{\theta_1\le|\theta|\le\theta_0}|\hat f|^2\Lambda_r\,d\theta\ge\eta^2\|f\|^2\min_{\theta_1\le|\theta|\le\theta_0}\Lambda_r$. Write $\Lambda_r=S_{\ne0}/(S_0+S_{\ne0})$: if $S_{\ne0}\ge S_0$ then $\Lambda_r\ge\frac12$; otherwise $\Lambda_r\ge S_{\ne0}/(2S_0)\ge\rho_-/2$. Under (M$_\lambda$), $(2\pi-|\theta|)/\lambda\in[\pi/\lambda,2\pi/\lambda]$ gives $|\widehat K((2\pi-|\theta|)/\lambda)|\ge|\widehat K(2\pi/\lambda)|$, while $\widehat K(\theta/\lambda)^2\le\mu^2\widehat K(0)^2$ and $(|\theta|/(2\pi-|\theta|))^{2r}\ge(\theta_1/2\pi)^{2r}$; so $\rho_-\ge\mu^{-2}A^2(\theta_1/2\pi)^{2r}$. Take square roots and apply the triangle inequality for $e$. Monotonicity of $A$: if $\lambda'<\lambda$ then $2\pi/\lambda'>2\pi/\lambda\ge\pi/\lambda$, and $|\widehat K|$ is non-increasing there. $\square$

For $r=0$ and a target concentrated at low frequency ($\eta\approx1$, $\mu=1$) the condition is $A(\lambda)\lesssim\sqrt2\,\epsilon_{\mathrm{tot}}$: the simple rule. For $r\ge1$ the exact admissibility boundary is where $\frac1{2\pi}\int|\hat f|^2\Lambda_r=\epsilon^2\|f\|^2$, and Corollary 2.2 gives its leading-order form $\sqrt2A(\lambda)(\theta_0/2\pi)^r=\epsilon$.

**Theorem 7 (selection, conditional on a monotone finite-precision model).** Fix $f$ band-limited as above and let $\mathcal A(\lambda):=\big(\frac1{2\pi}\int|\hat f|^2\Lambda_r(\cdot;\lambda)\big)^{1/2}/\|f\|$ be the exact aliasing floor. Suppose the computed relative error $\mathcal E(\lambda)$ of some solver satisfies, on an interval $I$ of bandwidths,

- (FP1) $\mathcal E(\lambda)\ge\max\{\mathcal A(\lambda),\ \varepsilon\}$, and
- (FP2) $\mathcal E(\lambda)\le\mathcal A(\lambda)+\varepsilon\,\Xi(\lambda)$ with $\Xi$ non-increasing on $I$.

Let $\lambda_\varepsilon:=\sup\{\lambda\in I:\mathcal A(\lambda)\le\varepsilon\}$. Then (i) for $\lambda>\lambda_\varepsilon$, $\mathcal E(\lambda)\ge\mathcal A(\lambda)>\varepsilon$, and $\mathcal A$ grows exponentially in $1/\lambda$ decrements; (ii) for $\lambda\le\lambda_\varepsilon$, $\mathcal E(\lambda)\le\varepsilon(1+\Xi(\lambda))$, and since $\Xi$ is non-increasing this upper bound is smallest at $\lambda=\lambda_\varepsilon$, where it equals $\varepsilon(1+\Xi(\lambda_\varepsilon))$; (iii) hence $\inf_I\mathcal E\in[\varepsilon,\varepsilon(1+\Xi(\lambda_\varepsilon))]$ and every $\lambda$ within a factor $(1+\Xi(\lambda_\varepsilon))$ of optimal lies in $(0,\lambda_\varepsilon(1+o(1))]$.

*Proof.* (i) is (FP1) with monotonicity of $\mathcal A$ (Theorem 6). (ii) is (FP2) with $\Xi$ non-increasing. (iii) combines them. $\square$

*What supports (FP2)'s monotonicity.* Theorem 4: the conditioning of the synthesis operator is $\asymp 1/B(\lambda)$, and $B$ is non-decreasing in $\lambda$ under (M$_\lambda$). The paper's Theorem 1: the halo term $e^{-c_3\lambda R}$ and stencil term $e^{-c_4\lambda K_c}$ are non-increasing in $\lambda$ at fixed $R,K_c$. *What is not proven:* that the measured floor is flat all the way up to $\lambda_\varepsilon$, i.e. that $\Xi(\lambda_\varepsilon)$ is $O(1)$. That is an empirical fact (expC03: flat from $\lambda\approx0.10$ to the wall) and depends on the truncated-SVD readout discarding the near-Nyquist modes it cannot resolve. The theorem's content is therefore: **the rule identifies the right end of the optimal set; the optimal set extends left of it by an amount the theory does not fix.** The measured argmins of expC03 and expC07 scattering left of the predicted wall is the expected signature.

---

## 7. Solving the rule

**Lemma 8 (exact covariances).**

(a) *Amplitude.* $A_{cK}=A_K$ and $\Lambda_r[cK]=\Lambda_r[K]$ for $c\ne0$.

(b) *Stretch.* For $K_s:=K(\cdot/s)$, $\widehat{K_s}(\xi)=s\widehat K(s\xi)$, hence $A_{K_s}(\lambda)=A_K(\lambda/s)$, $\Lambda_r[K_s](\theta;\lambda)=\Lambda_r[K](\theta;\lambda/s)$, and every rule solution scales: $\lambda^*_{K_s}=s\,\lambda^*_K$.

(c) *Sigmoid.* $\sigma'(x)=\tfrac14\mathrm{sech}^2(x/2)$, so $K_\sigma=\tfrac14(\mathrm{sech}^2)_2$ and by (a), (b) $\lambda^*_\sigma=2\lambda^*_{\tanh}$ exactly. Independently, $\sigma(\gamma(x-c))=\tfrac12+\tfrac12\tanh(\tfrac\gamma2(x-c))$: with a bias column the sigmoid network at $\gamma$ spans the tanh network at $\gamma/2$, so any correct rule must return the factor $2$.

(d) *Minimal order.* If $K$ satisfies (K1) to (K3) then $\widehat{K'}(\xi)=i\xi\widehat K(\xi)$ vanishes at $0$, so $K'$ violates (K3): the kernel order $r$ is the unique order at which the rule is defined. Differentiation multiplies $\widehat K$ by the polynomial $i\xi$, so across activations of different orders the tail's exponential class, not $r$, sets the solution (Proposition 9).

*Proof.* Direct from the definitions and $\widehat{K(\cdot/s)}(\xi)=s\widehat K(s\xi)$. $\square$

**Proposition 9 (asymptotics of the rule solution).** Suppose $|\widehat K(\xi)|/|\widehat K(0)|=c\,\xi^{q}e^{-a\xi^{p}}(1+o(1))$ as $\xi\to\infty$, with $a,p>0$, $q\ge0$. Let $\lambda^*(\varepsilon)$ be the largest solution of $A(\lambda)=\varepsilon$ and $L:=\ln(1/\varepsilon)$. Then as $\varepsilon\to0$,

$$\lambda^*(\varepsilon)=2\pi\Big(\frac{a}{L}\Big)^{1/p}\Big(1-\frac{(q/p)\ln(L/a)+\ln c}{p\,L}+O\big(L^{-2}\ln^2L\big)\Big).$$

*Proof.* With $\xi^*=2\pi/\lambda^*$ the equation reads $L=a\xi^{*p}-q\ln\xi^*-\ln c+o(1)$. Leading order $\xi^*=(L/a)^{1/p}$; substituting back, $a\xi^{*p}=L+\frac qp\ln(L/a)+\ln c+o(1)$, and $\lambda^*=2\pi\xi^{*-1}$ gives the expansion. $\square$

Consequences. (i) For pole-type kernels ($p=1$) $\lambda^*\approx 2\pi a/\ln(1/\varepsilon)$: linear in the distance $a$ of the nearest complex singularity of $K$ from the real axis. (ii) For entire kernels with $e^{-a\xi^2}$ tails, $\lambda^*\approx 2\pi\sqrt{a/\ln(1/\varepsilon)}$: only $\sqrt{\ln(1/\varepsilon)}$ grid points per kernel width are needed instead of $\ln(1/\varepsilon)$. (iii) The viable interval shrinks like $1/\ln(1/\varepsilon)$, which is the paper's Appendix C.1 statement. (iv) For $\mathrm{sech}^2$, $a=\pi/2$: the leading term is $\pi^2/L$; at $\varepsilon=5.65\times10^{-16}$ ($L=35.1$) this is $0.281$ and the first correction brings it to $0.247$, against the exact $0.250$. (v) In fp32 ($\varepsilon=2^{-24}$) the exact solution for tanh is $\lambda^*=0.485$. Untested.

---

## 8. The six kernels

Transforms are normalized by $\widehat K(0)$; each was checked against 40-digit quadrature in expC07 and re-derived here.

- **tanh, $\mathrm{sech}^2$** ($r=1,0$). $\widehat{\mathrm{sech}^2}(\xi)=\pi\xi/\sinh(\pi\xi/2)$. Derivation: integrate $\mathrm{sech}^2(z)e^{-i\xi z}$ around the rectangle $\mathbb R\to\mathbb R+i\pi$; $\mathrm{sech}^2$ is $i\pi$-periodic, so the two horizontal sides give $I(1-e^{\pi\xi})$; the only pole inside is the double pole at $z_0=i\pi/2$, where $\cosh(z_0+t)=i\sinh t$ gives $\mathrm{sech}^2(z_0+t)=-t^{-2}+\tfrac13+O(t^2)$ and residue $i\xi e^{\pi\xi/2}$; hence $I(1-e^{\pi\xi})=-2\pi\xi e^{\pi\xi/2}$ and $I=\pi\xi/\sinh(\pi\xi/2)$. Normalized: $(\pi\xi/2)/\sinh(\pi\xi/2)$, tail $\pi\xi e^{-\pi\xi/2}$ (pole distance $a=\pi/2$, $q=1$, $c=\pi$). Then

$$A_{\tanh}(\lambda)=\frac{\pi^2/\lambda}{\sinh(\pi^2/\lambda)}\approx\frac{2\pi^2}{\lambda}e^{-\pi^2/\lambda}.$$

- **sigmoid** ($r=1$). $\sigma'=\tfrac14\mathrm{sech}^2(x/2)$: normalized transform $\pi\xi/\sinh(\pi\xi)$, pole distance $\pi$. $A_\sigma(\lambda)=A_{\tanh}(\lambda/2)$.
- **Gaussian** ($r=0$). $K=e^{-x^2}$, $\widehat K=\sqrt\pi e^{-\xi^2/4}$, normalized $e^{-\xi^2/4}$; entire. $A(\lambda)=e^{-\pi^2/\lambda^2}$.
- **gelu** ($r=2$). $\mathrm{gelu}=x\Phi$, $\mathrm{gelu}''=2\varphi+x\varphi'=(2-x^2)\varphi$ with $\varphi$ the standard normal density; $\widehat\varphi=e^{-\xi^2/2}$, $\widehat{x^2\varphi}=-\partial_\xi^2e^{-\xi^2/2}=(1-\xi^2)e^{-\xi^2/2}$; so $\widehat K=(1+\xi^2)e^{-\xi^2/2}$, $\widehat K(0)=1$, mode at $\xi=1$ ($\mu=2e^{-1/2}=1.213$); entire.
- **swish** ($r=2$). $\mathrm{swish}=x\sigma$, $\mathrm{swish}''=2\sigma'+x\sigma''$; with $\widehat{\sigma'}=\pi\xi/\sinh\pi\xi$, $\widehat{\sigma''}=i\xi\widehat{\sigma'}$ and $\widehat{xg}=i\partial_\xi\hat g$: $\widehat K=2\pi\xi/\sinh\pi\xi-\partial_\xi(\pi\xi^2/\sinh\pi\xi)=\pi^2\xi^2\cosh(\pi\xi)/\sinh^2(\pi\xi)$, tail $2\pi^2\xi^2e^{-\pi\xi}$ (double pole at distance $\pi$, $q=2$), mode $\approx0.51$ ($\mu=1.169$).

| activation | $r$ | normalized $\widehat K$ | tail class $(a,p,q)$ | $\lambda^*$, simple rule | $B(\lambda^*)$ | measured argmin (expC07, hard cells) |
|---|---|---|---|---|---|---|
| tanh | 1 | $\frac{\pi\xi/2}{\sinh(\pi\xi/2)}$ | $(\pi/2,1,1)$ | 0.250 (anchor) | $1.1\times10^{-7}$ | 0.208 |
| $\mathrm{sech}^2$ | 0 | same | same | 0.250 | $1.1\times10^{-7}$ | 0.222 |
| sigmoid | 1 | $\frac{\pi\xi}{\sinh\pi\xi}$ | $(\pi,1,1)$ | 0.500 (exact) | $1.1\times10^{-7}$ | 0.437 |
| Gaussian | 0 | $e^{-\xi^2/4}$ | $(1/4,2,0)$ | 0.530 | $1.5\times10^{-4}$ | 0.494 |
| gelu | 2 | $(1+\xi^2)e^{-\xi^2/2}$ | $(1/2,2,2)$ | 0.707 | $1.1\times10^{-3}$ | 0.716 |
| swish | 2 | $\frac{\pi^2\xi^2\cosh\pi\xi}{\sinh^2\pi\xi}$ | $(\pi,1,2)$ | 0.455 | $3.6\times10^{-7}$ | 0.437 |

The simple-rule column solves $A(\lambda)=\varepsilon^*$ with $\varepsilon^*:=A_{\tanh}(0.25)=5.65\times10^{-16}$ (five times $2^{-53}$; using $2^{-53}$ itself gives $0.240$ for tanh, a four percent change, because $A$ is exponentially steep). The measured column is the median argmin over cells whose minimum sits above $10^{-14}$. Every measured value sits at or left of the prediction, as Theorem 7 says it should.

---

## 9. Numerical verification of the exact formulas

Script: `experiments/expC07_lambda_energy_rule/fiber_check.py`. Figure: `results/checkpoint_C_geometry/expC07_lambda_energy_rule/figures/fiber_theory_check.png`.

**Test A: the projection floor, three settings, $f=\sin\pi x$, $N=128$, $\theta_0=\pi h=0.049$.** Least-squares fits with a bias column and truncated SVD at $10^{-13}$, evaluated as relative $L^2$ on 8001 misaligned points. Prediction: $\Lambda_r(\theta_0;\lambda)^{1/2}$ from Theorem 2 / Corollary 2.1.

| $\lambda$ | periodic $\mathrm{sech}^2$ net fitting $f'$ ($r=0$): measured / predicted | periodic tanh net fitting $f$ ($r=1$) | interval + halo tanh net (repo setting) |
|---|---|---|---|
| 0.33 | $9.20\times10^{-12}$ / $9.20\times10^{-12}$ (1.000) | $7.20\times10^{-14}$ / $7.21\times10^{-14}$ (0.998) | $7.48\times10^{-14}$ (1.037) |
| 0.40 | $1.397\times10^{-9}$ / $1.397\times10^{-9}$ (1.000) | $1.090\times10^{-11}$ / $1.095\times10^{-11}$ (0.996) | $1.073\times10^{-11}$ (0.980) |
| 0.50 | $1.532\times10^{-7}$ / $1.532\times10^{-7}$ (1.000) | $1.194\times10^{-9}$ / $1.199\times10^{-9}$ (0.996) | $1.176\times10^{-9}$ (0.980) |
| 0.60 | $3.398\times10^{-6}$ / $3.398\times10^{-6}$ (1.000) | $2.649\times10^{-8}$ / $2.660\times10^{-8}$ (0.996) | $2.606\times10^{-8}$ (0.980) |
| 0.70 | $3.040\times10^{-5}$ / $3.040\times10^{-5}$ (1.000) | $2.368\times10^{-7}$ / $2.379\times10^{-7}$ (0.996) | $2.335\times10^{-7}$ (0.982) |

At $\lambda=0.30$ the tanh predictions ($4\times10^{-15}$) are below the fp64 floor and the measurements read the floor instead ($6.5\times10^{-15}$ periodic, $1.0\times10^{-13}$ interval). Everywhere above the floor the identity holds to $0.4\%$ on the torus and $2\%$ on the interval. The tanh column is $130\times$ below the bump column, which is the predicted ratio $\Lambda_1^{1/2}/\Lambda_0^{1/2}\approx\theta_0/2\pi=0.0078$, confirming the integration attenuation.

**Test B: the stored expC03 sweep** (interval + halo, truncated-SVD readout, sine $=\sin2\pi x$, sine\_8pi $=\sin8\pi x$). Prediction $\Lambda_1(\theta_0;\lambda)^{1/2}$ with $\theta_0=\omega_0h$; ratio measured/predicted:

| target | $N$ | $\theta_0$ | $\lambda=0.40$ | $\lambda=0.50$ |
|---|---|---|---|---|
| sine | 64 | 0.196 | 0.97 | 0.97 |
| sine | 128 | 0.098 | 0.99 | 0.98 |
| sine | 256 | 0.049 | 0.99 | 0.99 |
| sine | 1024 | 0.012 | 1.00 | 1.00 |
| sine\_8pi | 64 | 0.785 | 0.97 | 0.98 |
| sine\_8pi | 128 | 0.393 | 0.99 | 0.99 |
| sine\_8pi | 256 | 0.196 | 0.99 | 0.99 |
| sine\_8pi | 1024 | 0.049 | 1.00 | 1.00 |

Sixteen cells spanning a factor 64 in $\theta_0$ and $10^4$ in error, all within 3%. Note that sine at $N$ and sine\_8pi at $4N$ have the same $\theta_0$ and the same measured error, as the theory requires: the wall depends on the target and the width only through $\theta_0=\omega_0h$. The measured wall slopes in the stored sweep are $-3.9$ to $-4.1$ decades per unit of $1/\lambda$ across all targets and widths, against $-\pi^2/\ln10=-4.29$ for the bare exponential; the shortfall is the $2\pi^2/\lambda$ prefactor.

**Test C: the cardinal interpolant (Theorem 3).** The repo's fp64 QI construction on $\sin2\pi x$, $N=128$, halo sized per $\lambda$, evaluated as relative $L^2$; prediction $|1-M(\theta_0)|$:

| $\lambda$ | QI measured | $\lvert 1-M(\theta_0)\rvert$ | lstsq floor $\Lambda_1^{1/2}$ on the same geometry |
|---|---|---|---|
| 0.35 | $7.08\times10^{-11}$ | $7.21\times10^{-11}$ | $8.7\times10^{-13}$ |
| 0.40 | $2.081\times10^{-9}$ | $2.081\times10^{-9}$ | $2.5\times10^{-11}$ |
| 0.45 | $2.814\times10^{-8}$ | $2.813\times10^{-8}$ | $3.3\times10^{-10}$ |
| 0.50 | $2.238\times10^{-7}$ | $2.238\times10^{-7}$ | $2.6\times10^{-9}$ |
| 0.60 | $4.916\times10^{-6}$ | $4.916\times10^{-6}$ | $5.6\times10^{-8}$ |

Four-digit agreement, and an $85\times$ gap between the interpolation and projection floors on identical geometry, as predicted in Section 4.

**Figure.** `fiber_theory_check.png`: relative $L^2$ error against $\lambda$ on $[0.28,0.72]$, log scale. Solid black is $\sqrt2A(\lambda)$, dashed black is $\Lambda_1(\theta_0;\lambda)^{1/2}$. Red dots (periodic bump net) lie on the solid line; blue squares (periodic tanh net) and open green triangles (interval + halo tanh net) lie on the dashed line. The single green point off the dashed line, at $\lambda=0.30$, is the fp64 floor.

---

## 10. The rule, final form

**Exact.** For a target with spectrum $\hat f$ (grid units) and a network of kernel order $r$, the largest admissible bandwidth at precision $\varepsilon$ is the largest $\lambda$ with

$$\frac{1}{2\pi\|f\|^2}\int|\hat f(\theta)|^2\,\Lambda_r(\theta;\lambda)\,d\theta\ \le\ \varepsilon^2 .$$

**Leading order** for a resolved target at grid frequency $\theta_0=\omega_0h$ (Corollary 2.2):

$$\sqrt2\,A(\lambda)\Big(\frac{\theta_0}{2\pi}\Big)^{r}=\varepsilon .$$

**Simple rule** (the $r=0$ case, and a conservative choice for every $r$ since the attenuation factor is $\le1$):

$$A(\lambda^*)=\frac{|\widehat K(2\pi/\lambda^*)|}{|\widehat K(0)|}=\varepsilon .$$

For tanh in fp64 the simple rule gives $0.25$; the leading-order rule for $\sin2\pi x$ at $N=128$ gives $0.28$ and drifts slowly upward with $N$ (a factor $2\pi/\theta_0\approx 64$ in the anchor moves $\lambda^*$ by about ten percent). The paper's own interpolating construction, whose floor is $2A(\lambda)$ with no attenuation, is governed by the simple rule directly, which is why $0.25$ was found there first. The measured argmins sit left of all three because the floor is flat (Theorem 7).

**Recipe for a new activation.**

1. Find the kernel order $r$ and $K=\psi^{(r)}$ (Lemma 8(d): unique).
2. Compute $\widehat K$; only $\xi\in[8,40]$ matters.
3. Solve $A(\lambda)=\varepsilon$ by bisection on $\log A$, with $\varepsilon=\varepsilon^*=5.65\times10^{-16}$ (calibrated) or $2^{-53}$; or use the exact form with the target's $\theta_0$ if a less conservative value is wanted.
4. Set every inner weight to $\gamma=\lambda^*/h$ with $h$ the center spacing; on $[-1,1]$ with $N$ intervals, $\gamma=\lambda^*N/2$.

Shortcuts (Proposition 9): pole-type kernels with nearest singularity at distance $a$: $\lambda^*\approx2\pi a/\ln(1/\varepsilon)$; Gaussian-type $e^{-x^2/(2s^2)}$: $\lambda^*=\pi s\sqrt{2/\ln(1/\varepsilon)}$, exact for the Gaussian ($0.530$). Any stretch of an activation by $s$ multiplies $\lambda^*$ by $s$ exactly (Lemma 8(b)).

---

## 11. What is proven, what is modeled, what is open

**Proven** (Sections 1 to 7): the fiber projection identity; the exact aliasing floor $\Lambda_r$ for band-limited targets, valid against every coefficient vector and every polynomial part; its two-ghost and low-frequency forms including the $(\theta_0/2\pi)^r$ attenuation; the cardinal interpolant's floor $E\ge\Lambda_0$ with $E^{1/2}\to\sqrt6A$; the Riesz bounds and $\kappa\asymp1/B$; the identification of the paper's $c_{\mathrm{alias}}(\epsilon)$ as the band-edge value; admissibility as an interval $(0,\lambda_{\max}]$; the exact covariances; the asymptotic solution.

**Modeled** (Theorem 7): that the optimum is at the right end of the admissible interval rests on the finite-precision error being non-increasing in $\lambda$ away from the wall. Theorem 4 and the paper's halo and stencil terms support this; the flatness of the floor down to $\lambda\approx0.1$ is measured, not derived.

**Open.**

1. *Localization.* The theorems are on $\mathbb R$ (or the torus). The interval-plus-halo problem agrees with them to 2% numerically (Test A, Test B), but a rigorous $L^2([-1,1])$ statement needs a localization argument; the paper's halo term is the natural ingredient.
2. *Sup norm.* The paper's Theorem 1 is in $L^\infty$. The exact identities here are $L^2$. An $L^\infty$ lower bound of the same order should follow from the ripple identity for the constant target, but the optimal-coefficient version is not written.
3. *The left wall.* A quantitative model of the truncated-SVD readout in floating point, giving $\Xi(\lambda)$ and hence where the floor ends on the left, would complete Theorem 7. The fiber picture makes this tractable: truncation at relative level $\tau$ keeps the fibers with $\sqrt{G_\lambda(\theta)}\ge\tau\sqrt{\sup G_\lambda}$, i.e. roughly $|\widehat K(\theta/\lambda)|\ge\tau|\widehat K(0)|$, and the target is fully retained while $|\widehat K(\theta_0/\lambda)|\ge\tau|\widehat K(0)|$.
4. *Higher dimensions.* On a lattice in $\mathbb R^d$ the same fiber theorem holds with the dual lattice replacing $2\pi\mathbb Z$, so $A$ becomes $|\widehat K|$ at the shortest dual-lattice vectors. Ridge geometries are a different object and the observed drift of the 2D optimum with $N$ is not covered.
5. *References to verify before citing:* de Boor, DeVore, Ron (Trans. AMS 1994) for Theorem 1 at $r=0$; Unser and Daubechies (IEEE TSP 1997) and Blu and Unser (IEEE TSP 1999) for the least-squares versus interpolation error kernels; Aldroubi and Unser (1994) for sampling in shift-invariant spaces.
6. *Record hygiene.* `docs/theory_lambda_rule.md` quotes $A(0.30)\approx5\times10^{-15}$ and $A(1.5)\approx1.4\times10^{-3}$, which are the bare exponential; with the prefactor they are $3.4\times10^{-13}$ and $1.8\times10^{-2}$. Its fp32 figure $0.59$ should be $0.485$. The expC07 writeup's phrasing that the rule "predicts the measured minima" should become "predicts the wall; the minima sit on the floor's right shoulder", per Theorem 7 and Test B.
