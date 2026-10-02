# The lambda prediction rule (aliasing rule)

**Theory note for review. Companion: `theory_magnitude_rule.md` (the same Fourier tail governs both rules).**

**The rule in one line:** the optimal dimensionless bandwidth $\lambda^\*$ of an activation is the largest $\lambda$ at which its kernel's Fourier transform, evaluated at the first grid harmonic, has dropped below working precision:

$$A(\lambda) \;:=\; \frac{|\widehat K(2\pi/\lambda)|}{|\widehat K(0)|} \;=\; \varepsilon^\*, \qquad \varepsilon^\* = A_{\tanh}(0.25) = \frac{4\pi^2}{\sinh(4\pi^2)} = 5.65\times10^{-16} \approx \varepsilon_{\mathrm{fp64}}.$$

Everything below derives this from first principles and then evaluates it for the six activations.

## 1. Setup and notation

- Grid: $h = 2/N$, centers $c_k = -1 + kh$ (plus halo), inner weights $\gamma = \lambda/h$, so each neuron is $\psi(\gamma(x - c_k))$.
- Dimensionless (grid) coordinates: $u = x/h$, so a neuron is $\psi(\lambda(u - k))$ — the lattice has unit spacing and $\lambda$ is the only shape parameter. $1/\lambda$ is roughly the number of grid points per kernel width.
- Kernel: $K = \psi^{(r)}$, the lowest derivative of the activation that is an integrable bump. Orders: $r{=}0$ for $\mathrm{sech}^2$ and $e^{-x^2}$ (already bumps), $r{=}1$ for tanh and sigmoid ($\tanh' = \mathrm{sech}^2$), $r{=}2$ for gelu and swish (their second derivatives are bumps).
- Fourier convention: $\widehat K(\omega) = \int K(x)\,e^{-i\omega x}\,dx$, so $\widehat K(0) = \int K = M$ (the kernel mass).

The function space the model works in is spanned by lattice translates of the scaled kernel $K(\lambda u)$ (for $r \ge 1$ the derivative/difference structure converts activations into kernel translates — the paper's Section 3.2; the span is the same object).

## 2. Why aliasing is the irreducible error: the derivation

A quasi-interpolant reconstructs $f$ from grid samples with a cardinal function $L$ built from kernel translates: $(Qf)(x) = \sum_k f(c_k)\,L(x - c_k)$. Test it on a pure frequency $f(x) = e^{i\omega x}$. Since the samples $e^{i\omega c_k}$ are a geometric sequence on the lattice, Poisson summation gives exactly

$$\sum_k e^{i\omega c_k} L(x - c_k) \;=\; \frac1h\sum_{m\in\mathbb Z} \widehat L\!\Big(\omega + \frac{2\pi m}{h}\Big)\, e^{i(\omega + 2\pi m/h)x}.$$

Read this term by term. The $m{=}0$ term is $\frac1h\widehat L(\omega)e^{i\omega x}$ — the reconstruction we want (the construction normalizes $\widehat L(\omega) \approx h$ at low frequency; that is the Fourier partition-of-unity, Eq. 3 of the paper). Every $m \ne 0$ term is a **ghost copy** of the signal at the shifted frequency $\omega + 2\pi m/h$: the lattice cannot distinguish frequencies that differ by the grid harmonic $2\pi/h$. This is aliasing, and no choice of coefficients can remove it — it is a property of *sampling on a lattice*, not of the solve.

The amplitude of the first ghost ($m = \pm1$, the largest) relative to the signal is $|\widehat L(2\pi/h)| / |\widehat L(0)|$. Since $L$ is built from kernel translates, $\widehat L(\omega) = \widehat C(\omega)\widehat K_\gamma(\omega)$ with $\widehat C$ periodic and order-1, so the ratio is controlled by the kernel factor. The scaled kernel in physical coordinates is $K(\gamma x)$, whose transform is $\frac1\gamma\widehat K(\omega/\gamma)$; evaluating at $\omega = 2\pi/h$ and using $\gamma = \lambda/h$:

$$\frac{\omega}{\gamma}\Big|_{\omega = 2\pi/h} = \frac{2\pi/h}{\lambda/h} = \frac{2\pi}{\lambda} \qquad\Longrightarrow\qquad \text{ghost amplitude} \;\approx\; A(\lambda) = \frac{|\widehat K(2\pi/\lambda)|}{|\widehat K(0)|}.$$

**This is the whole mechanism.** The kernel's FT evaluated at $2\pi/\lambda$ — the first grid harmonic expressed in kernel units — is the fraction of the signal that leaks into an unremovable ghost. Higher harmonics ($m = \pm2, \dots$) sit at $4\pi/\lambda, \dots$ and are exponentially smaller. It matches the paper: for $K = \mathrm{sech}^2$, $A(\lambda) \approx e^{-\pi^2/\lambda}$, which is precisely the "intrinsic aliasing" term of Theorem 1 and `practical_implementation.tex`.

## 3. Why there is an optimum, and the anchor

$A(\lambda)$ is monotone decreasing in $1/\lambda$: wider kernels (small $\lambda$) alias less. But wide kernels make the basis ill-conditioned — the construction's prefactors and coefficient sizes diverge as $\lambda \to 0$ (the left wall of the U-curve; see the companion doc). So the optimum is the **largest $\lambda$ whose aliasing is invisible at working precision**:

$$\lambda^\* = \max\{\lambda : A(\lambda) \le \varepsilon_{\mathrm{work}}\}.$$

For tanh, $\widehat{\mathrm{sech}^2}(\omega) = \pi\omega/\sinh(\pi\omega/2)$ (derivation in Section 4), so $A(\lambda) = \frac{\pi\omega}{2\sinh(\pi\omega/2)}\big|_{\omega = 2\pi/\lambda}$ and

$$A_{\tanh}(0.25) = \frac{4\pi^2}{\sinh(4\pi^2)} = 5.65\times10^{-16},$$

which is fp64 machine epsilon. So the empirically known $\lambda^\* = 0.25$ for tanh is not an arbitrary constant: **it is exactly the bandwidth at which tanh's intrinsic aliasing hits the fp64 floor.** Two independent checks against known constants: $A_{\tanh}(0.30) = e^{-\pi^2/0.30} \approx 5\times10^{-15}$ — the documented error floor of the fp64 construction path at $\lambda = 0.30$; and $A_{\tanh}(1.5) \approx 1.4\times10^{-3}$ — the documented "$\lambda = 1.5$ does not work."

The rule for any other activation is then parameter-free: solve $A_K(\lambda) = \varepsilon^\*$ with $\varepsilon^\* = 5.65\times10^{-16}$.

## 4. The six kernels and their transforms, explicitly

**tanh and sech$^2$** share the kernel $K = \mathrm{sech}^2 x$ ($M = 2$). Its transform is the classical pair

$$\widehat{\mathrm{sech}^2}(\omega) = \frac{\pi\omega}{\sinh(\pi\omega/2)} \;\sim\; 2\pi\omega\,e^{-\pi\omega/2}.$$

(Route: $\mathrm{sech}\,x$ has simple poles at $x = i\pi(k+\tfrac12)$; contour integration gives $\widehat{\mathrm{sech}}(\omega) = \pi\,\mathrm{sech}(\pi\omega/2)$, and $\mathrm{sech}^2 = -(\tanh)'$ relates the two by an $i\omega$ factor.) The exponential rate $\pi/2$ is the distance from the real axis to the nearest pole. Solving $A(\lambda) = \varepsilon^\*$: **$\lambda^\* = 0.25$** for both (anchor for tanh; genuine prediction for the sech$^2$ activation, which spans a different space).

**sigmoid**: $K = \sigma' = \tfrac14\mathrm{sech}^2(x/2)$ ($M = 1$). Substituting $x = 2t$: $\widehat K(\omega) = \pi\omega/\sinh(\pi\omega)$ — the same shape with poles now at distance $\pi$, so the tail is $e^{-\pi\omega}$, twice the decay rate. Solving: $2\pi^2/\lambda = 4\pi^2 \Rightarrow$ **$\lambda^\* = 0.500$, exactly.** Cross-check from pure algebra: $\sigma(x) = \tfrac12(1 + \tanh(x/2))$, so with a constant column in the readout the sigmoid model at $\lambda$ spans exactly the tanh model at $\lambda/2$ — any correct rule *must* output $0.5$. The aliasing rule does, prefactors included.

**gaussian**: $K = e^{-x^2}$ ($M = \sqrt\pi$). Completing the square: $\widehat K(\omega) = \sqrt\pi\,e^{-\omega^2/4}$. This is an *entire* kernel — no poles, super-exponential tail. Solving $e^{-(2\pi/\lambda)^2/4} = \varepsilon^\*$: $(2\pi/\lambda)^2 = 4\ln(1/\varepsilon^\*) = 140.5 \Rightarrow$ **$\lambda^\* = 0.530$**.

**gelu**: $\mathrm{gelu}(x) = x\Phi(x)$, so $\mathrm{gelu}'' = 2\varphi(x) + x\varphi'(x) = (2 - x^2)\varphi(x)$ with $\varphi$ the standard normal pdf ($M = 1$, since $\mathrm{gelu}'(\infty) - \mathrm{gelu}'(-\infty) = 1$). Using $\widehat\varphi = e^{-\omega^2/2}$ and $x^2\varphi \leftrightarrow -\tfrac{d^2}{d\omega^2}e^{-\omega^2/2} = (1-\omega^2)e^{-\omega^2/2}$:

$$\widehat K(\omega) = (1 + \omega^2)\,e^{-\omega^2/2}.$$

Also entire. Solving $(1+\omega^2)e^{-\omega^2/2} = \varepsilon^\*$ gives $\omega = 8.887 \Rightarrow$ **$\lambda^\* = 2\pi/8.887 = 0.707 \approx 1/\sqrt2$**.

**swish**: $\mathrm{swish}(x) = x\sigma(x)$, $K = \mathrm{swish}'' = 2\sigma' + x\sigma'' = \tfrac14\mathrm{sech}^2(x/2)\big[2 - x\tanh(x/2)\big]$ ($M = 1$). Using $\widehat{\sigma'} = \pi\omega/\sinh(\pi\omega)$, $\widehat{\sigma''} = i\omega\,\widehat{\sigma'}$, and $\widehat{xg} = i\,\tfrac{d}{d\omega}\widehat g$:

$$\widehat K(\omega) = \frac{\pi^2\omega^2\cosh(\pi\omega)}{\sinh^2(\pi\omega)} \;\sim\; 2\pi^2\omega^2\,e^{-\pi\omega}.$$

Same pole distance as sigmoid ($\pi$, but a double pole $\Rightarrow$ the $\omega^2$ prefactor), so slightly below sigmoid: **$\lambda^\* = 0.455$**.

All five non-trivial transforms were verified against 40-digit mpmath quadrature (rel. error $\le 10^{-9}$, most $\le 10^{-30}$).

| activation | $r$ | kernel | tail type | $\lambda^\*$ predicted |
|---|---|---|---|---|
| sech$^2$ | 0 | $\mathrm{sech}^2x$ | pole at $\pi/2$ | 0.250 |
| gaussian | 0 | $e^{-x^2}$ | entire | 0.530 |
| tanh | 1 | $\mathrm{sech}^2x$ | pole at $\pi/2$ | 0.250 (anchor) |
| sigmoid | 1 | $\tfrac14\mathrm{sech}^2(x/2)$ | pole at $\pi$ | 0.500 (exact) |
| gelu | 2 | $(2-x^2)\varphi(x)$ | entire | 0.707 |
| swish | 2 | $2\sigma' + x\sigma''$ | double pole at $\pi$ | 0.455 |

## 5. Structural consequences

- **Derivative order barely matters.** $K = \psi^{(r)}$ means $\widehat K(\omega) = (i\omega)^r\widehat\psi(\omega)$: differentiation multiplies the transform by a *polynomial*, which shifts the solve only logarithmically. What sets $\lambda^\*$ is the kernel's **analyticity class**: for pole-type kernels (nearest complex singularity at distance $a$) the tail is $e^{-a\omega}$ and $\lambda^\* \approx 0.25\cdot a/(\pi/2)$, i.e. $\lambda^\*$ scales linearly with the pole distance; entire (Gaussian-type) kernels have $e^{-b\omega^2}$ tails, need only $\sqrt{\ln(1/\varepsilon)}$-many grid points per kernel width, and land at larger $\lambda^\*$. These are precisely the $p{=}1$ and $p{=}2$ cases of Theorem 1's aliasing exponent $e^{-c_2/\lambda^p}$.
- **Why energy matching fails.** Real-space energy $\int K^2$ is a low-order moment — two kernels can have identical widths and energies with Fourier tails differing by many orders of magnitude at $\omega \approx 12$–$36$ where the rule is decided (sech$^2$ vs $e^{-x^2}$: at $\omega = 12$, $e^{-\pi\omega/2} \approx 6\times10^{-9}$ vs $e^{-36} \approx 2\times10^{-16}$). Concretely: raw energy matching (with the $\lambda^r$ chain-rule amplitude) predicts $\lambda_{\mathrm{sigmoid}} = 2.0$, violating the exact affine constraint of $0.5$ by 4x; amplitude-normalized width matching passes sigmoid but predicts gelu at $0.23$ vs the measured $0.72$. Amplitude-sensitive invariants are ruled out a priori anyway, because the lstsq readout absorbs any amplitude rescaling.
- **Robustness caveat.** $A(\lambda)$ is nearly vertical on a log scale, so $\lambda^\*$ depends only logarithmically on the choice of $\varepsilon^\*$; the falsifiable content of the rule is the *ratios* between activations (and the absolute anchor coinciding with machine eps).

## 6. Numerical evidence to date

Source: expC07 (`experiments/expC07_lambda_energy_rule/run.py`), a sweep run *before* this rule was derived (out-of-sample): 6 activations x 4 targets ($\sin2\pi x$, $\sin8\pi x$, Runge, $e^x$) x $\lambda \in \mathrm{geomspace}(0.02, 1.5, 36)$ x $N \in \{64,128,256\}$, frozen uniform-grid geometry, SVD lstsq readout, eval rel $L_2$.

| activation | predicted | measured argmin (median over hard cells) |
|---|---|---|
| sech$^2$ | 0.250 | 0.222 |
| gaussian | 0.530 | 0.494 |
| tanh | 0.250 | 0.208 |
| sigmoid | 0.500 | 0.437 |
| gelu | 0.707 | 0.716 |
| swish | 0.455 | 0.437 |

- All six land within one $\lambda$-grid step; the gelu and swish cases are the discriminating ones (energy variants predict 0.23/0.33 there).
- Geometry of the match: the prediction marks the **right wall of the error basin** in every panel (`results/checkpoint_C_geometry/expC07_lambda_energy_rule/figures/`); measured argmins hug it from the left, exactly as tanh's own argmins scatter left of 0.25 — the basin floor extends toward small $\lambda$ until conditioning bites, and the rule pins where the floor ends.
- Internal exactness check: sigmoid at 0.500 is forced by the affine identity, and in the separate norm experiments (expA07/smoothness) tanh and sigmoid produce identical prediction ratios at their rule bandwidths — the equivalence is visible in independent data.
- Cross-checks against known constants: $\varepsilon^\* \approx$ fp64 eps at $\lambda = 0.25$; $5\times10^{-15}$ at $\lambda = 0.30$ (the fp64-path floor); $1.4\times10^{-3}$ at $\lambda = 1.5$ (known failure).

**Not yet tested** (candidate falsifiers): activations with designed pole distances (e.g. $\tanh(x/3)$-type stretches predicting exact ratios); an entire activation with different Gaussian width (erf: kernel $\propto e^{-x^2}$, predicts $\lambda^\* = 0.530$ identical to gaussian activation); lowering working precision (fp32: the rule predicts the whole optimum shifts to $\lambda^\*_{\tanh} = \pi^2/\ln(1/\varepsilon_{\mathrm{fp32}}) \approx 0.59$); overlaying the predicted full right-wall shape $C\cdot A(\lambda)$ against the measured error curve above the floor; the 2D ridge setting.
