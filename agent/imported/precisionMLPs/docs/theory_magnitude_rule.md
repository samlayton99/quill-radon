# The magnitude prediction rule (readout norm law)

**Theory note for review. Companion: `theory_lambda_rule.md` (the same kernel Fourier tail governs both rules).**

**The rule in one line:** on the frozen QI geometry, the min-norm least-squares readout has inner-band norm given by a single spectral integral,

$$\|v\|_2^2 \;=\; \frac{h^{2r-1}}{2\pi\,\lambda^{2r-2}}\int_{|\omega|\le\pi/h}\frac{\big|\widehat{f^{(r)}}(\omega)\big|^2}{\big|\widehat K(\omega h/\lambda)\big|^2}\,d\omega \qquad \text{(master formula)},$$

whose two regimes are: a **law regime** $\|v\| = \frac{\lambda^{1-r}}{M}\|f^{(r)}\|_{L_2}\,h^{r-1/2}$ when the target is smooth enough, and a **Nyquist-deconvolution edge regime** with slope $-(q+r-1)$, $q = s{+}1{-}r$, when the target has a singularity in its $s$-th derivative.

## 1. Setup and assumptions

Same geometry and notation as the lambda note: $h = 2/N$, $u = x/h$, neurons $\psi(\lambda(u-k))$ on the unit lattice, kernel $K = \psi^{(r)}$ with mass $M = \widehat K(0) = \int K$. We solve $[\Phi, \mathbf 1]\,[v; b] \approx y$ on a dense equispaced sample by minimum-norm least squares, and measure $\|v_{inner}\|_2$ over centers in $[-1,1]$.

Idealizations used by the derivation: (i) infinite lattice — the halo makes the finite problem close to this, and boundary effects are absorbed by free extension; (ii) dense samples — sums over $x_i$ become integrals; (iii) exact arithmetic — Section 7 treats what fp64 changes. $\widehat f(\omega) = \int f e^{-i\omega x}dx$ as before.

## 2. The elementary derivations first (the intuition)

Each kernel order has a two-line classical argument for what a *smooth* target forces the coefficients to be. These are worth internalizing before the general machinery; the master formula will reproduce all three.

**Order 0 (bumps: sech$^2$, gaussian) — Riemann sum.** For slowly varying coefficients $v_k \approx v(c_k)$, the lattice sum is a Riemann sum of a convolution: $\sum_k v_k K(\lambda(u-k)) \approx v(u)\sum_k K(\lambda(u-k)) \approx v(u)\int K(\lambda t)\,dt = v(u)\,M/\lambda$. Matching to $f$: $v_k = \frac{\lambda}{M}f(c_k)$. Coefficients are $O(1)$ samples of $f$, so the vector norm over $\sim N$ of them **grows**: $\|v\|^2 \approx \frac1h\int v^2 = \frac{\lambda^2}{M^2 h}\|f\|^2$, slope $+\tfrac12$.

**Order 1 (steps: tanh, sigmoid) — telescoping.** Each neuron is a smoothed step with total rise $\Delta = \psi(\infty) - \psi(-\infty) = \int\psi' = M$. Sweeping $x$ across center $c_k$ increments the model by $v_k\Delta$; matching the target's increment over one cell, $v_k\Delta = f(c_{k+1}) - f(c_k) \approx h f'(c_k)$, so $v_k = \frac{h}{M}f'(c_k)$ and $\|v\| = \frac{\sqrt h}{M}\|f'\| = \frac1M\sqrt{\tfrac2N}\|f'\|$, slope $-\tfrac12$. For tanh ($M{=}2$) this is the original $\|f'\|/\sqrt{2N}$.

**Order 2 (ramps: gelu, swish) — curvature matching.** Asymptotically $\psi(z) \to s\cdot z$ ($s = \int\psi'' = M$), so each neuron is a smoothed ramp contributing slope $v_k s\gamma$ once activated. The model's slope must increment by $f'$'s increment per cell: $v_k s\gamma = h f''(c_k)$, so $v_k = \frac{h^2}{M\lambda}f''(c_k)$ and $\|v\| = \frac{1}{M\lambda}(2/N)^{3/2}\|f''\|$, slope $-\tfrac32$.

One pattern: $\|v\| = \frac{\lambda^{1-r}}{M}\|f^{(r)}\|\,h^{r-1/2}$. Every extra kernel order trades one factor of $h$ of coefficient size against one derivative of the target.

## 3. First principles: the solve diagonalizes over grid frequencies

Now the real derivation, which also covers rough targets. The Gram matrix of lattice translates, $G_{jk} = \int g(u-j)\,g(u-k)\,du$ with $g(u) = \psi(\lambda u)$, depends only on $j - k$: the normal equations are a discrete convolution (Toeplitz). Discrete convolutions are diagonalized by the discrete-time Fourier transform: writing $V(\theta) = \sum_k v_k e^{-ik\theta}$ for $\theta \in [-\pi, \pi]$, Poisson summation turns the normal equations into independent scalar equations per $\theta$:

$$V(\theta)\sum_{m\in\mathbb Z}\big|\hat g(\theta + 2\pi m)\big|^2 \;=\; \sum_{m\in\mathbb Z}\overline{\hat g(\theta + 2\pi m)}\,F(\theta + 2\pi m),$$

where $F$ is the target's spectrum in grid units. Interpretation: at grid frequency $\theta$ the model owns *one* coefficient degree of freedom, whose physical response is spread over all aliases $\theta + 2\pi m$; the least-squares fiber matches it to the target across those aliases. (Min-norm: where the denominator vanishes, $V = 0$.)

When the principal band dominates ($|\hat g(\theta)| \gg$ side bands, target spectrum mostly in-band), the fiber solution is a pure **deconvolution**:

$$V(\theta) \;=\; \frac{F(\theta)}{\hat g(\theta)}.$$

The neuron's transform in terms of the kernel: $K = \psi^{(r)}$ gives $\widehat\psi(\omega) = \widehat K(\omega)/(i\omega)^r$, and scaling by $\lambda$ gives

$$\hat g(\theta) = \frac1\lambda\widehat\psi(\theta/\lambda) = \lambda^{r-1}\,\frac{\widehat K(\theta/\lambda)}{(i\theta)^r}.$$

Parseval ($\|v\|^2 = \frac{1}{2\pi}\int_{-\pi}^{\pi}|V|^2 d\theta$), a change of variables $\theta = \omega h$, and $|\widehat{f^{(r)}}| = |\omega|^r|\widehat f|$ then give the master formula stated at the top. Note what it says physically: **the coefficient spectrum is the target spectrum divided by the kernel spectrum**, and the norm is the energy of that quotient over the resolvable band $|\omega| \le \pi/h$.

## 4. Regime 1: the law

If $f^{(r)}$ is smooth, its spectrum is concentrated at low $\omega$ where $\widehat K(\omega h/\lambda) \approx \widehat K(0) = M$ (the kernel argument $\omega h/\lambda$ is tiny for fixed $\omega$ as $h \to 0$). The integral collapses to $\|f^{(r)}\|_{L_2}^2$ and

$$\|v\| \;=\; \frac{\lambda^{1-r}}{M}\,\|f^{(r)}\|_{L_2}\;h^{\,r-1/2},$$

recovering all three elementary laws with one prefactor. Per activation, at the aliasing-rule bandwidths:

| activation | $r$ | $M$ | $\lambda$ | prefactor $\lambda^{1-r}/M$ | slope in $N$ |
|---|---|---|---|---|---|
| sech$^2$ | 0 | 2 | 0.25 | 0.125 | $+1/2$ |
| gaussian | 0 | $\sqrt\pi$ | 0.530 | 0.299 | $+1/2$ |
| tanh | 1 | 2 | 0.25 | 0.5 | $-1/2$ |
| sigmoid | 1 | 1 | 0.50 | 1 | $-1/2$ |
| gelu | 2 | 1 | 0.707 | 1.414 | $-3/2$ |
| swish | 2 | 1 | 0.455 | 2.196 | $-3/2$ |

## 5. Regime 2: singularities pay Nyquist prices

**Spectrum of a jump.** If $f^{(s)}$ jumps by $J$ at a point $x_0$ (and is otherwise smooth), integrating $\widehat f = \int f e^{-i\omega x}$ by parts $s{+}1$ times leaves a boundary term at the jump: $\widehat f(\omega) \sim J\,e^{-i\omega x_0}/(i\omega)^{s+1}$. Hence $|\widehat{f^{(r)}}(\omega)| \sim J\,\omega^{-q}$ with

$$q = s + 1 - r \qquad \text{("target roughness relative to kernel order").}$$

Several separated jumps add in quadrature: $J^2 = \sum_i J_i^2$.

**Endpoint evaluation.** The deconvolution factor $1/|\widehat K(\omega h/\lambda)|$ grows exponentially toward the resolvable-band edge $\omega = \pi/h$, so for any polynomially-decaying target tail the integral is dominated by a thin layer at the edge (Laplace's method at the endpoint). With $\mu = -\frac{d}{d\omega_K}\ln|\widehat K(\omega_K)|$ evaluated at $\omega_K = \pi/\lambda$ (the local spectral decay rate; $\mu \approx a$ for a pole-type kernel with pole distance $a$, $\mu = \pi/(2b^{-1}\lambda)\cdot 2b$ for Gaussian-type), the layer width is $\lambda/(\mu h)$ and

$$\|v\| \;\approx\; J\Big(\frac{h}{\pi}\Big)^{q}\,h^{\,r-1}\,\lambda^{(3-2r)/2}\,\frac{1}{\sqrt{2\pi\mu}}\cdot\frac{1}{|\widehat K(\pi/\lambda)|}, \qquad \text{slope } -(q + r - 1).$$

The huge constant is the **Nyquist amplification** $1/|\widehat K(\pi/\lambda)|$: the kernel spectrum evaluated at *half* the first grid harmonic. This ties the two rules together: the aliasing rule sets $\lambda$ so that $\widehat K$ at $2\pi/\lambda$ is $\varepsilon^\*$; the magnitude edge term is then governed by $\widehat K$ at $\pi/\lambda$, which is roughly $\sqrt{\varepsilon^\*}$ for pole-type kernels and $(\varepsilon^\*)^{1/4}$ for Gaussian-type:

| activation | $|\widehat K(\pi/\lambda)|/M$ | Nyquist amplification |
|---|---|---|
| tanh / sech$^2$ | $1.06\times10^{-7}$ | $\approx 5\times10^{6}$ |
| sigmoid / swish | same order (pole at $\pi$, $\lambda$ doubled) | $\approx 10^{6}$--$10^{7}$ |
| gaussian | $1.5\times10^{-4}$ | $\approx 7\times10^{3}$ |
| gelu | $1.1\times10^{-3}$ | $\approx 10^{3}$ |

So a kink is *cheap in error but expensive in coefficients*, and far more expensive for pole-type kernels than for entire ones — the same analyticity class that sets $\lambda^\*$ sets the price of roughness.

**Predicted off-law slopes** ($r=1$): target $|x|$ has $s{=}1$, $q{=}1 \Rightarrow$ slope $-1$; target $x|x|$ has $s{=}2$, $q{=}2 \Rightarrow$ slope $-2$; target $|x|^3$ has $s{=}3$, $q{=}3$, but there the edge term is buried under the law (next section).

## 6. The crossover: when does the law lock?

The observed elbow in every norm-vs-$N$ curve is the width $N^\*$ where the (steeper) edge term falls below the law term. Setting them equal for tanh ($J$ from the jump strengths $2, 4, 12$ for $|x|, x|x|, |x|^3$):

| target | $q$ | predicted $N^\*$ | observed |
|---|---|---|---|
| $|x|^3$ | 3 | $\approx 310$ | lock at $\approx 300$ |
| $x|x|$ | 2 | $\approx 10^4$ | not locked by 1024, converging (39x above, slope $-2$) |
| $|x|$ | 1 | $\approx 10^{12}$ | never locks; slope $-1$ throughout |

Below $N^\*$ the norm is *still perfectly lawful* — it rides the edge power law — it just is not the $\|f^{(r)}\|$ law. "Losing prediction power" means the lock width runs away as the target gets rougher relative to the kernel order.

## 7. What finite precision adds (and the solver artifact)

The kernel spectrum spans from $M$ down to $|\widehat K(\pi/\lambda)|$ across the band — that ratio is the **physical spectral floor** of the feature matrix's singular values ($\sigma/\sigma_{max} \gtrsim 10^{-7}$ for sech$^2$, $10^{-4}$ for gaussian). Below the floor the true operator has (numerically) no content: those directions in an fp64 SVD are noise. Two consequences:

- The master formula's amplification is capped at the floor; nothing in exact arithmetic exceeds the Section 5 values.
- A solver whose truncation threshold falls *below* the floor can populate noise modes with enormous spurious coefficients that change the error not at all. This is exactly what LAPACK `gelsd` at default rcond did in the sweep: e.g. sech$^2$ on $|x|$ at $N{=}1024$ returned $\|v\| = 3\times10^8$, while a clean truncated SVD returns $\|v\| = 96$ at the *identical* eval error ($1.34\times10^{-5}$). The theory values are the truncated-SVD ones; gelsd norms above them are artifacts, worst for sech$^2$ because its exponentially graded spectrum leaves a dense cloud of borderline modes at the threshold.

## 8. Numerical evidence to date

Sources: expA07 main sweep (`experiments/expA07_inner_norm_rule/run.py`; tanh + gelu, 6 smooth-family targets, $N$ up to 1024) and the smoothness stress test (`experiments/expA07_inner_norm_rule/smoothness/run.py`; all 6 activations at their aliasing-rule $\lambda$, targets $|x|, x|x|, |x|^3$ and the $\sin(2\pi x)$-based set, $N = 32..1024$). Figures incl. the compensated ratio view under `results/checkpoint_A_numerics/expA07_inner_norm_rule/`.

**Law regime, verified:**
- tanh on all six smooth targets: tail slopes $-0.50$ to $-0.66$, prefactor ratios 1.000–1.030 at $N{=}1024$ (zero fitted parameters).
- Spot checks at $N{=}1024$ on $|x|^3$: tanh law $\|3x|x|\|/\sqrt{2048} = 0.0419$ vs measured $0.0421$; gelu law $(2/N)^{3/2}\sqrt{24}/0.707 = 5.98\times10^{-4}$ vs measured $5.96\times10^{-4}$.
- gelu at its aliasing-rule $\lambda = 0.707$ locks on $|x|^3$ (ratio 0.996) where the original $\lambda{=}0.25$ run broke by $10^4$ — the two rules compose.
- gaussian locks on $|x|^3$ from $N \approx 300$ (ratio scatter 0.75–1.3); sigmoid = tanh exactly (affine identity, visible as identical ratio columns).

**Edge regime, verified at $r{=}1$:**
- Slopes: $|x|$ predicted $-1$, measured $-0.98$; $x|x|$ predicted $-2$, measured $-2.00$ (both activations).
- Magnitudes at $N{=}1024$ (zero-parameter, endpoint-Laplace): $|x|$: predicted $\approx 940$, measured $610$; $x|x|$: predicted $\approx 1.2$, measured $1.4$. (O(1) edge-layer corrections expected; the exponentially large factor $5\times10^6$ is confirmed.)
- Crossovers: table in Section 6 — the elbow location, the "converging but not locked" cell, and the "never locks" cell all predicted.

**Spectral mechanism, verified:**
- SVD decomposition at $N{=}512$: within the physical singular band the solution norm matches theory (sech$^2$/$|x|$: $\approx 190$); the observed gelsd excess lives entirely below the spectral floor.
- Truncated-SVD re-solve at $N{=}1024$: sech$^2$/$|x|$: 96 vs gelsd $3\times10^8$ at identical error; sech$^2$/$|x|^3$: 1.14 $\approx$ law (gaussian: 2.7 vs law 3.6). Resolves the "sech$^2$ upturn" anomaly as a solver artifact.

**Not yet verified / known gaps:**
- $r{=}0$ and $r{=}2$ edge cells are noise-contaminated under gelsd; the sweep should be re-run with the truncated-SVD solver and the full two-term prediction (law$^2$ + edge$^2$) overlaid per cell. The formula predicts, e.g., gelu on $|x|$ has slope $-(q+r-1) = -1$ (gelsd data: $-1.26$, suggestive) and sech$^2$ on $|x|$ has slope $-1$ with value $\approx 40$–$100$ at $N{=}1024$.
- Prefactor accuracy in the edge regime is only factor-2 (endpoint Laplace + band-edge O(1) effects); a sharper constant would need the exact fiber solution near $\theta = \pi$.
- $J$-scaling untested (double the kink strength $\Rightarrow$ edge norm doubles, law regime unchanged); multi-kink quadrature addition untested quantitatively; boundary kinks (set B at $x = \pm1$) uncounted.
- The per-coefficient *profile* claim ($v_k \approx \frac{h}{M}f'(c_k)$ pointwise away from singularities, excess localized at the kink over the edge-layer width $\sim \lambda/(\mu h)$ centers) has only been spot-checked once, at tanh/$|x|$/$N{=}512$.
