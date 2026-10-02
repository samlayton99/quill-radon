# Why a sum of resolved 1-D ridges approximates a d-dimensional function: the operating theory

Status: the current theory of record for checkpoint H (written 2026-08-31 at Sam's request). Each claim is labeled theorem / exact-in-2D / measured / open. Sources: the Section 3 rewrite (1-D QI bound), the GPT direction-snapping construction (2026-08-29/30), expH01-expH06 measurements.

## Setup

Target $F$ analytic near the data ball $B_r(x_0)$; centered coordinates $z=x-x_0$. Dictionary: directions $V=\{v_1,\dots,v_M\}\subset S^{d-1}$ (antipodes identified) and, along each, a 1-D tanh QI block: offsets on the projected band $[-T,T]$, $T=1.25\,r$, spacing $h$, width $\gamma=\lambda^\star/h$. Readout: one truncated-SVD least squares. The question: why does $\sum_j q_j(v_j\cdot z)$, with each $q_j$ a resolved 1-D function, approximate a genuinely $d$-dimensional $F$ at all, and what sets the cost?

## Step 1 -- every function is an integral of ridges (theorem)

By Fourier inversion on the ball,

$$F(x_0+z)=\int_{\mathbb R^d}e^{i\xi\cdot z}\,d\mu(\xi),\qquad d\mu(\xi)=(2\pi)^{-d}e^{i\xi\cdot x_0}\widehat F(\xi)\,d\xi,$$

and each plane wave is a ridge: $e^{i\xi\cdot z}=e^{i\|\xi\|\,(\hat\xi\cdot z)}$ is a 1-D sinusoid of frequency $\|\xi\|$ composed with the projection onto $\hat\xi$. So $F$ is an integral over directions of 1-D profiles (the Fourier-polar form of the Radon/backprojection picture). This representation is the canonical gauge: raw Radon backprojection has a null space, the Fourier-polar decomposition does not.

## Step 2 -- direction snapping: the finite-$M$ certificate (theorem)

Given the finite set $V$, snap every frequency to its nearest available line, preserving the radial frequency: $S_V\xi=s(\xi)\,\|\xi\|\,v_{j(\xi)}$. Grouping frequencies by assigned direction gives an **exact finite ridge sum**

$$F_V(x_0+z)=\sum_{j=1}^{M}g_j(v_j\cdot z),\qquad g_j(t)=\int_{C_j}e^{i\,s(\xi)\|\xi\|\,t}\,d\mu(\xi),$$

whose error is pure phase mismatch on the ball: $|e^{i\xi\cdot z}-e^{i(S_V\xi)\cdot z}|\le\|z\|\,\|\xi-S_V\xi\|\le r\,\|\xi\|\,\theta(\hat\xi,V)$ with $\theta$ the projective angle to the nearest line. Hence

$$\boxed{\;\|F-F_V\|_{L^\infty(B_r)}\;\le\; r\int\|\xi\|\,\theta(\hat\xi,V)\,d|\mu|(\xi)\;\le\; r\,\Theta_V\int\|\xi\|\,d|\mu|(\xi),\;}$$

with $\Theta_V$ the covering angle of $V$ (for an even set, $\Theta_V\sim c_d\,M^{-1/(d-1)}$). Everything measured about directions follows from this one line:

- the direction cost scales with $k\cdot r$ ($k$ = the spectral scale $\int\|\xi\|d|\mu|/\|\mu\|$): the radius law (expH05, expH04);
- reaching angular accuracy $\varepsilon_{\rm ang}$ costs $M\sim(kr/\varepsilon_{\rm ang})^{d-1}$: the curse, and the measured 2.5 vs 1.4 orders per doubling in 3-D vs 4-D (resolution per direction is $M^{1/(d-1)}$);
- if $|\mu|$ is supported on $s$ lines (a true ridge sum), the bound is **zero** once those lines are in $V$: the atoms theorem, and why atom directions must be exact ($\delta$ of angle still costs $k\,r\,\delta$) -- the Gauss-Newton polish exists to drive $\delta\to$ fp64 zero;
- the optimal direction set for a given target is the weighted quantization $\min_V\int\|\xi\|\,\theta(\hat\xi,V)\,d|\mu|(\xi)$ -- the direction-space analogue of the 1-D optimal-center question (open).

The bound is first-order and ignores cancellation. In 2-D the cancellation is exact (Jacobi-Anger): for the mode $J_\ell(k\rho)e^{i\ell\phi}$ with $M$ even directions the residual is $\sum_{q\neq0}J_{\ell+qM}(kr)$, which is flat until $M-|\ell|\approx kr$ and then collapses super-exponentially -- the measured cliff (exact-in-2D; the $d\ge3$ spherical-harmonic analogue is open).

## Step 3 -- the 1-D leg (theorem, imported verbatim)

Each profile $g_j$ has spectrum equal to the radial spectrum of $\mu$ on its cone $C_j$, so its bandwidth is at most $F$'s. The 1-D QI theorem (Section 3 rewrite: resolution + aliasing + halo + stencil terms, all exponentially small at $\gamma h=\lambda^\star$) gives a block $q_j$ with relative error $\delta_N\le Ce^{-\alpha N}$. Writing $g_j=A_j\bar g_j$ with $A_j=|\mu|(C_j)$,

$$\Big\|F_V-\sum_j q_j(v_j\cdot z)\Big\|_\infty\;\le\;\sum_j A_j\,\delta_N\;=\;\|\mu\|\,\delta_N .$$

No factor of $M$: the offsets floor is **dimension-free and $M$-free** -- measured exactly so ($e_N$ flat in $M$, identical role in 2/3/4-D; the small $N^*$ seen in 4-D is an accuracy artifact of the direction wall, not a change in the 1-D requirement). Gradual (smoothly deformed) meshes inherit the same bound locally (measured exact in expH02; the deformation-stability theorem, with the 12-gap limit as its quantitative form, is open).

## Step 4 -- certificate to projection (the load-bearing principle)

Steps 2-3 exhibit **one** element of the span with error $\le e_M+e_N$. The final least squares is a (regularized) projection onto the span, so its error in the fitted norm is at most that of any member:

$$\text{measured error}\;\le\;\underbrace{r\!\int\!\|\xi\|\,\theta(\hat\xi,V)\,d|\mu|}_{e_M}\;+\;\underbrace{\|\mu\|\,C e^{-\alpha N}}_{e_N}\;+\;\underbrace{\text{numerical floor}}_{\text{fp64, rcond}}.$$

The construction is never used as weights -- only as an existence proof. This is why rank-deficient redundant dictionaries work, why the lstsq often lands far **below** the certificate (it re-optimizes all profiles jointly and exploits inter-direction cancellation: product-sines alignment, polynomial completeness), and why the numerical floor appears where it does (components kept above rcond$\cdot s_{\max}$; the wide-dictionary rcond lesson of expH06).

## Step 5 -- why the max, not the sum (lemma + measurement)

The angular truncation discards spectral energy at angular harmonics above the resolved degree; the longitudinal truncation discards radial frequencies above the resolved band. In an orthogonal angular$\times$radial decomposition the discarded sets overlap, giving

$$\max(e_M,e_N)\;\le\;e\;\le\;\sqrt2\,\max(e_M,e_N)$$

(rectangular-truncation lemma). Measurement sits at the lower end of the bracket -- $e\approx\max$ to within 10% on 46/48 cells in 3-D and all 4-D cells -- which is what makes the two floors a practical allocation instrument: measure $e_M(M)$ and $e_N(N)$ separately, take the crossing.

## Status ledger

- **Theorem**: Steps 1-4 as bounds; the 1-D four-term QI estimate; the $\sqrt2$ bracket.
- **Exact in 2-D**: the Jacobi-Anger aliasing series (the cliff mechanism).
- **Measured, no theorem yet**: tightness of the max; the 3-D/4-D cliff rates; exactness under gradual deformation; the 12-gap limit; the rcond/redundancy behavior; lstsq's margin below the certificate.
- **Open**: sharp $d\ge3$ angular aliasing; the weighted direction-quantization optimum (direction analogue of expH03's 1-D question); the data-weighted phase metric $d_P$ for non-uniform data; the deformation-stability theorem.
