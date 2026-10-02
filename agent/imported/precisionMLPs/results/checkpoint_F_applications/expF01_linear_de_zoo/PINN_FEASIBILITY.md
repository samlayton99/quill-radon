# Can the QI construction power a PINN?

Analysis + proof-of-concept, 2026-07-13. POC script: `experiments/expF01_linear_de_zoo/pinn_poc.py` (throwaway, numpy-only; ready to formalize as expF01 on request).

## TL;DR

- **Yes, and it needs no training.** Freeze the QI geometry, and for a linear PDE the physics residual is *linear in the readout*, exactly like the interpolation readout. Collocation + min-norm lstsq is the entire method. POC: 1D Poisson $3\times10^{-15}$, Helmholtz $5\times10^{-14}$, variable coefficients $6.5\times10^{-16}$, nonlinear $u''=u^3+f$ at $1.4\times10^{-15}$ (Newton), 2D Poisson on the disk $1.3\times10^{-15}$ with the expE01 Radon ridges. All rel $L_2$, all fp64, no oracle: the geometry is target-independent. Confirmed at scale in expF01/expF02 (9 linear + 9 nonlinear problems).
- **This is a replacement for PINN training, not a fix for it.** Gradient-trained PINNs will stall for the same reason interpolation does (Adam can't even solve the linear readout, expD01). The trained-PINN version of the recipe is the expD02/expD05 port: scale-aware init + final *PDE-collocation* refit.
- **The solve itself is not new** (it is ELM / random-feature collocation), **so the claim must be made against a random-feature baseline.** expF01 does: the structured QI/Radon geometry beats random features by 19x--685x at identical width. The geometry, not the frozen-feature trick, is the contribution.
- **A small residual certifies nothing.** An ill-posed problem produced a residual, rank, and singular spectrum indistinguishable from the well-posed one while being four orders less accurate. Judge a solve by nested-width self-consistency instead.
- Two PDE-specific bonuses: the halo needs no data outside $\Omega$ (interpolation's strongest assumption disappears), and residual evaluations are noise-free, so expB01's statistical floor never enters.

## Why it transfers

The repo's central decomposition is geometry $\to$ frozen features $\to$ linear readout, with all difficulty in the geometry and the readout a convex solve. A PDE does not disturb this. Freeze the QI geometry (uniform centers $c_m$ at spacing $h=2/N$ extended by the halo $R=\max(70, 0.4N)$, one shared $\gamma=\lambda/h$, $\lambda=0.25$) and write the model as a dictionary times a readout: $u(p) = \sum_m a_m\,\phi_m(p)$, so at any batch of points $p_j$,

$$u_j = \phi(p_j)^\top a, \qquad u = \Phi a, \qquad \Phi_{jm} = \phi_m(p_j),$$

rows $j$ over datapoints, columns $m$ over activations. Every derivative of a tanh activation is a polynomial in the activation itself ($\psi'=1-t^2$, $\psi''=-2t(1-t^2)$ with $t=\tanh$), so applying a linear operator $L$ to the dictionary yields another explicit matrix $L\Phi$, and the entire PINN objective collapses to one least-squares problem in $a$. The next section constructs every matrix in that sentence.

Everything Checkpoint A taught about the interpolation readout carries verbatim: the system is underdetermined and ill-conditioned by design, truncated SVD/lstsq handles it, never form normal equations. Nonlinear PDEs break the linearity in $a$ but not the structure: each Newton step turns out to be a *variable-coefficient linear* PDE for the correction, solved by the identical machinery (worked in full below). No learning rate, no epochs.

In 2D each neuron is the expE01 ridge $\tanh(\gamma(w_m^\top p - t_m))$, and since $\|w_m\|=1$ the Laplacian of a ridge is just $\gamma^2\psi''$: the collocation matrix is as cheap as $\Phi$.

## The solve, matrix by matrix

Setting for this whole section: the operator is

$$L \;=\; \sum_i s_i\, D_i,$$

where each $D_i$ is a partial derivative in any variables ($\partial_x$, $\partial_{tt}$, $\partial_{xt}$, and $D_i=\mathrm{id}$ for a zeroth-order term), and the $s_i$ are coefficients: constants for now, functions $s_i(x,t)$ two subsections down. Points are $p_j$ (in 1D just $x_j$, in space-time $(x_j,t_j)$); which points depends on the block: interior collocation points for PDE rows, boundary points for BC rows, $t=0$ points for IC rows, measurement locations for data rows.

### The four-part objective is one least-squares problem

The full PINN loss is

$$\mathcal L \;=\; \underbrace{\sum_{j\in\text{pde}}\big(L[u](p_j)-f_j\big)^2}_{\text{physics}} \;+\; w_b^2\!\!\sum_{j\in\text{bc}}\big(u(p_j)-g_j\big)^2 \;+\; w_0^2\!\!\sum_{j\in\text{ic}}\big(u(p_j)-u_{0,j}\big)^2 \;+\; w_d^2\!\!\sum_{j\in\text{data}}\big(u(p_j)-y_j\big)^2,$$

already a sum of squares. A PINN makes it nonconvex by letting the first layer move. Freeze the first layer and every residual is affine in $a$:

- BC/IC/data rows are plain dictionary rows: $u(p_j) - y_j = \phi(p_j)^\top a - y_j$.
- PDE rows are operator-applied dictionary rows: $L[u](p_j) - f_j = (L\Phi)_j\, a - f_j$, where

$$(L\Phi)_{jm} \;:=\; L[\phi_m](p_j)$$

means: apply $L$ to the activation *as a function*, then evaluate at the point. $L$ passes through the sum $u=\sum_m a_m\phi_m$ by linearity, which is the one and only property used.

So the loss is exactly $\mathcal L(a) = \|Aa - y\|_2^2$ with the four blocks stacked:

$$A \;=\; \begin{bmatrix} (L\Phi)/s \\[2pt] w_b\,\Phi_{\text{bc}} \\[2pt] w_0\,\Phi_{\text{ic}} \\[2pt] w_d\,\Phi_{\text{data}} \end{bmatrix}, \qquad y \;=\; \begin{bmatrix} f/s \\[2pt] w_b\,g \\[2pt] w_0\,u_0 \\[2pt] w_d\,y_{\text{data}} \end{bmatrix},$$

with $s$ a scalar normalizing the PDE block (why: next subsection) and $w_b, w_0, w_d$ the block weights. These weights are the "loss weights" every PINN paper hand-tunes; here they are literally row scalings of a linear system, and $\sqrt{n_{\text{pde}}/n_{\text{block}}}$ was sufficient everywhere in the POC. A Neumann BC changes nothing structural: its rows are $(D\Phi)_{\text{bc}}$ instead of $\Phi_{\text{bc}}$, still affine in $a$. Include actual measurements in the data block only if you have them; if they are noisy, that block reintroduces the expB01 statistical floor and should be down-weighted accordingly.

### The ingredient matrices

Every block above is $\Phi$ or an elementwise sibling of $\Phi$. All of them come from one argument matrix, built at whichever points the block uses:

$$Z_{jm} = \gamma\,(w_m^\top p_j - t_m) \qquad (\text{1D: } w_m=1,\ t_m=c_m, \text{ so } Z_{jm} = \gamma(x_j - c_m)).$$

Then

$$\Phi = \tanh(Z), \qquad (D_i\Phi)_{jm} \;=\; \gamma^{\,o_i}\;\pi_i(w_m)\;\psi^{(o_i)}(Z_{jm}),$$

where $o_i$ is the order of $D_i$, $\psi^{(o_i)}$ is the $o_i$-th tanh derivative (a polynomial in $t=\tanh$: $\psi'=1-t^2$, $\psi''=-2t(1-t^2)$, $\psi'''=-2(1-t^2)(1-3t^2)$, ...), and $\pi_i(w_m)$ is the product of direction components the chain rule pulls out ($\partial_{xt}$ on a ridge gives $w_{m,x}w_{m,t}$; in 1D every $\pi_i=1$). For constant coefficients the PDE block is the same combination as the operator:

$$L\Phi \;=\; \sum_i s_i\,(D_i\Phi).$$

Every entry is a known number at matrix-build time; $a$ appears nowhere. You could equally build $L\Phi$ with torch autograd on the frozen features, and it would agree to roundoff, since autograd is exact chain-rule differentiation, not finite differences; the closed forms are just cheaper.

Three practical facts about this system:

1. **Scales: each derivative order costs a factor $\gamma$.** $D_i\Phi$ carries $\gamma^{o_i}$, so a second-order PDE block runs $\gamma^2\sim10^3$ hotter than the BC/data blocks. The scalar $s=\max_{jm}|(L\Phi)_{jm}|$ normalizes the block to $O(1)$; nothing else was needed.
2. **Append a low-frequency polynomial block.** Monomials up to degree 3, as ordinary extra columns. For a pure-derivative operator these are exactly $\ker L$ (invisible to the PDE rows, pinned only by BC/IC rows), but *that is a special case, not the general story*: for Helmholtz $u''+k^2u$ their PDE entries are $k^2\cdot\mathbf 1$ and $k^2x$, and for $-\Delta+4I$ constants are emphatically not annihilated. Call it what it is -- a low-order supplement that supplies smooth modes a single sharp bandwidth $\gamma$ represents inefficiently. It is not cosmetic: ablating it costs 1--3 orders on the steady-2D problems (expF01).
3. **The RHS must not be all zeros.** If the PDE is homogeneous ($Lu=0$) and you kept only PDE rows, min-norm lstsq returns $a=0$ exactly, the smallest solution of $Aa=0$. The inhomogeneous rows (BC/IC/data) are what determine the solution; the PDE block constrains but never determines.

Dimensions, concretely: at $N=128$ in 1D ($R=70$, so $W=269$ tanh columns $+2$ kernel columns, $n=4W$ collocation points, 2 BC rows) $A$ is $1078\times271$.

### Why solve instead of descend

$\|Aa-y\|^2$ is a quadratic with Hessian $A^\top A$, whose condition number is $\mathrm{cond}(A)^2$. That squared conditioning is exactly what stalled Adam at $10^{-3}$ on the *interpolation* readout in expD01, and the PDE block is worse ($L\Phi$ inherits $\Phi$'s near-dependent halo columns, scaled by $\gamma^{o}$). The SVD route never forms $A^\top A$: with $A=U\Sigma V^\top$,

$$a \;=\; V\,\Sigma^+ U^\top y, \qquad \Sigma^+:\ \text{invert } \sigma_i > 10^{-13}\sigma_{\max},\ \text{zero the rest},$$

which is `numpy.linalg.lstsq(rcond=1e-13)`, identical to `solve_readout`'s svd path. The truncation matters for the Checkpoint A reason: $A$ is numerically rank-deficient (the expA03 null space of $[\Phi,\mathbf 1]$ persists under differentiation), and among the many $a$ fitting the residuals to the floor, truncated SVD returns the min-norm one, the same canonical choice as the interpolation readout. One distinction worth keeping straight: lstsq drives the *residual* $\|Aa-y\|$ to the fp64 floor; the *solution* error $\|u_a - u^*\|$ is that residual passed through the PDE's stability constant. For the well-posed problems tested the two floors coincide, which is what the POC table shows.

### Variable coefficients: proof that $s_i(x,t)$ costs nothing

The worry: a coefficient that varies over the domain feels like it should entangle with the unknown and break the "one matrix, one solve" structure. It does not, and the reason is worth having exactly.

**The invariant: the operator must be linear in the unknown $u$; it is allowed to be arbitrarily non-constant in the coordinates.** Linearity of $L=\sum_i s_i(p)D_i$ means $L[u+v]=L[u]+L[v]$ and $L[\alpha u]=\alpha L[u]$, which holds whatever the functions $s_i$ are, because at every fixed point they multiply by a number.

**Proof.** Fix a collocation point $p_j$ and apply the operator to $u=\sum_m a_m\phi_m$:

$$L[u](p_j) \;=\; \sum_i s_i(p_j)\, D_i[u](p_j) \qquad \text{(definition of } L \text{ evaluated at a point; } s_i(p_j) \text{ is now a scalar)}$$

$$D_i[u] \;=\; D_i\Big[\sum_m a_m \phi_m\Big] \;=\; \sum_m a_m\, D_i[\phi_m] \qquad \text{(differentiation is linear; the coefficients are not involved)}$$

Substitute and swap the two finite sums:

$$L[u](p_j) \;=\; \sum_m a_m \underbrace{\Big(\sum_i s_i(p_j)\,(D_i\Phi)_{jm}\Big)}_{=\;(L\Phi)_{jm}}.$$

Every factor inside the brace, $s_i(p_j)$ and $\gamma^{o_i}\pi_i(w_m)\psi^{(o_i)}(Z_{jm})$, is a known number once the collocation points are chosen. The readout $a$ appears once, linearly, outside. $\blacksquare$

In matrix form, let $S_i = \mathrm{diag}\big(s_i(p_1),\dots,s_i(p_n)\big)$. Then

$$L\Phi \;=\; \sum_i S_i\,(D_i\Phi),$$

and constant coefficients are the special case $S_i = s_i I$. A diagonal matrix on the left rescales *rows*; it never mixes columns and never touches $a$. That is the whole mechanism.

**Why it feels like it shouldn't work:** "variable coefficient" suggests varying with the thing being solved for. It varies with the *coordinates*, which are frozen data the moment you pick collocation points; $s_i(p_j)$ has exactly the same status as $f(p_j)$, an entry of the known problem data. What genuinely breaks the structure is the unknown appearing inside a coefficient or nonlinearly: $u\,u_x$, $u^3$, $\sin u$. That is the nonlinear case, next.

**Receipt.** $u'' + \sin(\pi x)\,u' + x^2 u = f$, manufactured $u^*=\sin 2\pi x$. The PDE-block row is

$$(L\Phi)_{jm} = \gamma^2\psi''(Z_{jm}) + \sin(\pi x_j)\,\gamma\psi'(Z_{jm}) + x_j^2\tanh(Z_{jm}),$$

with kernel columns $L[1]=x_j^2$ and $L[x]=\sin(\pi x_j)+x_j^3$ (revived by the lower-order terms, still known numbers). Result: rel $L_2 = 6.5\times10^{-16}$ at $N=64$, *below* the constant-coefficient Poisson row. In the POC this case is `L_var`.

### The nonlinear case: Newton, one linear solve per step

Running example $u'' = u^3 + f$. Define the residual operator $F(u) = u'' - u^3 - f$; we want $F(u)=0$. The residual vector $r(a) = \Phi''a - (\Phi a)^{\odot 3} - f$ (elementwise cube) is no longer affine in $a$, so there is no one-shot solve. The multistep scheme is Newton's method applied to the PDE itself, and each step lands back in the variable-coefficient linear class just proved.

**Step derivation.** Let $u_k = \Phi a_k$ be the current iterate and seek a correction $\delta$ with $F(u_k + \delta) = 0$. Expand the only nonlinear term:

$$(u_k+\delta)^3 = u_k^3 + 3u_k^2\,\delta + 3u_k\,\delta^2 + \delta^3,$$

and keep terms through first order in $\delta$:

$$F(u_k+\delta) \;=\; \underbrace{F(u_k)}_{=\,r_k,\ \text{known}} \;+\; \underbrace{\delta'' - 3u_k^2(x)\,\delta}_{\text{linear in }\delta} \;+\; O(\delta^2).$$

Setting the linear part to cancel the residual gives the correction equation

$$\delta'' \;-\; 3u_k^2(x)\,\delta \;=\; -\,r_k, \qquad \delta = 0 \text{ on the boundary (BCs already satisfied after step 1)}.$$

This is a **linear PDE for $\delta$ with variable coefficient $-3u_k^2(x)$**. The coefficient is known: evaluate the current iterate at the collocation points, $u_k(p_j) = \phi(p_j)^\top a_k$, square, negate, triple. So the previous subsection applies verbatim, and one iteration is:

1. Evaluate $u_k = \Phi a_k$ and the residuals: $r_k = \Phi'' a_k - u_k^{\odot 3} - f$ on the PDE rows, $\Phi_{\text{bc}}a_k - g$ on the BC rows.
2. Assemble the linearized system: $J_k = \Phi'' - 3\,\mathrm{diag}(u_k^{\odot 2})\,\Phi$ on the PDE rows; the BC rows are $\Phi_{\text{bc}}$ unchanged (they were linear all along).
3. Min-norm lstsq: $\delta a = \arg\min \|J_k\,\delta a + r_k\|_2$ (same block scaling, same rcond).
4. Update $a_{k+1} = a_k + \delta a$; stop when the residual hits the fp64 floor.

**Consistency check (the Gauss-Newton view).** Differentiate the residual vector directly: $\partial r_j/\partial a_m = \Phi''_{jm} - 3u_k(p_j)^2\,\Phi_{jm}$, which is $J_k$ above, and Gauss-Newton on $\min_a\|r(a)\|^2$ takes exactly the step in item 3. Linearize-the-PDE-then-collocate and collocate-then-linearize are the same algorithm; the first derivation says *what* each step solves, the second says it is still a least-squares step.

**Why the error squares each iteration.** Newton analysis on $F$: at the solution $F(u^*)=0$, and Taylor expansion around $u_k$ with the second-order remainder gives

$$0 \;=\; F(u_k) + F'(u_k)[u^*-u_k] + \tfrac12 F''[u^*-u_k]^2,$$

where for the cubic $F''(u)[v]^2 = -6u\,v^2$ is bounded near $u^*$. The step solves $F(u_k) + F'(u_k)[\delta] = 0$, so subtracting the two equations:

$$F'(u_k)\big[u_{k+1}-u^*\big] \;=\; \tfrac12 F''\big[u_k-u^*\big]^2 \quad\Longrightarrow\quad \|u_{k+1}-u^*\| \;\le\; C\,\|u_k-u^*\|^2,$$

provided the linearized PDE is well-posed ($F'(u_k)$ invertible with the BCs) and the start is close enough. Digits double per step. In least-squares language: the true Hessian of $\tfrac12\|r\|^2$ is $J^\top J + \sum_j r_j\nabla^2 r_j$, and Gauss-Newton drops the second term, which is proportional to the residual itself; our problems are zero-residual ($r\to$ fp64 floor), so the dropped term vanishes exactly where accuracy matters and GN inherits Newton's quadratic rate.

**Receipt.** From the zero initialization $a_0=0$ at $N=128$, the POC's per-iteration $\|r\|_\infty$:

$$4.0\times10^{1} \;\to\; 1.1\times10^{0} \;\to\; 1.2\times10^{-3} \;\to\; 1.3\times10^{-9} \;\to\; 1.9\times10^{-11}\ (\text{floor}).$$

The doubling of correct digits is visible (roughly $0\to1.5\to3\to9$); four steps from a zero start to the floor, final solution error $1.4\times10^{-15}$.

**Generalization and caveats.** For any pointwise smooth nonlinearity $N(u)$ the extra Jacobian term is $-\,\mathrm{diag}\big(N'(u_k)\big)\Phi$. Nonlinearity in derivatives works the same way: Burgers' $u\,u_x$ contributes $\mathrm{diag}(u_k)\,\Phi' + \mathrm{diag}(\partial_x u_k)\,\Phi$, so every step remains a variable-coefficient linear solve. Newton is local: a hard problem started far from the solution needs damping ($a_{k+1}=a_k+\alpha\,\delta a$), Levenberg-Marquardt ($+\,\mu\|\delta a\|^2$ in step 3), or continuation in a problem parameter. The cubic here needed none of that.

## Proof of concept

Setup: $\lambda=0.25$ fixed (no sweep in 1D), collocation on $4W$ uniform points, Dirichlet BCs as two weighted rows, `numpy.linalg.lstsq`. Values are rel $L_2$ against the exact solution on a dense grid ($L_\infty$ tracks within $\sim10\times$ everywhere).

| problem | $N{=}64$ | $N{=}128$ | $N{=}256$ | $N{=}1024$ |
|---|---|---|---|---|
| interpolation control, $\sin 2\pi x$ | $9.1\times10^{-15}$ | $3.5\times10^{-14}$ | $3.8\times10^{-14}$ | $2.6\times10^{-14}$ |
| Poisson $-u''=f$, $u^*=\sin 2\pi x$ | $3.3\times10^{-15}$ | $2.0\times10^{-15}$ | $4.3\times10^{-15}$ | $5.1\times10^{-13}$ |
| Helmholtz $u''+100u=f$, oscillatory $u^*$ | $4.4\times10^{-14}$ | $5.5\times10^{-14}$ | $8.0\times10^{-14}$ | $2.2\times10^{-13}$ |
| variable coeff. $u''+\sin(\pi x)u'+x^2u=f$ | $6.5\times10^{-16}$ | $2.3\times10^{-15}$ | $3.8\times10^{-14}$ | $7.9\times10^{-13}$ |
| boundary layer $\varepsilon u''+u'=1$, $\varepsilon=0.02$ | $5.6\times10^{-10}$ | $1.4\times10^{-12}$ | $5.0\times10^{-14}$ | $3.1\times10^{-14}$ |

- The PDE solves sit at or *below* the interpolation floor. Nothing about differentiation degrades the recipe; the mild upward drift at $N=1024$ ($\gamma=128$) is the expected $\gamma^{o}$-amplified roundoff and stays $\sim10^{-13}$.
- The boundary layer is the resolution term of Theorem 1 in action: error falls exponentially in $N$ until the kernel width $1/\gamma = 4h$ resolves the layer, then hits the floor. Steep-but-analytic solutions are a width question, not a method question.
- **Nonlinear** ($u''=u^3+f$, $N=128$): Newton with per-step lstsq, zero init, residual $4.0\times10^{1}\to1.1\times10^{0}\to1.2\times10^{-3}\to1.3\times10^{-9}\to$ floor; final error $1.4\times10^{-15}$. No damping needed at this mildness.
- **2D Poisson on the unit disk** (manufactured $u^*=e^{-2\|x\|^2}\sin(2x_1+x_2)$, Radon tensor ridges, small $\lambda$ sweep as in expE01):

| width | best $\lambda$ | rel $L_2$ |
|---|---|---|
| 256 | 0.15 | $4.3\times10^{-10}$ |
| 1024 | 0.15 | $4.5\times10^{-14}$ |
| 2304 | 0.20 | $1.3\times10^{-15}$ |

That last row is *below* expE01's interpolation floor on the same geometry family, again showing the PDE constraint costs nothing.

## What is possible

- **Linear PDEs in 1D and 2D to machine precision, zero training.** Poisson, Helmholtz, advection-diffusion, variable coefficients; anything whose solution is analytic on $\Omega$. Shown above.
- **Nonlinear PDEs via Newton / Levenberg-Marquardt** on the readout alone; each step is the linear solve. Shown for a mild case; stiffer cases need damping or continuation but face no structural barrier.
- **Time-dependent PDEs two ways.** (a) Space-time: heat or advection on $(x,t)$ is just an anisotropic 2D linear problem, and the Radon geometry already reaches the floor in 2D. (b) Method of lines: $u(x,t)=\sum_m a_m(t)\phi_m(x)$ with a classical stiff integrator on the collocation-projected ODE; better for long horizons.
- **Systems and vector outputs.** Shared geometry, per-component readout, exactly the expD02 $1\to\mathbb{R}^m$ recipe.
- **General domains, mesh-free.** The geometry never sees the domain (ridges cover a bounding disk); the collocation points define $\Omega$. Irregular 2D domains are free.
- **A trained-PINN rescue path.** If a trained network is required (e.g. as a component of something larger): scale-aware init ($\gamma=\lambda^*/h$, uniform centers with halo), train however, then a final PDE-collocation lstsq refit of the readout. This is the expD02/expD05 result with $\Phi$ replaced by the stacked $[L\Phi;\ \Phi_{\text{bc}}]$; the refit stays a single linear solve for linear PDEs.

## What is not possible

- **Machine precision from gradient-trained PINNs.** Nothing here fixes end-to-end training; the entire repo shows first-order methods cannot even solve the convex readout, and a trained PINN will drive $\lambda\to0$ the same way. The claim is "don't train, solve", and any experiment should present it that way.
- **Dimension $d\ge3$ at scale.** Uniform coverage costs $O((1/h)^d)$ features, and the dense lstsq costs its cube. 3D is marginal (say $10^4$–$10^5$ features); beyond that the uniform-geometry approach is out. This is the same frontier already flagged in Checkpoint F, not a new wall.
- **Non-analytic solutions.** Shocks, corner singularities on non-smooth domains, and free boundaries break the exponential rate; Theorem 1's resolution term becomes the whole story and decays algebraically at best. Steep-but-analytic is fine (boundary-layer row); genuinely non-smooth is not.
- **Guarantees for strongly nonlinear / stiff problems.** Newton inherits only local convergence; a turbulent or bifurcating problem may need continuation in a parameter, and there is no linear-case-style certainty.
- **Eigenvalue and inverse problems, as-is.** The unknown parameter multiplies the readout (bilinear), so the one-shot solve breaks; alternating solves or extended Gauss-Newton would be needed. Untested.
- **A priori width selection.** $N$ must resolve the unknown solution's scales, and high-order operators have a *finite* optimal width ($\gamma^r$ roundoff drift), so "as wide as you can afford" is wrong.
- **Certifying a solve from its residual.** Do not. expF01 found an ill-posed problem whose PDE residual ($6.2\times10^{-11}$), rank, and singular spectrum were indistinguishable from the well-posed one while the solution was four orders worse: the solver faithfully solves whatever problem you hand it, and every internal signal says success. The working $u^*$-free guardrail is **nested-width self-consistency** ($\|u_{W_2}-u_{W_1}\|$), which tracked the true error in both the well-posed and ill-posed cases.

## Relation to existing work

The solve itself is physics-informed extreme learning machines / random-feature collocation (PIELM; the random feature method of Chen & E; Bacho et al. 2025 "Operator learning at machine precision" is the operator-side cousin, already cited in the workshop paper). Those methods also report near-machine precision on smooth problems. **So the frozen-feature trick is not the contribution, and any claim must be made against a random-feature baseline, not in the abstract.** expF01 runs that comparison: at identical width and $\lambda$, the structured Radon/QI geometry beats random features by 19x--685x on all six 2D problems. That is the defensible claim -- the *geometry* (uniform/structured placement at a shared viable $\lambda$, with a halo) is what buys the last 1--2.5 orders, which is Checkpoint C's finding surfacing in the PDE setting.

What this repo supplies that those methods take on faith: *why* a frozen feature geometry works, which geometries do (Checkpoint C's failure modes), and the exponential-in-width expressivity guarantee (Corollary 1). What it does **not** yet supply: a stability theory for the collocation solve. QI says the dictionary *contains* an accurate approximant; it does not prove that minimizing $\|L\hat u - f\|^2$ recovers it stably, which additionally needs the continuous problem to be well posed, the residual to control the solution error, and the discrete system to inherit that. Position a paper claim as "QI geometry theory extended to PDE collocation, validated against a random-feature baseline," not as "lstsq solves Poisson" and not as a general-purpose solver replacement.

## Next steps (Checkpoint F mapping)

- `src` machinery: closed-form tanh derivative features (tested against autograd to $\sim$1 ulp) and a collocation-system builder (operator rows + BC rows + block scaling), reusing `solve_readout`.
- **expF01** 1D linear zoo: Poisson / Helmholtz / advection-diffusion over manufactured solutions from the 6-category target family, widths $\{32,\dots,1024\}$, seeds over collocation sampling; verify exponential descent to $\le10^{-13}$ and the $\gamma^{o}$ floor drift.
- **expF02** nonlinear 1D (Bratu, steady Burgers) via Newton, including a damping/continuation stress test.
- **expF03** 2D Poisson/heat on the disk and a non-trivial domain, reusing `expE01/geometries.py`; includes space-time heat.
- **expF04** the trained-PINN contrast: Adam-PINN and Adam+refit vs the direct solve, one figure, to make the "solve, don't train" point quantitatively.

## Reproduce

`uv run --extra dev python experiments/expF01_linear_de_zoo/pinn_poc.py` (single file, numpy only, ~1 min). The core is ~15 lines: build centers + $\gamma$, form the $L\Phi$ rows and two BC rows, one `lstsq` call.
