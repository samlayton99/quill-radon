# expF12 -- 3D unsteady Navier-Stokes via a tensor-product sech$^2$ basis (QI geometry + Gauss-Newton readout)

**Status:** draft-pending-Sam (single seed).

## TL;DR

- **The moonshot works, and it works as a solve, not as training.** Frozen QI geometry + Gauss-Newton collocation lstsq for the readout solves 3D unsteady Navier-Stokes (Beltrami, $\nu=1$) to velocity rel $L_2$ **1.5e-6** in **2 Newton steps** (step 1 = the Stokes solve already at 1.2e-6), with $\max|\nabla\!\cdot\!\mathbf u|\sim$ 7.5e-6 at fresh points -- within $7\times$ of the best its truncated basis can represent at all (2.2e-7).
- The supervised expressivity ceiling of the basis is velocity rel $L_2$ **5.6e-9** (N=32 per axis, $\lambda=0.125$), reached by a mode-wise Kronecker-SVD fit in seconds; it saturates at $\sim\sqrt{\varepsilon}$ because the fp64 conditioning budget divides multiplicatively across the four axes.
- Gradient training on the same architecture (Adam$\to$LBFGS on the readout) stalls orders above the solve -- the Checkpoint-D "optimizers can't solve the ill-conditioned readout" story reproduces in 4D on a PDE.
- One structural discovery: "Kronecker of pinvs" is NOT the right least-squares solve -- per-factor rcond truncation admits directions whose *product* singular value is $\sim\sigma_{\min}^4$ and amplifies roundoff. The SVD of a Kronecker product factorizes, so global product-singular-value truncation can be done mode-wise; this is what makes the fp64 fit work at all.

## Question

Does the frozen-geometry + solved-readout recipe survive the jump to $d=4$ (3D space + time) on a genuinely nonlinear system -- incompressible Navier-Stokes -- using a tensor-product architecture whose feature count ($N^4$) can never be materialized?

## Experiment design

**Architecture (the moonshot spec).** Input $(x,y,z,t)\in[-1,1]^3\times[0,1]$. Layer 1: each coordinate $i$ gets its own group of $N$ units with no cross-talk, $\phi^{(i)}_k(s)=\mathrm{sech}^2\!\big(w_{i,k}(s-c_{i,k})\big)$, plus one constant unit per group ($\tilde N=N{+}1$), so lower-order tensor products are exactly in the span. Layer 2: the order-4 outer product $v_{ijkl}=\phi^{(x)}_i\phi^{(y)}_j\phi^{(z)}_k\phi^{(t)}_l$ ($\tilde N^4$ features, no parameters). Layer 3: linear readout $A\in\mathbb R^{4\times \tilde N^4}$ to $(u,v,w,p)$.

- **Contraction, never materialization.** Pair $(x,y)\mapsto q_{12}\in\mathbb R^{\tilde N^2}$ and $(z,t)\mapsto q_{34}$, reindex $A$ as $(4,\tilde N^2,\tilde N^2)$; then $f_o=q_{12}^{\top}A_o\,q_{34}$ -- one GEMM plus a weighted sum, per-sample intermediate $B\tilde N^2$ instead of $B\tilde N^4$. Same function, same gradients (associativity). Verified against the naive materialized tensor to 1e-13.
- **Analytic derivatives.** With $\tau=\tanh z$, $\phi=1-\tau^2$: $\phi'=-2w\tau\phi$, $\phi''=w^2(6\tau^2-2)\phi$; any mixed partial is the same contraction with the derivative row substituted in the relevant axis. Verified against autograd.
- **QI geometry, frozen.** Per axis: uniform centers with a 2-center halo, $h=(\text{hi}-\text{lo})/(N-1-2n_{halo})$, $w_{i,k}=\gamma_i=\lambda/h$. Nothing in the first layer is ever trained.

**Problem.** Unforced incompressible NS, $\partial_t\mathbf u+(\mathbf u\!\cdot\!\nabla)\mathbf u+\nabla p-\nu\Delta\mathbf u=0$, $\nabla\!\cdot\!\mathbf u=0$, $\nu=1$, on $[-1,1]^3\times[0,1]$. Exact verifier: the Ethier-Steinman Beltrami flow ($a=\pi/4$, $d=\pi/2$), the standard exact 3D NS benchmark; a test pushes the closed form through autograd and confirms the NS residual vanishes to machine eps. Data given to all methods: velocity IC at $t=0$, Dirichlet velocity on the 6 faces over time, and a pressure "tap" (exact $p$ on one corner line over $t$) as gauge. Metrics: velocity rel $L_2$ (all 3 components jointly) and gauge-adjusted pressure rel $L_2$ on 20k fresh random space-time points; interior NS momentum residual RMS and $\max|\nabla\cdot\mathbf u|$ at fresh points.

**Method 1 -- supervised ceiling (context, not a solver).** Fit $A$ to the exact solution on an $(2N)^4$ tensor grid. The design matrix is $\Phi_x\otimes\Phi_y\otimes\Phi_z\otimes\Phi_t$ ($\Phi_a$ only $2N\times\tilde N$); since the SVD of a Kronecker product is the Kronecker product of the SVDs, the *global* min-norm truncated-SVD solution is computed mode-wise: apply $U_a^{\top}$ along each mode, zero entries with product singular value $s_is_js_ks_l<\mathrm{rcond}\cdot\max$, divide, apply $V_a$ back. The dense $16N^4\times\tilde N^4$ matrix is never formed. This is the honest "how precise can this basis be at all" reference. Swept $N\in\{8..32\}$, $\lambda\in\{0.1..0.25\}$, rcond $\in\{10^{-13},10^{-15}\}$.

**Method 2 -- the headline: Gauss-Newton collocation solve of the readout (no training).** The Checkpoint-F recipe (expF02/expF06 Newton-lstsq, expF09 multi-field blocks) carried to this architecture. Newton-linearize the advection at the current iterate $\mathbf u^0$: solve the linear system in the next iterate,
$$u_t+(\mathbf u^0\!\cdot\!\nabla)u+u\,\partial_x u^0+v\,\partial_y u^0+w\,\partial_z u^0+p_x-\nu\Delta u=(\mathbf u^0\!\cdot\!\nabla)u^0$$
(and cyclic), plus continuity, IC, BC, and gauge row blocks, each block rescaled to $O(1)$ max entry, stacked into ONE min-norm lstsq per Newton step. $\theta=0$ start makes step 1 exactly the Stokes solve; steps are damped on the nonlinear interior residual.
- **The reduced product-SVD basis is what makes this possible.** The linearized operator has variable coefficients ($\mathbf u^0\!\cdot\!\nabla$), which breaks Kronecker separability -- no mode-wise solve, and a dense lstsq over all $4\tilde N^4$ coefficients is out of reach. But the ceiling study shows fp64 only ever uses the directions whose product singular value survives truncation ($\sim$5-15%). So the solve runs in the top-$K$ product directions per field ($\psi_a=\Phi_a V_a S_a^{-1}$ per axis, tuples ranked by $s_is_js_ks_l$): $K_{vel}=3000$ per velocity field, $K_p=2000$, i.e. 11000 unknowns, $\sim$37k rows, one dense gelsy lstsq per Newton step. Config (oracle-selected, per-knob attribution in *Additional details*): $N=12$, $\lambda=0.1$ (top-$K$ supervised ceiling at this config: 2.2e-7), 6000 interior / 2000 IC / 2500 BC points, 3 Newton iterations, seed 0.
- **expF03 techniques carried over:** *cascade init* (solve at $K/4$ first -- in this nested basis the top-$K/4$ tuples are a prefix of the top-$K$ list, so the coarse solution embeds exactly by zero-padding, then Newton-polishes); rcond $10^{-15}$; condition blocks (IC/BC/gauge) weighted $w_{\rm mult}\sqrt{n_{\rm pde}/n_{\rm blk}}$ after $O(1)$ row scaling, $w_{\rm mult}=0.3$; the no-oracle tuning signal (stacked fresh residual incl condition rows) logged every iteration.
- Assembly derivatives verified against finite differences.

**Method 3 -- gradient PINN on the same architecture.** Adam (1500 steps, lr 2e-4, resampled batches 2048) $\to$ LBFGS (300 iters, fixed 12288/3072/4608 sets) on the same loss blocks; $A$ warm-started by a Kronecker fit to the $t$-constant extension of the IC (uses only given data); geometry frozen. $N=16$, $\lambda=0.2$.

**Method 4 -- literature baseline.** Standard tanh-MLP PINN (4$\to$64$\times$4$\to$4 as in expF07, autograd derivatives), same losses, Adam 3000 $\to$ LBFGS 200, matched-order wall-clock.

**Code & data.** `experiments/expF12_tensor_ns3d/`: `tensor_basis.py` (architecture, contraction, Kronecker-SVD fit), `beltrami.py`, `newton_solve.py` (reduced basis + Gauss-Newton), `pinn.py` (both trainers), `run.py` (stages: ceiling / newton / pinn / baseline / plots). Tests: `tests/test_expF12_tensor_ns3d.py` (7 tests: contraction==naive, analytic==autograd, Kronecker==dense lstsq, Beltrami satisfies NS, supervised precision, FD assembly check, tiny Newton solve). Results dir: `results/checkpoint_F_applications/expF12_tensor_ns3d/` (`ceiling.json`, `*_history.jsonl`, `*_final.json`, figures below).

## Results

| method | velocity rel $L_2$ | pressure rel $L_2$ | mom. res. RMS | max $\vert\nabla\cdot\mathbf u\vert$ | wall |
|---|---|---|---|---|---|
| **QI geometry + Gauss-Newton lstsq** ($K_{vel}$=3000, $K_p$=2000, $\lambda$=0.1, $w_{\rm mult}$=0.3) | **1.49e-6** | 3.6e-3 | 8.7e-5 | 7.4e-6 | 2 iters, ~6-15 min/iter |
| same, smaller budget ($K_{vel}$=2000, $K_p$=1000) | 1.63e-5 | 1.3e-2 | 1.4e-3 | 1.3e-4 | 2 iters, ~2-5 min/iter |
| same, larger budget ($K_{vel}$=5000; $\lambda$=0.1 / 0.125) | 6.8e-6 / 7.0e-6 | 6.9e-3 / 9.2e-3 | 6e-5 / 8e-5 | 6e-6 / 7e-6 | 2 iters, ~15 min/iter |
| tensor sech$^2$ PINN, Adam$\to$LBFGS (same architecture) | 2.7e-2 | 5.2e-1 | 9.2e-2 | 2.8e-1 | ~10 min |
| tanh-MLP PINN baseline | 2.1e-2 | 4.2e-1 | 8.0e-2 | 2.1e-1 | ~52 min |
| *supervised ceiling, same truncated basis ($K$=3000)* | *2.2e-7* | *9.1e-6* | -- | -- | *seconds* |
| *supervised ceiling, full basis ($N$=32)* | *5.6e-9* | *2.0e-8* | -- | -- | *9 s* |

**Supervised ceiling (best over $\lambda$, rcond $10^{-15}$):** rel $L_2$ velocity 2.5e-6 ($N{=}8$) $\to$ 1.0e-7 (12) $\to$ 3.1e-8 (16) $\to$ 1.4e-8 (20) $\to$ 8.6e-9 (24) $\to$ 5.6e-9 (32); $\lambda^*=0.125$ throughout, solve time $\le$10 s. The saturation at $\sim\sqrt{\varepsilon}$ is the 4-way conditioning product: each axis effectively gets only $(\mathrm{rcond})^{1/4}$ of relative singular-value depth, so the retained per-axis basis is shallow; below rcond $10^{-15}$ the coefficients blow up and the fit disintegrates (rcond $\to0$: rel $L_2\sim10^{18}$).

- **Newton converges in 2 steps and then sits at a fixed point** (metrics identical to 11 digits from iter 2 on). Step 1 (Stokes: advection off) already nails velocity (1.2e-6) but its pressure is $O(1)$ wrong -- exactly as physics demands: Beltrami flows have $(\mathbf u\!\cdot\!\nabla)\mathbf u=\nabla(|\mathbf u|^2/2)$, so the missing advection is *entirely* a pressure correction. Step 2 restores it (pressure 0.90 $\to$ 3.6e-3, momentum residual RMS 0.58 $\to$ 8.8e-5).
- **The optimizer-vs-solve gap is 4+ orders on identical architecture and geometry.** Adam$\to$LBFGS on the same frozen-basis readout stalls at 2.7e-2; the linear-algebra solve of the same objective reaches 1.5e-6. This is Checkpoint D's central finding (first-order methods cannot solve the ill-conditioned readout; lstsq can) reproduced in 4D on a nonlinear PDE.
- **What limits each number.** Velocity sits $\sim$7$\times$ above its truncated-basis ceiling (1.5e-6 vs 2.2e-7 supervised; the $K{=}2000$ run tracks its own ceiling the same way: 1.6e-5 vs 1.7e-6), and pushing $K$ further is *non-monotone* for the solve (see attribution) -- $K{=}3000$ at $\lambda{=}0.1$ is the oracle optimum. Pressure lands further from its value-space ceiling (3.6e-3 vs 9.1e-6) but its *gradient* error is what the momentum rows constrain (RMS 8.7e-5); the smooth low-gradient pressure modes are only weakly pinned by the one-line gauge tap, and upweighting the gauge does not move it. Growing $K_p$ 1000$\to$2000 bought $3.7\times$ in pressure and $16\times$ in momentum residual.
- **No time-stepping, no error accumulation mechanism** (`error_vs_time.png`): time is a coordinate and the IC is a row block. Absolute error stays flat-to-mildly-rising over $t$; the relative curve's late rise is mostly the decaying denominator (figure bullet below).

**Figures**

- `benchmark_error_vs_wallclock.png` -- the benchmark plot: velocity rel $L_2$ (log) vs wall-clock (log). Green squares: Gauss-Newton solve (each marker one Newton step; step 1 = Stokes). Blue: tensor sech$^2$ PINN (Adam then LBFGS; dotted vertical = phase switch). Orange: tanh-MLP PINN. Dashed lines: supervised ceilings (same-width and best); dash-dot: typical literature PINN accuracy ($\sim$1e-3).
- `ceiling_convergence.png` -- supervised ceiling vs per-axis $N$, one line per $(\lambda,\mathrm{rcond})$; log y. Shows the $\lambda$ basin and the rcond $10^{-13}\to10^{-15}$ gain.
- `error_vs_time.png` -- velocity rel $L_2(t)$ on a $25^3$ spatial grid per time slice, one line per method. The Newton solve stays 3-5 orders below both PINNs at every $t$; its rise toward $t{=}1$ (3e-7 $\to$ 5e-5) is partly the *relative* normalization -- the Beltrami field decays as $e^{-2d^2\nu t}$ ($\sim$0.007 energy by $t{=}1$), shrinking the denominator.
- `ns3d_slice.gif` -- animation over $t$ of the $z{=}0$ slice: exact $|\mathbf u|$, headline-model $|\mathbf u|$, its $\omega_z$ vorticity, and $\log_{10}$ pointwise velocity error.

## Additional details

- **Per-knob attribution (oracle-selected, one change at a time from the winning cell unless noted).** *Init is fixed-point-irrelevant at Re $=$ 1:* $\theta=0$, an IC-extension warm start, and cascade all converge to the identical fixed point (matching to 10 significant digits) -- the init ladder is basin insurance for higher Re, exactly expF03's nonlinear setting, not this one. *$\lambda$:* 0.1 beats 0.125 in the solve (1.5e-6 vs 7.0e-6) even though 0.125 has the better supervised ceiling (5.9e-8 vs 1.8e-7 at $K$=5000) -- the scattered-point collocation pays for wide-kernel conditioning in a way the aligned tensor-grid fit does not. *$K$ is non-monotone for the solve:* 2000 $\to$ 3000 gains 11$\times$, but 3000 $\to$ 5000 *loses* 4.6$\times$ (6.8e-6) -- the near-floor SVD directions (largest $1/s$ scalings) help the grid fit but inject amplified fp64 noise at scattered points. *$w_{\rm mult}$:* 0.3 beats 1.0 by 1.3$\times$ (1.49e-6 vs 1.96e-6), the expF03 two-sided finding; pressure is insensitive to the gauge upweight (~3.6e-3 either way), so the pressure floor is not gauge-weighting-limited. *rcond $10^{-15}$ + cascade:* neutral here (within 1.3$\times$). *Tensor-grid interior collocation* (9$^4$) is strictly worse (v 3.1e-6, p 0.70) -- too few distinct per-axis values. The momentum residual RMS is pinned at $\sim$6-10e-5 across every config while ceiling models reach 1.5e-5: the solve is collocation-conditioning-limited, not basis-limited.
- **The Kronecker-pinv trap.** $(\Phi_1\otimes\cdots\otimes\Phi_4)^+=\Phi_1^+\otimes\cdots\otimes\Phi_4^+$ holds for exact pinvs, but per-factor rcond truncation keeps tuples where every factor passes the cut while the product is $\sim\sigma_{\min}^4\approx10^{-30}$ -- those directions amplify fp64 roundoff catastrophically (first implementation stalled at 3e-6 no matter the width). Global product truncation (exactly the dense-lstsq rcond semantics, computed mode-wise) removed the stall. The same effect is why the ceiling saturates near $\sqrt{\varepsilon}$ rather than reaching the 1D QI floor: precision in a product basis pays conditioning to the 4th power.
- **Cost anatomy.** Evaluation + derivatives are $O(B\cdot 4\tilde N^4)$ flops but only $O(B\tilde N^2)$ memory via the pairing; a naive einsum order would be $B\tilde N^3$ memory. The Gauss-Newton step is dominated by the dense lstsq ($\sim$30k$\times$7k), a few minutes on 4 CPU cores; total solve wall-clock is minutes, vs tens of minutes for the gradient PINNs to stall far higher.
- $\nu=1$ (Re $=$ 1): advection is mild, Newton converges without continuation (expF06 needed $\nu$-continuation at Re $\ge$ 20 in 2D steady Burgers; a Reynolds sweep is future work, not claimed here).

## Conclusions

*(proposed, pending Sam's review)* The frozen-QI-geometry + solved-readout recipe survives the jump to $d=4$ and a nonlinear constrained system: 3D unsteady Navier-Stokes falls to 1.5e-6 velocity rel $L_2$ in two Gauss-Newton lstsq steps with zero training, while gradient optimization of the identical architecture stalls 4 orders higher. The remaining gap to fp64 is not the PDE or the optimizer but the tensor-product conditioning budget (the $\sqrt{\varepsilon}$ product-truncation floor and the top-$K$ solve budget).

## Open questions

- Reynolds sweep: where does plain Newton stop converging, and does $\nu$-continuation (expF06) rescue it? Turbulent-regime Taylor-Green is the real stress test.
- The $\sqrt{\varepsilon}$ tensor-product floor: can the per-axis factors be computed in extended precision (the repo's mpmath path, per axis only -- the 1D factors are tiny) to push the product truncation deeper?
- Adaptive $K$: the top-$K$ product-direction budget is spent globally; per-field or residual-guided allocation may buy an order.
