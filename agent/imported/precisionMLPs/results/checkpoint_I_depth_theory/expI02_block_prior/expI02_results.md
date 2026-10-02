# expI02 -- a first prior on what the QI blocks are good for

**Status: draft, speculative prior for Sam's real tests (2026-09-05). Single seed unless stated; nothing here is definitive.**

## TL;DR

- The corrected block (fixed mesh at every depth, free projection, one calibration pass plus a weak band penalty, smooth init, head solved inside the forward pass) trains as well as the notebook's range-tracked recipe and needs no forward normalization. The band penalty is neither help nor harm across seeds.
- The fixed-feature wins (capped whitening 3-4 orders, heavy-ball momentum 30-300x over Adam, function-value coordinates another 3-10x under momentum) do not transfer to the block at a first-order budget of 2000-4000 steps. Raw coordinates with Adam and VarPro win; whitening is neutral to harmful under Adam and seed-unstable; heavy ball is 10x worse; the learning rate moved the result 6x, more than any mechanism.
- The block is optimization-limited, not capacity-limited: every learned number sits three to six orders above its oracle floor, no size knob moves it at fixed steps, three- and four-layer stacks do not train first order, and a plain MLP's width ladder scales past it on composition.
- Head to head at $d=5$ at matched parameters and FLOPs, first order the block is within 3x of the shallow ridge-QI and of the MLP either way, and the MLP wins the random-ridge control. Gauss-Newton with a dense Jacobian, which does not scale, buys the block one to two and a half orders on product peak, three bumps and composition.
- Where the structure pays is wherever a head can be solved on QI-resolved features: the Lorenz flow map (5x over the MLP), 1-D Poisson through the operator-system solve ($3\times10^{-7}$ where the matched MLP's features cannot solve at all), kin8nm at parity. Friedman1 and concrete go to the MLP by 10-15%, Fashion-MNIST by 0.6-1 point.
- Two implementation facts that would have corrupted the prior: whiten on the occupied band of the mesh, not the whole mesh (20-100x); never truncate a solved head below 100x the working dtype's roundoff (fp32 at rcond $10^{-14}$ made the block look worse than ridge regression).

## Question

With the geometry fixed at every depth ($\gamma h=\lambda_\star$, centers never moved, no forward normalization) and utilization controlled by one calibration pass plus a weak band penalty: which coefficient coordinates, preconditioning, optimizer, head treatment, rank and depth make the compositional QI block train best, and where does it beat a standard MLP at matched parameters and FLOPs?

## The prior, stated before the data, and its scorecard

1. Capped whitening of the center-index coefficients is the largest optimization gain in the block ($\ge2$ orders at 2000 first-order steps on the $d=3$ targets); function-value coordinates add $\le1$ order and can hurt under Adam. **Held on fixed features, failed in the block.**
2. Heavy-ball momentum beats Adam on whitened coefficients; Adam stays better on the directions; the mixed optimizer wins. **Held on fixed features, failed in the block.**
3. Calibration plus the weak band penalty matches the old range tracking within $3\times$; the failure mode is a channel collapsing toward constant. **Held; no channel collapsed.**
4. The block beats the MLP by $\ge3$ orders on factorizable $d=5$ targets at matched parameters and FLOPs, and ties or loses on random ridges. **Failed on the first half at this budget; held on the control.**
5. Friedman1: both at the noise floor; Lorenz map: block 1-2 orders ahead; kin8nm and concrete: parity. **Mixed: Lorenz 5x, kin8nm parity, Friedman1 and concrete to the MLP.**
6. Fashion-MNIST: within one point of the matched MLP. **Held.**
7. Linear PDEs with the operator-system head solve: block $\ge3$ orders ahead; Burgers: both near $10^{-3}$, the MLP possibly better. **Held in 1-D only; in 2-D and on Burgers neither model trained in 3000 Adam steps.**

Two facts measured before the studies and built into them: whitening the bank Gram on the whole mesh $[-1,1]$ instead of the occupied band $[-0.8,0.8]$ costs 20-100x in the fixed-feature ladder (the whitener uses the occupied band); the cosine schedule costs under 2x against a constant step and is kept for uniformity.

## Experiment design

**The corrected block.** Layer 1: free projection $p=Vx+b_V$ ($V\in\mathbb R^{M\times d}$, not normalized). Every layer: bank $H_{rj}=\tanh(\gamma(p_r-c_j))$ with $c_j$ cell-centered on $[-1,1]$, $N$ cells, $h=2/N$, $\gamma=\lambda_\star/h$, $\lambda_\star=0.25$; mixer $z_k=b_k+\sum_{rj}C_{rjk}H_{rj}$ with $C_{r,:,k}=P\,\theta_{r,:,k}$. Coordinates $P\in\{I,\ D,\ W_\kappa,\ DW_\kappa\}$: $D$ the differencing map $w_j=(a_j-a_{j-1})/2$ (the bump basis $b_j=(t_j-t_{j+1})/2$, a partition of unity on the interior whose coefficients are smoothed local heights; the sum's own constant $-a_{N-1}/2$ stays inside the bank); $W_\kappa$ the whitener of the bank Gram on uniform occupancy of $[-0.8,0.8]$, $W_\kappa=V\,\mathrm{diag}(\min(1/\sigma,\kappa/\sigma_{\max}))$, $\kappa=10^3$, so the whitened basis has singular values $\min(1,\kappa\sigma/\sigma_{\max})$. $P$ acts on parameters once per step, never on activations. Rank: full $C$ or $C=\sum_s a^{(s)}\otimes P\Phi^{(s)}$. Depth: channel layers on the $K$ channels, residual $z+\Delta$ optional. Readout: the last mixer, solved by QR then SVD of $R$ at rcond $\max(10^{-14},100\,\varepsilon_{\rm dtype})$ for regression; trained for cross-entropy.

**Init.** Every profile is the same smooth random function whatever $P$ (a degree-3 Chebyshev sum with coefficients $\xi_m/(m+1)$, unit sd on the occupied band, projected onto the bank; $\theta=P^{-1}w$), so coordinate arms differ only in optimization. Never zero.

**Band control.** One calibration pass: each bank input gets mean 0 and sd $s_\star=0.4$ (projection rows and $b_V$; channel profiles and biases); residual increments sd 0.1. Training adds $\beta R_{\rm band}$ with $R_{\rm band}=\sum_k([|\mu_k|-0.25]_+^2+[0.15-s_k]_+^2+[s_k-0.6]_+^2+\mathbb E[(|z_k|-0.9)_+^2])$ on the actual batch statistics, $\beta=10^{-2}$; at the calibrated state a Gaussian channel carries about 0.04 of excursion baseline.

**Optimizers.** Adam (lr $5\times10^{-3}$ on the ladder, $2\times10^{-2}$ downstream after the lr check); heavy ball (momentum 0.9, lr $=1/L$ with $L$ the top Hessian eigenvalue of the stepped group by power iteration at init); mixed (Adam on $V,b_V$, heavy ball on $\theta$). All with 50-step warmup and cosine decay. Heads: VarPro (the head solved inside the forward pass on that pass's own features, the Kaufman gradient, zero extra bank evaluations), periodic (replaced by the solve every 250 steps, trained by Adam in between), trained (a gradient parameter; the solved-head error still logged at every log point). Gauss-Newton (variable projection, Kaufman Jacobian, dense) only as a finisher with its own column.

**Studies.**
- A0: frozen 1-D bank (80 cells, data on $[-0.8,0.8]$), head trained from zero for 1000 full-batch steps, four targets, $P\times\{\text{GD, momentum, Adam}\}$, the solved head as the floor.
- A1: the ladder on the two-level block, $d=3$, targets gauss bump / fast waves / composition / three bumps, $(M,N_1,K,N_2)=(6,32,2,64)$, $n=2560$, 2000 steps: old (notebook recipe: unit directions, range tracking, zero profiles) $\to$ smooth init $\to$ fixed mesh + calibration + penalty $\to$ $D$ $\to$ $W$ $\to$ $DW$ $\to$ heavy ball / mixed; then on the winner: trained and periodic heads, $\beta=0$, rank 1 and 2, lr 0.001 and 0.02 (also for the whitened rung); seeds 1, 2 on the top three; Gauss-Newton (8 iterations) on every rung; the oracle floor (factorization given, channel solved against the standardized known $P$, readout solved).
- A2: scaling at $d=4$ with the winner at lr 0.02, 2000 steps: $N_1$, $N_2$, $M$, $K$ one at a time from $(8,32,2,64)$, depth 2/3/4 with and without residual, the tanh MLP width ladder 16-256, the oracle floor at the base config.
- B1: $d=5$, six targets including the random-ridge control, 4000 steps (the expI01 E2 budget), block / shallow ridge-QI at the same units / tanh MLP at the same parameters / at the same FLOPs; the MLP gets the same final head solve; 3 seeds; Gauss-Newton (20 iterations) on the QI arms at seed 0.
- B2: Friedman1 ($d=10$, $\sigma=1$), Lorenz flow map ($3\to3$, RK4, $\Delta t=0.05$, train and test from different trajectories), kin8nm, concrete (OpenML); block vs MLP at matched parameters vs ridge regression; 3 splits; fp32 except the noise-free Lorenz map; on noisy labels every solved head picks its truncation by two-fold validation under a readout-norm cap, before training for the VarPro head and again at the end.
- B3: Fashion-MNIST, fp32, cross-entropy, Adam, batch 256, 10 epochs; block at two sizes against tanh and ReLU MLPs at matched parameters and logistic regression; test error per epoch with the trained head and with a one-hot least-squares refit.
- B4: 1-D Poisson (manufactured $u=\sin2\pi x+0.3\sin8\pi x$), 2-D Poisson (manufactured $u=\sin\pi x\sin\pi y+\tfrac12\sin3\pi x\sin2\pi y$), Burgers ($\nu=0.01/\pi$, Cole-Hopf oracle from expF13), all in scaled coordinates on $[-1,1]^d$; block vs MLP at matched parameters, Adam (lr $10^{-3}$) on residual $+10\times$ boundary loss, 3000 steps; on the linear problems both models then get the head solved on the stacked operator system (feature Laplacians by forward-over-reverse autodiff). A second protocol (lr $3\times10^{-3}$, 6000 steps, the head replaced by the operator solve every 250 steps for both models) was worse on every problem and is kept only as a failed variant.

**Compute.** Mac mini M4, CPU; fp64 for A0-A2, B1, B4 and the Lorenz map; fp32 otherwise. About three hours of runs; wall-clock numbers in the JSON were partly taken while another study shared the machine and are not comparable across studies.

**Code & data.** `experiments/expI02_block_prior/{qi2.py, tasks.py, run.py, plots.py, tables.py}`; tests `tests/test_expI02_block_prior.py` (24: the mesh, the coordinate maps as exact reparametrizations, the capped spectrum, the 1-D floor through the stack, the expH05 cliff cell, calibration, the band penalty, the in-pass VarPro solve, the PDE residuals, the expF01 Poisson floor through the operator solve). Results and figures under `results/checkpoint_I_depth_theory/expI02_block_prior/` (`a0.json` .. `b4.json`, `*_v1_lr5e-3.json` and `b4_v2_periodic_opsolve.json` the superseded passes, `figures/fig1_ladder.png` .. `fig6_pinn.png`). Spec: `docs/superpowers/specs/2026-09-05-expI02-block-prior-design.md`.

## Results

### A0: on frozen features, whitening is the lever and momentum beats Adam

Trained head after 1000 full-batch steps, relative $L_2$ on the test interval (the solved head sits at $10^{-14}$ on every target but Runge, $5\times10^{-11}$):

| target | raw, Adam | raw, momentum | values, momentum | raw + whitening, momentum | values + whitening, momentum | values + whitening, Adam |
|---|---|---|---|---|---|---|
| mixed sine | 2.2e-1 | 2.1e-1 | 8.0e-2 | 3.4e-5 | 4.8e-6 | 1.9e-3 |
| exp | 5.6e-4 | 2.1e-3 | 6.1e-3 | 5.4e-6 | 4.5e-6 | 5.8e-4 |
| Runge | 5.9e-3 | 1.9e-2 | 1.1e-2 | 2.3e-5 | 2.9e-6 | 2.0e-4 |
| gauss | 1.9e-3 | 2.5e-3 | 2.5e-3 | 6.2e-6 | 3.8e-6 | 2.3e-4 |

Whitening buys three to four orders under every optimizer. On the whitened problem heavy-ball momentum beats plain GD by about 10x and Adam by 30-300x. Function-value coordinates do nothing on their own and add a factor of 3-10 only on top of whitening under momentum; under Adam they are neutral to harmful. The best first-order result is still six to nine orders above the solved head: the head must be solved.

### A1: in the block, none of that transfers at a 2000-step budget

Two-level block, $d=3$, seed 0, solved-head test error after 2000 steps (geometric mean over the four targets; the oracle floors at this size are $9\times10^{-9}$ for the gauss bump and composition and $10^{-7}$ for fast waves, five to six orders below every learned number):

| rung | gauss bump | fast waves | composition | three bumps | geo. mean | +GN (8 it.) |
|---|---|---|---|---|---|---|
| old (notebook recipe) | 2.0e-2 | 1.5e-1 | 1.3e-3 | 9.8e-4 | 7.9e-3 | 3.6e-3 |
| + smooth init | 5.4e-3 | 3.0e-1 | 3.5e-3 | 1.3e-3 | 9.3e-3 | 7.8e-4 |
| + fixed mesh, calibration, penalty (**mesh**) | 1.1e-3 | 1.7e-1 | 3.0e-3 | 1.3e-3 | 5.3e-3 | 1.0e-3 |
| + function-value coordinates | 2.7e-3 | 3.6e-1 | 5.3e-3 | 1.4e-3 | 9.2e-3 | 1.1e-3 |
| + whitening | 3.0e-4 | 5.7e-1 | 2.3e-3 | 1.3e-2 | 8.6e-3 | 1.8e-3 |
| + values and whitening | 9.7e-4 | 5.9e-1 | 3.1e-3 | 1.3e-2 | 1.2e-2 | 2.6e-3 |
| whitening, heavy ball | 1.2e-2 | 5.7e-1 | 4.7e-2 | 4.7e-2 | 6.3e-2 | 1.5e-2 |
| whitening, mixed (Adam on $V$) | 6.1e-3 | 5.7e-1 | 7.6e-3 | 1.8e-3 | 1.5e-2 | 2.3e-3 |
| mesh, lr 0.02 | 1.3e-3 | 2.2e-3 | 8.3e-4 | 2.6e-4 | **8.8e-4** | -- |
| whitening, lr 0.02 | 2.2e-4 | 1.0e-3 | 7.8e-3 | 7.8e-2 | 3.4e-3 | -- |

- **The corrected band control matches the old range tracking.** Fixed mesh plus one calibration pass plus the weak penalty is within a factor of two of the notebook recipe either way (gauss bump 17x better, composition 2x worse), and $\beta=0$ is indistinguishable from $\beta=10^{-2}$ across three seeds (geometric means 1.7e-3, 8.2e-3, 9.4e-3 against 5.3e-3, 2.3e-3, 9.6e-3). No channel collapsed.
- **Whitening is not a win under Adam in the block.** It helps the gauss bump 4x and hurts three bumps 10x at the ladder's lr; at lr 0.02 it gives the best first-order gauss and fast-waves numbers of the study (2.2e-4, 1.0e-3) and the worst three-bumps number (7.8e-2). Across seeds it is less stable than raw (8.6e-3, 1.4e-2, 6.5e-2 against 5.3e-3, 2.3e-3, 9.6e-3).
- **Heavy ball is 10x worse than Adam in the block**, and the mixed optimizer, after a bug fix in its curvature estimate (the first pass measured $L$ over all parameters instead of the heavy-ball group), is still 3x worse. The step $1/L$ from the top Hessian eigenvalue at init is too small once the directions and two layers share one scale, and Adam's per-coordinate scaling is doing real work.
- **The learning rate matters more than any mechanism tested.** Raw coordinates at lr 0.02 beat lr 0.005 by 6x in geometric mean (composition 3.6x, three bumps 5x, gauss unchanged). A coordinate change alters the parameter scale, so one lr is not one setting; this confounds every whitening row, and it is why the downstream studies carry lr 0.02.
- **Head treatment.** VarPro beats an Adam-trained head with a final solve by 2.5x in geometric mean, and the trained head catches its own solved head within 300 steps (figure 2b): the gap the solved-head diagnostic looks for closes early, and the representation is what limits. Periodic replacement of the head every 250 steps with Adam continuing is dead (2.3e-1, and Gauss-Newton diverges from it): the freshly solved raw-coordinate head has coefficients of order $10^{-2}$ and Adam's first normalized steps move each by the lr, the kill-list mechanism "Adam's scale-free step next to an exact solve" (`docs/REQUIREMENTS.md`, section 5). The same mechanism sank the periodic operator solve in B4.
- **Rank.** The KAT tie (rank 1) and rank 2 are 2.5-3x worse than the untied tensor at this size.
- **Gauss-Newton** (8 iterations, dense Jacobian) buys 5-100x on the gauss bump (best 3.3e-6, from the whitened rung), 3-10x on composition, little on three bumps, nothing on fast waves.
- **Fast waves is a seed lottery**: 2e-3 on some seeds, 1e-1 to 6e-1 on others, for every arm; the radial quadratic channel is either found or not. Any claim about it needs seeds.

### A2: no size knob moves the block at fixed steps; depth breaks it; the MLP scales past it

$d=4$, 2000 steps, seed 0, solved-head test error, base $(M,N_1,K,N_2)=(8,32,2,64)$ at 683 parameters:

| target | base | $N_1$ 8..64 | $N_2$ 32..128 | $M$ 4..16 | $K$ 1..4 | depth 3, 4 | depth 3r, 4r | MLP 369p .. 67k p | oracle floor |
|---|---|---|---|---|---|---|---|---|---|
| fast waves | 1.3e-3 | 1.1e-2, 5.9e-3, 1.3e-3, 1.1e-3 | 1.3e-3, 1.3e-3, 2.2e-3 | 1.6e-3, 1.3e-3, 3.1e-3 | 2.2e-3, 1.3e-3, 2.4e-3 | 4.8e-1, 4.8e-1 | 4.8e-1, 4.9e-1 | 2.0e-1 .. 1.9e-2 | 6.8e-6 |
| composition | 2.5e-3 | 3.8e-3, 2.3e-3, 2.5e-3, 3.4e-3 | 3.2e-3, 2.5e-3, 3.6e-3 | 5.9e-3, 2.5e-3, 6.2e-3 | 4.0e-3, 2.5e-3, 8.1e-4 | 2.1e-1, 3.3e-3 | 4.1e-3, 7.1e-2 | 1.6e-2 .. 5.4e-5 | 1.6e-7 |
| product peak | 1.6e-2 | 2.4e-2, 1.3e-2, 1.6e-2, 4.9e-2 | 1.2e-2, 1.6e-2, 2.9e-2 | 3.4e-2, 1.6e-2, 7.1e-3 | 1.1e-2, 1.6e-2, 7.5e-3 | 8.0e-1, 8.0e-1 | 1.6e-1, 4.3e-2 | 9.4e-2 .. 2.1e-2 | 1.3e-3 |

- Past $N_1=32$ nothing moves by more than 3x in any direction; $K=4$ helps composition 3x and $M=16$ helps product peak 2x. The learned error is set by the optimizer, not by capacity: the oracle floors are two to four orders lower on the same sizes.
- Three- and four-layer stacks do not train first order (0.2-0.8 on all three targets, one exception at depth 4 on composition); residual streams recover part of it on composition and product peak but never beat depth 2. Depth needs its own initialization and optimizer study before it means anything.
- The MLP width ladder scales cleanly with parameters and crosses the block on composition at about 5k parameters (block 2.5e-3 at 683, MLP 1.9e-3 at 4.5k and 5.4e-5 at 67k). On fast waves the block at 683 parameters beats the MLP at 67k by 15x; on product peak it beats it by 1.3x.

### B1: at $d=5$ the block ties the shallow model and the MLP first order; Gauss-Newton is the differentiator

4000 steps, matched parameters (831) and FLOPs, median of three seeds with the min-max range, Gauss-Newton (20 iterations) on seed 0:

| target | block | block + GN | shallow ridge-QI, same units | MLP, same params | MLP, same FLOPs |
|---|---|---|---|---|---|
| gauss bump | 3.5e-2 (9e-4 .. 2e-1) | 6.6e-3 | 3.0e-3 | 1.1e-2 | 1.0e-2 |
| fast waves | 2.3e-1 (8e-3 .. 4e-1) | 4.3e-1 | 1.7e-1 | 9.3e-2 | 9.1e-2 |
| composition | 2.0e-3 (2e-3 .. 8e-3) | 8.8e-4 | 2.2e-3 | 7.2e-3 | 7.0e-3 |
| product peak | 4.0e-2 (4e-2 .. 6e-2) | 1.4e-4 | 6.5e-2 | 6.1e-2 | 3.8e-2 |
| three bumps | 7.8e-3 (1e-3 .. 5e-2) | 2.1e-4 | 8.3e-3 | 2.1e-2 | 2.1e-2 |
| random ridges (control) | 3.4e-1 | 1.6e-1 | 2.0e-1 | 5.8e-2 | 6.5e-2 |

First order, the block is within 3x of the shallow model on every target and within 3x of the MLP in either direction, with a much larger seed spread (two orders on the gauss bump). The MLP wins the control by 5x, as a control should. Gauss-Newton is what separates the block: 300x on product peak, 40x on three bumps, 2x on composition, nothing on fast waves and the control. That finisher materializes an $n\times P$ Jacobian and is not the scalable route.

### B2: solved heads pay on noise-free structure, not on noisy tabular data

Test RMSE in standardized units, median of three splits:

| dataset | $n$, $d$ | ridge | block | MLP, same params |
|---|---|---|---|---|
| Friedman1 | 2000, 10 | 0.498 | 0.281 | 0.249 |
| Lorenz flow map (noise free) | 4000, 3 | 0.304 | 0.002 | 0.010 |
| kin8nm | 8192, 8 | 0.757 | 0.245 | 0.255 |
| concrete | 1030, 8 | 0.643 | 0.336 | 0.286 |

All models beat ridge by 2-150x, so the structure is real. The block wins the noise-free flow map by 5x (1.4e-3 to 2e-3 against 1e-2, seeds agree), ties kin8nm, and loses Friedman1 and concrete by 13-17%. A first pass with the block in fp32 solving its head at rcond $10^{-14}$ put it *below ridge* on three of four sets; the truncation must sit above 100x the dtype's roundoff and, on noisy labels, be validated before the VarPro head trains against it. The California-housing read of expI01 stands: the block is not a tabular-data model.

### B3: Fashion-MNIST, within a point of the matched MLPs

Cross-entropy, Adam, batch 256, 10 epochs, fp32, one seed:

| model | params | test accuracy, trained head | test accuracy, least-squares refit head |
|---|---|---|---|
| logistic regression | 7850 | 84.0 | 81.1 |
| QI block, small | 35914 | 86.4 | 85.8 |
| tanh MLP, same params | 36087 | 87.4 | 86.6 |
| ReLU MLP, same params | 36087 | 86.9 | 85.0 |
| QI block, large | 88202 | 87.4 | 86.5 |
| tanh MLP, same params | 88615 | 88.0 | 87.6 |
| ReLU MLP, same params | 88615 | 88.1 | 87.0 |

The block trails the matched MLPs by 0.6-1.0 points at both sizes and takes about six epochs to catch its own least-squares-refit head, which already scores 75% on the calibrated random features before any training.

### B4: the operator solve is the block's PINN, and it worked only in 1-D here

Relative $L_2$ against the exact solution after 3000 Adam steps, then with the head solved on the stacked operator system:

| PDE | block, Adam | block, operator solve | MLP, Adam | MLP, operator solve |
|---|---|---|---|---|
| 1-D Poisson | 7.7e-1 | 3.4e-7 | 4.9 | 5.9 |
| 2-D Poisson | 7.2e-1 | 7.7e-1 | 4.4e-1 | 4.6e-1 |
| Burgers | 7.9e-1 | -- | 3.7e-1 | -- |

Neither model trains far on the PINN loss in 3000 Adam steps at these sizes (263-533 parameters), and the block trains worse than the MLP on 2-D Poisson and Burgers. On 1-D Poisson the operator solve on the block's features reaches $3\times10^{-7}$ while the matched MLP's 15 features cannot solve the system at all; on 2-D Poisson the block's features after training were not good enough for the solve to help. The shallow QI geometry reaches $10^{-12}$ on the same 1-D problem with no training (test), so the block's 1-D number is its channel resolution, not the method's floor. Replacing the head by the operator solve during training (every 250 steps, Adam continuing) made both models worse on every problem, the same kill-list mechanism as the periodic head in A1.

### Figures

- **`fig1_ladder.png`** (A0) -- one panel per target, test error of the trained head against step on a fixed $[10^{-15},10^{1}]$ axis; colors are the four coordinate systems, solid lines heavy-ball momentum at lr $1/L$, dotted lines Adam, dashed grey the solved head. Read the vertical spread between blue/green and red/purple (whitening), then solid against dotted (momentum against Adam), then the six-order gap between the best line and the dashed floor.
- **`fig2_mechanisms.png`** (A1) -- (a) solved-head test error after 2000 steps against the rung of the ladder, one line per target, filled markers first order, hollow markers after eight Gauss-Newton iterations, dotted horizontals the oracle floors, small dots the extra seeds where run; the two columns right of the dashed line are the lr 0.02 check. Read the flatness of every line across the mechanism rungs, the drop at the lr column, and the five-order gap to the dotted floors. (b) On the gauss bump with the mesh recipe: test error with the Adam-trained head against the same features with the head solved, the VarPro run, and the periodic-replacement run. Read the blue and red lines merging by step 300, the dashed VarPro line 10x below them, and the dotted line jumping up at every replacement.
- **`fig3_scaling.png`** (A2) -- one panel per target at $d=4$, solved-head error against parameters; colored lines sweep one knob each (annotated with the knob's value), black squares depth 2/3/4, hollow squares depth with residual streams, grey triangles the MLP width ladder, dotted the oracle floor at the base config. Read the flat colored lines (no knob moves the error), the black squares jumping up at depth 3, and the grey line crossing the block on composition.
- **`fig4_headtohead.png`** (B1, B2) -- (a) $d=5$: per target, the median over seeds with a min-max bar, one marker per arm, hollow red the block after Gauss-Newton. Read the four first-order markers sitting within a decade of each other on every target, the MLP alone below on the control, and the hollow markers one to two and a half orders down on product peak and three bumps. (b) test RMSE divided by the ridge RMSE per dataset, block against the matched MLP. Read the Lorenz point far below the MLP and the other three at parity or slightly above.
- **`fig5_fashion.png`** (B3) -- (a) test accuracy against epoch for the large size, the block's least-squares-refit head dotted; (b) accuracy after ten epochs against parameters for both sizes, logistic regression as the dash-dot line. Read the constant 0.6-1 point gap and the refit head's 75% at epoch 0.
- **`fig6_pinn.png`** (B4) -- one panel per PDE, relative $L_2$ against the exact solution during Adam training for the block and the matched MLP, hollow markers the head solved on the operator system at the end. Read the flat training curves everywhere, and the single red marker at $3\times10^{-7}$ on 1-D Poisson.

## Additional details

- **The whitener's reference occupancy.** The fixed-feature probe that motivated the design used a bank on $[-1.25,1.25]$ with the Gram taken on the data band; the first implementation whitened on the whole mesh $[-1,1]$ with the data on $[-0.8,0.8]$ and lost 20x (raw + whitening) to 100x (values + whitening) under momentum. The reference occupancy must match where the calibrated data sit.
- **The in-pass VarPro solve.** Solving the head on the current pass's detached features before forming the output gives exactly the Kaufman variable-projection gradient, costs no extra bank evaluation, and made the per-step cost of the ladder drop from about 250 ms to under 10 ms. The record's QR-then-SVD solver is used at every logged point and at the end; the per-step solve uses LAPACK gelsd with the same truncation.
- **fp32 and the truncation.** A head solved at rcond $10^{-14}$ in fp32 keeps garbage singular directions ($\varepsilon_{\rm fp32}\approx10^{-7}$); the floor is now $100\,\varepsilon_{\rm dtype}$, derived from the working dtype, never a constant (requirements checklist item 6).
- **Timing.** Nothing here is a fair wall-clock comparison: another study shared the machine during A0 and the first pass of A1.

## Conclusions

*Speculative, single-seed, for Sam's real tests.* In this prior the block's differentiator is the solvable head on QI-resolved features (the Lorenz map, the 1-D operator solve, the Gauss-Newton column at $d=5$), not first-order trainability: with Adam at a few thousand steps it sits three to six orders above its own oracle floors, within a factor of three of a shallow ridge-QI or a plain MLP, and does not improve with size or depth. The corrected band control (fixed mesh, calibration, weak penalty) is a clean replacement for range tracking and costs nothing; coefficient coordinates and preconditioning, decisive on frozen features, are not decisive inside the block under Adam.

## Open questions

- The E1-to-E2 gap is the whole question: what first-order-class method closes the three-to-six-order gap to the oracle floor in the channel layer? The checkpoint-D step-2 solver applied to the channel tensor, or a scalable Gauss-Newton, are the candidates; coordinates and normalization are not.
- Whitening under Adam with the learning rate tuned per coordinate system (it won the gauss bump and fast waves at lr 0.02 and lost three bumps); a 2-D lr-by-cap grid would settle whether it is ever a win.
- Why depth 3-4 does not train: the calibrated init, the residual increment scale, or Adam on a deeper stack; a per-layer learning-rate and init study before any depth claim.
- The fast-waves seed lottery: what in the direction and profile init decides whether the radial quadratic channel is found.
- PINN training of the block: the operator solve must be applied without Adam kicking the head afterwards (a scale-aware restart, or VarPro on the operator system every step at a cost that is affordable only with a cheaper feature-Laplacian).
