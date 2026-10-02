# expI01 -- the compositional ridge-QI block: plan and checklist

**Status:** plan (2026-09-01). Nothing run yet beyond a CPU smoke test of the notebook. Sam runs the full study on Colab.

## Question

Can one hidden QI layer be composed with a second QI layer so that depth is used in the way the theory says it can be: a few scalar channels, each a ridge sum of 1-D QI profiles, each followed by a 1-D QI outer function? Concretely: (1) does precision improve as $N_1$, $N_2$, $M$, $K$ grow, and does it scale better than the shallow ridge-QI model; (2) at matched parameter count and matched FLOPs, does the block beat a plain tanh MLP on regression targets that have the structure the theory needs?

## Model (the law: constraints are structural, never penalties)

$$\hat f(x)=b_0+\sum_{k=1}^{K}\sum_{\ell=1}^{N_2}\Psi_{k\ell}\tanh\!\big(\gamma_2[u_k(x)-\tilde c_\ell]\big),\qquad u_k=\frac{P_k-m_k}{s_k},\qquad P_k(x)=b_k+\sum_{r=1}^{M}\sum_{j=1}^{N_1}C_{rjk}\tanh\!\big(\gamma_1[v_r^\top(x-x_0)-c_j]\big).$$

| element | rule | why |
|---|---|---|
| directions $v_r$ | unit length, normalized in the forward pass | removes the $\gamma$ vs $\|v\|$ ambiguity |
| level-1 centers $c_j$ | fixed, cell-centered on $[-T,T]$, $T=1.25\,r_{\rm data}$ | direction independent because $\|v\|=1$; collar is load-bearing (expH01) |
| $\gamma_1$ | $\lambda/h_1$, $h_1=2T/N_1$, $\lambda=0.25$ | aliasing rule (expC07) |
| channel tensor | rank 1 (KAT tie $a_r\Phi_{jk}$) first; `rank=S`; untied as control | Sam: start rigid |
| level-2 coordinate | $u_k=(P_k-m_k)/s_k$, $(m_k,s_k)$ read from the training data with a 25% collar on a fixed schedule | range tracking; the QI scale relation holds in $u$ exactly |
| level-2 centers | fixed, cell-centered on $[-1,1]$; $\gamma_2=\lambda N_2/2$ | same rule as level 1 |
| readout $\Psi,b_0$ | solved: QR then SVD of $R$, rcond $10^{-14}$ | never normal equations (expA01); never gradient (expD01) |
| function coefficients | zero at init; $b_k$ distinct small random | Sam's spec; $b_k$ breaks the channel symmetry at $\Phi=0$ |
| precision | float64 everywhere | GPU runtime; TPUs refused |

Counts: params $\approx Md+MS+SN_1K+K+KN_2$; tanh units $MN_1+KN_2$; FLOPs per sample $Md+MN_1+MN_1S+SN_1K+2KN_2$.

## Arms

1. **oracle**: factorization known ($P=\rho^2$ on the coordinate axes, or the three composition ridges); channel readout solved against the known $P$; regrid; readout solved. No learning. Measures the dividend and the composed floor.
2. **adam**: Adam on directions, channel coefficients, readout; final readout solve.
3. **varpro**: Adam on the nonlinear parameters only; readout re-solved every step.
4. **gn**: variable-projection Levenberg-Marquardt with the Kaufman Jacobian, readout re-solved at every trial; run after varpro.

Baselines: **shallow** ridge-QI at the same unit count with the same direction learning; **MLP** two hidden tanh layers at (a) the same parameter count and (b) the same FLOPs, standard init, Adam, final readout solve on its last layer.

## Experiments (notebook sections, in order of value per minute)

| id | what | knobs | read |
|---|---|---|---|
| E0a | 1-D block, solved readout, $\sin2\pi x$ | $N_1=128$ | must be $\lesssim10^{-12}$ |
| E0b | shallow, expH05 geometry and data ($d=2$, $r=0.4$, 128 offsets) | $M\in\{4,8,12,16\}$ | must reproduce the cliff: fast waves and gauss $<10^{-10}$ at $M=12$, radial Runge at $M=16$ |
| E1 | oracle composition, $d\in\{2,3,4,5,8\}$, five targets | $N_1\in\{8..64\}$, $N_2\in\{32..256\}$ | the dividend and the composed floor; units against the recorded shallow cost (9216 in 3-D, a wall in 4-D) |
| E2 | **headline**: error vs $d$ at matched cost, $d\in\{2,3,4,5,8\}$, five targets | base config $(M,N_1,K,N_2)=(\max(6,2d),32,2,64)$ | compositional rank 1 and untied vs shallow (same units) vs MLP@params vs MLP@FLOPs, all learned from zero. Rank 1 with a shared direction weight vector cannot represent three bumps (three channel centers) or the Genz product (a profile per axis); the untied arm shows what the architecture can do, rank 1 what the most rigid form does |
| E3 | which arm learns it, $d=3,5$, four targets | rank $\{1,2,\text{untied}\}$ x {adam, varpro(+gn)} | curves of refit test error vs step; gap to E1 is the optimization problem |
| E4 | one knob at a time in $d=4$, three targets | $N_1,N_2,M,K$ | compositional vs shallow vs the two MLPs; error vs params and vs FLOPs |
| E5 | control: 24 random ridges, $d=3,5$ | | shallow must win or tie |
| E6 | California housing ($d=8$; diabetes fallback) | $K\in\{1,2,4\}$ | test RMSE vs shallow and param-matched MLP; noise-floored, not a precision test. Inputs winsorized at $3\sigma$ into the unit ball (no test point leaves the band); the QI arms pick their truncation by two-fold validation under a readout-norm cap of 100, because a $10^{-14}$ truncation interpolates the noise with $\|w\|\sim10^9$ and blows up on rare test points a validation split cannot see (measured during the build: val 0.49, test 12) |

Targets: gauss bump, radial Runge, fast waves ($g(\rho^2)$, one quadratic channel), composition ($\exp$ of a ridge sum; $d$-generic cyclic form for $d\ge4$), product peak (Genz; $\exp\sum\log$, one channel of axis profiles), three bumps ($K=3$), packet (two anchors), product sines (four exact ridges, no dividend), random ridges (control). Anchors are expH05's in $d=2$ and expH06's in $d=3$; generated deterministically otherwise.

## Persistence and progress (nothing is lost if compute runs out)

- `MOUNT_DRIVE=True` puts `results.json` and every figure on Google Drive; every finished run is written immediately (atomic replace).
- Every plot is rebuilt from `results.json` after each target, shown in place (`clear_output`) with a running table, and saved.
- Re-running any cell skips runs already in the file, so a disconnected session resumes where it stopped.
- `BUDGET_MIN` stops starting new runs after a wall-clock budget; the last cell rebuilds all figures from the file.

## Confound checklist (the law)

- [ ] E0a and E0b pass before any other number is read.
- [ ] fp64 confirmed on the runtime (`torch.get_default_dtype()`, matmul timing printed). No TPU.
- [ ] $\gamma h=0.25$ asserted at both levels; $A_{\tanh}(0.25)=5.7\times10^{-16}$ printed.
- [ ] Every model in a comparison sees the same training set, test set and seed (`make_data(seed)`), the same step budget, the same final readout solve, the same rcond.
- [ ] Test error is scored on the inner 0.9 ball only; the collar is never scored.
- [ ] $n_{\rm train}\ge8\times$ units for the QI models and $\ge4096$ always; rows per column reported when a solve is wide.
- [ ] Parameter and FLOP counts come from one function per model (`counts()`), and MLP widths are chosen by that function, not by hand.
- [ ] MLP gets the final readout solve too (its last layer), so "solved readout" is not the confound.
- [ ] No per-target tuning of $\lambda$, learning rate, rcond, or step budget.
- [ ] Range tracking schedule identical across arms; regrid always followed by a readout solve in the solved arms.
- [ ] Learned arms report the refit error (fresh readout solve), never the raw Adam readout, so the arms are compared on the same footing.
- [ ] Error axes fixed to $[10^{-15},10^{1}]$; trajectories plotted, not endpoints only.
- [ ] Single seed by default; add seeds (`SEEDS`) before claiming any factor under 10x.
- [ ] Control target (E4) included; the block must not win there.
- [ ] Oracle arm (E1) separates representation from learning: if E1 reaches the floor and E2 does not, the gap is optimization, not architecture.
- [ ] Composed floor recorded: compare E1's best error with $\varepsilon_{\rm mach}(1+\sum_k L_k\|P_k\|_\infty)$.
- [ ] Results JSON and figures saved (`expI01_out/`); the run's config cell copied into the writeup.

## What would count as a result

- Scaling: on the compositional targets, E3's compositional curve descends toward $10^{-13}$ along $N_1$ and $N_2$ at unit counts an order below the shallow model's, and is flat in $M$ beyond the factorization's direction count.
- Architecture: at matched params and at matched FLOPs the block beats the tanh MLP by orders on the compositional targets and does not beat the shallow model on the random-ridge control.
- Learnability: the gap between E1 (oracle) and E2 (learned) is the optimization problem; which arm closes it, and whether rank 1 suffices, is the finding.

## Not in scope here

Rank $S>1$ sweeps beyond the untied control, shared outer function, more than two levels, residual streams, learned centers, non-uniform data, noise. All are follow-ups once E0 to E4 are read.

## Code

`experiments/expI01_compositional_qi/build_notebook.py` writes `compositional_qi_colab.ipynb` (self-contained, no repo imports) and `--smoke` executes every cell at QUICK size on CPU. Upload the notebook to Colab, set a GPU runtime, run all.
