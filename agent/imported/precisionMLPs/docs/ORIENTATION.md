# Orientation: where the optimizer program stands, and what to build next

Read this second. **`docs/REQUIREMENTS.md` comes first**: it is the gate that every proposed mechanism passes before implementation, and it holds the complexity budget, the GPU cost model, and the kill list. This document is the *state*: what is settled, what is open, what was tried and failed, and which written claims turned out to be wrong. `docs/INDEX.md` maps every document in the repo; if something is not in the index, treat it as historical.

## 1. The program

The founding fact: the quasi-interpolant construction builds a network by hand that reaches $10^{-15}$ relative error, while training the same architecture stalls at $10^{-3}$ to $10^{-10}$. Twelve digits exist in the architecture and gradient descent cannot reach them.

The cause is structural. A network is *exactly linear* in some of its parameters and curved in the rest. For the linear ones, fitting is a linear least-squares problem solved in closed form to machine precision. Gradient descent on that same subproblem needs about $(\sigma_1/\sigma_j)^2$ steps for the direction with singular value $\sigma_j$, and these feature matrices have $\sigma_{400}/\sigma_1 \approx 3\times10^{-19}$.

Sam's three-step program, in `docs/motivation.md` (the north star, read it):

1. **A general optimizer.** Adam. Done.
2. **An iterative least-squares solver that scales like Adam.** Substantially done, with one hard negative result.
3. **Lobotomize Adam and stitch the solver in.** The current phase.

Step 3 decomposes into three sub-problems:

| | status |
|---|---|
| **What to solve.** Which parameters enter the least-squares set $L$ | **solved**, `results/checkpoint_D_optimizers/expD15_inclusion_score/METHOD_L_selection.md` |
| **How hard to solve.** The damping $\mu$ and its schedule | **partly specified**, and the spec was violated. §3.1 |
| **How Adam and the solve coexist.** Tempering Adam; not stunning its learning | **open, and confounded.** §3.2, §3.3 |

## 2. What is settled

**The step-2 solver.** The recipe reaches machine epsilon on a frozen feature matrix in fp64 at $O(d)$ state (`expD09_recipe_results.md`), hardened across 6 targets, 4 widths, noise, batching, fp32, and 2-D (`expD10_hardening_results.md`). The full spec, including the damping control rules, is `results/checkpoint_D_optimizers/expD12_mu_ladder/STEP2_SOLVER_SPEC.md`.

**The step-2 wall, and it is real.** No $O(d\cdot k)$ iterative solver reaches the fp64 floor on QI feature matrices (`expD11_results.md`, sharpened in `lead_B_results.md`). Reaching the floor requires stored orthogonality of size $c\approx r$, so $\Theta(d^2)$ state. The binding constraint is the **spectrum**, not batching or drift: the required window tracks the number of distinct singular-value scales, and QI spectra are gapless (median consecutive ratio 1.03), while a 2-level spectrum reaches $10^{-15}$ with zero stored state. `lead_B` localizes it to one sentence: a preconditioner reaches $\kappa = O(1)$ only by matching $R^{-1}$ to additive relative accuracy $1/\kappa$, and $R^{-1}$ has $\varepsilon$-rank $\approx d$ at that tolerance, so every $O(dk)$ family is a truncation that buys nothing. This forces the tiered design: a cheap recurring solver banking $10^{-7}$ to $10^{-9}$, plus a one-time finisher.

**The linear block separates by rows** (approved by Sam). A weight matrix $W\in\mathbb R^{o\times h}$ is $o$ *independent* least-squares problems sharing one design matrix, so state is $h^2/2 + ho$, not $(oh)^2/2$. The governing size is the layer's **input** width. In `results/checkpoint_D_optimizers/expD12_mu_ladder/STEP2_SOLVER_SPEC.md`, marked APPROVED.

**Which parameters to solve.** Four working mechanisms with measured tradeoffs, in `results/checkpoint_D_optimizers/expD15_inclusion_score/METHOD_L_selection.md`. A parameter is in $L$ iff its Jacobian column does not move when $L$ is perturbed. The cheapest mechanism costs $O(\#\text{tensors})$ probes (7 to 28 measured), holds 100% precision on every architecture tested, and correctly refuses attention weights, layer-norm parameters and earlier blocks. Its limitation: the best set is sometimes a *mixed* set spanning several tensors by index, worth up to $40\times$.

**Adam preserves a good geometry but does not find one.** From the QI construction, an optimizer that solves the readout exactly reaches $\sim10^{-16}$; from random init it does not.

## 3. The three open questions

### 3.1 $\mu$ and its schedule, which is more specified than it looks

$\mu$ is the damping in the block solve, $d_\mu = \arg\min_d \|J_L d + r\|^2 + \mu\|d\|^2$. Parameterize by $\alpha = \sqrt\mu/\sigma_1(J_L)$, never by $\mu$ directly.

**Read `results/checkpoint_D_optimizers/expD12_mu_ladder/STEP2_SOLVER_SPEC.md` sections IV.5-IV.7 and V.5-V.8 before designing anything here.** It already contains three control rules, and the last one is the guard the next phase most needs:

$$\text{(a) stop a level when}\quad \text{test}_2 = \frac{\|A^\top r - \mu x\|}{\|A\|\|r\|} \le \alpha, \qquad \text{(b) cap}\quad T_{\max} = \min(T_{\text{hard}},\ 5\alpha^{-3/4}),$$

$$\text{(c) damping floor}\quad \boxed{\ \alpha \;\ge\; r_{\text{entry}} = \|y - \Phi_k w\|/\|y\|\ }\quad\text{measured at level entry.}$$

Rule (c), *never damp finer than the residual you walked in with*, is derived in V.6 and it is load-bearing. expD13 measured what happens without it: the observable error and the true error diverge by up to **19,849×** when drift outruns damping, with no error signal warning you. It also unifies three guards into one, since $r_{\text{entry}}$ is bounded below by geometry drift, label noise, and feature-space approximation error alike. And it is free.

Four measured laws constrain any schedule:

| law | measurement |
|---|---|
| $\kappa_\mu = 1/\alpha$ **exactly** | ratio $1.000$ at every $\alpha$ tested (expD12) |
| solver iterations $\approx 3\alpha^{-0.7}$ | 2, 8, 46, 243, 557 at $\alpha = 10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}, 3{\times}10^{-5}$ |
| a single damped solve lands at eval error $\approx 0.3\alpha$ | over ten decades (expD14 T1) |
| the terminal solve costs $0.87$-$0.93\,r$ iterations | 7% spread across 1-D, 2-D, random, $d$ from 261 to 2060 |

So $\alpha$ *is* the accuracy you are choosing to bank, it sets solver cost superlinearly, and there is a hard floor below which solving is not just wasted but actively misleading.

**What expD14 got wrong, and it was already written down.** Section V.7 of the spec says plainly: *"I hypothesized $\mu$ should be matched to the drift as a variance budget. **Measured false.** Sweeping $\eta \in \{0, 0.1\alpha, \alpha, 10\alpha\}$ shows no sweet spot at matching, less drift is monotonically better, and the terminal solve washes out the difference entirely. What survives is only the one-sided version: don't drive $\alpha$ below the drift."* expD14 then built exactly the two-sided drift-matching rule, measured it as 4 orders worse than a fixed low $\alpha$, and it also **caused weight blowup**, inflating $\|v\|$ from 0.55 to 402. That is one of the three violations the project exists to remove.

**What is genuinely open.** A schedule driven by a *measured* signal rather than a preset ladder, respecting floor (c). The obvious untried candidate is the classical Levenberg-Marquardt ratio (§4). Also unresolved: expD13 tested per-level stopping rules (`stopping.jsonl`: `test2<=a`, `budget 3a^-0.7`, held-out plateau, against an oracle) and those results were never written up.

### 3.2 Tempering Adam so it accepts a solve

The problem is precise. Adam's update is $\Delta\theta_i = -\eta\,\hat m_i/(\sqrt{\hat v_i}+\epsilon) \approx -\eta\,\mathrm{sign}(g_i)$, so **the step size is $\eta$ regardless of how small the gradient is**. After an exact solve the readout is optimal for the current features; Adam then moves the geometry by $\eta$ per coordinate and re-injects error proportional to $\|v\|\eta$. Measured across six decades of $\eta$ and five of $\|v\|$, the data lie on that line. This is the sawtooth in every periodic-solve run.

One fix is measured and works: give Adam's direction the **exact-line-search step length** $t^\star = -(r\cdot Jp_A)/\|Jp_A\|^2$, capped by Adam's own length. One JVP, falls out of machinery already needed, and linear in the residual so it vanishes as the problem is solved. Without it, three of four cells end their run diverging (final error $3\times10^2$ on one).

Measured and neutral: resetting Adam's moments on solved coordinates (no effect), and letting Adam also step the solved block (neutral to mildly positive).

Open: whether a line search is the right general answer, and what the hand-off signal should be. The corrected Adam SNR in §4 is the natural candidate.

### 3.3 Preventing the solve from stunning Adam

The hardest of the three, and currently **confounded**.

Solving $L$ exactly makes the residual orthogonal to everything the current features can express, so the gradient reaching the geometry is only the part of the error the features *cannot* explain. Measured consequences: from random init, plain Adam followed by one terminal solve beats the in-training solve on 3 of 4 targets; on a two-layer bench, pinning the readout every 200 steps damaged feature learning by three orders.

Under-solving recovers it. The arm that deliberately leaves residual for Adam produced a geometry worth $5.8\times10^{-7}$ on `runge`, $16000\times$ better than its start and $100\times$ better than Adam's own. The same rule swung six orders across widths on `sine_8pi`, so the mechanism is real and the rule is not.

**The confound, and it must be cleared first.** Every arm in expD14 carried two throttles never tested with them off: the line-searched $A$ step and gain-allocated energy. So "the in-training solve suppresses feature learning" is indistinguishable from "the throttles suppressed it." One run per cell settles it: exact solve, Adam's geometry step fully unthrottled, no energy split, scored on whether the geometry improved. **Do this before anything else in this sub-problem.**

## 4. Signals worth trying, from the three passages

Sam pasted three literature summaries on "how do you know a model is done learning." Most is calibrated for regimes this project does not have: no minibatch noise (full batch, noiseless targets), no generalization gap (the eval grid is the true function), no classification, no weight decay. Discard neural collapse, HT-SR's tuned $\alpha$ range, gradient disparity, the noise scale, and the EB criterion in its minibatch form.

Four ideas survive.

**The four clocks** (interpolation, representation, readout, generalization) and the claim that they desynchronize. Our failure is grokking with the sign flipped: in grokking the readout clock *lags* the representation clock, and here we run it to $+\infty$ instantly. This is the right vocabulary for §3.3, and it reframes $\mu$, $\rho$ and $\tau$ as three crude ways of slowing the readout clock.

**The realized-over-predicted ratio**, $\rho = G/I$ with $I$ the first-order predicted improvement of the step actually taken and $G$ the realized one. $\rho \approx 1$ means the local model is accurate; $0 < \rho \ll 1$ means the direction is useful but curvature or step size is eating it. This is the classical Levenberg-Marquardt ratio and the obvious driver for $\mu$ in §3.1. It was **never tried**: expD14 invented drift-matching and gain-allocation instead, against the repo's own rule that battle-tested mechanisms win design toss-ups.

**Adam's own SNR, noise-floor corrected.** $\hat m_i^2/\hat v_i$ is the fraction of the gradient's second moment attributable to a persistent direction. The correction is usually omitted and matters: with $\beta_1 = 0.9$, pure noise gives $\mathbb E[m^2]/\mathbb E[v] \approx q = (1-\beta_1)/(1+\beta_1) = 0.0526$, so the RMS of $m/\sqrt v$ under no signal is $0.229$, not zero. The corrected estimate is $\hat s = [(\overline{m^2/v} - q)/(1-q)]_{[0,1]}$. Free, per-coordinate, and the natural hand-off signal for §3.2: it says when Adam's steps have become diffusion rather than motion.

**Coherent travel.** Over a window $W$, $D(W) = \|\sum_{t\in W}\Delta\theta_t\| / \sum_{t\in W}\|\Delta\theta_t\|$, with $D\approx1$ coherent and $D\approx1/\sqrt{|W|}$ a random walk. Restricted to the geometry block it answers §3.3 as a *number* rather than an inference from the final floor. Two accumulators, $O(d)$ state; the window must exceed Adam's $\beta_1$ memory or momentum manufactures alignment. Never instrumented, and it should be.

One caution from the same material: local optimizer statistics cannot certify that no further transition is coming. Any hand-off rule built from them is a policy, not a proof.

## 5. The test matrix

The four geometry-and-data regimes are the discriminating core. Build them by taking a correct geometry with uniform data and perturbing exactly one thing:

| case | what is wrong | why it is in the set |
|---|---|---|
| `qi` | nothing | geometry already right; the solve should engage immediately |
| `clustered` | centers bunched, sparse middle; data uniform | a **geometry** defect |
| `datagap` | centers uniform; 96% of data removed from the middle | a **data** defect |
| `random` | first layer randomized | nothing is right |

`clustered` and `datagap` are the pair that matters: any criterion measuring only linearity must score them identically.

Beyond those, all called out by Sam: **2-D** (the Radon ridge geometry from expE01, the most important extension, and most step-3 work so far is 1-D), 2-layer deep-linear from random init, and the architectures already built in `experiments/expD15_inclusion_score/` (skip connection, fixed basis, depth-2, 5-block ResNet, two conv variants, 2-block pre-norm transformer).

Report all four metrics separately, because they are not interchangeable:

1. Eval relative $L_2$, **split by region** when data coverage is non-uniform. A global average hides the entire effect.
2. **The floor of the final geometry**, the error one terminal exact solve would give. This separates "solved the readout" from "learned the geometry" and is the metric §3.3 turns on.
3. Passes, counted honestly, **including probes**. expD14's cost table omitted the probe and understated its headline configuration by $82\times$.
4. Coherent travel on the geometry block, once instrumented.

## 6. Next steps, in order

1. **Clear the confound (§3.3).** Exact solve, unthrottled Adam geometry step, no energy split, scored on the final geometry's floor. One run per cell. Everything downstream depends on the answer.
2. **Swap the discovery in.** expD14 used iteration 11's dense curvature probe: $\sim4m$ forward passes, never counted, and it collapses entirely in bf16 (its $10^{-3}$ relative step is below bf16's unit roundoff of $3.9\times10^{-3}$). `results/checkpoint_D_optimizers/expD15_inclusion_score/METHOD_L_selection.md` replaces it at two passes with better accuracy. Straight substitution.
3. **Drive $\mu$ by the LM ratio**, against a fixed-$\alpha$ baseline and the expD12 ladder, **respecting the floor $\alpha \ge r_{\text{entry}}$**. Instrument coherent travel and the corrected Adam SNR at the same time; both are nearly free and they are the diagnostics for §3.2 and §3.3.
4. **Run the deployable path in its shipped form.** The cheap $O(d)$ solver has never been composed with a discovered $L$; both numerical guards currently live only in the reference branch, so this is not a formality.
5. **2-D throughout.**

## 7. Repository map

**Read these.** All are tracked.

| document | what it is |
|---|---|
| `docs/REQUIREMENTS.md` | **the gate.** Requirements, complexity budget, GPU cost model, kill list, pre-build checklist. Run its section 8 before proposing anything |
| `docs/INDEX.md` | the map: every surviving document, what it is for, where it lives |
| `docs/motivation.md` | the three-step program, Sam's framing. Start here |
| `docs/requirements_and_lessons.md` | the evidence behind the gate: the requirements as originally written, the litmus tests, eight measured lessons with their measurements |
| `results/checkpoint_D_optimizers/expD12_mu_ladder/STEP2_SOLVER_SPEC.md` | the step-2 solver in full: the $\mu$ control rules, the damping floor, the terminal solve. Also the only writeup of expD12/expD13 |
| `results/checkpoint_D_optimizers/expD15_inclusion_score/METHOD_L_selection.md` | which parameters to solve: four methods, tradeoffs, costs |
| `results/checkpoint_D_optimizers/expD09_2nd_order_regime/DAMPED_GAUSS_NEWTON.md` | the $\mu$ math written out |
| `results/checkpoint_D_optimizers/expD10_step2_hardening/batching_test.md` | lessons T1-T12. Read before touching solver code. **T1 is corrected in place** |
| `results/.../expD10_step2_hardening/expD10_hardening_results.md` | the hardening ledger and the tier structure |
| `results/.../expD11_batching/expD11_results.md` + `lead_B_results.md` | why no $O(dk)$ solver reaches the fp64 floor |
| `results/.../expD09_2nd_order_regime/expD09_recipe_results.md` | the step-2 recipe |
| `results/.../expD08_qi_init_nlcg/iteration_11/iteration_11_results.md` | the certificate, the coupling law, the settling regime |
| `results/.../expD14_lobotomy/iteration_0/iteration_0_results.md` | the first stitching attempt. **Read its correction header** |
| `results/.../expD15_inclusion_score/expD15_results.md` | discovery, experimental record |
| `results/checkpoint_D_optimizers/expD11_batching/SAM_SPEC_superseded.md` | Sam's $O(m{+}n)$ spec. Superseded by expD11's negative result; kept for framing |

**Live code.** `experiments/expD15_inclusion_score/` (discovery, current), `experiments/expD14_lobotomy/iteration_0/` (stitching), `experiments/expD09..expD13` (solver, $\mu$ ladder, drift ladder), `experiments/expD08_qi_init_nlcg/{run,iter11}.py` (imported by expD14 and expD07; the other 24 scripts were deleted).

**What was removed, and why.** The expD08 campaign ran eleven optimizer iterations. Iterations 1-10, their per-iteration writeups, the tether documents, the working notes, the optimizer audit and 24 scripts were deleted, because iteration 11 supersedes all of them and the reusable content was already promoted into `requirements_and_lessons.md`. One nugget from the deleted audit is worth recording: it flagged a **per-tensor sampled probe** ($O(\#\text{tensors})$ forwards instead of $O(m)$) as "the requirement-5-compatible probe; should become THE probe everywhere" in July. expD15 independently rediscovered and validated exactly that, and it is now the recommended mechanism.

Four documents were **untracked and nearly lost** (the old ignore rules tracked only `*_results.md` under `results/`): the motivation, the requirements and lessons, the damped-Gauss-Newton math, and the step-2 spec. All four were rescued. Documents now live **next to the experiment's writeup under `results/`** (`STEP2_SOLVER_SPEC.md` in `results/checkpoint_D_optimizers/expD12_mu_ladder/`, `METHOD_L_selection.md` in `results/checkpoint_D_optimizers/expD15_inclusion_score/`, and so on), because Sam reads `results/`, not `experiments/`; `experiments/` holds only code, `docs/` only program-level material. Since August 2026 **all `*.md` under `results/` is tracked**, so a new doc there is safe by default.

## 7b. Test-suite state

`uv run --extra dev python -m pytest -q -m "not slow"` gives **262 passing, 17 failing**. The 17 are pre-existing and confined to `test_expD09` (7), `test_expF10_qi_operator` (4), `test_expF11_qi_fno_init` (3), `test_expG03_extrapolation` (2), `test_expF12_tensor_ns3d` (1). None was touched by the cleanup; they are checkpoint F/G and step-2 tests that were already red. Fix or triage them before trusting the suite as a gate.

One class of bug was fixed during the cleanup and is worth knowing about, because it will recur: **several experiments each define a `run.py`, and a bare `import run` picks up whichever one is already in `sys.modules`.** In a full pytest session that silently binds the wrong module and previously aborted collection for the entire suite. Four files now load their `run.py` by explicit path (`importlib.util.spec_from_file_location`); do the same in any new experiment rather than relying on `sys.path` order.

## 8. Claims in the record that are wrong

Do not build on these.

- **expD14's headline cost.** The arm producing $10^{-16}$ costs 572 passes per step against Adam's 4. Its cost table describes a cheaper arm never run in that configuration. Correction header is on the file.
- **"The certificate is cheap."** Iteration 11's per-parameter curvature probe costs $2(2m{+}1)$ forwards, 5,538 at $N{=}256$, never counted in any cost table. Superseded by `results/checkpoint_D_optimizers/expD15_inclusion_score/METHOD_L_selection.md`.
- **"Precision-agnostic"** (expD14's solver). False: nine hardcoded fp64 constants, none derived from dtype. In bf16 the probe's relative step rounds away and the certificate admits the whole network. The discovery rule in `results/checkpoint_D_optimizers/expD15_inclusion_score/METHOD_L_selection.md` does not share this defect; its zeros are bitwise zeros and survive any dtype.
- **T1 in `batching_test.md`** ("steering degrades gracefully, baked-in whitening is poisoned by staleness"). Corrected in place by expD11 `lead_B`: scored against the drifted problem's *own* floor, stale block-QR sits at 0.5-1.5× that floor while block-Jacobi steering is 180× to $2\times10^6$× above it. The original reading was measuring the problem's floor moving.
- **expD09 round 6's double-double result** read as a standalone existence proof. `lead_B` contradicts it: unpreconditioned dd-LSQR is within 2-5× of fp64-LSQR at matched iterations; extra precision only pays where a block preconditioner has already clustered the spectrum.
- **"Recall is free"** in the discovery work. True at $10^{-3}$ precision, false at $10^{-8}$ where losing 3% of the set cost a factor of 47.
- **"Coverage degrades with depth."** Tested across three families at five depths and not supported; non-monotone, and a ResNet shows no trend at all. What looked like depth was contamination probability in the probe batch.
- **A reported $23\times$ data-gap localisation** was an artifact of a halo-less geometry where 12 centers sat over 2 data points.
- **Ratios of 1.00 in the discovery work** are measured against the best *tensor-aligned* affine set, which is not the best affine set. A mixed set beats it by up to $40\times$.
