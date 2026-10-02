# The gate: requirements, complexity, and what is already dead

**Read this before proposing any mechanism. Every design idea gets checked against this file first, and most ideas die here rather than after a week of implementation.** The single most expensive failure mode in this project has been building something that works numerically and cannot possibly scale, or rebuilding something already measured dead.

Section 8 is a checklist. Run it on paper before writing code.

---

## 1. What we are actually building

A quasi-interpolant construction builds a network **by hand** that reaches $10^{-15}$ relative error. Training the same architecture stalls at $10^{-3}$ to $10^{-10}$. Twelve digits exist in the architecture and gradient descent cannot reach them.

The cause is structural: a network is exactly linear in some parameters and curved in the rest. For the linear ones, fitting is a linear least-squares problem with a closed-form solution. Gradient descent needs $(\sigma_1/\sigma_j)^2$ steps for the direction with singular value $\sigma_j$, and here $\sigma_{400}/\sigma_1 \approx 3\times10^{-19}$.

So the deliverable is an optimizer that runs ordinary gradient training on the curved parameters and solves the linear ones exactly, discovering which is which by measurement. Sam's three steps: a general optimizer (Adam, done), a least-squares solver that scales like Adam (mostly done), and the lobotomy that stitches them together (current).

**Calibration, and this corrects an earlier framing.** The target is *not* machine epsilon on production models. It is an optimizer for MSE regression generally, feasible across regimes, that breaks past Adam's $10^{-3}$ barrier wherever a least-squares structure exists. Machine-epsilon capability is the *capability* being demonstrated, not the deployment bar.

**Success criterion** (from `CLAUDE.md`): across $N\in\{32,64,128,256,\dots\}$ on the 6-category target family, over 3-5 seeds, error falls at $O(\log(1/\varepsilon))$ in width and reaches eval relative $L_2 \le 10^{-13}$ with $L_\infty$ at machine epsilon, **without** initializing from the construction.

## 2. The five hard requirements

Every candidate is judged against all five. A candidate that fails any one is dead, not "promising with caveats."

1. **First-order and architecture-blind.** Cost in Adam's class: gradients and gradient-cost autodiff products only. No assumptions about the architecture. (Knowing that parameters arrive in *named tensors* is allowed; every framework provides that. Knowing which tensor is "the head" is not.)
2. **MSE loss for now.** Arbitrary differentiable losses are deferred, not a gate on current work.
3. **Solves least squares to the floor.** On least-squares problems it reaches direct-solve (SVD) precision, $\sim10^{-14}$ relative $L_2$.
4. **Generalizable, not finicky.** One hyperparameter story across problems, like Adam. No special regimes, no mode switches, no per-problem tuning. Works under batching. Survives Occam's razor.
5. **Scales.** No exploding memory or per-iteration cost as the problem grows.

**Sam's practicality gate, binding, on top of those.** Every proposed mechanism must make the method *more* practical at real-world scale, not less:

- No state that grows with dataset size. No per-sample caches, no per-batch parameter-sized vectors.
- Optimizer state stays $O(m)$ in Adam's class. Adam stores $2m$.
- Never patch one impracticality by adding another.
- Battle-tested mechanisms (EMA, momentum, Levenberg-Marquardt) win any design toss-up over novel ones.

**Three litmus tests** a candidate must pass before promotion: the expD07 `dl_test` real-data multilayer suite (stable, sustained descent, competitive order of magnitude, not required to win); the batching grid (no blowups, sustained improvement at every batch level); and the full-batch machine-epsilon floor on the toy targets, which is the founding result and is never given back.

## 3. The complexity budget, stated as numbers

This is the table that kills most proposals. $m$ or $P$ is the parameter count, $n$ the dataset size, $d$ the size of the solved block, $r$ its numerical rank.

| resource | budget | consequence of exceeding |
|---|---|---|
| persistent optimizer state | $O(m)$, Adam's class ($2m$) | requirement 5, hard |
| per-step compute | 1 forward + 1 backward, plus $O(1)$ extra passes | requirement 1 |
| per solve-event compute | $k$ passes, amortized over a $T\!\sim\!200$-step cadence | expD10 cost model |
| one-time / terminal solve | $O(d\cdot r)$ transient allowed, CPU-parkable | expD10 tier 3 |
| anything scaling as **passes $\propto P$** | **forbidden** | killed the curvature probe |
| anything scaling as **state $\propto n$** | **forbidden** | Sam's gate |
| materializing $J$ ($n\times m$) | **forbidden** | $n{=}10^6, m{=}10^9$ is $10^{15}$ entries |

**The scaling honesty rule.** $d$ can grow $10\times$ beyond any current width. A method whose cost or memory needs $k/d$ fixed is not sustainable. State this ratio explicitly for any proposal.

**Worked example of the rule biting.** The iteration-11 certificate cost $2(2m{+}1)$ forward passes per probe: 5,538 at $N{=}256$, and $4\times10^9$ at $m{=}10^9$. It was never counted in any cost table, and expD14's headline number understated its own configuration by $82\times$ as a result. The replacement costs two passes and is independent of $P$. **Count the probes.**

## 4. GPU reality

Analysis of the computation's shape; only pass counts are measured.

**The cost model is sequential dependent passes, not FLOPs and not bandwidth.** Probe passes cannot overlap each other or the training step. Counting arithmetic will mislead you. When comparing designs, count passes that cannot be fused.

**Reductions are the largest scaling risk in the whole program.** Krylov methods need four to six inner products per iteration, each a device-wide reduction and, multi-GPU, a collective. At hundreds of iterations per solve event that is thousands of latency-bound collectives, which is the standard reason Krylov underperforms on GPUs. Communication-avoiding variants buy fewer syncs by trading numerical stability, which is the wrong trade for a method whose purpose is machine precision. **Any design built on many Krylov iterations must state its reduction count.**

**Index sets must be masks.** `nonzero`, `union1d`, `setdiff1d`, and random choice over index arrays are uncoalesced gathers *and* host-side set operations. Use a length-$m$ mask and multiply instead of gathering.

**No branching on device scalars.** A tolerance comparison, a convergence check, or a "did membership change" test forces a host-device synchronization that stalls the pipeline. Make them masked arithmetic on device, or run them on a lagged schedule so the transfer overlaps compute.

**Elementwise $O(m)$ work is free.** Posterior updates, masks, decays and comparisons fuse into one or two bandwidth-bound kernels. Adam already touches five such arrays per step.

**Precision.** The repo is fp64 and GPU fp64 runs at between one half and one thirty-second of fp32 depending on the device, so fp64 carries a large constant before anything else is considered. The method must be **precision-agnostic**: it runs in whatever dtype training uses and drives error to *that* dtype's floor. Full double-double as the method is a no-go; compensated double-word arithmetic in the native dtype as a localized implementation detail is fine.

**One thing that survives any dtype, and it is the reason the discovery rule works.** If a Jacobian column does not depend on the perturbed parameters, both evaluations run identical operations on identical inputs, so the difference is *bitwise* zero, not "small." That holds in fp64, fp32 and bf16 alike. Contrast the curvature probe, whose $10^{-3}$ *relative* step sits below bf16's unit roundoff of $3.9\times10^{-3}$: in bf16 the perturbation rounds away, every curvature reads zero, and the test admits the entire network. **Ask of any new signal: is its zero exact or merely small?**

## 5. The kill list

Each of these was measured dead. Do not rebuild them without a specific reason the measurement does not apply.

**Solver and preconditioning**

- **Any $O(d\cdot k)$ iterative solver reaching the fp64 floor on QI spectra.** Reaching the floor needs stored orthogonality $c\approx r$, i.e. $\Theta(d^2)$. The mechanism is the spectrum, not batching or drift: the required window tracks the number of distinct singular-value scales, and QI spectra are gapless (median consecutive ratio 1.03). A 2-level spectrum reaches $10^{-15}$ with zero stored state. (expD11)
- **Any preconditioner making $\kappa(AM) = O(1)$ here.** It must match $R^{-1}$ to additive relative accuracy $1/\kappa$, and $R^{-1}$ has $\varepsilon$-rank $\approx d$ at that tolerance. Every $O(dk)$ family (band, block-diagonal, hierarchical, cascaded, multi-level) is a truncation that reduces $\kappa$ by nothing. Also individually killed: sparse inverse-Cholesky/Vecchia, HODLR compression, cascaded/butterfly whitening, Tikhonov continuation. (expD11 lead_B)
- **Extended precision as the lever.** Unpreconditioned dd-LSQR is within $2$-$5\times$ of fp64-LSQR at matched iterations; the barrier is Krylov convergence rate, not rounding. dd only pays where a block preconditioner has already clustered the spectrum.
- **Gradient-history memory below the effective rank.** A 64-vector window on a rank-900 problem underperformed plain CG. Memory is all-or-nothing with respect to the rank.
- **Batch-local state below the $1/\sqrt b$ agreement scale.** The tail eigenspace is scrambled between batches, so no batch-local state (conjugacy, memory, exact steps) makes progress below it. Only exact aggregation across batches extracts that information.

**Control and damping**

- **Any control decision that compares loss values.** The loss is quadratic in the residual, so loss differences are unresolvable in fp64 around relative $10^{-12}$; the gradient is linear in the residual and stays informative to $10^{-16}$. Line search on loss, loss-based trust regions, and loss-based stopping all cap the method near $10^{-10}$. Watch gradient norms instead.
- **A gauge acting on readings below its own noise floor.** Hit three independent times (loss comparisons, the drift gauge, the gradient-space gain ratio, which ratcheted damping to $10^{23}$). Every measured control signal needs an explicit noise-floor guard.
- **$\mu$ matched to the geometry drift as a variance budget.** Measured false: sweeping drift $\eta \in \{0, 0.1\alpha, \alpha, 10\alpha\}$ shows no sweet spot at matching, less drift is monotonically better, and the terminal solve washes out the difference. Only the one-sided version survives: never drive $\alpha$ *below* the drift. expD14 rebuilt the two-sided rule anyway, measured it 4 orders worse than a fixed low $\alpha$, and it caused weight blowup ($\|v\|$: 0.55 to 402).
- **A trust-region clip on the least-squares step.** Costs 5-8 orders; the damping already controls the step length.
- **Adam's scale-free step next to an exact solve.** $\Delta\theta_i \approx -\eta\,\mathrm{sign}(g_i)$ does not shrink as the gradient does, so it re-injects $\|v\|\eta$ of error after every solve. Without an exact-line-search cap, three of four cells end their run diverging.

**Discovery of the linear block**

- **Per-parameter curvature scores from stochastic estimators.** The target value for an affine coordinate is *exactly zero* and every estimator has a noise floor, so the score reports its own variance. Worse, that variance scales with Gram row mass, largest for the readout, so the exactly-affine block scored as the *most* nonlinear.
- **Soft per-coordinate damping from estimated curvature** (solve everything, weight by $\mu_i$). Exact curvature matches the oracle in all cells; a cheap estimate at 4, 16 or 64 probes is no better than random. Same reason: the damping is set by the estimator's noise, not the truth.
- **Global uniform damping** (the no-selection baseline). Sweeping one $\mu$ over fourteen orders reaches $2.7\times10^{-3}$ where the oracle reaches $1.8\times10^{-8}$.
- **Set-level group testing to isolate offenders.** The exact second-difference test is real and cheap per test, but it reports only *that* a set contains an offender, so isolating them costs $O(\#\text{offenders})$ tests, roughly a thousand passes.
- **Ranking without verification.** Purity is all-or-nothing: adding 2% wrong parameters costs $5\times$, 5% costs four orders, while dropping 20% of the correct set costs nothing measurable at $10^{-3}$ precision. A good ordering is worth nothing; you need a separating gap.

**Reconstructed from a deleted document.** The expD08 optimizer audit (untracked, lost in the July cleanup) carried a graveyard of eleven mechanisms killed in one day. What I can recover from having read it: the carried residual, exact Gauss-Newton step lengths from one JVP, PR+ conjugate directions, the no-loss-comparisons rule and NaN-reject were **locked in**; the full orthogonal memory basis $B$ and its exhaustion escape were **discard-pending** for failing requirement 5 ($\text{rank}\times m$ memory, fragile under batching); the trip/ratchet/basis-reset mechanism was killed by Sam directive ("resetting and ratcheting are wrong and must go") after measuring non-terminating on sine at $N{=}256$; and the fixed tether $T=10^6$ violates requirement 1 by construction and survived only as a measuring stick. It also flagged the **per-tensor sampled probe** as "the requirement-5-compatible probe; should become THE probe everywhere," which expD15 independently rediscovered and validated a month later.

## 5b. Locked in: positive findings, do not re-litigate

The mirror image of the kill list. These were measured, they hold, and rediscovering them costs weeks.

- **Exact step lengths are available at first-order cost.** The Gauss-Newton curvature along a direction $d$ is one JVP, $c = \frac{2}{n}\|Jd\|^2$, giving the exact quadratic step with nothing to tune. Near-exact step lengths are non-negotiable for conjugacy, so this is both required and free.
- **Trust gradient components down to $10^{-24}$ relative energy.** Deep in a run the true signal is a $10^{-12}$ fraction of the raw gradient. Any threshold that distrusts tiny projected gradients (bailing out when a projection removes 99.9999%) silently recreates the $10^{-10}$ stall.
- **Function space is resample-stable; parameter space is not.** Half-data solutions agree functionally at $\sim10^{-14}$ but their readout parameters differ at relative $O(1)$, connected by a flat valley. Optimizer state living in parameter space (memory bases, conjugacy directions) chases functionally meaningless displacements under batching. Prefer function-space quantities.
- **Safeguard value is regime-dependent.** The trust window was harmful full-batch and mildly useful under noise. Ablate in the target regime or the result is meaningless.
- **Discovery by exact-zero column test.** A parameter is in $L$ iff its Jacobian column does not move when $L$ is perturbed. The per-tensor sampled version costs $O(\#\text{tensors})$ probes, two passes each, and held 100% precision on a transformer, a 5-block ResNet, conv nets, and both geometry regimes, correctly refusing attention weights, layer-norm parameters, and earlier blocks.

## 6. What the record got wrong

Claims that were written down and later measured false. If you find one of these repeated somewhere, fix it.

- expD14's headline cost: the arm producing $10^{-16}$ costs 572 passes/step against Adam's 4.
- "The certificate is cheap": $2(2m{+}1)$ forwards, never counted.
- "Precision-agnostic" (expD14's solver): nine hardcoded fp64 constants, none derived from dtype.
- `batching_test.md` T1 ("baked-in whitening is poisoned by staleness"): corrected in place; stale block-QR tracks the moving floor at $0.5$-$1.5\times$ while steering is $180\times$ to $2\times10^{6}\times$ above it.
- expD09 round 6's double-double result read as a standalone existence proof.
- "Recall is free" in discovery: true at $10^{-3}$, false at $10^{-8}$ where losing 3% cost $47\times$.
- "Coverage degrades with depth": not supported, non-monotone across three families.
- Discovery ratios of 1.00 are against the best *tensor-aligned* set, which is not the best affine set.

## 7. Two nuances that look like contradictions

**Carried versus fresh residual.** Lesson 2 says recomputing the residual fresh each iteration caps convergence at $10^{-10}$, because the subtraction re-rolls $10^{-16}$ noise every step; carrying it freezes the noise into a one-time offset. Iteration 11 says the opposite: carry nothing, recompute fresh. Both are right in their own regime. Carrying is correct for an *iterative recurrence* where the noise compounds; recomputing is correct for a *one-shot direct solve*, which is unbothered by a slightly noisy right-hand side, and where a carried residual would feed the optimizer a stale linearization. Know which regime you are in.

**Under-solving.** The whole program exists to solve exactly, yet deliberately *under*-solving measurably improves the geometry a run learns (a geometry $16000\times$ better than its start on one target). These are not in conflict: precision is banked at the end, and mid-training the residual left unsolved is what keeps Adam's learning signal alive. The open question is the schedule, not the principle.

## 8. The pre-build checklist

Answer all of these in writing before implementing. If any answer is "unknown," the design is not ready.

1. **Passes.** How many forward and backward passes per step, per solve event, and once? Which are sequentially dependent and therefore unfusable? Does any count scale with $P$ or $n$?
2. **State.** How many $m$-sized arrays persist? Is that within Adam's $2m$ class? Does anything scale with the dataset?
3. **The $k/d$ ratio.** If the design has a memory or block parameter $k$, does it need $k/d$ fixed as $d$ grows $10\times$? If yes, it is dead.
4. **Reductions.** How many device-wide reductions or collectives per step? Krylov-based designs must state this.
5. **Exact or small?** If the design tests a quantity against zero, is that zero exact (bitwise) or an estimate with a noise floor? Estimators cannot detect exact zeros.
6. **Precision.** Does it run in bf16 and drive to bf16's floor? Are any constants hardcoded to fp64? Is any relative perturbation smaller than the dtype's unit roundoff?
7. **Control signals.** Does any decision compare loss values? Does any gauge act below its own noise floor? Is there an explicit guard?
8. **Kill list.** Is this on the list in section 5? If it resembles something there, state precisely why the measurement does not apply.
9. **Battle-tested alternative.** Is there a classical mechanism (LM ratio, EMA, momentum, line search) that does this job? If so, it wins the toss-up and must be the baseline.
10. **The litmus tests.** How will it be run on `dl_test`, on the batching grid, and at the full-batch floor?
11. **Falsification.** What measurement would show this is wrong, and is it cheap enough to run first?

## 9. Where the rest lives

`docs/INDEX.md` maps every surviving document. Start with `docs/ORIENTATION.md` for the current state and the open questions.
