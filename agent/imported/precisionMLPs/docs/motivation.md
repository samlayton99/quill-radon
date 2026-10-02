# Checkpoint D motivation: the lobotomy program

**Sam's framing, 2026-07-29. The north star for checkpoint D. Moved here from `results/` because `results/` is gitignored except `*_results.md` and this file was untracked.**

**Sam's framing, 2026-07-29. This is the design north star for the rest of checkpoint D.**

## The three-step program

1. **Step 1: a general-purpose optimizer.** Adam/AdamW for now (Muon or others later). The point is generality, not the specific choice. Trivial.
2. **Step 2: an iterative least-squares optimizer** that works very well and scales the way the general-purpose optimizer scales -- in model size, data size, batching, and error. ~85% solved (expD07/D08/D09 campaign).
3. **Step 3: lobotomize the general optimizer and stitch in the least-squares solver.** Every iteration updates all parameters: some governed by (modified) Adam, some by least-squares updates. Must be **as reliable as the general optimizer, monotonically better in all cases**, and solve the ideal QI-init cases to the floor in 1-D and 2-D. That is the goalpost.

Step 2 was manufacturing a custom part. Step 3 is the assembly work.

## Why this program exists

The QI failure analysis identified four requirements for trained machine precision: (1) correct $\gamma$ regime, (2) regularly spaced centers, (3) identical $\gamma$s, (4) the ability to solve a least-squares problem -- which Adam fails at (expD08 iter 11 Sec 3: gradient methods need $\sim(\sigma_1/\sigma_j)^2$ steps; $10^{22}$ for the directions machine precision needs).

We have shown Adam does not destroy a good geometry, and that an optimizer performing an exact least-squares solve finishes the problem (iter 11: all 30 cells at/below the direct-solve floor).

In higher dimensions the curse of dimensionality forbids *placing* centers with nice geometric properties. But if a general optimizer can *learn* the centers/hyperplanes while a least-squares solve runs alongside (increasingly as the geometry settles), barrier (4) is removed -- especially with initialization in the right $\gamma$ regime.

**Candidate application:** initialize a transformer FFN in the right regime; with large $\gamma$s and the residual stream as target, solve least-squares problems on the back half of the transformer.

## Calibration of the target (correcting an earlier framing)

- The target is **not** machine epsilon on production diffusion models. It is an optimizer for **MSE regression tasks generally** that is feasible across a whole range of regimes and can break past Adam's $\sim10^{-3}$ barrier where a least-squares structure exists. Machine-epsilon capability is the *capability*, useful for scaling laws and stability, not the deployment bar.
- $d$ can grow 10x beyond any current width; a method whose cost or memory needs $k/d$ fixed is not sustainable at scale. Scaling honesty is a hard requirement.
- The solve applies to a **subset** of parameters, repeatedly, inside training. Adam moves the geometry, but if Adam is tamed or annealed, the solves get more and more stationary (iter 11's settling regime shows this happening). Staleness is a cost curve, not an execution.

## Step-3 design questions (the open list)

- Which parameters go in the least-squares set $L$ (measurement, threshold, top-k, cap)? What signals -- curvature, relative error?
- How often to refresh $L$?
- How much step size / "energy" do $L$-steps get, and how does that anneal?
- How are non-$L$ ($A$) parameters updated? Zero their momentum?
- Graceful handoff when parameters move $A \leftrightarrow L$ (Adam moment handling)?
- When/how often to precondition the least-squares solve? What can be cached?
- How much memory $k$ does the solver get?
- Gradient/step accumulation before a solve?
- Interleaving: does $L$ run several solver iterations between Adam steps?

**Likely hyperparameters:** standard Adam's; solver memory budget $k$; $L$-selection (linearity sensitivity, cap, top-k); energy allocation; preconditioning schedule; $L$-refresh frequency.

**Prior art to mine:** expD08 iteration 11 (`iteration_11_results.md`) -- the certificate (linear + informed), the coupling law ($\|v\|\eta$ re-injection), min-norm as protection, the settling regime, the feature-pinning failure mode, and the two selection bugs.

## Where step 2 stands

expD09's recipe (`expD09_recipe_results.md`) solved the frozen-$\Phi$ subproblem at machine epsilon in fp64 at $O(d)$ state. Remaining hurdles before step 2 is closed: practicality of $O(d)$ vs the $k/d$ iteration law, the $O(d\cdot r)$ of SPIR, batched formation of the whitening (QR over batches), memory placement of the whitened operator $B$ (with $n\sim4d$, any materialized $n\times d$ array is $\sim4d^2$), and eliminating the contiguity requirement.
