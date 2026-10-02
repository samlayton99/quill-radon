# Checkpoint H: the open questions (Sam, 2026-08-29)

Written down to be pursued. Sam's framing; numbers and pointers added only where they exist.

## The eight questions

1. **The direction cliff in 2-D.** The 2-D ridge lattice does not resolve, then suddenly does, as the number of directions grows. Pinpoint it, recreate it, check it is reliable. Does gradualness matter in 2-D the way it did in 1-D (expH02)? Is there a radial (across-direction) equivalent of $\lambda$?
   - What exists: `results/checkpoint_H_highdim/expH04_mesh_finding/figures/directions_vs_radius_d2.png` (data `directions_vs_radius_d2.json`, script `experiments/expH04_mesh_finding/directions_vs_radius.py`). Fast concentric waves (task 2.3's function), data in a ball of radius $r$ about $(0.35,-0.25)$, offsets confined to the data's projection band, 48 per direction, one seed. On-data relative $L_2$ vs number of directions $M$: at $r=0.1$, $M=8\to12$ takes $10^{-7}\to3\times10^{-13}$; at $r=0.8$, $M=12\to16\to24$ takes $10^{-4}\to6\times10^{-9}\to5\times10^{-12}$. The same shape in 3-D (`directions_vs_radius_d3.png`, 32 per direction) but only to $10^{-10}$.
   - Reliability: one function, one seed, one offsets-per-direction count, error on the data ball only. Not yet checked: other functions, seeds, whether the cliff position depends on the per-direction count, whether it is a cliff or a steep slope on a finer $M$ grid, and gradualness across directions (even angles only so far).
2. **The optimal center distribution.** Given a function and a data distribution, what is the right center density along a direction? Known so far: gradualness is required (expH02; mesh map band-limited above about 12 gaps, `mesh_map_scale.png`), and high resolution in a data gap fails catastrophically (expG). The optimum itself has not been identified; theory should lead.
3. **The interaction of 1 and 2.** In a 2-D ridge lattice, can we be picky about which directions get ridges, and what is the correct way? How to allocate ridges between directions and along directions is open. Answers to 1 and 2 separately will inform this, but their interaction is a different problem.
4. **Global and local neurons together.** Some neurons carrying global structure, some local. Local structure matters but cannot be the only tool because of the curse of dimensionality. How would a single network do both?
5. **A theory of multiple hidden layers.** Depth is what makes real deep learning work. What could it give the theory? How to think geometrically about ridges transformed by a second layer is not understood.
6. **Gated MLPs (SwiGLU).** Possibly useful; what they would give is unknown.
7. **3-D to the floor for analytic functions.** An engineering problem, and a big result if done. Current state: even ridge lattice at $B=4096$ bottoms at $10^{-7}$ (waves) to $10^{-2}$ (spike, bursts) for every directions/centers split (`split_d3.png`); on a data ball of radius 0.15-0.3 the recentered lattice reaches $10^{-10}$ with 2k-8k units (`directions_vs_radius_d3.png`). Nothing at the floor on a genuinely 3-D target.
8. **Is the Radon / integral-over-$v$ picture the right theory?** It was the first idea. A cleaner way to represent a higher-dimensional QI may exist, and finding it is the whole goal.

## Bookmarked experiment (not run)

2-D data with a dense region, a sparse region, and a real hole; ridge lattice only; per direction an expH02-style mesh limited by the projected data; one ridge origin per region; each region scored against its own standalone ceiling (fit on that region's data alone); the hole scored separately. Tests: resolves where needed, resolves where the data allows, does not overload the hole, stays gradual. Decides whether the ridge construction can be sparse where there is no data.


## The program (Sam, 2026-08-31): trim the fat in 1-D..4-D before touching higher d

Recorded verbatim in intent, lightly ordered. Push one hidden layer first. The gifts we hold: gradualness along a direction (expH02) and free placement of extra spokes against an unmoving even background. Both together mean we can vary along $v$ and vary $v$ itself, and pack ridges more efficiently. We do not yet understand this well enough in low dimensions.

1. **Get 4-D to the floor if we can.**
2. **Trim the fat at uniform data**: fewer neurons for the same floor, using the function's frequency content to place $v$ (asymmetric functions especially).
3. **Trim the fat when the data varies**: frequency, $y$-noise (expected $\varepsilon_y/\sqrt{n}$ behavior), the data distribution; matching gradual 1-D meshes and $v$ placement to all three. The $M$-vs-$N$ tradeoff acquires an interaction term with $v$ placement: global spokes may get less $N$, learned atom directions more.

Only after the fat-trimming is understood in 1-D..4-D do we think about higher $d$. The larger agreed picture: directions must be learned; the hypothesis is that learned $v$ + 1-D resolution along it is more efficient than arbitrary neurons; and GPT's compositional-QI theory (whole 1-D QI constructions as reusable macro-blocks composed in a small graph) is the favored candidate for how depth attacks manifold and direction placement later.


### Amendments (Sam, 2026-08-31, second pass)

- Efficiency denominator (units saved at matched error vs the two-floor crossing) and identifiability tracking ($\|w\|$, rows-per-column, predicted noise floor $\|w\|\varepsilon_y/\sqrt n$): agreed, adopted. Seeds: prompt Sam when a claimed factor is small; do not get pedantic.
- Angular trimming IS in scope, against Claude's caution: the point is to move away from global even space-filling. Wanted: a better mechanism than the frozen even background -- ideally learned, e.g. a space-filling pull inside the loss (a repulsion/coverage regularizer on the directions), so data-fit and coverage trade off inside one training signal.
- Depth stays in the program and will eventually need its own explanation.
- expH03 (distribution matching) is the vehicle for the 1-D optimal-center-distribution question.

### Sam's moonshot guess (recorded 2026-08-31, expected to be refined)

Per layer: a set of learned unit directions $v$, with the biases along each direction ALSO learned but heavily regularized -- $\lambda$ held constant, jitter under control (the gradualness constraint as a penalty), a minimum block size around $N=12$. Allocation (how many blocks, how many units each) is suspected NOT to be gradient-learnable. Each layer built this way yields learned QI composition blocks, in a compressed format (a block shares one unit direction, ~12x fewer parameters than free neurons). One least-squares solve at the end. Training injected with efficiency principles: hierarchical resolution / residual staging, plus terms for noise, frequency, and data density.
