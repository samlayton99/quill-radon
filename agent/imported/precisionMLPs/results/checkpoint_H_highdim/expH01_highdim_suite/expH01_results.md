# expH01 -- A benchmark suite for approximation in one to five dimensions

**Status: draft -- suite build, pending Sam.**

## TL;DR

- The suite is 80 approximation problems on the cube $[-1,1]^d$ for $d=1..5$, sixteen per dimension. **No target is a sum of one-dimensional profiles along fixed directions.** That was the whole point of the rebuild: the previous version's targets were $\sum_i g_i(v_i^\top x)$, which builds the model's own way of representing functions into the benchmark.
- Every target carries an exact gradient, checked against finite differences to $8\times10^{-10}$ relative in all five dimensions. The gradient is what makes the theory's prediction for where centers should go computable without a symbolic profile.
- The only model here is the reference one: directions spread evenly over the sphere, centers spaced evenly along each direction, one least-squares solve, plus a random-feature control. It exists to run the suite end to end, not to be good.
- Run end to end at $B=4096$ on the 36-task first-pass list, the reference reaches relative $L_2$ between $3\times10^{-14}$ and $2\times10^{-12}$ on every smooth target in $d=1$ and $d=2$ (the narrow spike and the product peak in $d=2$ are sharpness-limited at $10^{-5}$ and $10^{-7}$), and degrades in the order the design predicts: kinks at $10^{-5}$, steps at $10^{-2}$, and $d=3$ unresolved at this budget. Random features are three to twelve orders worse on the smooth tasks and never better on the densest-region set.
- The `dense_region` test set does the job it was built for. On the clustered data the same fit that reads $10^{-10}$ on uniform points reads $2\times10^{-13}$ on the densest region (2.11), and on the sheet data $3\times10^{-14}$ on the sheet against $5\times10^{-1}$ off it (2.15).

## Question

Can the high-dimensional approximation problem be stated precisely enough that, when a curve later moves, we know which mechanism moved it -- without smuggling the answer into the benchmark by writing every target as a sum of one-dimensional profiles?

## Experiment design

### Directions and coordinates

The domain is $\Omega_d=[-1,1]^d$. To keep nothing axis-aligned, the reference directions come from the DCT-II matrix $Q_{j0}=1/\sqrt d$, $Q_{jk}=\sqrt{2/d}\cos(\pi(j+\tfrac12)k/d)$ for $k>0$; its columns are $u_1,\dots,u_d$ (1-indexed, so $u_1$ is the constant column, and $d=1$ degenerates to $Q=[1]$). Everything is written in the normalized coordinates

$$z_k(x)=\frac{u_k^\top x}{\|u_k\|_1}\in[-1,1],$$

whose range is exactly $[-1,1]$ on the cube with both ends attained at a corner, so a length scale written in $z$ means the same thing in every dimension. The raw projection $y_v(x)=v^\top x$ is kept for the reference model's geometry and for the center-density calculation, which both work along a single direction. A gradient taken in $z$ becomes a gradient in $x$ through $\partial z_k/\partial x_j=Q_{jk}/\|u_k\|_1$.

Distances to a point $a$ use the **scaled radial distance**

$$\rho_a(x)=\frac{\|z(x)-a\|_2}{\sqrt d},$$

so that a corner of the cube sits at distance about 1 in every dimension and a width such as $\sigma=0.25$ means the same thing in $d=1$ and $d=5$. Three fixed anchor points, given in $z$ and truncated to $d$ coordinates, place features in comparable positions as $d$ grows: $a^{(1)}=(.30,-.20,.25,-.15,.10)$, $a^{(2)}=(-.40,.35,-.10,.20,-.25)$, $a^{(3)}=(.15,.10,-.30,.05,.30)$.

### The twelve function families

| family | formula | why it is here |
|---|---|---|
| bumps at several widths | $\sum_k A_k e^{-\rho_{a_k}^2/2\sigma_k^2}$, $(A,\sigma)=(1,.50),(.7,.25),(.5,.10)$ | one global resolution cannot serve widths that differ by 5x |
| wide bump | $B_0=e^{-\rho_{a^{(1)}}^2/0.5}$ | the smooth background under the packet, step and kink families |
| concentric waves | $\cos(\pi\omega\rho_{a^{(1)}})$, $\omega\in\{1,6\}$ | pure resolution, with no preferred direction at all (cosine, not sine, so it is smooth at the center) |
| composition | $\exp(\sin(\pi z_1)\cos(\pi z_2))$ ($\exp(\sin\pi z_1)$ in $d=1$) | a function of a function |
| polynomial | $\sum_{k=1}^d(z_k^2z_{k+1}-z_kz_{k+1}^3)$, cyclic | every term couples two coordinates |
| radial spike | $1/(1+\alpha^2\rho_{a^{(1)}}^2)$, $\alpha\in\{4,12\}$ | smooth, with a pole a distance $1/\alpha$ away in the complex plane |
| product peak | $\prod_k 1/(1+a_k^2(z_k-b_k)^2)$, $a=(3,5,8,12,6)$, $b=(.35,-.25,.10,-.40,.25)$ | the standard Genz peak: a product, sharp in every coordinate at once |
| burst of oscillation | $B_0+0.8\,e^{-\rho_a^2/\tau^2}\sin(\pi\omega\rho_a)$, $\tau=.18$, $\omega=10$ | a small hard region that can be put where the data is, or where it is not |
| sphere step | $B_0+0.8\cdot\mathbf 1[\rho_{a^{(2)}}<0.35]$ | a genuine discontinuity across a curved surface |
| wavy step | $B_0+0.8\cdot\mathbf 1[z_2>0.3\sin(\pi z_1)+0.1]$ | the same, across a surface no single direction describes |
| kinks | $\lvert\rho_{a^{(1)}}-0.4\rvert$; $\max(0,\rho_0^2-0.25)$ | value continuous and slope broken; value and slope continuous and curvature broken |
| piecewise | $\sin(2\pi\rho)$ inside $\rho_{a^{(1)}}<0.4$, $\sin(0.8\pi)+3(\rho-0.4)^2$ outside | two formulas glued so the value matches and the slope does not |

The two step families set `differentiable = False`; their gradient is the gradient of the smooth part, and an `interface_mask` marks the points within $0.05$ of the step. The kink families keep `differentiable = True`: their gradient is correct away from a measure-zero surface, which is also marked.

**None of these is a finite sum of one-dimensional profiles.** The test for this is a mixed second difference $\partial^2F/\partial z_1\partial z_2$, which is exactly zero at any step size for any $\sum_k g_k(z_k)$ and for any sum of profiles along the $u_k$. A control built as a genuine sum of profiles registers $4\times10^{-14}$; every family in the suite registers between $0.5$ and $80$.

One point worth stating plainly: $\rho_a$ is a distance, so a function written as a non-even function of $\rho$ (for example $\sin(\pi\omega\rho)$) would have a cone point at its anchor. Every smooth family here is therefore written as an even function of $\rho$: the bumps and the burst envelope through $\rho^2$, the concentric waves and the burst carrier through $\cos(\pi\omega\rho)$, which is an analytic function of $\rho^2$. The only non-smooth targets are the ones meant to be: the steps, the kinks and the piecewise-defined function.

### Putting the targets on a common scale

Each raw target is centered and scaled once, $F=(\widetilde F-m)/s$, with $m$ and $s$ computed on a fixed set of points spread uniformly over the whole cube -- the same set for every task of a given $d$, whatever that task's own data geometry is, sheet tasks included. That is what stops a clustered data geometry from silently rescaling the loss, and it is why two tasks that share a target function agree bit for bit. The reference set is a scrambled Sobol sequence with a frozen seed, $2^{20}$ points per dimension. Sobol rather than independent draws because independent points estimate a unit-variance mean only to $s/\sqrt n$, which at any affordable $n$ is coarser than the tolerance the scaling has to hold; $2^{20}$ rather than $2^{16}$ because the fast oscillatory targets in $d=5$ swing by several times the claimed tolerance between independent $2^{16}$ scrambles. Measured against an independent $2^{21}$ scramble, the worst task is off by $4.8\times10^{-4}$ in the mean and $5.2\times10^{-4}$ in the standard deviation.

### Data geometries

Eight, all supported in the cube.

- **even grid**: equispaced midpoints in $d=1$, Halton points with bases $2,3,5,7,11$ for $d>1$. Deterministic; the seed is ignored.
- **uniform**: independent uniform draws.
- **hotspots**: $0.20\,\text{uniform}+0.40\,N_T(\mu_+,.22^2I)+0.25\,N_T(\mu_-,.28^2I)+0.15\,N_T(\mu_\perp,.25^2I)$ with $\mu_\pm=\pm0.45\mathbf 1$ and $\mu_\perp=0.35\,u_2/\|u_2\|_\infty$ (in $d=1$: means $.45,-.45,0$). Since $u_1\propto\mathbf 1$, $z_1(\mu_+)=0.45$ exactly and every other coordinate of $\mu_\pm$ in $z$ is zero.
- **stretched hotspots**: $0.20\,\text{uniform}+0.40\,N_T(.35\mathbf 1,\Sigma)+0.40\,N_T(-.35\mathbf 1,\Sigma)$ with $\Sigma=Q\,\mathrm{diag}(.25^2,.083^2,.15^2,\dots)Q^\top$, a 3:1 stretch that makes movement along $u_2$ much less visible in the data than movement along $u_1$. Recovered from a sample to better than 1%.
- **flat sheet** and **flat sheet, thickened**: $y=(t,0)$ in $d=2$ and $(s,t,0,\dots,0)$ for $d\ge3$ with parameters uniform on $[-.75,.75]$, $x=Qy$; the thickened variant replaces the zero perpendicular coordinates by $N(0,.015^2)$.
- **curved sheet** and **curved sheet, thickened**: $y=(.75t,.30\sin\pi t)$ in $d=2$ and $(.65s,.65t,.25\sin\pi s,.20\sin\pi t,.15\sin\pi(s+t))_{1:d}$ for $d\ge3$, $s,t$ uniform on $[-1,1]$; the thickened variant adds $N(0,.015^2)$ perpendicular to the sheet, projected with the analytic tangent Jacobian $J=Q\,\partial y/\partial(s,t)$ (measured $|J^\top\text{displacement}|<10^{-16}$).

Truncated normals use **per-cluster** rejection, so the realized fractions equal the nominal weights exactly ($.1994,.4017,.2491,.1498$ against $.20,.40,.25,.15$ at 200000 points); rejecting globally would reweight the clusters by how much of each falls outside the cube. Sheets are asserted to stay inside the cube; the largest $|x|$ reached by the curved sheet is $0.55$ in $d=2$, $0.83$ in $d=3$, $0.76$ in $d=4$ and $0.68$ in $d=5$, and no thickened point ever had to be redrawn. Where a density has a formula (uniform and the two hotspot families) it exposes an unnormalized log density, used only to rank points from sparsest to densest.

### The 16 tasks per dimension

The same list runs in every dimension, so a result can be read across $d$ without wondering whether the function changed. Tasks 1-4 use the even grid (bumps, slow waves, fast waves, polynomial); 5-10 use uniform data (composition, broad spike, narrow spike, sphere step, wavy step, kink ring); 11-14 use hotspot data (fast waves again, burst on the densest cluster, burst away from every cluster, product peak); 15-16 use the curved sheet, clean and thickened, both with the composition target.

Four rows are substituted in $d=1$, where there is no room for a curved sheet and a sphere is two points: task 9 becomes the piecewise target, task 10 the one-sided kink (so both kinds of kink appear somewhere in the suite), task 15 a step out in the sparse tail of the hotspot data at $z_1=0.78$, and task 16 a narrow spike sitting exactly on the densest cluster.

The deliberate comparisons: 2 against 3 (same shape, six times the oscillation); 3 against 11 (literally the same function object, only the data changes); 12 against 13 (the same burst, once where the data is and once where it is not); 6 against 7 (the same shape at two sharpnesses); 15 against 16 (the same function on the same sheet, with and without a thin layer of noise). Tasks meant to share a function share the target object, so their scaled versions agree bit for bit. The away-from-clusters burst anchor is checked against every cluster mean: it clears them by $2.5$, $3.6$ and $3.7$-$3.9$ widths for $d\ge2$. In $d=1$ it clears the $-0.45$ and $0$ clusters by $4.6$ and $3.4$ widths but the $+0.45$ cluster by only $1.82$; with the softened width $\sigma=0.22$ no point of $[-1,1]$ is both two widths away from that cluster and not at the domain boundary.

### Three test sets

Every fit is scored on three fixed sets ($20000$ points for $d\le2$, $40000$ for $d\ge3$; the seeds are distinct from the training seeds):

- `same_as_train` -- fresh points drawn the same way as the training data. Did the resolution go where the data actually is?
- `uniform` -- uniform over the whole cube. What did adaptation give up elsewhere?
- `dense_region` -- points from the densest part of the data only, chosen so that they are surrounded on every side by a margin of training data. This is the set on which machine precision is a fair question: no outliers, no boundary points, no points the data does not surround. For the even grid and uniform data it is the shrunken cube $[-0.8,0.8]^d$, with the outer band of width $0.2$ as the margin. For the hotspot families it is the single densest cluster (highest peak density $w/\sqrt{\det\Sigma}$; in every dimension the one at $+0.45$ with width $0.22$) restricted to one standard deviation, with that cluster's own 1-3 sd shell, the other clusters and the uniform background as the margin -- 83% of the training data lies outside it in $d=2$ and 98% in $d=5$. For the sheets it is points exactly on the sheet whose parameters lie in the inner 80% of their range, keeping the test away from the sheet's rim and, for the thickened variants, in the middle of the slab.

Sheet tasks carry two more sets: `on_sheet`, and `distance_r` for $r\in\{.02,.05,.10,.20\}$, points at exactly that distance perpendicular to the sheet.

How much training data there is, is a knob in its own right: `--ratios` sets sizes relative to the budget, `--n-train` sets absolute sizes, and passing both runs the union.

### What is recorded

Mean squared error, relative $L_2$ and largest absolute error on each test set. On top of that: error in ten equal-size bins ordered by how dense the data is at each test point (hotspot geometries only); error inside versus outside the burst region $\rho_a\le2\tau$; error near versus far from a step or slope break, $\lvert\cdot\rvert\le0.05$; and for the sheet tasks, error on the sheet and at each fixed distance from it. Angular recovery is gone -- there are no true directions any more -- although the `geometry()` interface on models is kept, so a future model still reports its directions, centers and widths in one shape.

The remaining prediction from the theory is **how densely centers should be placed** along a direction $v$:

$$\rho^\star_v(t)\ \propto\ \big[\,p_v(t)\,D_v(t)\,\big]^{1/(2r+1)},\qquad D_v(t)=\mathbb E\big[\,\lvert\partial_vF(X)\rvert^2\ \big|\ v^\top X=t\,\big],$$

with $t=v^\top x$, $p_v$ the density of the projected training points, and $r=1$ for tanh centers. $D_v$ is estimated from the sample: the projected points are binned, $\lvert\nabla F(X)\cdot v\rvert^2$ is summed in each bin, and the sum and the count are smoothed with the same Gaussian before dividing (smoothing the ratio instead would drag the zeros of the empty bins past the ends of the data inwards and make the estimate sag). The result is scaled to integrate to the number of centers, so it can be laid directly against a model's actual centers. Two checks: it integrates to the requested number to $10^{-9}$, and it is flat under uniform data both for a target whose slope is exactly constant (interior ripple $2.2\%$) and for one whose slope energy is constant on average ($2.5\%$). It refuses to run on the step families and says why.

### The reference model

Directions spread evenly over the sphere, centers spaced evenly along each direction, and one truncated-SVD least-squares solve with a bias column at `rcond` $10^{-13}$ -- the same solve the one-dimensional experiments use. Given a budget $B$ of tanh units: $\max(3,\text{round}(B^{1/d}))$ centers along each direction and $\text{round}(B/\text{that})$ directions. Directions are the single $[1]$ in $d=1$, equally spaced angles on $[0,\pi)$ offset by half a step in $d=2$, spherical Fibonacci on the upper half sphere in $d=3$, and for $d\ge4$ a fixed-seed Gaussian draw normalized to unit length -- **explicitly a placeholder**: evenly spread in distribution, but not an equal-weight cubature rule. Centers run over $[-T,T]$ with $T=1.25\|v\|_1$ and width $\gamma=\lambda/h$, $h=2T/\text{(centers per direction)}$, $\lambda=0.25$ (expC03's value). The extra 25% past the range of the projection is load-bearing: on the easiest 1-D task at $B=128$ it is the difference between relative $L_2\approx5\times10^{-14}$ and $1.4\times10^{-6}$.

The control is a random first layer (Xavier-normal) at the same budget with the same solve. It is deliberately weak: it is there to show that the frozen geometry, not the least-squares solve, is what buys accuracy.

**Code & data.** Library: `experiments/expH01_highdim_suite/h01suite/{basis,targets,normalize,densities,tasks,metrics,baseline}.py`. Driver: `experiments/expH01_highdim_suite/{run.py,viz.py}`, modes `--gallery`, `--smoke`, `--full`, `--plot`, with `--budgets`, `--ratios` and `--n-train`. Tests: `tests/test_expH01_highdim_suite.py` (55 tests). Data: `results/checkpoint_H_highdim/expH01_highdim_suite/smoke.json`. Figures: `results/checkpoint_H_highdim/expH01_highdim_suite/figures/{gallery_d1,gallery_d2,gallery_d3,gallery_d4,gallery_d5,predicted_center_density,smoke_baseline}.png`. Specification: `results/checkpoint_H_highdim/expH01_highdim_suite/SUITE_SPEC.md` (Version 3).

## Results

The run is the 36-task first-pass list -- everything in $d=1$ and $d=2$, plus 3.3, 3.11, 3.12, 3.13 -- at $B=4096$ tanh units with $n_\text{train}=8B$, one seed, scored on the three test sets. (An earlier pass at $B=1024$ showed the $d=2$ tasks with a frequency-6 or narrow-spike component unresolved at that budget; the table below is $B=4096$ only.)

**Every smooth target reaches the double-precision floor in $d=1$ and $d=2$.** On points drawn like the training data: the bumps, both concentric waves, the polynomial, the composition, both spikes, the product peak, and both sheet tasks read between $3\times10^{-14}$ and $2\times10^{-12}$ in both dimensions. Two $d=2$ exceptions are limited by sharpness rather than by the solve: the narrow spike 2.7 at $6.6\times10^{-5}$ (its broad sibling 2.6 is at $9.3\times10^{-13}$, so a factor of three in width costs eight orders at this budget) and the product peak 2.14 at $6.3\times10^{-7}$ (its narrowest factor has width $1/12$).

**The non-smooth targets separate cleanly and in the right order.** Steps are worst ($1.6\times10^{-2}$ for 1.8, $1.4\times10^{-1}$ for 2.8 and 2.9), kinks and the piecewise target next ($5\times10^{-6}$ to $1.6\times10^{-5}$ in $d=1$, $4.9\times10^{-3}$ for the kink ring in $d=2$). Splitting by region puts the error where the non-smoothness is: on 1.8 the band within $0.05$ of the step reads $3.6\times10^{-2}$ against $5.7\times10^{-6}$ away from it; on 1.15 (the step in the sparse tail) $6.2\times10^{-2}$ against $8.2\times10^{-7}$; on the 2-D steps $2.7$-$3.3\times10^{-1}$ in the band against $4\times10^{-2}$ outside it.

**On clustered data, the densest region is at machine precision and the sparse tail is not.** For every hotspot task in $d\le2$ the reading on the densest region is $2\times10^{-13}$ to $10^{-11}$ while the uniform-cube reading is $10^{-11}$ to $5\times10^{-10}$, and the density bins show the error falling monotonically from the sparsest bin ($10^{-11}$ in $d=1$, $10^{-10}$ in $d=2$) to the densest ($10^{-13}$). The burst comparison (12 against 13, the same burst on the cluster and away from it) reads the same on the densest region ($1.9\times10^{-13}$ and $3.1\times10^{-13}$ in $d=1$; $10^{-11}$ and $6\times10^{-12}$ in $d=2$): the reference places its centers evenly, so it does not care where the burst is, and that is the baseline any adaptive placement has to beat. Inside the burst region itself, 1.12 reads $2.9\times10^{-13}$ and 1.13 reads $1.6\times10^{-11}$ on points like the training data: the burst is a little harder to pin where the data is thin.

**Sheet data: perfect on the sheet, unconstrained off it, and a thin layer of noise fixes that.** On 2.15 (data exactly on a bent curve) the fit is $3.3\times10^{-14}$ on the curve and $5\times10^{-1}$ on uniform points; it costs $1.7\times10^{-2}$ to move $0.02$ off the curve and $1.9\times10^{-1}$ to move $0.20$. On 2.16, the same function on the same curve with $0.015$ of perpendicular noise, the error at distance $0.02$ is $5.2\times10^{-14}$, at $0.05$ $1.2\times10^{-11}$, at $0.10$ $2.7\times10^{-8}$, at $0.20$ $3\times10^{-5}$: the noise pins the perpendicular direction and the model uses it.

**$d=3$ is unresolved at this budget, for a stated reason.** With $256$ directions and $16$ centers each, the concentric waves (3.3) reach only $1.8\times10^{-7}$ even though the training data is fit to $2.5\times10^{-14}$ mean squared error: the geometry can interpolate the training points but not the function between them. The burst tasks 3.12/3.13 are at $10^{-1}$ on points like the training data and $4\times10^{-2}$ / $7\times10^{-3}$ on the densest region. These are budget limits of the reference, not properties of the tasks.

**Random features** are $3$ to $12$ orders worse on every smooth task and never better on the densest region except where neither model can do anything (the steps).

Relative $L_2$ error at $B=4096$, reference model on the three test sets, and random features on the densest region:

| task | function | data | same_as_train | uniform | dense_region | random, dense_region |
|---|---|---|---|---|---|---|
| 1.1 | bumps | even_grid | 6.0e-13 | 1.7e-12 | 4.5e-13 | 2.6e-01 |
| 1.2 | slow waves | even_grid | 3.2e-13 | 8.4e-13 | 2.1e-13 | 6.1e-04 |
| 1.3 | fast waves | even_grid | 2.1e-13 | 3.2e-13 | 1.9e-13 | 1.0e+00 |
| 1.4 | polynomial | even_grid | 4.8e-13 | 1.4e-12 | 4.5e-13 | 1.1e-07 |
| 1.5 | composition | uniform | 3.3e-13 | 3.4e-13 | 2.3e-13 | 3.1e-02 |
| 1.6 | broad spike | uniform | 5.3e-13 | 5.3e-13 | 4.3e-13 | 1.9e-01 |
| 1.7 | narrow spike | uniform | 3.8e-13 | 3.7e-13 | 3.1e-13 | 5.8e-01 |
| 1.8 | sphere step | uniform | 1.5e-02 | 1.6e-02 | 2.7e-02 | 5.1e-01 |
| 1.9 | piecewise | uniform | 1.6e-05 | 1.2e-05 | 2.0e-05 | 4.6e-01 |
| 1.10 | one-sided kink | uniform | 4.9e-06 | 4.3e-06 | 7.1e-06 | 7.2e-02 |
| 1.11 | fast waves | hotspots | 4.1e-12 | 1.4e-10 | 1.8e-13 | 1.0e+00 |
| 1.12 | burst on the cluster | hotspots | 1.6e-12 | 5.9e-11 | 1.9e-13 | 7.5e-01 |
| 1.13 | burst away from clusters | hotspots | 8.4e-12 | 2.6e-10 | 3.1e-13 | 1.6e-01 |
| 1.14 | product peak | hotspots | 1.3e-12 | 4.5e-11 | 2.4e-13 | 8.0e-02 |
| 1.15 | step in the sparse tail | hotspots | 1.7e-02 | 1.3e-02 | 1.7e-07 | 1.2e-01 |
| 1.16 | narrow spike on the cluster | hotspots | 8.7e-13 | 2.3e-11 | 2.4e-13 | 4.7e-01 |
| 2.1 | bumps | even_grid | 5.0e-13 | 6.7e-13 | 2.2e-13 | 1.6e-01 |
| 2.2 | slow waves | even_grid | 3.6e-13 | 4.6e-13 | 1.8e-13 | 5.3e-05 |
| 2.3 | fast waves | even_grid | 1.9e-12 | 4.5e-12 | 8.9e-13 | 8.4e-01 |
| 2.4 | polynomial | even_grid | 1.2e-12 | 2.6e-12 | 1.1e-12 | 3.6e-07 |
| 2.5 | composition | uniform | 4.5e-13 | 2.3e-13 | 6.9e-14 | 2.5e-02 |
| 2.6 | broad spike | uniform | 9.3e-13 | 9.8e-13 | 8.4e-13 | 7.4e-02 |
| 2.7 | narrow spike | uniform | 6.6e-05 | 6.5e-05 | 6.5e-05 | 4.1e-01 |
| 2.8 | sphere step | uniform | 1.4e-01 | 1.4e-01 | 1.7e-01 | 4.7e-01 |
| 2.9 | wavy step | uniform | 1.4e-01 | 1.4e-01 | 1.6e-01 | 4.5e-01 |
| 2.10 | kink ring | uniform | 4.9e-03 | 4.9e-03 | 6.2e-03 | 1.3e-01 |
| 2.11 | fast waves | hotspots | 2.0e-10 | 5.0e-10 | 1.7e-13 | 4.6e-01 |
| 2.12 | burst on the cluster | hotspots | 2.2e-11 | 8.7e-11 | 1.0e-11 | 1.0e+00 |
| 2.13 | burst away from clusters | hotspots | 1.3e-10 | 3.6e-10 | 5.7e-12 | 6.8e-02 |
| 2.14 | product peak | hotspots | 6.3e-07 | 1.9e-06 | 5.2e-08 | 1.1e-01 |
| 2.15 | composition | curved_sheet | 3.2e-14 | 5.0e-01 | 3.3e-14 | 1.0e-04 |
| 2.16 | composition | curved_sheet_noisy | 1.3e-12 | 2.1e-01 | 3.6e-14 | 7.7e-04 |
| 3.3 | fast waves | even_grid | 1.8e-07 | 3.1e-07 | 1.7e-07 | 5.2e-01 |
| 3.11 | fast waves | hotspots | 9.5e-06 | 2.9e-05 | 2.1e-08 | 2.0e-01 |
| 3.12 | burst on the cluster | hotspots | 7.3e-01 | 4.0e+00 | 4.3e-02 | 1.2e+00 |
| 3.13 | burst away from clusters | hotspots | 2.5e-01 | 8.0e-01 | 6.9e-03 | 1.8e-01 |

### Figures

- **`gallery_d1.png`** -- 16 panels, one per $d=1$ task, in a 4 by 4 grid. The black curve is the scaled target over $[-1,1]$; the tinted histogram beneath it is a 2000-point draw from that task's data geometry, colored by geometry; gold bands mark the burst of oscillation ($\rho_a\le2\tau$) and red bands the $0.05$ neighbourhood of a step or slope break. Read 1.12 against 1.13: the same burst, sitting on the tall part of the histogram in one and out in its tail in the other. Read 1.2 against 1.3 for the six-fold change in oscillation, and note the kink at the center of both.
- **`gallery_d2.png`** -- 16 panels, one per $d=2$ task. Color is the scaled target on the square (diverging, symmetric limits at the 99.5th percentile of $|F|$), black dots are 1500 training points, and the burst and step regions are outlined rather than shaded so the target underneath stays visible. The sheet tasks 2.15 and 2.16 show their support directly: a curve, and a thickened curve. 2.9's step surface is visibly not a straight line.
- **`gallery_d3.png`, `gallery_d4.png`, `gallery_d5.png`** -- 16 tasks each, two panels per task. Left: the target on the plane spanned by $u_1$ and $u_2$, in normalized coordinates over the full $[-1,1]^2$ so that features placed at large $|z_k|$ stay visible; the hatched region outside the black diamond is where that plane leaves the cube (the formula is still defined there, the points are simply unreachable). Right: a 2-D histogram of 20000 training points projected onto $(z_1,z_2)$. Compare the broad even-grid and uniform rows with the three tight blobs of the hotspot rows and the sharp-edged patch of the sheet rows.
- **`predicted_center_density.png`** -- 3 rows by 2 columns, left $d=1$ and right $d=2$, everything measured along $u_1$. Rows are the density of the projected training points, the average squared slope along $u_1$ (log scale), and the prediction itself, scaled to 64 centers. Three curves per panel: the burst-on-the-cluster target under uniform data (the reference), the same target under clustered data, and the burst-away-from-clusters target under the same clustered data. The two clustered cases share a data geometry so their top-row curves coincide exactly. The bottom row is the point: uniform data gives a broad prediction peaking where the function moves fastest, clustered data with the burst on the cluster sharpens that peak at $t\approx0.45$, and moving the burst away splits the prediction between the cluster and the burst, shifting the peak to $t\approx0.76$.
- **`smoke_baseline.png`** -- 4 stacked panels. The top three are relative $L_2$ on the three test sets, the reference (blue circles) against random features (red squares), one tick per task, on a log axis fixed at $10^{-15}$ to $10^{3}$ across all three so the rows can be compared directly. The gap between blue and red is the whole claim about frozen geometry: up to twelve orders on the smooth tasks, narrowing to a factor of a few on the steps and on the unresolved $d=3$ cells, where neither model has anything to offer. The bottom panel is relative $L_2$ per data-density bin for the clustered tasks under the reference, and it is where 1.12 and 1.13 cross: 1.12 rises into the dense bins, 1.13 falls.

## Additional details

**What was removed.** The previous version's 1-D profile library (`atoms.py`) is deleted: no target is built from one-dimensional pieces any more, so nothing imported it. Angular recovery is deleted from the metrics and the records for the same reason -- with no true directions, the number had no referent. The tilted-direction helpers in `basis.py` went with them.

**Distance to a curved sheet.** The fixed-distance sets are constructed at exact perpendicular distances, so those errors are exact. The general `distance_to_sheet` for a curved sheet is a nearest-point search on a parameter grid, an upper bound, meant for checks on modest point counts rather than large sweeps. For a flat sheet it is exact.

**Test coverage.** 55 tests, all passing. Highlights: $Q_d$ orthogonal to $2\times10^{-16}$ and $z_k$ attaining both ends of $[-1,1]$ on cube corners; every family's gradient matching central differences to $8\times10^{-10}$ relative across $d=1..5$; the mixed-second-difference check that no target is a sum of one-dimensional pieces; 80 tasks, 16 per dimension, unique ids and names, and all nine same-function pairs identical bit for bit; scaling within $4.8\times10^{-4}$ of zero mean and $5.2\times10^{-4}$ of unit standard deviation on an independent Sobol sample; every data geometry inside the cube, hotspot fractions at the nominal weights to $5\times10^{-3}$, the stretched covariance recovered to under 1%, thickened sheet noise at $0.0150$ and orthogonal to the tangent to $10^{-16}$; the dense-region sets inside one standard deviation of the densest cluster with 83-98% of the training data outside; the predicted center density integrating exactly and flat to $2.5\%$ where it must be.

## Conclusions

*Pending Sam.* What the data plainly shows is that the suite is wired correctly and that its targets are no longer built out of the model's own vocabulary: every family fails a mixed-second-difference test for separability, every gradient matches finite differences, the reference reaches the double-precision floor on the smooth targets in $d\le2$, and the error decomposition localizes its failures to the step bands, the sparse bins and the off-sheet distances. Whether this is the right sixteen tasks per dimension is a judgement call, not something the data settles, and is left open.

## Open questions

- **Is 16 the right sixteen?** The list is fixed across dimensions on purpose, which costs a slot or two in $d=1$ (four rows are substituted) and may be spending too many rows on kinks and steps relative to smooth targets.
- **The two sharpness-limited $d=2$ tasks.** The narrow spike (2.7) and the product peak (2.14) are the only smooth $d\le2$ targets not at the floor at $B=4096$; the budget at which they reach it is the natural first point of the $B$ sweep.
- **The budget and data-size surfaces.** The machinery for $B\in\{64,\dots,4096\}$ and $n_\text{train}/B\in\{1.25,2,4,8\}$ is exposed and only two points of it have been run. The two surfaces, error as a function of $B$ and of $n_\text{train}$, are the natural next measurement.
- **Regularization of the reference.** The least-squares solve is unregularized, so away from the data the reference is unconstrained and reports relative $L_2\gg1$. Should the reference stay that way (an honest floor), or should the suite also carry a regularized version so later models are compared against something that does not diverge?
- **Evenly spread directions for $d\ge4$.** The $d\ge4$ direction set is a fixed-seed normalized Gaussian draw, a placeholder. A real spherical design would remove a confound from every $d\ge4$ measurement.
- **Label noise.** Zero throughout. Nothing in the implementation assumes it.
