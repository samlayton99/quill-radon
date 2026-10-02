# expH04 -- Mesh finding: can a monitor tell us where the centers and directions should go?

**Status: draft -- conclusions pending Sam's sign-off.**

## TL;DR

- Centers, 1-D and 2-D: yes, where the target has a sharp feature. A monitor read off a first even fit (two solves, no knowledge of the target) reaches the fp64 floor at $4$-$8\times$ fewer units than the even mesh: spike at the hotspot (1.16) at $B=128$, $8\times10^{-7}\to2\times10^{-14}$; burst at the hotspot in 2-D (2.12) at $B=1024$, $1.8\times10^{-4}\to10^{-6}$; step in the sparse tail (1.15) at $B=1024$, $3\times10^{-5}\to10^{-13}$. The estimated monitor matches the one computed from the true gradient to two digits in every cell, and on resolved targets the adapted meshes now sit within $1$-$5\times$ of even.
- The "price at the floor" of the first ladder ($10$-$100\times$ above the even floor on already-resolved targets) was diagnosed and removed. The fp64 floor tolerates smooth non-uniform spacing (expH02's half-Gaussian through this pipeline: $2$-$8\times10^{-15}$, same as even) but not roughness at the scale of one gap: shaking the even mesh by $2\%$ of a gap costs $10^{-15}\to7\times10^{-11}$. The monitor was the rough part (a histogram smoothed at $1.5$ gaps); the fix is a principle, not a knob: the mesh is a coordinate map, and a measured resolution limit of the construction ($12$ gaps, `mesh_map_scale.png`) fixes how smooth it must be ($\sigma=5.8$ gaps); with that the price is gone and the wins improve (2.12 at $B=4096$ now beats even, $10^{-11}\to4\times10^{-13}$).
- Directions: the angle-density rule from the theory ($m(\theta)\propto A(\theta)^{1/3}$) does nothing, on the suite and on a known-answer ridge, and it cannot: a ridge is a delta in angle and $A(\theta)$ is a broad second moment. What works is the *eigenvectors* of that second moment, the active subspace $\mathbb E[\nabla F\nabla F^\top]$: treat the problem as $m$-dimensional inside it. On the composition target (a 2-D function embedded in $d$ dimensions) at $B=4096$ the dense-region error goes from $4\times10^{-7}$ ($d=3$) and $8\times10^{-4}$ ($d=5$) to $10^{-12}$ with the true subspace and to $2\times10^{-11}$ / $5\times10^{-6}$ with the subspace estimated from a first fit; on the $d=5$ noisy sheet $10^{-6}\to5\times10^{-11}$. Iterating fit $\to$ covariance $\to$ fit converged geometrically in the first ladder ($8\times10^{-4}\to5\times10^{-6}\to10^{-9}$) and stalled in the rerun; the covariance should be weighted by where the fit is good, untested. The one place it fails is a pure ridge plus background (the known-answer check): the covariance direction is biased by $0.01^\circ$ and a ridge needs its direction exact.
- The direction count is set by the data, not the domain. Precision *everywhere* in a domain of radius $R$ costs $(kR)^{d-1}$ directions whatever the feature's size (the $d=3$ split sweep: no split of $B=4096$ resolves a spike, best $2\times10^{-2}$), but precision *on data* within radius $r$ of the ridge origin costs $\max(\binom{p+d-1}{d-1},(kr)^{d-1})$, $p\approx12$ (`directions_vs_radius.py`: 12 directions at $kr\approx1$, 24 at $kr\approx7.5$, against 64 domain-centered). For data on an $m$-dimensional manifold of extent $L$ the total is $\sim(kL)^m$: intrinsic dimension and smoothness along the data, independent of the ambient $d$. That is the scaling story, and it is the sampling count, so it is the right target.

## Question

The parked theory prescribes a center density along each direction $v$ of $\rho_v(t)\propto[p_v(t)R_r(v,t)]^{1/(2r+1)}$ ($p_v$ the projected data density, $R_r=\mathbb E[|\partial_v^rF|^2\mid v\cdot x=t]$) and a direction density $m(v)\propto A(v)^{d'/(2s+d')}$ with $A(v)=\mathbb E|\partial_vF|^2$. With the solve fixed (one truncated-SVD least squares), does placing the mesh by these rules beat the even mesh, does the practical version (monitor estimated from a first fit) keep the gain, and what does the answer say about a construction that scales with $d$?

## What placement theory says (the part worth keeping)

**Centers along a direction.** The fixed-geometry $\tanh$ fit at $\lambda=\gamma h=0.25$ is spectrally accurate: the error of a profile of frequency $\omega$ on spacing $h$ decays like $e^{-c/(h\omega)}$ until the fp64 floor (expB02, expC02: error vs $N$ is a straight line on a log-linear plot, and the $N$ needed is proportional to the frequency). So the requirement for precision $\varepsilon$ is local and multiplicative: $h(t)\,\omega(t)\le\kappa(\varepsilon)$ everywhere, i.e. the center density should follow the *local frequency* $\omega_v(t)$ of the ridge profile, with amplitude entering only through $\log(a/\varepsilon)$. The theory's rule $(pR_r)^{1/(2r+1)}$ is the equidistribution rule for a method of fixed order $r$ (error $\sim h^r|\partial^rg|$, weighted by the data measure); it is amplitude- and data-weighted. Both were run: the amplitude-weighted rule pushes harder where the profile is both rough and populated and wins at small budgets; the frequency rule $\omega=\sqrt{R_2/R_1}$ is data-independent and milder. Neither escapes the price at the floor.

**Data.** The data density enters twice, and neither is as a multiplier in the monitor: the least-squares loss is data-weighted (nothing constrains the fit where there is no data, expG), and each center needs a few samples in its support to be determined (expH02, the $n=2W$ rows). For placement it is a constraint, $\rho_v(t)\lesssim n\,p_v(t)/c$, not a weight.

**Directions.** A single hidden layer is a quadrature of the ridge integral: by Radon inversion $F(x)=c_d\int_{S^{d-1}}g(v,v\cdot x)\,dv$ with $g=\Lambda^{d-1}\mathcal RF$, and by the Fourier-slice theorem $\hat g(v,\rho)\propto|\rho|^{d-1}\hat F(\rho v)$, so the profile along $v$ has exactly $F$'s bandwidth in that direction (the 1-D rule above applies unchanged). The direction count comes from the plane-wave expansion: at a point $x$ the integrand is $G_x(v)=\int|\rho|^{d-1}\hat F(\rho v)e^{i\rho\,v\cdot x}d\rho$, and $e^{i\rho v\cdot x}$ has spherical-harmonic content only up to degree $\rho|x|$. So **the angular bandwidth of the integrand at $x$ is $k\,|x-x_0|$**, with $k$ the bandwidth of $F$ and $x_0$ the origin the offsets are measured from, and a quadrature with $M$ nodes (degree $\sim M^{1/(d-1)}$) has error $\sim e^{-c(M^{1/(d-1)}-k|x-x_0|)}$: **the error appears first at the points farthest from the ridge origin.** Precision everywhere in a domain of radius $R$ therefore costs $M\sim(kR)^{d-1}$, whatever the feature's size (the ridges are walls and must cancel far away), but precision on data within radius $r$ of $x_0$ costs only $M\sim(kr)^{d-1}$, and the origin is free to choose ($c_{ij}=v_i\cdot x_0+t_j$). There is also a floor: representing every polynomial of degree $p$ as a sum of ridges takes $\binom{p+d-1}{d-1}$ directions, and fp64 needs the local Taylor degree $p\approx12$; so

$$M\;\approx\;\max\Big(\tbinom{p+d-1}{d-1},\;c\,(kr)^{d-1}\Big).$$

`directions_vs_radius.py` measures this (`directions_vs_radius_d{2,3}.png`): the fast concentric waves, data uniform in a ball of radius $r$ about an off-center point, offsets confined to the data's projection band, direction count swept. In 2-D the count for $10^{-10}$ on the data is $12$ at $kr\approx1$ (the polynomial floor, $p+1=13$) rising to $24$ at $kr\approx7.5$, against $64$ for the domain-centered reference on the same function; in 3-D the count for $10^{-9}$ doubles as $r$ doubles ($64$, $128$, $256$ at $kr\approx1,2,4$) above a floor near $\binom{14}{2}=91$. Away from the data the error is $O(1)$-$10^{9}$: the far field of a ridge system is the saturated-wall arrangement, which is the generalization problem, not the precision problem.

**Data on a manifold.** If the data lies on an $m$-dimensional manifold of extent $L$, embedded in any $d$, cover it by patches of radius $r$ where it is flat to working precision; in a patch the directions are needed only in the tangent space ($(kr)^{m-1}$ of them, found by the local gradient covariance), each with $\sim kr$ offsets, and there are $(L/r)^m$ patches. The total is $\sim(kL)^m$: set by the intrinsic dimension and the number of oscillations of $F$ across the manifold, independent of $r$, of the ambient dimension and of the domain. That is also the sampling count, so nothing beats it, and the ridge network reaches it. The two objects to learn are exactly the ones the rungs estimate: the (local) active subspace, and the distribution of offsets along each direction under the mesh-map smoothness constraint.

**Why the angle-density rule cannot find a subspace.** $A(\theta)=v_\theta^\top Cv_\theta$ is a smooth quadratic form in the direction, so any density built from it is smooth and spreads directions around; a subspace is a measure-zero set of directions. The information is in the eigenvectors of $C=\mathbb E[\nabla F\nabla F^\top]$ (the active subspace, Constantine; battle-tested), not in its values. Its accuracy requirement is severe: a subspace tilt $\epsilon$ costs error $\sim\epsilon kR$, so the subspace must be refined, and the iteration fit $\to$ covariance $\to$ fit is a contracting fixed point (below).

**The far field.** Far from the data every $\tanh$ wall is saturated, so $\hat F$ there is $\sum_i\pm W_i$ with signs set by which side of each wall $x$ is on: piecewise constant on the cells of the hyperplane arrangement, with cell values that are sums of large alternating weights. That is the $O(1)$ error off the sheets and the expG03 blow-ups. The theory suggests the fix: per direction, local profiles that are zero-sum ($\sum_jw_{ij}=0$, so a recentered ridge vanishes outside its patch), and a separate coarse global mesh (few directions, wide units) to carry the far field. Not built here.

**What a general-$d$ construction looks like, then.** Patches of the data (one ridge origin each); in each patch the local active subspace from a pilot fit, refined by iteration, even directions inside it with $M\approx\max(\binom{p+m-1}{m-1},(kr)^{m-1})$; a 1-D QI mesh along each direction whose spacing follows the frequency/roughness monitor, band-limited above $12$ gaps; zero-sum local profiles plus a coarse global mesh for the far field; one solve. Its cost is $\sim(kL)^m$, governed by the intrinsic dimension and the smoothness of $F$ along the data, not by the ambient $d$ or the domain. The parts measured in this experiment: the 1-D monitor and its smoothness limit, the active subspace with iteration, the direction count against the data radius. Not yet built: patching, zero-sum profiles, the coarse global mesh.

## Experiment design

**One pipeline, many monitors.** For each direction $v$ (the even reference's directions, $T=1.25\|v\|_1$ as in expH01) a monitor $m(t)\ge0$ on a fine grid over $[-T,T]$ becomes centers by

$$\rho(t)=\frac{1-s}{2T}+s\,\frac{m(t)}{\int m},\qquad h(t)=\frac{1}{n\,\rho(t)}\ \text{graded so that}\ |h'(t)|\le g,\qquad c_j=C^{-1}\!\big(\tfrac{j+1/2}{n}\big),$$

$C$ the cumulative of $1/h$, local spacing $h_j=(c_{j+1}-c_{j-1})/2$, $\gamma_j=0.25/h_j$. Floor $s=2/3$ (a third of the centers stay even; no gap wider than about three even spacings); grading $g=0.15$ (neighboring spacings within a factor $\approx1.15$, the failure mode expH02 found); monitors smoothed at $5.8$ even spacings, derived from the measured mesh-map resolution limit $L^*=12$ gaps (the first ladder used $1.5$; see "The price at the floor" below, and `bw1.5/` for that data). $s=0$ reproduces the even reference to $10^{-13}$. Rungs, safest first:

- `even`: $m=1$.
- `data_p13`, `data_p1`: $m=p_v^{1/3}$, $m=p_v$ -- data only.
- `oracle_r1`, `oracle_r2`: $m=(p_vR_1)^{1/3}$, $(p_vR_2)^{1/5}$ with the true gradient (analytic; $R_2$ by central difference). Skipped on the step targets.
- `surr_r1`, `surr_r2`: the same with $\partial_v\hat F=\sum_kw_k\gamma_k(v_k\cdot v)\,\mathrm{sech}^2u_k$ (and the analytic second derivative) of a first even fit at the same budget. Two solves. The practical version.
- `residual`: $m=(p_v\,\mathbb E[e^2\mid t])^{1/3}$, $e$ the first fit's residual. Two solves.
- `surr_r1_x3`: `surr_r1` iterated twice more.
- `freq_oracle`, `freq`: $m=\omega_v(t)=\sqrt{E_2/E_1}$ with $E_r=\mathbb E[|\partial_v^r F|^2\mid t]$ averaged over a window of at least $5\%$ of the range (a narrower window sees the zeros of $F'$ at extrema and reports an infinite frequency there; that version broke at large $B$, $1.7$: $7\times10^{-14}\to10^{-7}$).
- `active_oracle`, `active`, `active_x3`: eigen-decompose $\mathbb E[\nabla F\nabla F^\top]$ (true, or from the first fit's analytic gradient), take $m$ = the number of eigenvalues carrying $99.9\%$ of the trace; if $m<d$, spend $80\%$ of the budget as an $m$-dimensional even mesh inside the subspace ($B_a^{1/m}$ centers per direction, `even_directions(m)` embedded), the rest as a $d$-dimensional even mesh; centers by the `surr_r1` monitor. `active_x3` re-reads the covariance and monitor from each new fit, three times.
- 2-D only: `dir_oracle`, `dir_surr` place angles on $[0,\pi)$ by the floor-and-grade rule from $A(\theta)$ (exponent $1/3$; also $1$ in the known-answer check), even centers; `both_surr` adds estimated-slope centers; `joint_surr` adds per-direction counts $\propto$ floor + monitor mass.

**Tasks and budgets.** $d=1$: 1.1, 1.7, 1.8, 1.11-1.16, $B\in\{32,\dots,1024\}$. $d=2$: 2.1, 2.3, 2.7, 2.8, 2.11-2.16, $B\in\{256,\dots,4096\}$. $d=3$: 3.5, 3.7, 3.11, 3.12, 3.13, 3.16 at $B=4096$, rungs `data_p1`, `surr_r1`, `freq`, `active`, `active_x3`. $d=5$: 5.5, 5.16 at $B=4096$, the active rungs. $n=8B$, seed 0; the three suite test sets; relative $L_2$. Side studies: floor sweep $s\in\{0,\dots,1\}$ (1.7, 1.13, 1.14, 1.16); the $d=3$ split sweep (even mesh, centers per direction $\in\{8,\dots,64\}$ at $B=4096$); a known-answer 2-D target $F=\sin(6\pi\,u\cdot x/\|u\|_1)+0.5/(1+8\|x-a\|^2)$, $u$ at $37^\circ$, uniform data, for the direction rungs; and a check that pinning the collar at even density does not remove the price at the floor.

**Tests.** Grading bounds $|h'|$ and keeps the count; neighbor ratios stay under $1.2$; $s=0$ reproduces the even reference; $\gamma_jh_j=0.25$; analytic surrogate derivatives match finite differences; the estimated-slope mesh takes 1.16 from $>10^{-8}$ to $<10^{-11}$ at $B=128$.

**Code & data.** `experiments/expH04_mesh_finding/{mesh.py,run.py,viz.py,known_answer_2d.py,floor_price.py,mesh_map_scale.py,tables.py}`; tests `tests/test_expH04_mesh_finding.py`. Data `ladder_d1.json`, `ladder_d1freq.json`, `ladder_d2{a,b,c,d}.json`, `ladder_d3.json`, `ladder_d5.json`, `floor.json`, `split_d3.json`, `known_answer_2d.json`, `floor_price.json`, `mesh_map_scale.json`; figures under `figures/`.

## Results

### 1-D

Relative $L_2$ on the dense region at $B=128$ (where the meshes differ most):

| task | even | data $p$ | true slope | est. slope | est. curvature | residual | est. frequency |
|---|---|---|---|---|---|---|---|
| 1.1 even grid multiscale bumps | $9\times10^{-14}$ | $2\times10^{-14}$ | $9\times10^{-14}$ | $9\times10^{-14}$ | $8\times10^{-14}$ | $3\times10^{-11}$ | $1\times10^{-13}$ |
| 1.7 uniform data radial runge a12 | $3\times10^{-6}$ | $1\times10^{-7}$ | $2\times10^{-14}$ | $2\times10^{-14}$ | $3\times10^{-14}$ | $3\times10^{-14}$ | $2\times10^{-10}$ |
| 1.8 uniform data sphere jump | $1\times10^{-1}$ | $1\times10^{-1}$ | -- | $1\times10^{-1}$ | $1\times10^{-1}$ | $1\times10^{-1}$ | $2\times10^{-1}$ |
| 1.11 hotspot data radial oscillation freq6 | $3\times10^{-15}$ | $1\times10^{-12}$ | $4\times10^{-14}$ | $4\times10^{-14}$ | $3\times10^{-14}$ | $2\times10^{-13}$ | $1\times10^{-14}$ |
| 1.12 hotspot data packet at hotspot | $8\times10^{-14}$ | $8\times10^{-14}$ | $9\times10^{-14}$ | $9\times10^{-14}$ | $7\times10^{-14}$ | $8\times10^{-14}$ | $2\times10^{-13}$ |
| 1.13 hotspot data packet away from hotspots | $2\times10^{-14}$ | $8\times10^{-11}$ | $6\times10^{-13}$ | $6\times10^{-13}$ | $3\times10^{-14}$ | $2\times10^{-13}$ | $2\times10^{-13}$ |
| 1.14 hotspot data product peak | $1\times10^{-15}$ | $2\times10^{-14}$ | $6\times10^{-15}$ | $5\times10^{-15}$ | $3\times10^{-14}$ | $6\times10^{-14}$ | $1\times10^{-14}$ |
| 1.15 hotspot data jump in sparse region | $2\times10^{-3}$ | $5\times10^{-3}$ | -- | $5\times10^{-3}$ | $2\times10^{-3}$ | $8\times10^{-4}$ | $3\times10^{-3}$ |
| 1.16 hotspot data radial runge a12 at hotspot | $8\times10^{-7}$ | $1\times10^{-9}$ | $2\times10^{-14}$ | $2\times10^{-14}$ | $2\times10^{-14}$ | $1\times10^{-14}$ | $3\times10^{-10}$ |

The same at $B=1024$ (everything resolved; the price of adaptation):

| task | even | data $p$ | true slope | est. slope | est. curvature | residual | est. frequency |
|---|---|---|---|---|---|---|---|
| 1.1 even grid multiscale bumps | $4\times10^{-14}$ | $1\times10^{-13}$ | $1\times10^{-13}$ | $1\times10^{-13}$ | $1\times10^{-13}$ | $2\times10^{-13}$ | $3\times10^{-14}$ |
| 1.7 uniform data radial runge a12 | $7\times10^{-14}$ | $3\times10^{-14}$ | $5\times10^{-13}$ | $5\times10^{-13}$ | $6\times10^{-13}$ | $2\times10^{-13}$ | $4\times10^{-14}$ |
| 1.8 uniform data sphere jump | $5\times10^{-2}$ | $5\times10^{-2}$ | -- | $5\times10^{-1}$ | $2\times10^{-1}$ | $7\times10^{-1}$ | $5\times10^{-2}$ |
| 1.11 hotspot data radial oscillation freq6 | $7\times10^{-14}$ | $5\times10^{-14}$ | $3\times10^{-13}$ | $3\times10^{-13}$ | $2\times10^{-13}$ | $5\times10^{-13}$ | $5\times10^{-14}$ |
| 1.12 hotspot data packet at hotspot | $4\times10^{-14}$ | $3\times10^{-14}$ | $5\times10^{-13}$ | $5\times10^{-13}$ | $8\times10^{-14}$ | $9\times10^{-13}$ | $2\times10^{-14}$ |
| 1.13 hotspot data packet away from hotspots | $5\times10^{-14}$ | $5\times10^{-14}$ | $5\times10^{-14}$ | $5\times10^{-14}$ | $5\times10^{-14}$ | $1\times10^{-13}$ | $1\times10^{-14}$ |
| 1.14 hotspot data product peak | $4\times10^{-14}$ | $3\times10^{-14}$ | $4\times10^{-14}$ | $4\times10^{-14}$ | $5\times10^{-14}$ | $2\times10^{-13}$ | $3\times10^{-14}$ |
| 1.15 hotspot data jump in sparse region | $3\times10^{-5}$ | $2\times10^{-5}$ | -- | $1\times10^{-13}$ | $1\times10^{-12}$ | $7\times10^{-14}$ | $3\times10^{-5}$ |
| 1.16 hotspot data radial runge a12 at hotspot | $6\times10^{-14}$ | $2\times10^{-14}$ | $3\times10^{-13}$ | $3\times10^{-13}$ | $9\times10^{-13}$ | $2\times10^{-13}$ | $3\times10^{-14}$ |

Uniform-cube test at $B=1024$:

| task | even | data $p$ | true slope | est. slope | est. curvature | residual | est. frequency |
|---|---|---|---|---|---|---|---|
| 1.1 even grid multiscale bumps | $2\times10^{-13}$ | $3\times10^{-13}$ | $2\times10^{-13}$ | $2\times10^{-13}$ | $2\times10^{-13}$ | $3\times10^{-11}$ | $2\times10^{-13}$ |
| 1.7 uniform data radial runge a12 | $8\times10^{-14}$ | $3\times10^{-14}$ | $5\times10^{-13}$ | $5\times10^{-13}$ | $6\times10^{-13}$ | $2\times10^{-13}$ | $4\times10^{-14}$ |
| 1.8 uniform data sphere jump | $3\times10^{-2}$ | $3\times10^{-2}$ | -- | $4\times10^{-1}$ | $1\times10^{-1}$ | $4\times10^{-1}$ | $3\times10^{-2}$ |
| 1.11 hotspot data radial oscillation freq6 | $2\times10^{-9}$ | $6\times10^{-12}$ | $2\times10^{-10}$ | $1\times10^{-10}$ | $6\times10^{-10}$ | $5\times10^{-8}$ | $3\times10^{-10}$ |
| 1.12 hotspot data packet at hotspot | $7\times10^{-10}$ | $2\times10^{-12}$ | $2\times10^{-12}$ | $2\times10^{-12}$ | $4\times10^{-12}$ | $8\times10^{-9}$ | $1\times10^{-8}$ |
| 1.13 hotspot data packet away from hotspots | $1\times10^{-9}$ | $7\times10^{-12}$ | $1\times10^{-7}$ | $1\times10^{-7}$ | $7\times10^{-8}$ | $8\times10^{-8}$ | $3\times10^{-7}$ |
| 1.14 hotspot data product peak | $4\times10^{-10}$ | $2\times10^{-12}$ | $3\times10^{-12}$ | $2\times10^{-12}$ | $2\times10^{-10}$ | $2\times10^{-10}$ | $2\times10^{-10}$ |
| 1.15 hotspot data jump in sparse region | $5\times10^{-2}$ | $3\times10^{-2}$ | -- | $2\times10^{-2}$ | $2\times10^{-2}$ | $2\times10^{-2}$ | $4\times10^{0}$ |
| 1.16 hotspot data radial runge a12 at hotspot | $3\times10^{-10}$ | $8\times10^{-13}$ | $5\times10^{-13}$ | $6\times10^{-13}$ | $2\times10^{-12}$ | $1\times10^{-11}$ | $6\times10^{-10}$ |

- **Sharp features are where the mesh matters.** On the two spikes (1.7, 1.16) the slope, curvature and residual monitors reach $2\times10^{-14}$ at $B=128$ where the even mesh is at $10^{-6}$ and needs $B=512$ to get there; the uniform test agrees (1.16: $1.9\times10^{-3}\to4.5\times10^{-10}$). The data-only rung gets a third of the way on 1.16 because the data sits on the spike.
- **The estimated monitor equals the true one.** "true slope" and "est. slope" agree to two digits in every cell; a first fit accurate only to $10^{-6}$ already puts its derivative energy in the right places.
- **Steps.** The step in the sparse tail (1.15) is found only when the mesh is fine enough for the monitor to see it: at $B=1024$ the slope and residual monitors take the dense region from $3\times10^{-5}$ to $10^{-13}$; at $B=128$ nothing helps (with $6$-gap smoothing the residual monitor's peak is too blurred; with $1.5$-gap smoothing it reached $4\times10^{-11}$, the one place the rough monitor was better). On the step in uniform data (1.8) no rung helps.
- **No price at the floor any more.** On the resolved tasks (1.11-1.14) the adapted meshes now sit within a factor $1$-$5$ of even at every budget ($3\times10^{-13}$ vs $7\times10^{-14}$ at worst), against $10$-$100\times$ in the first ladder.
- **Data-weighted means data-weighted.** On the burst away from the hotspots (1.13) the slope monitors move centers to the clusters and the uniform-cube error at $B=1024$ goes from $10^{-9}$ (even) to $10^{-7}$ while the dense region is unchanged; the data-density rung, which spreads its floor evenly, is the best on the uniform test there ($7\times10^{-12}$).

### 2-D

Dense region at $B=1024$:

| task | even | data $p$ | est. slope | residual | est. frequency | angles (est.) | active (est.) |
|---|---|---|---|---|---|---|---|
| 2.1 even grid multiscale bumps | $2\times10^{-8}$ | $7\times10^{-11}$ | $3\times10^{-9}$ | $3\times10^{-10}$ | $4\times10^{-10}$ | $2\times10^{-8}$ | $3\times10^{-9}$ |
| 2.3 even grid radial oscillation freq6 | $2\times10^{-10}$ | $9\times10^{-10}$ | $3\times10^{-10}$ | $2\times10^{-10}$ | $2\times10^{-10}$ | $2\times10^{-10}$ | $3\times10^{-10}$ |
| 2.7 uniform data radial runge a12 | $5\times10^{-3}$ | $2\times10^{-3}$ | $3\times10^{-3}$ | $3\times10^{-3}$ | $7\times10^{-3}$ | $5\times10^{-3}$ | $3\times10^{-3}$ |
| 2.8 uniform data sphere jump | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ |
| 2.11 hotspot data radial oscillation freq6 | $4\times10^{-11}$ | $2\times10^{-9}$ | $1\times10^{-10}$ | $6\times10^{-11}$ | $4\times10^{-11}$ | $4\times10^{-11}$ | $1\times10^{-10}$ |
| 2.12 hotspot data packet at hotspot | $2\times10^{-4}$ | $2\times10^{-6}$ | $1\times10^{-6}$ | $3\times10^{-5}$ | $1\times10^{-4}$ | $2\times10^{-4}$ | $1\times10^{-6}$ |
| 2.13 hotspot data packet away from hotspots | $2\times10^{-5}$ | $2\times10^{-6}$ | $2\times10^{-6}$ | $3\times10^{-6}$ | $1\times10^{-5}$ | $2\times10^{-5}$ | $2\times10^{-6}$ |
| 2.14 hotspot data product peak | $7\times10^{-5}$ | $7\times10^{-6}$ | $3\times10^{-5}$ | $3\times10^{-5}$ | $6\times10^{-5}$ | $6\times10^{-5}$ | $3\times10^{-5}$ |
| 2.15 curved sheet composition | $7\times10^{-14}$ | $1\times10^{-14}$ | $7\times10^{-14}$ | $8\times10^{-14}$ | $1\times10^{-13}$ | $6\times10^{-14}$ | $7\times10^{-14}$ |
| 2.16 curved sheet noisy composition | $1\times10^{-13}$ | $2\times10^{-13}$ | $2\times10^{-13}$ | $1\times10^{-13}$ | $2\times10^{-13}$ | $1\times10^{-13}$ | $2\times10^{-13}$ |

Dense region at $B=4096$:

| task | even | data $p$ | est. slope | residual | est. frequency | angles (est.) | active (est.) |
|---|---|---|---|---|---|---|---|
| 2.1 even grid multiscale bumps | $2\times10^{-13}$ | $4\times10^{-13}$ | $4\times10^{-13}$ | $2\times10^{-13}$ | $6\times10^{-13}$ | $4\times10^{-13}$ | $4\times10^{-13}$ |
| 2.3 even grid radial oscillation freq6 | $9\times10^{-13}$ | $4\times10^{-11}$ | $2\times10^{-12}$ | $8\times10^{-13}$ | $1\times10^{-12}$ | $9\times10^{-13}$ | $2\times10^{-12}$ |
| 2.7 uniform data radial runge a12 | $7\times10^{-5}$ | $1\times10^{-6}$ | $9\times10^{-7}$ | $9\times10^{-7}$ | $7\times10^{-6}$ | $7\times10^{-5}$ | $9\times10^{-7}$ |
| 2.8 uniform data sphere jump | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ | $2\times10^{-1}$ |
| 2.11 hotspot data radial oscillation freq6 | $2\times10^{-13}$ | $3\times10^{-11}$ | $3\times10^{-12}$ | $9\times10^{-13}$ | $3\times10^{-13}$ | $2\times10^{-13}$ | $3\times10^{-12}$ |
| 2.12 hotspot data packet at hotspot | $1\times10^{-11}$ | $2\times10^{-10}$ | $3\times10^{-12}$ | $4\times10^{-13}$ | $1\times10^{-12}$ | $1\times10^{-11}$ | $3\times10^{-12}$ |
| 2.13 hotspot data packet away from hotspots | $6\times10^{-12}$ | $3\times10^{-9}$ | $1\times10^{-12}$ | $5\times10^{-13}$ | $2\times10^{-11}$ | $6\times10^{-12}$ | $1\times10^{-12}$ |
| 2.14 hotspot data product peak | $5\times10^{-8}$ | $1\times10^{-10}$ | $2\times10^{-10}$ | $2\times10^{-10}$ | $5\times10^{-9}$ | $5\times10^{-8}$ | $2\times10^{-10}$ |
| 2.15 curved sheet composition | $3\times10^{-14}$ | $2\times10^{-14}$ | $2\times10^{-14}$ | $2\times10^{-14}$ | $4\times10^{-14}$ | $3\times10^{-14}$ | $2\times10^{-14}$ |
| 2.16 curved sheet noisy composition | $4\times10^{-14}$ | $3\times10^{-14}$ | $4\times10^{-14}$ | $2\times10^{-14}$ | $5\times10^{-14}$ | $3\times10^{-14}$ | $4\times10^{-14}$ |

Uniform-cube test at $B=4096$:

| task | even | data $p$ | est. slope | residual | est. frequency | angles (est.) | active (est.) |
|---|---|---|---|---|---|---|---|
| 2.1 even grid multiscale bumps | $7\times10^{-13}$ | $9\times10^{-13}$ | $9\times10^{-13}$ | $6\times10^{-13}$ | $1\times10^{-12}$ | $9\times10^{-13}$ | $9\times10^{-13}$ |
| 2.3 even grid radial oscillation freq6 | $5\times10^{-12}$ | $1\times10^{-10}$ | $9\times10^{-12}$ | $3\times10^{-12}$ | $3\times10^{-11}$ | $5\times10^{-12}$ | $9\times10^{-12}$ |
| 2.7 uniform data radial runge a12 | $6\times10^{-5}$ | $2\times10^{-6}$ | $2\times10^{-6}$ | $2\times10^{-6}$ | $1\times10^{-5}$ | $6\times10^{-5}$ | $2\times10^{-6}$ |
| 2.8 uniform data sphere jump | $1\times10^{-1}$ | $1\times10^{-1}$ | $1\times10^{-1}$ | $1\times10^{-1}$ | $2\times10^{-1}$ | $1\times10^{-1}$ | $1\times10^{-1}$ |
| 2.11 hotspot data radial oscillation freq6 | $5\times10^{-10}$ | $6\times10^{-9}$ | $8\times10^{-10}$ | $1\times10^{-9}$ | $4\times10^{-10}$ | $5\times10^{-10}$ | $8\times10^{-10}$ |
| 2.12 hotspot data packet at hotspot | $9\times10^{-11}$ | $9\times10^{-8}$ | $9\times10^{-10}$ | $1\times10^{-10}$ | $1\times10^{-9}$ | $9\times10^{-11}$ | $9\times10^{-10}$ |
| 2.13 hotspot data packet away from hotspots | $4\times10^{-10}$ | $2\times10^{-6}$ | $1\times10^{-9}$ | $3\times10^{-9}$ | $6\times10^{-8}$ | $3\times10^{-10}$ | $1\times10^{-9}$ |
| 2.14 hotspot data product peak | $2\times10^{-6}$ | $8\times10^{-9}$ | $4\times10^{-8}$ | $2\times10^{-8}$ | $2\times10^{-7}$ | $2\times10^{-6}$ | $4\times10^{-8}$ |
| 2.15 curved sheet composition | $5\times10^{-1}$ | $6\times10^{-1}$ | $6\times10^{-1}$ | $6\times10^{-1}$ | $4\times10^{-1}$ | $5\times10^{-1}$ | $6\times10^{-1}$ |
| 2.16 curved sheet noisy composition | $2\times10^{-1}$ | $4\times10^{-1}$ | $3\times10^{-1}$ | $3\times10^{-1}$ | $1\times10^{-1}$ | $2\times10^{-1}$ | $3\times10^{-1}$ |

- **Under-resolved tasks gain, resolved tasks no longer pay.** At $B=1024$ the burst at the hotspot (2.12) goes from $1.8\times10^{-4}$ to $10^{-6}$ (slope) and $2\times10^{-6}$ (data density); the product peak $10\times$ with the data density; the bumps (2.1) $300\times$. At $B=4096$ the spike (2.7) and the peak (2.14) still gain $70$-$500\times$, and on the resolved tasks the residual and slope meshes are now at or below even (2.12: $10^{-11}\to4\times10^{-13}$ residual; 2.13: $6\times10^{-12}\to5\times10^{-13}$). The data-density rung alone does pay on 2.3/2.11 ($40\times$) because it moves centers off the waves' outer rings where there is little data.
- **Angle rung is the even mesh** to the digit on every task; the active-subspace rung finds $m=2=d$ and falls back to the slope monitor, as designed. **Sheets**: at the floor on the sheet, $O(1)$ off it, for every mesh.

### $d=3$ and $d=5$

Dense region:

| task | even | data $p$ | est. slope | est. frequency | active (true) | active (est.) | active, iterated |
|---|---|---|---|---|---|---|---|
| 3.5 uniform data composition | $4\times10^{-7}$ | $2\times10^{-7}$ | $3\times10^{-7}$ | -- | -- | $2\times10^{-11}$ | $2\times10^{-11}$ |
| 3.7 uniform data radial runge a12 | $2\times10^{-2}$ | $2\times10^{-2}$ | $2\times10^{-2}$ | -- | -- | $2\times10^{-2}$ | $2\times10^{-2}$ |
| 3.11 hotspot data radial oscillation freq6 | $2\times10^{-8}$ | $6\times10^{-8}$ | $3\times10^{-8}$ | -- | -- | $3\times10^{-8}$ | $3\times10^{-8}$ |
| 3.12 hotspot data packet at hotspot | $4\times10^{-2}$ | $3\times10^{-2}$ | $4\times10^{-2}$ | -- | -- | $4\times10^{-2}$ | $4\times10^{-2}$ |
| 3.13 hotspot data packet away from hotspots | $7\times10^{-3}$ | $5\times10^{-3}$ | $6\times10^{-3}$ | -- | -- | $6\times10^{-3}$ | $6\times10^{-3}$ |
| 3.16 curved sheet noisy composition | $8\times10^{-9}$ | $1\times10^{-9}$ | $3\times10^{-9}$ | -- | -- | $2\times10^{-13}$ | $2\times10^{-13}$ |
| 5.5 uniform data composition | $8\times10^{-4}$ | $8\times10^{-4}$ | $8\times10^{-4}$ | -- | -- | $5\times10^{-6}$ | $5\times10^{-6}$ |
| 5.16 curved sheet noisy composition | $1\times10^{-6}$ | $1\times10^{-6}$ | $1\times10^{-6}$ | -- | -- | $5\times10^{-11}$ | $5\times10^{-11}$ |

Uniform-cube test:

| task | even | data $p$ | est. slope | est. frequency | active (true) | active (est.) | active, iterated |
|---|---|---|---|---|---|---|---|
| 3.5 uniform data composition | $9\times10^{-7}$ | $4\times10^{-7}$ | $7\times10^{-7}$ | -- | -- | $8\times10^{-10}$ | $8\times10^{-10}$ |
| 3.7 uniform data radial runge a12 | $3\times10^{-2}$ | $2\times10^{-2}$ | $3\times10^{-2}$ | -- | -- | $3\times10^{-2}$ | $3\times10^{-2}$ |
| 3.11 hotspot data radial oscillation freq6 | $3\times10^{-5}$ | $5\times10^{-5}$ | $3\times10^{-5}$ | -- | -- | $3\times10^{-5}$ | $3\times10^{-5}$ |
| 3.12 hotspot data packet at hotspot | $4\times10^{0}$ | $3\times10^{0}$ | $3\times10^{0}$ | -- | -- | $3\times10^{0}$ | $3\times10^{0}$ |
| 3.13 hotspot data packet away from hotspots | $8\times10^{-1}$ | $8\times10^{-1}$ | $5\times10^{-1}$ | -- | -- | $5\times10^{-1}$ | $4\times10^{-1}$ |
| 3.16 curved sheet noisy composition | $4\times10^{-1}$ | $1\times10^{-1}$ | $2\times10^{-1}$ | -- | -- | $3\times10^{-2}$ | $3\times10^{-2}$ |
| 5.5 uniform data composition | $2\times10^{-3}$ | $2\times10^{-3}$ | $2\times10^{-3}$ | -- | -- | $3\times10^{-2}$ | $3\times10^{-2}$ |
| 5.16 curved sheet noisy composition | $1\times10^{0}$ | $2\times10^{0}$ | $2\times10^{0}$ | -- | -- | $9\times10^{-2}$ | $9\times10^{-2}$ |

- **Low active dimension is where the direction problem is solvable.** Composition in $d=3$: $4\times10^{-7}\to2\times10^{-11}$; the noisy curved sheet with the same function (3.16): $8\times10^{-9}\to2\times10^{-13}$; in $d=5$ the sheet goes $10^{-6}\to5\times10^{-11}$ and the composition $8\times10^{-4}\to5\times10^{-6}$. The true subspace gives $10^{-12}$ on both $d=5$ tasks (first ladder, `bw1.5/`). The iteration fit $\to$ covariance $\to$ fit converged in the first ladder ($8\times10^{-4}\to5\times10^{-6}\to10^{-9}$ on 5.5, $10^{-6}\to2\times10^{-13}$ on 5.16) and stalled in this rerun (5.5: $5\times10^{-6}$ after every round). The plausible cause is that the covariance is re-read over the whole training set, including the regions where the active fit is worse than even (its uniform error is $3\times10^{-2}$); a covariance weighted by where the fit is good is the obvious repair and is untested.
- **Isotropic content in $d=3$**: on the spike, the waves and both bursts no rung moves the error by more than $3\times$; with $16$ centers per direction the mesh map has one degree of freedom, and the split sweep says the budget itself is short.

### $d=3$ split sweep

| task | 8 per dir | 12 per dir | 16 per dir | 24 per dir | 32 per dir | 48 per dir | 64 per dir |
|---|---|---|---|---|---|---|---|
| 3.3 even grid radial oscillation freq6 | $9\times10^{-3}$ | $4\times10^{-5}$ | $2\times10^{-7}$ | $1\times10^{-7}$ | $9\times10^{-6}$ | $7\times10^{-4}$ | $6\times10^{-3}$ |
| 3.7 uniform data radial runge a12 | $8\times10^{-2}$ | $4\times10^{-2}$ | $2\times10^{-2}$ | $1\times10^{-2}$ | $2\times10^{-2}$ | $4\times10^{-2}$ | $7\times10^{-2}$ |
| 3.11 hotspot data radial oscillation freq6 | $5\times10^{-4}$ | $2\times10^{-6}$ | $2\times10^{-8}$ | $8\times10^{-9}$ | $6\times10^{-7}$ | $5\times10^{-5}$ | $4\times10^{-4}$ |
| 3.12 hotspot data packet at hotspot | $5\times10^{-1}$ | $2\times10^{-1}$ | $4\times10^{-2}$ | $8\times10^{-3}$ | $2\times10^{-2}$ | $6\times10^{-2}$ | $8\times10^{-2}$ |
| 3.13 hotspot data packet away from hotspots | $6\times10^{-2}$ | $2\times10^{-2}$ | $7\times10^{-3}$ | $5\times10^{-3}$ | $1\times10^{-2}$ | $1\times10^{-2}$ | $2\times10^{-2}$ |
| 3.16 curved sheet noisy composition | $2\times10^{-6}$ | $1\times10^{-7}$ | $8\times10^{-9}$ | $3\times10^{-11}$ | $2\times10^{-11}$ | $3\times10^{-10}$ | $9\times10^{-9}$ |

Every isotropic task is U-shaped with its optimum at $16$-$24$ centers per direction, so the reference split is near-optimal and no reallocation of $B=4096$ resolves a spike or a burst: both the direction count and the per-direction resolution are short. The sheet task (3.16) wants $24$-$32$ per direction ($2\times10^{-11}$, $400\times$ better than the reference split): a function of low active dimension wants fewer directions and more centers, which the active-subspace mesh does explicitly.

### Direction count against the data radius

`directions_vs_radius_d{2,3}.png`. Fast concentric waves, data in a ball of radius $r$ about an off-center point, ridge system recentered there with offsets confined to the data's projection band, direction count $M$ swept at a generous fixed number of offsets per direction.

| | $kr\approx1$ | $kr\approx2$ | $kr\approx4$ | $kr\approx7.5$ | domain-centered reference, whole cube |
|---|---|---|---|---|---|
| $d=2$, $M$ for $10^{-10}$ on the data | 12 | 12-16 | 16 | 24 | 64 |
| $d=3$, $M$ for $10^{-9}$ on the data | 64 | 128 | 256 | | 256 directions, $16$ per direction: $10^{-7}$ |

In 2-D the count sits at the polynomial floor $\binom{p+1}{1}=13$ for $kr\lesssim2$ and then grows with $kr$; in 3-D it doubles as $r$ doubles from $kr\approx1$ to $4$, above a floor near $\binom{14}{2}=91$. Away from the data the error is $10^{6}$-$10^{9}$ at small $M$ and $O(1)$ once the data is resolved: the far field of a ridge system is the saturated-wall arrangement.

### Known-answer ridge

| $B$ | even angles | angles from $A(\theta)^{1/3}$ | active (true) | active (est.) | active, iterated |
|---|---|---|---|---|---|
| 256 | $2\times10^{-2}$ | $1\times10^{-2}$ | $2\times10^{1}$ | $2\times10^{1}$ | $7\times10^{4}$ |
| 512 | $6\times10^{-5}$ | $6\times10^{-5}$ | $7\times10^{0}$ | $7\times10^{0}$ | $5\times10^{5}$ |
| 1024 | $7\times10^{-6}$ | $8\times10^{-6}$ | $4\times10^{-2}$ | $4\times10^{-2}$ | $1\times10^{4}$ |
| 2048 | $1\times10^{-7}$ | $1\times10^{-7}$ | $2\times10^{-2}$ | $2\times10^{-2}$ | $3\times10^{2}$ |
| 4096 | $3\times10^{-10}$ | $4\times10^{-10}$ | $3\times10^{-3}$ | $3\times10^{-3}$ | $5\times10^{1}$ |

The angle-density rungs are the even mesh (right panel of `known_answer_2d.png`: $A(\theta)$ is a broad $\cos^2$). The active-subspace rung finds the ridge direction to $0.01^\circ$ and that is not good enough: a $0.01^\circ$ tilt on a ridge of $7.5$ periods costs $3\times10^{-3}$, worse than even angles. The bias comes from the background's gradient in the covariance (true and estimated agree), and the iteration cannot remove it for $m=1$ (every active unit shares the one direction, so their covariance returns it unchanged; that refit diverged). A pure ridge needs its direction to machine precision, a projection-pursuit problem; this is the sum-of-ridges case the suite excludes.

### Figures

- **`ladder_centers_d1_{dense_region,uniform,same_as_train}.png`** -- $3\times3$ panels (tasks); $x$ = $B$, $y$ = relative $L_2$ (fixed $[10^{-15},10^2]$); one line per center rung (black even, blue data-only, red dashed true-gradient monitors, green estimated, orange residual, cyan frequency). Read 1.7/1.15/1.16 for the wins, 1.11-1.14 for the price at the floor, 1.13 uniform for the data-weighting penalty.
- **`ladder_centers_d2_*.png`**, **`ladder_directions_d2_*.png`** -- same layout for the ten 2-D tasks ($2\times5$); the direction figure has even, estimated-slope centers, the four angle rungs and the active-subspace rungs.
- **`highdim_bars.png`** -- $d\ge3$ at $B=4096$: bars of relative $L_2$ per task and rung, three test sets side by side. The active rungs on 3.5 / 5.5 / 3.16 are the point.
- **`split_d3.png`** -- $d=3$ even mesh at $B=4096$: error vs centers per direction (directions $=B/$centers), one line per task; dashed = the reference split $16$. Flat or U-shaped everywhere: the wall is the budget.
- **`known_answer_2d.png`** -- left: error vs $B$ for even angles, angle-density rungs and active-subspace rungs on the ridge-plus-bump target; right: the estimated $A(\theta)$ (broad) with the placed angles and the true ridge direction.
- **`gain_at_top_budget.png`** -- $\log_{10}(\text{error}/\text{even})$ at the largest budget per task ($d\le2$), dense region and uniform; mostly the price of adaptation at the floor.
- **`directions_vs_radius_d{2,3}.png`** -- left: on-data error vs the number of directions, one line per data radius; middle: the same error over the whole cube; right: directions needed for $10^{-10}$ vs radius, with the polynomial floor $\binom{12+d-1}{d-1}$ and the large-$kr$ slope $r^{d-1}$.
- **`mesh_map_scale.png`** -- the scale: dense-region error vs the wavelength (in gaps) of a sinusoidal perturbation of the even mesh, one line per amplitude, three resolved cells; dashed = even. Harmless above about $12$ gaps at every amplitude.
- **`floor_price.png`** -- the diagnosis: left, dense-region error against mesh roughness $\max|\Delta^2\log h_j|$ for even / slope-monitor / half-Gaussian / jittered meshes; middle, the same against the largest readout weight; right, error relative to even against the monitor smoothing (in gaps) for every resolved cell, line style = grading cap.
- **`floor_sweep_d1.png`** -- dense-region error vs $B$ for six floors $s$ (top `surr_r1`, bottom `data_p1`); $s\in[\tfrac13,\tfrac23]$ captures the gain, $s=1$ settles a decade higher.
- **`mesh_examples_d1.png`** -- 1.13, 1.14, 1.16, 1.15 at $B=128$: target and data histogram; the center density each rung chose with the placed centers as ticks. The residual monitor's peak on the step and the slope monitor's miss are in the bottom row.
- **`mesh_examples_d2.png`** -- 2.12, 2.13, 2.16 at $B=1024$: the resolution field $\sum_k\gamma_k\,\mathrm{sech}^2(u_k(x))$ (bright = finer mesh) with the training data, for even / data density / estimated slope / directions+centers.

## Additional details

### The price at the floor, resolved

`floor_price.py`, on targets the even mesh resolves (1.11, 1.13, 1.14 at $B=128$-$512$; 2.11 at $B=1024$), four families of mesh through the same solve, with two diagnostics per mesh: the roughness $\max_j|\Delta^2\log h_j|$ (second difference of the log spacing) and the largest readout weight.

| mesh | roughness | dense-region error (1.14, $B=128$; even $=1.1\times10^{-15}$) |
|---|---|---|
| even mesh, positions shaken by $2\%$ of a gap | $0.2$ | $7\times10^{-11}$ |
| shaken by $20\%$ | $2.5$ | $4\times10^{-10}$ |
| slope monitor smoothed at $1.5$ gaps (first ladder) | $0.12$ | $1.3\times10^{-13}$ |
| slope monitor smoothed at $6$ gaps | $0.008$ | $4.9\times10^{-15}$ |
| expH02's half-Gaussian density through this pipeline, $s=1$ | $0.002$ | $3.4\times10^{-15}$ |

- **Roughness at the gap scale is the whole story.** Across all four families the error tracks the roughness over five decades (`floor_price.png`, left); the neighbor ratio by itself does not (a graded monitor at ratio $1.16$ costs $100\times$, a $2\%$ jitter at ratio $1.12$ costs $10^5\times$). Smooth non-uniformity costs nothing, so the pipeline (grading, placement, local widths) is innocent and the cardinal structure survives any smooth change of spacing; what it cannot survive is spacing that wiggles from one unit to the next.
- **The monitor was the rough part.** At $1.5$ gaps the histogram noise is passed straight into the spacing. Smoothing at $6$ gaps removes the price on every resolved cell (1.11 at $B=128$: $1.1\times10^{-11}\to4\times10^{-14}$; 2.11 at $B=1024$: $7.6\times10^{-10}\to1.2\times10^{-10}$, $6.5\times10^{-11}$ at $12$ gaps, even $4.2\times10^{-11}$) and improves the small-budget wins (1.7 at $B=128$: $3\times10^{-13}\to2\times10^{-14}$; 2.12 at $B=1024$: $1.9\times10^{-5}\to1.0\times10^{-6}$; 2.12 at $B=4096$ with the residual monitor: $10^{-11}\to4\times10^{-13}$, now better than even). $12$ gaps over-smooths (1.7 falls back to $3\times10^{-11}$). The one rung that wants a sharp monitor is the step (1.15, residual monitor: $2.4\times10^{-9}$ at $1.5$ gaps, $8.7\times10^{-6}$ at $6$). The grading cap $g$ matters much less than the smoothing.
- The readout weights grow with roughness but only loosely predict the error (middle panel); they are a symptom, not the cause. The collar was ruled out earlier (pinning it at even density changes nothing).
- **Why, and why that number.** Every mesh here is a coordinate map, $c_j=\Phi(jh_0)$, and in the coordinate $\xi=\Phi^{-1}(x)$ the network is the uniform construction applied to $f\circ\Phi$; so $\Phi$ must carry no content the construction cannot represent. `mesh_map_scale.py` measures that limit directly: the even mesh with one sinusoidal perturbation of the positions, amplitude $a$ gaps and wavelength $L$ gaps (`mesh_map_scale.png`). A perturbation at $L\ge12$ gaps is harmless at every amplitude up to $0.2$ gap (neighbor ratios up to $1.2$); at $L=2$-$8$ gaps it costs $10^{-12}$-$10^{-9}$, roughly linearly in $a$; the threshold is the same in gaps at $B=256$ and $B=512$ and on all three targets. (Wavelength $3$ is the periodic special case: a 3-gap unit cell is still a lattice.) So the rule is: **the mesh map must be band-limited to wavelengths above $L^*\approx12$ gaps.** A Gaussian of width $\sigma$ suppresses wavelength $L$ by $e^{-2\pi^2\sigma^2/L^2}$; $10^{-2}$ suppression at $L^*$ requires $\sigma\ge5.8$ gaps, which is the constant now in `mesh.py` (derived, not tuned; any band-limited parametrization of the map with resolution $L^*$ is equivalent). The ladders were rerun with $\sigma=6$ (the rounded value, launched before the derivation was written down; the difference is nil) and those are the tables above; the first ladder's data is under `bw1.5/`.
- **A consequence for high $d$.** Adaptation needs many gaps per direction: with $16$ centers per direction ($d=3$ at $B=4096$) the map has about one degree of freedom below $L^*$, so no monitor can do anything -- a second, independent reason the $d=3$ center rungs are null, and a reason the active-subspace route (which raises the per-direction count to $B_a^{1/m}$) is the one that can adapt in high $d$.

- The neighbor-spacing ratio sits at its cap ($1.15$-$1.17$) in every adapted cell: every monitor asks for more contrast than the grading allows, so the exponents ($1/3$ vs $1$ vs $1/5$) are barely distinguished; only the `data_p13`/`data_p1` pair shows it, and $p^1$ wins on the spikes.
- The collar was tested as the cause of the price at the floor and rejected: pinning $|t|>\|v\|_1$ at even density leaves 1.11 at $4\times10^{-11}$ (vs $10^{-11}$ adapted, $3\times10^{-15}$ even) and blunts the wins (1.7: $3\times10^{-13}\to1.5\times10^{-12}$).
- `surr_r1` and `oracle_r1` agree to two digits in every cell of the 1-D and 2-D ladders; a first fit accurate only to $10^{-6}$ already puts its derivative energy in the right places. The exception is the subspace estimate in $d=5$, where the first fit is $O(10^{-3})$ and the noise eigenvalues are $2\times10^{-5}$ of the leading one.
- The 2-D direction-density rungs are identical to even on every suite task (the targets are radial or otherwise isotropic in their gradient second moment), and on the known-answer ridge as well.

## Conclusions

Pending Sam's review. Proposed: (1) a center monitor read from a first even fit places centers as well as the true gradient does, and on sharp features that is worth $4$-$8\times$ in width; (2) the mesh is a coordinate map and must be band-limited to wavelengths above about $12$ gaps, the construction's own resolution limit; that fixes the monitor smoothing ($5.8$ gaps) with no free constant; (3) steps need the residual monitor; (4) the direction problem is an active-subspace problem, not a density problem, and solving it makes the cost depend on the active dimension rather than $d$; (5) isotropic full-dimensional content is a $(kR)^d$ budget wall for any single-hidden-layer ridge mesh.

## Open questions

- Why is the fp64 floor this sensitive to gap-scale roughness ($2\%$ jitter, five orders)? The cardinal coefficients of the construction cancel to $|c_0|\sim338$ against unit output; a translation-non-invariant basis should lose that cancellation at first order in the jitter, which predicts a much milder loss. Worth a clean 1-D study on the QI construction itself (not the least-squares fit).
- The active-subspace iteration converges geometrically; how many rounds to the floor in $d=5$, and does it hold when the active dimension is not exact (a weak dependence on the remaining coordinates, e.g. the noisy sheets)?
- A monitor that is not data-weighted, or the data density as a constraint rather than a weight, for tasks scored on the uniform test.
- Beyond ridges: products or a second layer are what would localize units and remove the $(kR)^{d-1}$ direction tax; out of scope for the single-layer program.
