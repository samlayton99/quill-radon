# expE01 -- 2D ridge-geometry zoo: extending the recipe to R^2 -> R

**Status: draft -- conclusions pending Sam's sign-off.**

## TL;DR

- The 1D recipe (fixed geometry + fp64 lstsq) carries to 2D: a smooth target (gauss_bump) reaches the fp64 floor (~$6\times10^{-14}$) under a structured, training-free ridge geometry.
- Placement decides it, as in 1D: the Radon (direction x offset) geometries are best or tied at the largest width on 3 of 4 targets. runge2d (a central spike) stays resolution-limited.

## Question

Does the 1D "fixed geometry + lstsq" result extend to $\mathbb{R}^2\to\mathbb{R}$, and which ridge geometry (if any) reaches the floor on a smooth 2D target?

## Experiment design

Each neuron is a ridge $\tanh(\gamma(w_m^\top x - t_m))$ -- a smoothed step across the line at signed distance $t_m$ in unit direction $w_m$, so a geometry is just a recipe for the $(w_m,t_m)$ pairs (the Radon/line view). For a placed point $c_m$ the radial tangent rule sets $w_m=c_m/\|c_m\|$ and $t_m=\|c_m\|$, giving preactivation $\gamma(\hat n_m^\top x - r_m)=\gamma\cdot\text{signed-dist}(x,\ell_m)$ -- the 2D analog of a 1D center. Dimensionless bandwidth $\lambda=\gamma\,h_\text{ref}$ uses a common reference spacing $h_\text{ref}=2.8/\sqrt N$, so $\lambda$ is comparable across geometries. Six geometries: three point-based (`hex`, `mode_radial`, `sixn_rings`, radial tangents), `random_ridges` (random points, decoupled orientations), and two that place lines directly in $(\theta,t)$ (`radon_tensor` uniform grid, `radon_interlaced` half-shifted on alternate directions). Sweep: 4 targets x widths $\{64,\dots,8192\}$ x a $\lambda$ grid (best eval-$L_\infty$ cell kept); fit on ~8k area-uniform disk samples, eval on a disk-clipped grid; fp64 lstsq with a bias column. Point geometries take a collar $R=1.6$, the Radon ones a node-count halo $R=2.5$. (Supersedes the hex-only study, now the `hex` geometry.)

**Code & data.** `experiments/expE01_geometry_zoo_2d/` (`run.py`, `geometries.py`, `extra_analysis.py`); geometry in `src/construction/hex_geometry.py` (tests in `tests/test_hex_geometry.py`). Data: `data.json` (+ per-cell MLPs in `weights/`). Figures: `geometry_suite.png` (deliverable), `error_vs_lambda.png`, `error_vs_lambda_anim.{html,mp4,gif}`, `targets_3d.png`, `geometry_viz_sample.png`, `geometry_viz.ipynb`.

## Results

- **The recipe reaches the 2D floor on a smooth target:** on gauss_bump the Radon geometries hit ~$10^{-14}$ by $N\approx576$--$1024$ (radon-tensor $5.9\times10^{-14}$ at $N=8192$; best non-Radon ~$60\times$ worse).
- **Radon wins at scale; points win small.** At $N=8192$ the Radon geometries are best or tied on gauss_bump, sine2d, and mixed2d; at small $N$ the point geometries win (too few directions otherwise). The $\lambda$ optimum drifts down with $N$, as in 1D.
- **runge2d stays resolution-limited** for all six; random_ridges does best (~$8\times10^{-9}$), plausibly because its centers cluster toward the spike.

### Figures

- **`geometry_suite.png`** (deliverable) -- 4 targets (rows) x {rel $L_2$, $L_\infty$} (cols), one line per geometry vs width. Read each panel for the lowest geometry and the scaling slope; the gauss_bump row is where the two Radon lines separate and reach ~$6\times10^{-14}$.
- **`error_vs_lambda.png`** (+ `error_vs_lambda_anim.*`) -- the U-shape per geometry at fixed $N$ (animated over width); the Radon lines are flatter and lower on the smooth targets.
- **`targets_3d.png`** -- the four target shapes (gauss_bump/mixed2d smooth, runge2d a spike, sine2d an oscillation).
- **`geometry_viz_sample.png`** / **`geometry_viz.ipynb`** -- the spatial ridge layouts and a pointwise error heatmap on the disk (see *where* each geometry interpolates well).

## Additional details

- The readout is underdetermined in 2D the same way as 1D; truncated lstsq still produces a low-error function. eval $L_\infty$ on a finite grid is a lower bound; the gauss_bump Radon tail sits at the fp64 floor and may be conditioning/grid noise (no conditioning logged this run).

## Conclusions

*Proposed, pending Sam.* The fixed-geometry + lstsq recipe extends to 2D -- gauss_bump reaches the fp64 floor under the Radon geometry, which is best/tied on 3 of 4 targets at scale. Placement decides precision as in 1D; runge2d's resolution limit and random_ridges' edge there look incidental, not a property of the recipe.

## Open questions

- **Concentrate coverage where the target needs it** (Sam): place uniform coverage near the bumps/high-curvature regions rather than uniformly over the disk (random_ridges' runge win is likely incidental center-clustering), and check whether the scaling laws then descend cleanly. Same curvature-clustering lead as expC05, in 2D.
- Whether extended precision and larger $N$ push the other smooth targets to the floor.
