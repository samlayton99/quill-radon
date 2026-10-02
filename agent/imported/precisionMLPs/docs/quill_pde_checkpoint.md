# QUILL/Radon PDE checkpoint: what works, how it works, and what remains

Consolidated from the saved expF19 experiments, October 1, 2026. This is the
checkpoint for the Radon/PDE branch of the investigation, not a claim to resolve
the earlier general theory of learned neural geometry. The evidence index below
identifies the underlying code, metrics, and figures. Reported errors are sampled
relative norms unless otherwise stated; refinement differences are labeled as
such. Timings are recorded local runs, not hardware-independent benchmarks.

We can construct accurate neural representations without Adam or a large neural
readout fit, and use them inside solvers for several linear and nonlinear PDE
classes. On one controlled nonlinear elliptic benchmark, this was dramatically
more accurate and faster than three tested Adam/L-BFGS PINNs. A matched classical
spectral solver was faster still and equally accurate. We have removed a neural
optimization bottleneck in supported settings. We have not eliminated numerical
PDE solving, established superiority over classical methods, or solved PINNs in
general.

## 1. The construction and the unknown solution are separate problems

The construction answers: given a suitable function, how do we represent it by
neurons accurately without fitting their weights? A PDE solver answers: given
initial/boundary conditions and a differential equation, which function is the
solution? Knowing how to do the first does not automatically answer the second.

For a scalar one-dimensional function, the constructed network has real centers
\(c_j\), positive inverse widths \(\gamma_j\), real readouts \(w_j\), and bias
\(b\):

\[
\widehat f(x)=b+\sum_j w_j\tanh(\gamma_j(x-c_j)).
\]

Its derivative is

\[
\widehat f'(x)=\sum_j w_j\gamma_j\operatorname{sech}^2(\gamma_j(x-c_j)).
\]

The kernel \(\gamma\operatorname{sech}^2(\gamma s)/2\) integrates to one.
On a uniform grid of spacing \(h\), choosing approximately
\(w_j=(h/2)f'(c_j)\) therefore makes the derivative a quadrature approximation
to a smoothed version of \(f'\). This explains the derivative-shaped readouts.
Finite width, sampling, and boundaries require corrections for high accuracy;
plain derivative samples are not the full high-precision formula.

The current QUILL encoder uses an analytic profile \(g\), with a valid complex
neighborhood on which it can be evaluated. For a common inverse width
\(\gamma\), define the shift distance \(d=\pi/(2\gamma)\) and density

\[
\rho_d(z)=\frac{g(z+id)-g(z-id)}{2id},\qquad
w_j^{\mathrm{base}}=\operatorname{Re}\left[\frac h2\rho_d(c_j)\right].
\]

The implementation adds explicit contour-moment corrections to the halo weights
and anchors the bias. Those corrections use quadrature and stable scaled partial
fractions, not a least-squares fit. Their derivation and exact implementation are
linked in the evidence index. As \(d\) tends to zero, \(\rho_d\) tends to \(g'\),
connecting the precise construction back to the derivative picture.

In the PDE solver, the analytic functions supplied to this encoder are known
basis functions, such as sines, cosines, and polynomials. We do not query an
unknown PDE solution at complex inputs. A naive attempt to apply this density
formula directly to the evolving tanh network fails because the complex shifts
can hit its poles. Encoding known basis functions and analyzing real PDE values
avoids that problem.

### Geometry and resolution

For the integration-level convention, \(N\) is the number of interior centers,
\(h=(b-a)/(N-1)\), \(\lambda=\gamma h\), and each side receives
\(R=\lceil\sqrt N\rceil\) corrected halo centers. The low-level
`quill_boundary.encode` instead accepts an interval-cell count, so it has one
more interior center and its default halo uses that cell count. Wrappers pass
the interior-count halo explicitly. This convention matters when reproducing
counts; halo nodes are not included when calculating interior spacing.

The original expF19 runs used long uncorrected halos and fixed lambda .25. The
current corrected construction supersedes that protocol. We verified the
correction against an independent historical implementation at 60-digit
precision. Bandwidth sweeps check the derivatives the PDE needs, not just the
function value. Values around .2 worked well in several cases; .2 is not a
universal optimum. The generic periodic backend tests .15/.20/.25/.30, while
some specialized stiff tests use .25.

The halo correction has a substantial measured effect: for the sine diagnostic
with 128 interval cells and lambda .25, plain square-root halos gave maximum
value error \(1.14\times10^{-4}\); correcting the same geometry gave
\(4.44\times10^{-16}\). Under the construction's analytic assumptions, its
exact-arithmetic boundary-error bound has scale
\(h\exp(-\lambda R^2/4)\). That is only the boundary contribution, not a bound
on total floating-point, lattice, or PDE error.

In multiple dimensions, a ridge \(g(\omega\cdot x)\) is a one-dimensional
profile along direction \(\omega\). Radon-style representations combine these
profiles. With \(M\) directions and \(N\) interior centers per direction, the
flat construction uses roughly \(M(N+2R)\) neurons. There are two different
resolution limits: missing directions and inadequate resolution along a
direction. Increasing one cannot generally repair the other.

An empirical rule using the larger of the independently measured angular and
center error floors predicted 26 held-out 3D Gaussian value errors above
\(10^{-12}\) within a factor 1.027, and 27 Laplacian errors within 1.155. This is
a useful measured allocation rule for that family, not a general error theorem.

## 2. The actual nonlinear PDE procedure

Start with known physical basis functions \(\psi_1,\ldots,\psi_P\), chosen for
the domain and boundary conditions. Their unknown amplitudes form a vector
\(a(t)\in\mathbb R^P\). Construct neural approximations
\(\widetilde\psi_\ell\) once, then represent the evolving solution as

\[
\widehat u(x,t)=\sum_{\ell=1}^P a_\ell(t)\widetilde\psi_\ell(x).
\]

The functions are known; their amplitudes are determined by the PDE. This is
the central distinction that makes the method tractable.

For a flat neural implementation, let \(\phi(x)\in\mathbb R^J\) contain the
fixed neural features, including the constant feature. The explicit encoding
matrix \(E\in\mathbb R^{J\times P}\) contains the constructed readouts for the
basis functions. Then

\[
\widetilde\psi_\ell(x)=\sum_j\phi_j(x)E_{j\ell},\qquad
w(t)=Ea(t),\qquad \widehat u(x,t)=\phi(x)^TEa(t).
\]

Thus hundreds of thousands of neural readouts need not be independent unknowns.
The large vector \(w\) is obtained from the physical amplitudes \(a\) by a
known map. Later shared-bank/product architectures factor this computation
instead of storing an expanded flat network; their tanh counts do not count
all product features or independent PDE amplitudes.

This removes one particular large solve; it does not establish inference cost
as the only computational limit. Physical-coordinate count, quadrature,
nonlinear updates, cached arrays, and validation can still dominate.

To state the evolution precisely, take \(Q\) quadrature points. The synthesis
matrix \(S\in\mathbb R^{Q\times P}\) evaluates the constructed basis there:
\(S_{q\ell}=\widetilde\psi_\ell(x_q)\). The analysis matrix
\(A\in\mathbb R^{P\times Q}\) maps sampled function values to physical
amplitudes using explicit orthogonal-basis quadrature. For example, if
\(B_{q\ell}=\psi_\ell(x_q)\), quadrature weights form a diagonal matrix
\(W\), and squared orthogonal-basis norms form a diagonal matrix \(H\), then
\(A=H^{-1}B^TW\). Here \(H^{-1}\) is scalar normalization in each coordinate,
not a dense neural Gram-matrix inverse. This simple analysis is available
because the tested bases have known structure; arbitrary irregular features do
not automatically have such a map.

For an equation \(u_t=\mathcal F[u]\), the implemented coordinate update has
the form

\[
\boxed{\dot a=A\,\mathcal F[\widehat u],\qquad w=Ea.}
\]

The right-hand side uses actual constructed neural values and analytic spatial
derivatives. Nonlinear products are evaluated before analysis, so different
directions interact. This is modified spectral/Galerkin evolution through a
constructed neural representation. Neural encoding and quadrature errors mean
it is not automatically identical to an exact-basis Galerkin method; in
well-resolved tests the two agree very closely. In particular, analysis composed
with neural synthesis is approximately, not necessarily exactly, the identity.

For viscous Burgers, the complete forward recipe is:

1. Supply the initial function \(u_0\), viscosity \(\nu\), domain, and boundary
   conditions. No interior solution labels are required.
2. Choose physical modes and construct their neural features. Check value and
   first/second derivative encoding accuracy.
3. Analyze the prescribed initial function by quadrature to obtain \(a(0)\).
4. Evaluate the current neural field \(\widehat u\), its slope, and curvature.
5. Form the physical increment
   \(-\widehat u\widehat u_x+\nu\widehat u_{xx}\).
6. Analyze that increment with \(A\); advance \(a\) with a numerical time
   integrator. Construct the corresponding neural readouts from \(a\).
7. Repeat, checking independent residuals, time refinement, and spatial
   refinement. Increase resolution or report failure when needed.

Navier–Stokes uses the same idea with a three-component velocity, nonlinear
advection, viscosity, and a pressure projection that enforces incompressibility.
Stationary elliptic problems instead use preconditioned residual iteration or
Newton-CG for physical amplitudes. Inverse problems optimize a declared vector
of physical unknowns around these forward solves.

There is no large unconstrained neural readout fit in this procedure. There is
still quadrature, numerical integration, and sometimes nonlinear iteration or
small linear solves. Most time-dependent outputs are spatial networks indexed
by time, not one fixed network taking space and time as inputs. The special
linear wave construction does produce a fixed four-input network.

## 3. What the results establish

### Direct construction, nonlinear solving, and comparison to PINNs

The first experiments encoded analytic or classically evolved profiles. These
established representation accuracy, not an independently neural PDE algorithm.
Later native experiments evaluated the constructed field inside every PDE
update without loading a reference trajectory. Classical solvers then supplied
independent validation. Both stages remain in the repository and must not be
confused.

| Experiment | Measured result | What it establishes / limitation |
|---|---|---|
| Linear 3D wave, fixed four-input network | Roughly \(10^{-14}\) error | Constructive spacetime representation for a supported linear IVP |
| Anisotropic 3D heat | Roughly \(2\times10^{-15}\) | Numerically propagated initial Radon profiles; not zero numerical work |
| Cole–Hopf Burgers | Roughly \(6\times10^{-15}\) | Special nonlinear equation with a known linearizing transform |
| Native QUILL Burgers, final time .5 | \(2.50\times10^{-13}\) at step .001; \(2.52\times10^{-14}\) at .0005 | Actual neural RHS evolution; smooth supported periodic case |
| Cubic B-spline/ReLU-cubed control | Second-order convergence in the tested formulation | Geometry-based explicit coefficient updates also work outside tanh; classical spline collocation |
| Native NS time polynomial | Degree-8/time-integration difference \(2.6\times10^{-15}\), spatial error about \(1.97\times10^{-9}\) | Local Taylor recurrence for a finite projected ODE; not a global nonlinear solution formula |

The strongest head-to-head test is the nonlinear Dirichlet problem on
\([0,\pi]^2\):

\[
-\Delta u+5u^3=3\sin x\sin y+0.3\sin(3x)\sin(2y),\qquad u|_{\partial\Omega}=0.
\]

| Method | Relative solution error | Recorded time |
|---|---:|---:|
| Constructed QUILL, 12 modes per axis | \(6.38\times10^{-8}\) | .0159 s |
| Constructed QUILL, 20 modes per axis | \(1.17\times10^{-12}\) | .0884 s |
| Matched classical sine basis, 20 modes per axis | \(1.17\times10^{-12}\) | .0326 s |
| DeepXDE PINN, width 32, seed 0 | \(3.20\times10^{-5}\) | 83.0 s |
| DeepXDE PINN, width 32, seed 1 | \(3.72\times10^{-5}\) | 95.2 s |
| DeepXDE PINN, width 64, seed 0 | \(2.45\times10^{-5}\) | 281.2 s |

The constructed times include setup, solving, and an independent residual
check. PINN times include diagnostic callbacks. Imports and reference generation
are excluded; concurrent machine activity can affect timings. PINNs used
DeepXDE 1.15, PyTorch float64, four tanh hidden layers, Glorot initialization,
hard boundary factors, 15,000 Adam steps at .001 followed by nominal 5,000
L-BFGS iterations. Actual completed counts are recorded. References enter
evaluation, not either solver's training/updates. This is one PDE and three
baseline configurations, not an exhaustive modern-PINN comparison.

At 12 modes per axis, sharing 1D QUILL banks and using exact product gates
reduced the tanh count from 122,400 unmerged flat-ridge neurons to 442, with 144
physical amplitudes. At 20 modes there are 714 tanh units and 400 amplitudes.
This changes the architecture and exploits tensor structure; it is not a free
reduction of every dimension of the computation. The classical control is also
faster to evaluate. The data support avoiding a costly neural search, not an
intrinsic accuracy/speed advantage of tanh over sine functions.

### Boundaries, stiffness, and nonsmooth solutions

On a smooth manufactured annulus problem, the constructed polar-coordinate
basis reached \(2.19\times10^{-12}\) relative value error with 190 physical
coordinates and 546 tanh units. Separate nonzero Dirichlet, mixed Robin/Dirichlet,
and diffusion-contrast 10,000 tests passed the recorded checks. Small Newton-CG
and PDE preconditioner solves are part of the method. A mapped annulus is not
evidence for arbitrary CAD geometry or arbitrary boundary conditions.

Stiff Allen–Cahn front errors decreased from .0263 to .00275 to
\(2.82\times10^{-5}\) as mode cutoff increased from 32 to 64 to 128. The next
reported difference, \(5.62\times10^{-7}\), was below the reference refinement
difference \(1.68\times10^{-6}\); it is not a certified exact error. Coarse
solutions violated physical bounds despite finishing integration, motivating
sampled invariant guards.

Shock tests use conventional Rusanov finite volume and SSPRK3, with the field
represented as the derivative of a ReLU primitive. They refined approximately
first order in L1, conserved to below \(1.7\times10^{-15}\), and had nonpositive
recorded entropy defects. This is a different backend, not a demonstration
that smooth QUILL alone handles shocks.

A quadratic-reaction problem has a theoretical blowup-time bracket
\([5/6,1]\). Numerically reaching amplitude 10,000 around .838906 is a stopping
event, not a proof of the precise blowup time. Positivity, amplitude, and
resolution checks prevent silently presenting such trajectories as reliable.

### Adaptive directions and centers

In periodic 2D nonlinear reaction–diffusion, a finite integer-frequency
dictionary supplies candidate ridge directions. We select active directions
from PDE increments, then choose the smallest tested center count and a
bandwidth passing value and Laplacian checks. This is adaptive spectral
selection; continuous angles are not learned.

At the tested cutoff 18, fixed allocation used 408 directions and 80,056 tanh
units, reaching \(1.53\times10^{-7}\) relative error in 9.62 s. Adaptation with
rejected-window replay used 110 directions and 19,098 units, reaching
\(2.18\times10^{-7}\) in 3.28 s. Its exact-trigonometric control reached the
same error in .790 s. A one-direction control used 117 units.

The important failure was temporal: adding missing directions only after a
window had evolved left \(7.82\times10^{-5}\) error. Restoring the beginning
of that window, adding the directions, and replaying reduced the error about
358-fold. Later adequate resolution cannot automatically recover dynamics lost
earlier. The replay run used 260 actual RK steps rather than 120 nominal steps.
Candidate analysis still stores dense arrays; sparse output is not a fully
sparse discovery algorithm or a global error guarantee.

### Three-dimensional Navier–Stokes and the animation

Supported flow is smooth incompressible flow on a periodic cube. The algorithm
uses constructed 1D QUILL banks, exact product gates, quadrature, a Fourier
pressure projection, and RK integration. The native field and its derivatives
enter every stage. It is a numerical fluid solver in a structured neural
representation, not an Adam PINN or a closed-form Navier–Stokes solution.

Taylor–Green tests through time 1 at viscosities .05 and .01 differed from
finer spatial runs by \(2.38\times10^{-6}\) and \(7.22\times10^{-6}\).
Time-step-halving differences were about \(2\)–\(4\times10^{-12}\). Sampled
divergence was below \(3.06\times10^{-15}\); sampled physical residuals were
roughly \(7.5\times10^{-5}\) to \(1.9\times10^{-4}\). This distinguishes
spatial error from much smaller time error.

The animation uses a more complicated mixture of Taylor–Green, ABC, and cyclic
second-harmonic initial velocity, viscosity .075, and final time 4. It evolves
480 steps and saves 121 frames. The largest of five sampled differences from
the finer spatial run was \(3.71\times10^{-6}\); the largest time-halving
difference was \(5.99\times10^{-10}\). Maximum sampled physical residual was
\(1.14\times10^{-4}\), divergence \(3.80\times10^{-15}\), and relative energy
balance defect \(5.11\times10^{-16}\). These are finite numerical checks, not
certificates over every point and time.

That representation uses 867 shared tanh units but 59,049 velocity coefficient
entries, along with product operations. Calling it an 867-parameter fluid
solver would be incorrect. Passive sheets and tracers use interpolated velocity
for visualization; this interpolation does not drive the PDE. The roughly
23.7-second recorded run includes particle evolution and recording, so it is
not the same timing scope as the earlier benchmark table. We have not tested
industrial turbulence, complicated walls, or general high-Reynolds-number flow.

### Dimensions: structure matters more than the input count

| Test | Result | Essential qualification |
|---|---|---|
| Broad angular representation of a known smooth Gaussian | Near roundoff in 4D, about \(6\times10^{-11}\) in 5D with millions of neurons | Known-function encoding, not solving a generic unknown PDE |
| Linear heat/advection with three supplied ridge profiles | Through 100D, 663 neurons, roughly \(10^{-15}\) on the unit-ball test | Directions and sparse structure are provided |
| Interacting nonlinear reaction–diffusion dimension sweep | Tested smooth 5D problem resolved; 6D refinement hit an 800 MB experiment budget | Classical Fourier Galerkin followed by neural compilation, not native high-D QUILL dynamics |

There is no single demonstrated maximum dimension. A supplied low-complexity
representation can survive high ambient dimension; broad directional coverage
and nonlinear mode mixing remain expensive. The 800 MB limit was a chosen
experiment budget, not a physical limit of the laptop. No curse-of-dimensionality
removal or automatic manifold discovery was established.

### Measurements, inverse problems, and uncertainty

Measurements are supported by fitting a small physical parameter vector around
the forward solver. They are not magically enforced by the construction. The
inverse stage explicitly uses nonlinear optimization/least squares. This is
distinct from fitting all neural readouts, and it can still be expensive.

- Six diffusion-tensor parameters from 36 observations: noiseless relative
  parameter error about \(2.8\times10^{-9}\); 1% noise gave roughly 1.2–5.3%
  error across eight runs.
- Burgers viscosity and two initial amplitudes from 24 measurements: noiseless
  viscosity error \(1.14\times10^{-11}\), held-out field error
  \(2.81\times10^{-13}\). At .1%/1% noise, median viscosity error was
  1.21%/12.1%, while field error was about .033%/.324%.
- A 17-coefficient unknown Burgers initial profile from 48 measurements:
  at .5% noise, early observations produced about 2.2–4.3% initial-profile
  error; late observations produced about 10.2–10.7%. Late-data future
  predictions were much better, about .18–.28%, because predicting a smoothed
  future does not require recovering all earlier fine structure.

The profile problem uses a smoothness prior and known noise to select its
strength. Regularization supplies otherwise unrecovered information; it does
not reverse information loss. A constructed inverse-crime control achieved
near-zero data loss but failed independent forward refinement. Sensor-offset
mismatch was also flagged. These checks matter more than optimizer success.

Local uncertainty estimates were tested for the small parameter problem, with
27–29 of 30 runs covered by nominal 95% intervals depending on the parameter.
These are small-sample checks under a correct model and known noise, not a
general Bayesian posterior. Active bounds and deficient local rank disable
the covariance estimate. General unknown coefficient fields, arbitrary
observation operators, and universal data assimilation are not established.

### The forced-Navier–Stokes blowup moonshot

The complete numerical forcing evaluator for the September 2026 manuscript
has not been implemented. Its blowup solution has not been reproduced.

The separate synthetic test prescribed a shrinking divergence-free Gaussian
swirl. Moving the centers and widths with its known scale maintained about
\(1.4\times10^{-15}\) core representation error with 1,275 tanh units while
the width contracted 100-fold. This demonstrates representation adaptivity,
not discovery of that geometry from dynamics. Its required force is singular
in the limit, so it is not a substitute for the manuscript's claimed smooth-force
construction. A fixed finite energy-stable Galerkin space cannot show literal
infinite speed under bounded smooth forcing; resolving concentration requires
increasing resolution or changing geometry.

## 4. Verification is part of the method

We deliberately constructed examples where apparently good diagnostics fail:
two successive truncations missed the same forcing component despite 70.7%
solution error; regularly spaced time probes missed a forcing pulse despite
33.3% error; a 3D force that vanished at the endpoint produced a 7.65% error
while endpoint-only checks passed.

The current checks combine independent spatial residual points, irregular
temporal and accepted-step probes, initial-condition checks, spatial and time
refinement, and relevant physical invariants. For fluids, these include
divergence and energy balance. For inverse problems they include forward
refinement at the inferred parameters and observation discrepancy. Passing
sampled checks is evidence, not a continuum certificate. A final residual alone
does not control accumulated trajectory error. A tiny projected residual can
coexist with a poorly resolved physical solution.

## 5. Where it fits among PINNs and numerical solvers

A conventional residual-trained PINN chooses network parameters to reduce a
combination of PDE, initial/boundary, and measurement losses. These are numerical
solvers too. Solving from the equation and initial/boundary conditions without
interior solution labels is not unique to PINNs; classical PDE solvers do it.

Our best-supported advantage is to prescribe a good representation and solve in
structured coordinates, avoiding simultaneous discovery of geometry and
readouts through a large neural optimization. Its price is reliance on a useful
basis, coordinate map, quadrature, boundary treatment, and PDE-specific solver.
The present common interface dispatches among supported classes; it does not
turn an arbitrary PDE declaration into a reliable solution automatically.

| Desired PINN capability | Current position |
|---|---|
| Forward solve without interior solution labels | Demonstrated on supported linear and nonlinear classes |
| High precision | Demonstrated for several smooth cases; resolution and conditioning still matter |
| Continuous differentiable field | Available for smooth neural snapshots; a fixed global spacetime network is not general |
| Sparse data and physical parameter inference | Demonstrated on stated finite-dimensional inverse problems |
| Arbitrary geometry and boundary conditions | Limited boxes, periodic domains, and a mapped annulus; not general |
| High-dimensional problems | Strong when useful structure is supplied; generic scaling unresolved |
| Shocks and singularities | Separate shock backend and detection/refinement studies; no universal smooth-network solution |
| Reliable uncertainty | Local conditional estimates only |
| Many-query or parameterized solution operator | No amortized neural-operator result established |
| Faster inference than competing methods | Not established; matched classical evaluation was faster |
| Automatic discovery of good neural geometry | Finite-dictionary direction selection; no general learned manifold or moving-geometry solver |

There is substantial related work. [Evolutional Deep Neural Networks](https://arxiv.org/abs/2103.09959)
evolve neural parameters using PDE dynamics after initialization.
[Neural Galerkin](https://arxiv.org/abs/2203.01360) combines sequential residual
projection with adaptive sampling. The [Random Feature Method](https://arxiv.org/abs/2207.13380)
combines prescribed neural features with numerical PDE discretization; its
[time-dependent extension](https://arxiv.org/abs/2304.06913) solves least-squares
systems. Avoiding conventional end-to-end Adam training is therefore not itself
a new contribution.

The specific contribution to investigate here is the explicit high-accuracy
QUILL encoder, its derivative/boundary control, and how it removes a large
neural readout fit when a tractable physical-coordinate analysis is available.
This checkpoint does not establish priority or publication novelty. Such a
claim needs a narrower comparison with these methods, not just ordinary PINNs.

The limitations of our PINN comparison also matter. The [expert training guide](https://arxiv.org/abs/2308.08468)
documents substantial effects of architecture and training choices. Conversely,
a [systematic PINN/FEM comparison](https://arxiv.org/abs/2302.04107) found FEM
better in solution time and accuracy for its tested problems. Neither result
licenses a universal claim that PINNs always fail or that our baseline is their
best attainable performance.

## 6. What is useful, and what would establish a stronger result

The present method is most defensible for smooth low-dimensional or supplied
low-complexity problems with convenient bases, especially when an explicit
neural representation is useful downstream. It is also an experimental tool
for separating representational capacity from neural optimization difficulty.
Its inverse wrappers benefit from a fast forward solver, although the matched
classical solver often supplies that benefit more cheaply.

The neural representation's potential convenience for composition,
differentiation, or integration with learned models is a plausible niche, not
a measured downstream advantage in these tests. If the only desired output is
a solution on a simple domain, our matched classical solver is currently the
stronger practical default.

The next scientifically decisive work is not another easy smooth PDE:

1. Find a case where geometry discovered from the PDE concentrates the
   representation more efficiently than a matched adaptive classical basis.
   The prescribed shrinking-vortex test suggests what to measure, but has not
   solved this problem.
2. Compare explicit construction against random-feature and Neural Galerkin
   methods, including setup, all physical amplitudes, derivative errors,
   quadrature/storage, inference, and independent validation cost.
3. Extend one application that actually needs the neural representation, then
   demonstrate its downstream benefit rather than assuming one.
4. Strengthen reliability and domain support while retaining explicit failure
   states. A polished universal-looking interface must not hide unsupported
   mathematics.

We did build something technically substantial: a constructive route from
interpretable basis geometry to accurate neural PDE solutions, with real
nonlinear dynamics, sparse-data inversion, and explicit negative controls.
We did not discover a universal nonlinear PDE formula or establish that neural
representations improve on the underlying classical discretization. Those
boundaries define the next research question rather than diminish the results
already measured.

## Evidence and reproduction index

All paths below are relative to this file. Earlier exploratory outputs are
retained; the stages and caveats above determine which claims they support.

| Topic | Code or instructions | Measurements / figures |
|---|---|---|
| Experiment entry point and history | [Experiment README](../experiments/expF19_radon_direct_pde/README.md) | [Cumulative status](../results/checkpoint_F_applications/expF19_radon_direct_pde/status.md) |
| Exact encoder and historical derivation | [Encoder](../experiments/expF19_radon_direct_pde/quill_boundary.py), [September construction](appendix_notes/paper_v5_replacements/organized_v7_package/sections/7_appendix/01_Construction/09_26_sl.tex) | [Boundary method](../results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review/boundary_method.md), [allocation predictions](../results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review/allocation_prediction.json) |
| Linear and early nonlinear construction | [Experiment README](../experiments/expF19_radon_direct_pde/README.md) | [Linear results](../results/checkpoint_F_applications/expF19_radon_direct_pde/linear/), [nonlinear report](../results/checkpoint_F_applications/expF19_radon_direct_pde/nonlinear/report.md) |
| Native dynamics | [Native tanh](../experiments/expF19_radon_direct_pde/native_tanh.py) | [Native dynamics results](../results/checkpoint_F_applications/expF19_radon_direct_pde/native_dynamics/) |
| Supported API and reproduction commands | [Solver README](../experiments/expF19_radon_direct_pde/solver/README.md) | [API metrics](../results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver/api_metrics.json) |
| PINN and classical comparisons | [Baseline runner](../experiments/expF19_radon_direct_pde/general_baseline.py) | [Comparison metrics](../results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver/comparison_metrics.json), [figure](../results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver/comparison.png) |
| Curved boundaries | [Domain study](../experiments/expF19_radon_direct_pde/general_domains.py) | [Domain report](../results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver/domain_report.md) |
| Shocks, stiffness, reaction blowup | [Hard-case runner](../experiments/expF19_radon_direct_pde/general_hard.py) | [Hard-case report](../results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver/hard_results.md) |
| Dimensions | [Dimension study](../experiments/expF19_radon_direct_pde/general_dimensions.py) | [Encoding scaling](../results/checkpoint_F_applications/expF19_radon_direct_pde/scaling/report.md), [PDE frontier](../results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver/dimension_frontier.json) |
| Adaptive directions | [Adaptive study](../experiments/expF19_radon_direct_pde/adaptive_ridge_study.py) | [Adaptive metrics](../results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup/adaptive_metrics.json) |
| Negative controls and reliability | [Verification study](../experiments/expF19_radon_direct_pde/verification_study.py) | [Verification metrics](../results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup/verification_metrics.json) |
| NS refinement | [Extended NS study](../experiments/expF19_radon_direct_pde/native_ns_extended.py) | [NS report](../results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup/ns_report.md) |
| Animated fluid run | [Animation simulation](../experiments/expF19_radon_direct_pde/fluid_animation_sim.py) | [Validation](../results/checkpoint_F_applications/expF19_radon_direct_pde/fluid_animation/validation.json), [video](../results/checkpoint_F_applications/expF19_radon_direct_pde/fluid_animation/preview.mp4) |
| Inverse parameters and local uncertainty | [Inverse runner](../experiments/expF19_radon_direct_pde/general_inverse_uq.py) | [Inverse metrics](../results/checkpoint_F_applications/expF19_radon_direct_pde/capabilities/inverse_metrics.json), [UQ metrics](../results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver/inverse_uq_metrics.json) |
| Unknown initial profile | [Profile study](../experiments/expF19_radon_direct_pde/inverse_profile_study.py) | [Profile metrics](../results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup/inverse_profile_metrics.json) |
| Blowup moonshot limits | [Synthetic geometry probe](../experiments/expF19_radon_direct_pde/moonshot_geometry.py) | [Source audit](../results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup/moonshot_source_audit.json), [probe metrics](../results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup/moonshot_geometry_metrics.json) |

This consolidation did not rerun the experiments. It checked the saved metrics,
implementation, and study disclosures. Focused solver tests are in
`tests/test_quill_general_solver.py`, `tests/test_quill_verified_solver.py`, and
`tests/test_quill_ns_guards.py`; execution instructions are in the solver README.
