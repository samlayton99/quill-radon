# Radon catalogue audit: pure-MLP repair and cached native solvers

Read-only research audit, 2026-10-01. Scope assigned by root: route 1 and
cached route 2, including sparse/angular/active-axis variants and their
application controls. Latest direct-profile, frame-iteration, streamed and
nonredundant-disk methods are outside this audit. No new experiments run.

## Symbols and costs

- `d`: input dimension; `p`: polynomial/profile degree; `P=binom(p+d,d)`:
  independent full total-degree coordinates. `N`: interior centers;
  `R=ceil(sqrt(N))`; `H=N+2R`; `M`: directions; `B=MH`: tanh neurons.
- `Q`: all physics/condition evaluation points; `E`: inference points;
  `D`: number of derivative channels, field count held fixed; `K`: total
  Krylov products across the nonlinear solve. `b`: bounded preconditioner
  block size. These costs do not assume K stays bounded with resolution.
- `C_enc(N,p)`: explicit QUILL encoding of p+1 known profiles, including
  contour quadrature and halo partial fractions. Its retained bank is
  O(Hp). It is not a target fit. Keeping this term explicit avoids hiding
  contour quadrature order and high-precision construction costs.
  In the implementation, m=(R+1)//2, contour count q_H=max(96,8mk+32),
  k=ceil(delta/(8*tau)), tau=pi/(2gamma). For polynomial profiles, a safe
  implemented upper bound is O(q_H^3+(H+q_H)p+m*q_H*p+m^3+m^2*p): the
  first term permits dense Legendre-quadrature setup, the last two are
  current scaled-partial-fraction construction. Working storage is
  O(q_H^2+Hp+m*q_H+q_H*p). At present small resolutions q_H is often96;
  asymptotically it must not silently be treated as a universal constant.
- Every cached solve below adds O(K D Q P) iterative product work and
  O(D Q P) feature-cache storage. Bounded Gram preconditioning adds
  O(D Q P b + P b^2) work per build and O(Pb) factors. If b grows to P,
  that advantage disappears. No full global residual Jacobian is assembled,
  but the dense feature jets remain expensive.
- Any flat ordinary exported MLP takes O(E B d) dense-layer work for fixed
  outputs and O(Bd) model storage. Geometry-aware evaluation can share the
  dot product within a direction, O(E M(d+H)); that optimized implementation
  is not the same operation count as ordinary dense Torch export. Batched
  evaluation adds O(batch*B) activation workspace; fixed-order derivatives
  multiply the relevant channel count.

## Distinct implemented/proposed methods

### 1. Route 1: deep tanh polarization compiler + neural physics solve

Known one-dimensional Legendre profiles are encoded with QUILL. Their
products are implemented inside an ordinary MLP by square subnetworks and
`ab=B0^2[S((a+b)/(2B0))-S((a-b)/(2B0))]`, where S approximates the square.
There are no external product gates in the deployed forward graph. A
balanced tree handles products across coordinates. The PDE coefficients
are solved from zero using actual neural derivatives; this is not a solved
polynomial field compiled after the fact.

Let `a` be a power of two covering the maximum number of nonzero coordinate
factors per feature (`a=O(min(d,p))`). Neurons are O(H(d+Pa)), depth
O(log a). The current unfused implementation stores *dense* affine layers,
with parameter count `W=O(H[d^2+Pad+P^2 a^2])`; many entries are zero but
their memory/arithmetic is not automatically sparse. Construction costs
O(W+C_enc), evaluation O(EW), model memory O(W), activation workspace
O(batch*H(d+Pa)). Cached PDE setup adds O(D Q W) work, O(D Q P) storage;
then the common Krylov/preconditioner costs above. Affine fusion is optional
and can enlarge dense layers, so do not claim it automatically reduces cost.

Evidence: `solver/pure_mlp.py`, `pure_mlp_depth_study.py`,
`results/.../pure_mlp_depth/metrics.json` and `pde_only_resources.json`.
Cold nonlinear elliptic solve `-Delta u+5u^3=f` on a square: p4, P15,
N257, lambda .2, 9,312 tanh units, 3.367 MB dense model; ordinary relative
field error 6.847e-16, raw AD PDE RMS 3.562e-15. One 2D PDE floor success,
not a broad consistency demonstration. 4D and 8D checks test a constant
and a coordinate product only, not full PDE spaces. Full 4D p4 estimate:
123,384 units, 43,231,450 stored parameters, 345.85 MB weights,
651.84 MB for one 1000-point widest-layer activation. No full 4D depth-PDE
trial. Lambda/N tests show value-floor accuracy does not guarantee
second-derivative floor (e.g. N129,.2 maximum second derivative discrepancy
3.109e-9; N257,.2 7.816e-14 in the known degree3 bank). No general
neuron-error scaling exponent was established. Successful small proof of
architecture, expensive as presently implemented.

### 2. Exact squared-ReLU product compiler (proposal, not tested)

User permitted a separate squared-ReLU comparison. The theoretical square
identity `x^2=ReLU(x)^2+ReLU(-x)^2` makes multiplication exactly representable
by affine maps and squared-ReLU units. No RePU PDE experiment or implementation
was found in this route. Tanh/GELU remained the main requested activation
family; the implemented route1 is tanh, not GELU. Do not report a result,
empirical rate, or tested complexity for the proposed RePU/GELU routes.
Potential exact-arithmetic arithmetic-circuit scaling is not measured dense
MLP cost and does not eliminate the subsequent PDE coefficient solve.

### 3. Cached disk ridge coordinates (Logan--Shepp construction)

`RidgePINNFeatures` uses degree-n ridge polynomials on n+1 reference angles,
and an explicit common-grid angular reproducing identity with M>=p+1.
QUILL encodes the known U_n profiles. The resulting actual model is flat
Linear/Tanh/Linear, with readouts w=Ea and only P coordinates solved.
The optional `backend='polynomial'` is an independent exact-basis control,
not the same architecture and never its warmstart.

Disk angular maps occupy O(MP), not the ball method's O(MpP). Their current
degree-wise recurrence construction is O(Mp^3) arithmetic, plus C_enc and
known-profile checks. Cached neural feature evaluation/setup is
O(D Q M(Hp+P+d)); model compilation adds O(MP+MHp+Bd), peak persistent
storage O(MP+Hp+DQ P+Bd). The generic solve term O(K D QP) remains.
Inference after compilation has the flat-MLP cost above.

Evidence: `solver/ridge_pinn_features.py`, `ridge_pinn_study.py`,
`results/.../ridge_pinn/ridge_pinn_resolution_quill_*.json`.
Cold nonlinear disk PDE `-Delta u+u^3=f`, target for manufactured data
`exp(.3x)cos(1.3y)+.1xy`, N257,.2:

| p | M | B | ordinary field relative L2 |
|---|---|---|---|
| 4 | 5 | 1,455 | 8.991e-3 |
| 8 | 9 | 2,619 | 2.796e-6 |
| 12 | 13 | 3,783 | 8.720e-11 |
| 16 | 17 | 4,947 | 8.764e-16 |
| 18 | 19 | 5,529 | 4.049e-16 |

p18 raw Torch PDE RMS 5.256e-15. This is rapid/spectral-like convergence
in p for this analytic target until roundoff, not a universal rate in B.
At fixed p, extra N only improves profile encoding, not angular function
space. M<p+1 is an explicitly allowed negative control, not an admissible
equivalent resolution. Successful disk method; failure on a space-time
square exposed domain mismatch (sampled basis condition p28 about3.09e9,
versus3.98 on the disk). Strong disk-coordinate Burgers follow-up reached
2.44e-7 space-time /1.19e-5 final-time errors, iteration limit.
At p12,N257, M2/4/8/13 yielded field errors4.4e-2/1.3e-3/1.5e-7/8.7e-11;
at p12,M13, N33/65/129/257 yielded6.2e-7/1.2e-9/8.7e-11/8.7e-11.
Those separate sweeps are the clearest directly measured M/N tradeoff.

### 4. Enclosing-ball/box moment coordinates + tensor sphere cubature

`BallRidgeFeatures` maps a physical box into its enclosing unit ball,
constructs the coefficients of known Legendre-coordinate features from
algebraic ball moments, applies tensor spherical cubature, and explicitly
encodes Gegenbauer directional profiles with QUILL. Ordinary forward and
actual residuals contain no products/polynomials; polynomial expressions
are construction coordinates only. Full angular rule has M=(p+1)^(d-1).

Unlike method3 it retains `profile_map[M,p+1,P]`: O(MpP) storage. Cached
jet construction costs O(D Q M(Hp+pP+d)), plus algebraic map construction
G(d,p,M), then O(K D QP) solve work. Persistent construction/solve memory
O(MpP+Hp+DQ P+Bd), with additional moment/cubature workspaces. Algebraic
map arithmetic includes a dense O(MP^2) contraction across monomials and
coordinate features; moment-expansion work and arbitrary precision also
matter. For clarity, keep G explicit rather than imply this constructor
is O(B). p>8 uses >=50 decimal digits for the *known map only*; PDE and
export stay float64. With balanced p~N,M~N^(d-1),P~N^d,Q~P (fixed d),
cached jet contraction alone is O(N^(3d)); map storage O(N^(2d)). Thus
this version never solved the asymptotic construction/memory wall.

Evidence: `solver/ball_ridge_features.py`, `pure_mlp_repair_summary.py`,
`results/.../pure_mlp_repair/flat_ridge_ns4d_*.json`,
`route2_generality/burgers/route2_burgers_summary.json`.
4-input manufactured Navier--Stokes control, p4, M125,P70 per field,
four outputs: N129,lambda .25,B19,125, ordinary velocity error7.675e-14,
pressure2.506e-14; raw momentum RMS1.903e-14/1.219e-14/1.217e-14.
N257,.2 B36,375 has *worse* ordinary velocity1.622e-13, despite stable
grouped field2.27e-15. Do not call this consistent1e-14 ordinary NS floor.
All spatial face traces determine the ideal low-degree velocity, making
this architecture/conditioning control substantially easier than a hard IVP.

Box coordinates improve smooth unforced Burgers, but did not reach floor:

| p | B (N257) | ordinary space-time relative L2 | final-time relative L2 |
|---|---|---|---|
| 24 | 7,275 | 1.058e-7 | 5.007e-6 |
| 28 | 8,439 | 6.132e-9 | 7.272e-7 |
| 32 | 9,603 | 2.989e-10 | 1.010e-8 |
| 36 | 10,767 | 1.676e-11 | 4.444e-9 |

Best raw AD interior PDE RMS5.753e-11; status line_search_failed. Here
roughly16--21x field improvement per1164neurons over the displayed range,
not a proven asymptotic rate. N513 representation-only control and nine
lambda/N profile calibrations show N is not the remaining dominant error.
The independent exact polynomial p20 control also remains~3.05e-6, helping
separate profile encoding from approximation/solve limitations.

### 5. Low-degree sparse angular cubature

Same ball-coordinate constructor and PDE solve, different angular rule:
degree<=2 uses M=d^2; degree3 uses
M=d+2*binom(d,2)+4*binom(d,3). Established signed Stroud/Smolyak-type
rules preserve required even angular moments. Signed weights enter map
construction, not a signed PDE loss. This changes exponential-in-d
direction cost to polynomial-in-d at those fixed low degrees.

Use method4's compute/memory formulas with the smaller M. p2 has
P=O(d^2),M=O(d^2),so map storageO(d^4);p3 hasP=O(d^3),M=O(d^3),so
map storageO(d^6). Q~P gives O(D N d^4 +D d^6) cached setup at p2 and
O(D N d^6+D d^9) at p3, plus map construction and iterative solves.
This is not a dimension-independent method and is not implemented at
arbitrary degree. Standard dense export adds O(dMN) model storage/work.

Evidence `route2_scaling_features.py`, `route2_scaling_study.py`,
`results/.../route2_generality/scaling/summary.json`:
10D quadratic15,300neurons/66coordinates:1.537e-15 field;
20D quadratic61,200/231:6.916e-15 field,2.622e-15 rawPDE;
10D cubic88,740/286:8.527e-15 field,5.276e-15 rawPDE.
Successful controlled low-degree tests; all-face traces already determine
those ideal polynomials, so no general hard high-D inference claim.
Derivative-profile sharing reduced the same20D run11.01s to2.64s with
bitwise-identical coefficients,~445MB process peak. This is a constant-factor
implementation improvement, not a new approximation or convergence theorem.

### 6. Declared-active-axis ridge construction

`ActiveAxisRidgeFeatures` applies method4 only to k explicitly supplied
active coordinates; inactive first-layer columns are exactly zero. The
PDE still determines unknown readouts from zero. This is a disclosed prior,
not learned manifold discovery. Replace d by k in M/P/constructor/solver
costs; storing input points costs Qd and the ordinary dense export still
stores/executes Bd (unless sparse structure is exploited). This distinction
must appear in any claimed high-dimensional scaling.

10D ambient, two known active inputs, exponential manufactured solution,
p8: N129 B1377 ->4.589e-10 field; N257 B2619 ->2.791e-12. N257 rawPDE
3.811e-11, line_search_failed. No floor or learnt compression demonstrated.
Evidence `route2_scaling_features.py`, scaling/d10_active2_*.json.

### 7. Sequential time-slab ridge networks

Uses method4 on each of L prescribed time slabs; the previous network's
terminal trace is the next slab's initial condition. Each new readout
starts zero. No classical marcher supplies the states. Representation is
an explicitly indexed collection of local MLPs, not a single global MLP.
Total solve work sum_l T_l, peak one slab's solver+retained models, total
deployment storage L*Bd (shared geometry can reduce metadata); inference
chooses one slab, therefore one model's cost, not L times it.

Four p24/N257 Burgers slabs:4*7275neurons total, space-time2.287e-9,
final8.876e-9,23.99s total; first3 slabs line_search_failed, lastconverged.
It improved over globalp24, but no matched-accuracy cost advantage or floor
was shown. Evidence route2_burgers_study.py, burgers/route2_burgers_slabs_*.json.

## Controlled modifications and failures to preserve

- **Additional PDE collocation on the faces:** same p36 box network,
  more residual constraints, no new representation. Under the bounded
  solve it worsened Burgers to1.376e-7 space-time/9.277e-5 final,
  budget_exhausted. Previous p36 geometry map reused, but solution readouts
  stillzero;2second setup claim would be false (original map18.06s).
- **More centers without more function-space degree:** N513 fixed solved
  coefficients was a representation diagnostic, not a new PDE solve and
  not evidence that overresolution fixes the outstanding Burgers issue.
- **Ill-conditioned disk coordinates on a square:** failed geometry choice
  corrected by box coordinates; it did not invalidate disk construction.
- **Naive high-degree moment arithmetic:** overflow/cancellation fixed with
  Python integer factorials/parityzeros/highprecision known-map construction;
  old preflight failures are historical, not current errors. Some degree24
  known-coordinate derivative errors still reached2.14e-5, so construction
  conditioning remained limited. Ordinary combined solution can be more
  accurate than worst basis column; report both when interpreting floor.
- **Strong-residual smooth shock approximation:** not a PDE solution method.
  tanh shock sequence has L1 errorO(1/gamma), strong Burgers residual
  O(sqrt(gamma)); entropy separates compressive vs expansive jumps with
  same flux balance. No entropy-aware shock solver was implemented here.
- **Underresolved angular cubature:** d4,p6 known ridge with M64 instead
  of343 has0.817maximum function error. Additional centers cannot repair it.
- **Old interleaved Sobol face sampling:** coverage bug; corrected sparse/
  active-axis results use independent face sequences. Older fulltensor
  runs are marked legacy_interleaved and should be cost controls rather
  than strongest accuracy evidence.
- **Original cell-count halo:** archived under pure_mlp_depth/
  initial_cell_halo_convention. Current public N means interiorcenters,
  and halo isceil(sqrt(N)); do not accidentally cite archived figures.

## Coverage, not separate representations

The cached residual framework adds unknown physical parameters, sparse
measurement residuals, boundary derivatives, explicitly aggregated integral
constraints and inequality hinges. Small joint parameter variables do not
change the dominant feature-cache costs. Dense arbitrary aggregation data
can add their own memory; low-rank/noisy identifiability is not solved by
calling the residual engine. Degree continuation reuses a previous *neural*
solution, not a different PDE solver. The declarative driver routine is
an integration/adaptivity method, not evidence of universal convergence.

Annulus with variable diffusion, cubic reaction, mixed outerDirichlet/
innerRobin: field2.03e-15. Neumann plus integral gauge:1.95e-15. Joint
inverse with12exactanchors:3.15e-15 and kappa1.7. No-anchor inverse has
an exact scale ambiguity and is correctly labeled unidentifiable;
different starting parameters yield different solutions despite tinyPDE
residual. Noisy anchors have nonzero residuals and may terminate
line_search_failed; floor on exactdata does not imply noise-floor inference.
All use p10,N257,M11,B3201,66fieldcoordinates. Evidence constraints/
forward_constraints_summary.json and inverse_summary.json.

Parameterized family: one(x,mu) MLP solves `-mu*u_xx+u=1`, zeroendpoints,
mu in[1,2]; p18,N257,M19,B5529. No solution labels. Field1.773e-12,
parameter derivative4.827e-10,rawPDE6.932e-12,1.47s. Degree6/10/14/18
continuation reduces fresh PDE1.61e-3/5.32e-6/1.216e-8/1.852e-11.
Useful reuse, not floor or general neural operator. Evidence
route2_parametric_study.py and route2_generality/parametric/metrics.json.

## Rate interpretation

There is no single demonstrated error-vs-neurons law across these methods.
For analytic targets, degree refinement often produced spectral-like
decrease until truncation/conditioning/solver/roundoff intervened. Increasing
N at fixed degree only refines the known univariate profile realization.
Uniform tensor directions scale M~p^(d-1); if N~p then B~p^d, and an
ideal analytic error exp(-c p) becomes exp(-c B^(1/d)). This is a conditional
approximation expectation, not a measured universal PDE solve rate.

Explicit known-function ball-frame reproduction is an additional diagnostic
in `ridge_frame_dimension_study.py` and pure_mlp_repair/
ridge_frame_dimensions.json. It reconstructs a single known Gegenbauer
ridge via the angular identity, does no fit and solves no PDE. d2--4,p4/6,
N129/257 show ordinary values~1e-15--1e-13 dependingcase, with derivative
errors higher. It should accompany the known-function catalogue, not count
as proof of general physics solution.
