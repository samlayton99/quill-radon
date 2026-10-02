# Radon catalogue audit: early PDE and non-MLP branches

Read-only source/result audit, October 1, 2026. This file is a catalogue input,
not a new experiment. Scope excludes later pure-MLP depth compilation, flat
ridge/frame route 2, and the new streamed directional/disk solver.

## Counting and complexity conventions

- `d`: input dimension, including time for a spacetime representation.
- `M`: ridge directions; `N`: interior centers; `R=ceil(sqrt(N))`; `H=N+2R`;
  `B=MH` actual flat tanh units. Historical uncorrected-halo runs have their
  explicitly recorded H rather than this newer convention.
- `p`: maximum profile/polynomial degree; `P`: independent field coordinates.
  In full total-degree product spaces, `P=binom(d+p,p)`. In full periodic tensor
  spaces, `P=(2K+1)^d`, with frequency cutoff K. P is not a tanh-neuron count.
- `Q`: all fitting/quadrature points; `J`: derivative jet channels; `L`: time
  stages; `I`: total iterative linear products; `G`: nonlinear iterations;
  `b`: bounded preconditioner-block size. Field count is suppressed when fixed.
- `Cenc(s,N)` and `Senc(s,N)` mean actual corrected encoder setup work/storage
  for s known profiles, including complex profile evaluations, contour quadrature,
  partial fractions, and quadrature-rule creation. These must not be mislabeled
  automatically linear in N. `quill_boundary.py` currently has repeated elementary
  polynomial convolutions, O(R^3) scalar work, plus O(R^2 s + R q_h s + N s)
  work and O(Ns + Rq_h + q_h s) arrays, before profile cost and `leggauss` setup.
  The default q_h can grow like N asymptotically; tested modest N often has
  k=1 and q_h=max(96,O(R)). Root has the independent construction audit.
- All solve costs below INCLUDE the iterative/evolution stage. Evaluation means
  ONE point after the state/readout has been obtained, unless otherwise noted.
  A factorized flat ridge evaluation costs O(Md+B), stores O(Md+B), and can have
  O(H) scratch; a literal dense MLP stores/evaluates O(dB). Derivative outputs can
  add d or d^2 factors; fixed required jet channels are suppressed below.
- None of the empirical refinement tables establishes a universal rate. A
  conditional analytic approximation model is exp(-c p), giving
  exp(-c P^(1/d)) for tensor/total-degree families, PLUS encoding, integration,
  solve, and floating-point errors. Fixed smooth double-precision models cannot
  promise arbitrary precision merely by adding neurons.

## Distinct methods and outcomes

### E01. Analytic linear-IVP propagation and explicit ridge encoding

**Setup:** supplied initial function, constant-coefficient advection, two-mode
Airy, simple heat, analytic anisotropic Gaussian heat. Advection and Airy use
fixed two-input tanh networks; heat uses spatial networks with explicitly
time-dependent weights. The 3D heat profiles come from analytically propagated
filtered Radon transforms.

**Other solve:** none when the supported analytical propagation formula is
used. This is specialized equation knowledge, not a general nonlinear PINN.
Independent reference integration is evaluation-only.

**Costs:** initial profile generation + M encodings, approximately O(M Cenc(1,N))
for corrected equivalents, storing O(Md+B). Analytic heat recompiles readouts
for each requested time, so this construction cost belongs to each time query.
Flat evaluation as above. Historical code uses simpler uncorrected profiles.

**Outcome:** smooth linear values reached near roundoff. High-order derivatives
amplify rounding: Airy can have value errors near 1e-16 while scaled PDE residual
is 1e-12 to 1e-10. Angular and center error floors are separate. A neuron which
satisfies the PDE identically does not ensure the correct initial condition.

**Evidence:** `linear.py`, `linear/README.md`, `linear/metrics.json`,
`linear/independent_autograd_checks.json` under expF19/results.

### E02. Independent FFT evolution of Radon heat profiles

**Setup:** unknown solution of 3D constant-coefficient heat with prescribed IC;
sample each initial filtered directional profile, FFT it, multiply by its exact
heat propagator, inverse FFT, encode explicit tanh readouts.

**Other solve:** YES: a one-dimensional spectral PDE solution on each spoke.
No neural LS or training. This was rejected as the general intended PINN route,
while remaining a legitimate dimensional-splitting solver/compiler.

**Costs:** with q offsets per spoke, O(M q log q) propagation plus encoding;
O(Mq+B+Md) storage (can stream spokes). Evaluation flat-ridge cost. q is an
additional resolution count, not automatically N. There are no nonlinear
cross-direction interactions in this linear special case.

**Outcome:** about 2e-15 heat solution error, including a 1,048,576-neuron
evaluation. No generic nonlinear applicability. A tiny heat residual can coexist
with a bad angular approximation of IC.

**Evidence:** `linear_numerical.py`, `linear/numerical_evolution_metrics.json`.

### E03. Fixed spacetime wave construction

**Setup:** full-space 3-spatial-dimensional wave IVP, zero initial velocity.
D'Alembert on each Radon profile replaces its spatial direction w with the two
spacetime directions (w,+c) and (w,-c). Fixed four-input ordinary tanh MLP.

**Other solve:** none in production; only known IC profiles and explicit
quadrature. Kirchhoff quadrature is an independent validator.

**Costs:** two copies of M profile banks, O(M Cenc(1,N)) construction, O(Md+2B)
stored factorized parameters; evaluation O(Md+2B). No time stepping or query-time
coefficient solve.

**Outcome:** roughly 1e-14 solution error. Independent ordinary Torch network
with 32,128 neurons agrees with factorized output to 5.93e-16; AD PDE RMS
4.59e-15. Supported linear geometry/IC assumptions matter; no arbitrary BC or
nonlinear extension established.

**Evidence:** `linear_wave.py`, `linear/wave_metrics.json`, README.

### E04. Cole–Hopf Burgers plus explicit neural encoding

**Setup:** periodic 1D viscous Burgers; exact Cole–Hopf reduction to heat, Bessel
coefficients from the sine IC, quotient of heat solution/derivative, FFT of the
resulting known profile, then explicit inverse-smoothing readouts.

**Other solve:** uses a specialized analytic linearizing transform and spectral
evaluation; no nonlinear iterative solve, no LS, no Adam. It is not a formula
for generic nonlinear PDEs.

**Costs:** spectral profile evaluation O(q k + q log q) if k retained terms are
summed directly; encoding in historical Fourier formula O(Nk), memory O(q+Nk)
if formed densely, bounded-chunk variants O(q+N+k). Network eval O(N), model
O(N); a new output time requires recomputing profile/readouts.

**Outcome:** 309 neurons, value error 5.83e-15, neural PDE RMS 2.25e-13.
At spacing h=.16 and unresolved retained modes, error was 1.29e4; proper cutoff
gave 9.94e-7; h=.08 gave 1.52e-12; h=.04 gave 5.79e-15. Frequency aliasing can
catastrophically amplify rather than simply discard an unresolved tail.

**Evidence:** `nonlinear_pde.py`, `nonlinear/report.md`,
`nonlinear/burgers_metrics.json`, `burgers_bandwidth_ablation.json`.

### E05. Classical Fourier PDE evolution then flat Radon compilation

**Setup:** interacting periodic 3D Taylor–Green NS; later high-dimensional
quadratic reaction–diffusion. A classical dealiased Fourier Galerkin/RK solver
discovers the solution, then grouping collinear wavevectors supplies profiles
for explicit network readouts. Mode pruning and later corrected halos improve
the compiler, not the underlying discovery method.

**Other solve:** YES, complete exterior classical forward solve. Rejected as a
native PINN achievement. The resulting spatial snapshots ARE ordinary tanh MLPs.

**Costs:** grid size Q=q^d, O(L Q log Q) evolution (d/vector factors suppressed),
O(Q) state; grouped encoding O(H P) plus bank construction. Total storage
O(Q+B+Md+HP) if retaining full bank; direct Fourier compiler may stream profiles.
Final MLP evaluation O(Md+B). Q/P remain exponential in d for dense coverage.

**Outcome:** NS 16^3/24^3/32^3/48^3 grid errors 9.78e-5/4.06e-6/5.10e-8/3.66e-11
against finer grids. Neural conversion adds only 1.4–1.8e-15. Counts
337,025/668,525/801,125/1,003,425 neurons. Therefore NOT a machine-floor PDE solve.
48^3 forward11s, 64^3 reference27s; 48^3 encoding+128-point derivative evaluation
.57s. Later corrected NS compiler uses 358,872 neurons for 2.2e-15 value
conversion, 519,420 for 1.6e-15 Laplacian conversion. Pruned 207,480 neurons have
3.2e-11 value and 2.5e-9 Laplacian conversion error.

High-D nonlinear branch resolves a selected smooth 5D case; 6D refinement hits
800MB budget. This is not native neural high-D dynamics. The nonlinear support
generates new directions, so sparse IC support alone does not fix final cost.

**Evidence:** `nonlinear/report.md`, `solver/highdim.py`, `general_dimensions.py`,
`general_solver/dimension_nonlinear.json`, `quill_review/ns_report.md`.

### E06. Native cubic-B-spline / ReLU-cubed coefficient dynamics

**Setup:** periodic viscous Burgers, explicit kernel coefficients as evolving
state. Nodal convolution (1,4,1)/6 has known inverse sqrt(3) rho^|j|,
rho=-(2-sqrt(3)). A fixed 57-tap stencil returns RHS values to coefficients.

**Other solve:** no exterior trajectory, no FFT, no LS/factorization; RK4 still
numerically integrates the native coefficient ODE. Production evaluator uses
compact piecewise cubics, admitting but not executing a literal ReLU^3 MLP.

**Costs:** O(rN) IC construction, O(L r N) integration, O(N+r) RAM, fixed r=28;
local evaluation O(1) per point, O(N) model, vs O(N) time for literal dense MLP.

**Outcome:** N16/32/64/128/256 errors 9.89e-3/1.35e-3/3.19e-4/7.91e-5/1.97e-5.
Second-order spatial rate, NOT a practical floor method. All ~25–31ms. Literal
ReLU^3 evaluator at N64 differs from stable compact evaluator by4.72e-11 due to
cancellation. Fourier control reaches2.5e-13 at comparable cost.

**Evidence:** `native_burgers.py`, `native_dynamics/burgers_native_results.md`.

### E07. Native tanh readout evolution with explicit analysis/synthesis

**Setup:** periodic Burgers; known trigonometric profiles explicitly encoded in
E, actual tanh derivative matrices S; evolve theta'=E A F(S theta). First runner
evolves full theta and caches C=EA. Later reduced-coordinate form evolves
b'=A F(SEb), theta=Eb. These are algebraically identical native fields but
different storage/work implementations.

**Other solve:** no exterior trajectory or global readout LS; numerical RK4
coefficient integration and predetermined trigonometric analysis remain.

**Costs:** full form Cenc(P,N)+O(HPQ) setup, O(QH+HP+QP) storage,
O(LQH) integration. Reduced form same worst setup to form actual SE, O(QP+HP)
storage plus bounded QH temporary; O(LQP) integration. Export O(HP), then O(H)
evaluation and model for one direction. Spatial approximation has P=2K+1.

**Outcome:** B153/291/559/1091 with K8/16/32/64 yield
4.77e-3/1.01e-4/5.70e-8/2.50e-13. Halving dt from.001 to.0005 at K64 yields
2.52e-14; near target scale, not universally exact. Ordinary matched spectral
coordinates agree around1e-15. Fixed dt becomes the floor. This is a modified
spectral method in neural coordinates, no neural numerical advantage established.

**Evidence:** `native_tanh.py`, `native_dynamics/tanh_metrics.json`.

### E08. Native flat-ridge Navier–Stokes projected dynamics

**Setup:** periodic 3D Taylor–Green; corrected true tanh field/gradient/Laplacian
in every RHS, FFT analysis and Leray pressure projection, reduced amplitudes,
RK4. Cached actual neural basis tables, not an exterior reference trajectory.

**Other solve:** numerical time integration + Fourier analysis/global pressure
projection. Reference-only spectral run generated after native runs.

**Costs:** per-direction encodings + O(QHP) table construction, O(JQP+HP+B)
persistent RAM and O(QH) direction scratch. Each stage O(JQP+QlogQ), total L
times that; exported flat network eval O(Md+B), model O(Md+B). With Q~P~K^d,
cache behaves like K^(2d), explaining the memory wall despite compressed state.

**Outcome:** T=.05, K2/3/5, B10829/32045/167907, errors2.18e-4/3.98e-6/1.94e-9.
Native-vs-same-K spectral1.5e-15 is representation equivalence, NOT PDE error.
K5 cache186.3MB, peakRSS400.2MB, build15.136s, evolution.780s; conventional
spectral evolution.177s. Successful native mechanism, inaccurate at continuum
floor and poor cache scaling.

**Evidence:** `native_ns_pilot.py`, `native_ns_refine.py`, native NS report.

### E09. Native NS Taylor-in-time recurrence

**Setup:** same native projected NS ODE b'=Lb+B(b,b), generate time coefficients
(n+1)b[n+1]=L b[n]+sum_j B(b[j],b[n-j]) directly, then Horner evaluation.

**Other solve:** no production time steps/LS, but this IS a numerical truncated
Taylor solution of a finite spatial ODE. Not a universal closed form.

**Costs:** E08 spatial cache; degree T recurrence stores O(TP+TQ+JQP), work
O(T JQP + T QlogQ + T^2 Q) with cached coefficient fields and fixed d; Horner
state O(TP), then O(HP) export and ordinary flat eval. Local convergence radius
requires control; longer times need restart/slabs, not demonstrated here.

**Outcome:** degree8 discrepancy2.60e-15 against same spatial cutoff, but total
error1.97e-9 versus refined spatial reference. Degree12 construction .098s after
15.2s spatial cache; exact-mode Taylor .027s. Thus time floor only.

**Evidence:** `native_taylor.py`, `native_dynamics/taylor_metrics.json`.

### E10. Flat-ridge elliptic shifted residual iteration

**Setup:** -Delta u+5u^3=f on [0,pi]^2, zero Dirichlet; encode products using
sin(mx)sin(ny)=[cos(mx-ny)-cos(mx+ny)]/2. Iterate coefficients using only the
scalar inverse (m²+n²+shift)^-1 and actual neural PDE residuals.

**Other solve:** no exterior solve/no global LS, but problem-specific inverse
reference Laplacian and nonlinear fixed-point iteration. This is a true flat
ridge field; the physical coordinate basis is prescribed.

**Costs:** P=K², B=2PH unmerged; Cenc(O(K),N)+O(QPH) setup, O(QP+HP+QH)
peak/cache; O(GQP) iteration. After collapsing geometry, eval O(Md+B), storage
O(Md+B). The actual source shares profile encodings but repeats ridge evaluations.

**Outcome:** K4/8/12 errors4.08e-3/1.60e-5/6.37e-8; K12 B122400, P144,
22updates, build1.76s, iteration.00232s, off-gridPDE3.03e-6. Not floor in this
runner. Shift0 diverged, iteration spectralradius2.714; shift5 radius.261.

**Evidence:** `native_elliptic.py`, `capabilities/elliptic_metrics.json`.

### E11. Shared-bank product elliptic box solver

**Setup:** same smooth cubic elliptic PDE; two shared 1D QUILL sine banks,
explicit product gates and affine boundary corrections, K² physical readouts.
THIS IS NOT A PURE TANH MLP or original flat Radon network.

**Other solve:** same scalar-preconditioned fixed-point iteration as E10; no
external trajectory/global readout LS.

**Costs:** Cenc(K,N)+O(Q(dHK+P)) basis setup, O(QP+HK) cache, O(GQP) solve.
One-point eval O(dHK+P), model O(HK+P), tanh activations dH. Product gates and P
coefficients must be counted. K-specific changes do not have a neurons-only rate.

**Outcome:** K12 P144 B442 error6.38e-8 in.0159s; K20 P400 B714 error1.17e-12
in.0884s. Spectral-like decrease, stopped above1e-14 in tested range. Matched
classical sine control same accuracy in.0326s. Baseline PINNs 2.45–3.72e-5 in
83–281s; this one comparison does not prove general superiority.

**Evidence:** `solver/tensor_box.py`, `general_solver/comparison_metrics.json`.

### E12. Mapped annulus Galerkin / Newton–CG

**Setup:** variable-coefficient cubic elliptic PDE on annulus; polar map,
QUILL polynomial banks, trigonometric coordinate features, boundary factors,
product gates. Nonzero Dirichlet and inner Robin supported.

**Other solve:** global small P×P diffusion/Robin preconditioner Cholesky;
matrix-free nonlinear Newton–CG. No exterior solution or readout LS. Architecture
is mapped products, not a flat/pure tanh MLP.

**Costs:** bank cost + O(QP²+P³) preconditioner setup, O(QP+P²) RAM;
Newton/Krylov O(I(QP+P²)) aftersetup, plus nonlinear scalar work. Evaluation
O(H(p_r+p_theta)+P), model O(H(p_r+p_theta)+P). P=p_r(2p_theta+1).

**Outcome:** P28/66/120/190, tanh306/342/442/546, errors
1.11e-2/1.85e-5/9.25e-9/2.19e-12. FinestPDERMSrelative8.92e-12, .0376s.
Contrast10000error3.91e-12. Diagonal-only preconditioner capped30CG failed;
diffusion preconditioner succeeded with up to42CG. Not arbitrary geometry/floor.

**Evidence:** `solver/elliptic.py`, `general_solver/domain_report.md`.

### E13. Stiff QUILL reaction–diffusion with BDF

**Setup:** native reduced trigonometric/QUILL field for stiff Allen–Cahn and
quadratic blowup reaction, actual neural derivatives. Implicit BDF with dense
P×P Newton matrices, physical invariant and underresolution guards.

**Other solve:** YES internal dense implicit state solves; no external trajectory.
Different problem-specific backend, not general pure-MLP spacetime solution.

**Costs:** E07 reduced setup plus O(P²) matrices, O(QP+P²+HP) RAM. Across L
implicit stages and F factorizations, O(LQP + F(QP²+P³)) conservative cost.
Snapshots collapse to H-unit tanh readout, O(H) inference. State trajectories
and reporting add storage if retained.

**Outcome:** K32/64/128 Allen–Cahn errors .0263/.00275/2.82e-5; K256 difference
5.62e-7 lies below reference uncertainty1.68e-6. Not floor. Blowup threshold
10000 near.838906 is an event, not true singularity time. Analytical bracket
5/6≤T*≤1 independent of solver. Coarse invariant violations were real failures.

**Evidence:** `solver/reaction.py`, `general_hard.py`, hard_results.md.

### E14. Conservative finite volume with ReLU primitive

**Setup:** inviscid Burgers shocks/rarefactions; Rusanov flux + SSPRK3; evolving
cell values represented as derivative of a ReLU primitive. Classical finite
volume, not smooth QUILL nor a tanh network.

**Other solve:** classical numerical PDE evolution IS the whole method. No
neural fitting. Included because it was used to claim broader PDE capabilities.

**Costs:** O(N) initialization, O(LN) evolution, O(N) RAM; local field lookup
O(logN) or O(1) uniform-cell indexing, vs O(N) literal primitive-network eval.

**Outcome:** roughly first-order L1 convergence, conservation<1.7e-15, recorded
entropy defect nonpositive. Maximum error at a discontinuity stays large. This
does not reach machine-floor solution accuracy by a moderate neuron increase.

**Evidence:** `solver/hyperbolic.py`, `general_solver/hard_results.md`.

### E15. Adaptive native Fourier-ridge selection, with/without replay

**Setup:** periodic2D nonlinear reaction–diffusion; fixed candidate integer
frequency dictionary, add/remove active directions from PDE increments, calibrate
N/lambda per known profile. No continuous angle search or external trajectory.

**Other solve:** native RK stages + dense quadrature analysis. Successful version
rejects a time window, restores its initial state, and replays with expanded
directions; original no-replay version is a documented failed design.

**Costs:** full candidate analysis storage O(Q P_c), actual neural basis caches
O(Q P_a), plus banks; active evolution O(L Q P_a), discovery perwindow O(Q P_c)
and sorting. New feature setup O(QH deltaP). Replay multiplies real L. Final
flat eval O(M_active d+B_active), but sparse final network does not make discovery
or construction sparse in dimension.

**Outcome:** fixed408directions/80056neurons error1.53e-7,9.62s; replay
110directions/19098neurons error2.18e-7,3.28s; no-replay7.82e-5. Actual260RKsteps
vs120nominal. Matched trig algorithm.790s; no intrinsic tanh speedup. One-direction
control117neurons. Not floor, still dense candidate bottleneck.

**Evidence:** `solver/adaptive_ridges.py`, `adaptive_followup/adaptive_metrics.json`.

### E16. Shared-product native NS and fluid animation

**Setup:** periodic3D velocity, three shared actual QUILL sine/cosine banks,
tensor readouts/product gates, FFT analysis, Leray projection, RK4. Changed
architecture, no longer original flat/pure tanh MLP.

**Other solve:** native numerical Galerkin evolution; no supplied reference
trajectory. Same physical algorithm as spectral method, altered representation.

**Costs:** bank Cenc(K,N), 1D tables O(d qHK); P=(2K+1)^d, Q=q^d.
Tensor synthesis O(d K Q) for q~K plus O(QlogQ) analysis perstage; memory
O(P+Q+dqK+HK), not QP. Per arbitrary point O(dHK+dP) (d fixed; tensor contraction),
model O(HK+P). Total L stages. This beats dense basis caching but retains K^d.

**Outcome:** selectedK11/K13 at T1 viscosities.05/.01: finer-reference differences
2.38e-6/7.22e-6; timehalving~2–4e-12.867tanh but36501/59049velocitycoefficients.
T4 animation3.71e-6 spatialdifference,1.14e-4physicalresidual. Not PDE floor.
Matched trig products faster by~13–26%; apparent FFT advantage is implementation.

**Evidence:** `solver/ns_extended.py`, adaptive NS report, fluid animation scripts.

### E17. Generic cached total-degree QUILL-product residual solver

**Setup:** arbitrary declared smooth residual equations, coupled fields,
IC/BC/gauges/observations, implicit domains and inverse parameters. Total-degree
Legendre products encoded through shared QUILL banks. Damped GN + LSMR, diagonal
or bounded parity blocks; no named-equation dispatch. Not original flat Radon
nor pure tanh MLP because exact multiplication remains.

**Other solve:** no external PDE trajectory; DOES globally solve physical basis
coefficients iteratively. Matrix-free Jacobian does not imply low basis memory.

**Costs:** bank Cenc(p,N)+O(JQ(dHp+dP)) feature setup; O(JQP+Hp+Pb) RAM;
O(I JQP) Krylov work, plus preconditioner build O(QPb+Pb²) per refresh and local
nonlinear AD. Full P=binom(d+p,p), often Q~P, giving O(JP²) cache. Field evaluation
O(dHp+dP), model O(Hp+P); actual current `evaluate` materializes query_count×P.

**Outcome:** thirteen generic smooth/inverse declarations gave field errors
1e-8–1e-13 at default tolerance1e-6; polynomial controls similar. Stronger
manufactured 4D transientNS coldp10→p12: velocity1.31e-15 pressure3.54e-15,
387s, PDE1.95e-15. Endpointpressure1.42e-14. Thus floor on favorable controls,
not consistent universal floor. Fourinputs p14 fullscalarP3060 leads~3.60GB
interior basis cache alone at six samples/feature,eightderivativechannels.

Failures: coldhighdegree conditioning, highcontrast180.69s failureerror6.53e-5
before blockpreconditioning achieved5.77e-8 in2.30s; boundarylayerdegree40failed,
continuationdegree44passed. These are failures of runs/defaults, not inability
to represent every target. Polynomial basis/QUILL coding both exponential in p
conditionally, not a rate in hidden tanh count alone.

**Evidence:** `solver/general_features.py`, general_residual.py,
`docs/quill_general_solver_checkpoint.md`, general_solver_summary.json,
ns_symbolic_precision_cold_p10_p12.json.

### E18. Sparse-support precision product solver

**Setup:** E17 with selected Legendre multiindices, downward-closed expansion,
pruning/refinement based on earlier PDE-solved coefficients; anchored halo
evaluation, tighter stopping/defect refinement and trace-sized BC sampling.
Known-solution capacity diagnostics inform research resolution choices but
reference coefficients do not initialize the solve or choose support.

**Other solve:** same global iterative native PDE solve. Sparse support removes
some product dimensions, not underlying d-dependence or feature-cache issue.

**Costs:** replace fullP by selectedS in E17: O(JQS) cache and Krylov products;
bank still depends on maxdegreep. Candidate closure/enrichment has own combinatorial
cost. O(Hp+S) model; O(dHp+dS) arbitrarypoint eval. Not small just because tanh
bank count is small.

**Outcome:** unforcedABC spacetime: p10 polynomialvelocity1.50e-11 pressure5.78e-10;
p12velocity1.19e-14 pressure4.22e-13; p14S1795poly2.46e-15/9.13e-14;
p14S2701actualQUILL5.74e-16/1.50e-14,445s last correction. Endpointpressure
1.22e-13 still not floor. Some rows retained plateau/iteration-limit status.
Reference-only coefficients could achieve4.87e-16/1.39e-15, but this is capacity,
not discovery. MoreN (1025 vs257) can worsen roundoff.

**Evidence:** `solver/general_sparse.py`, general_ns4d_sparse_precision.py,
general_residual/ns4d_sparse_mixed14_summary.json, general checkpoint.

### E19. Exact-polynomial control residual solver

**Setup:** same GN/LSMR residual solver, use exact Legendre products instead of
their QUILL approximants. Classical spectral collocation/control, NOT a neural
or Radon method. It was a materially important tested route, not to erase.

**Other solve:** no exterior solution; same coefficient solve. Basis formation
O(JQ(dp+dP)), memoryO(JQP+Pb), I-productsO(IJQP), evaluationO(dp+dP), modelO(P).
No neural bank encoding cost or tanh approximation error.

**Outcome:** thirteen matched controls similar to QUILL; precisionNS entries
above. Demonstrates that most observed gains derive from constructed coordinates
and numerical solver, not proven intrinsic benefit of tanh itself.

**Evidence:** exactpolynomial files and generalcheckpoint.

### E20. Restricted weak/entropy front-family solve

**Setup:** translating tanh front with 2parameters or spreading piecewise-linear
ReLU ramp with3; integrate conservation balances and add entropy inequalities.
Same generic GN engine tested on these custom integral residuals.

**Other solve:** tiny nonlinear leastsquares over supplied front geometry. Not
automatic Radon front discovery; not the default smooth QUILL geometry.

**Costs:** residual quadrature/analytic-integral cost O(Q s), tinyGN O(Qs²+s³)
with s=2or3; memoryO(Qs+s²); simplefield evaluationO(1), modelO(1).

**Outcome:** conservation alone accepts nonphysical expansion shock with weak
residual9.6e-17 but L1error.2. Entropy conditions yield rarefactionerror2.5e-6,
limited by initial smoothing,6GNsteps. Shock speed.5 or.333 recovered~2.7e-12.
No generic floorclaim; important counterexample to residual-only success.

**Evidence:** general_shock_analysis.py, general_weak_study.py, weak_engine_metrics.json.

### E21. Analytical heat-parameter inverse wrapper

**Setup:** six independent entries of SPD diffusion tensor,36noisy/exact
observations; explicit analytic Radon heat construction recomputed perparameter.
Six-parameter bounded nonlinear leastsquares with analytic sensitivities.

**Other solve:** small physical inverse LS, no global neural readout LS; forward
relies on supported analytical heat solution. Not arbitrary unknown fields.

**Costs:** Iinv forward/sensitivity constructions+sensor evaluations, each
roughly O(s(Qobs B+B)), cached kernelsO(Qobs B), smallGN O(Qobs s²+s³).
Finalgeometry/readout eval standardflat. Exact costs depend sharedtimes/cache.

**Outcome:** noiselesstensorrelative2.8e-9,1%measurementnoise1.2–5.3% across8draws.
Did not reach parameterfloor. Noise/identifiability limit independent ofnetwork.

**Evidence:** inverse.py, inverse/tensor_metrics.json.

### E22. Native Burgers physical-parameter inverse with tangents/UQ

**Setup:** viscosity+twoICamplitudes,24observations. Native reduced QUILL
forwardstate and three tangentODEs, boundedsmallparameterLS. Local covariance,
rank and model-mismatch checks are extensions of same method, not new forward
solvers.

**Other solve:** repeatednativeforward numerical integration + small3parameterLS.
External solver only generates synthetic measurements/reference audit.

**Costs:** reducedE07setup, Iinv L Q P(s+1) tangentforwardwork, O(QP+P(s+1)+HP)
state/cache plus O(Qobs s²+s³) perinverseupdate. At fixed smalls scales like
severalforwardruns. Finalsnapshot evalO(H), exportedmodelO(H).

**Outcome:** refinedK64(1091neurons,P129) viscosityrelative1.14e-11,
withheldfield2.81e-13,futurefield1.97e-10. MainK32viscosity2.06e-6.0.1/1%noise
medianviscosity1.21/12.1%; field0.0328/0.324%. IC-onlydata has exactzero viscosity
sensitivity: tinydataresidual doesnot identifyparameter.30seedconditionalUQ
coverage29/29/27of30, not globalBayesianproof.

**Evidence:** native_inverse.py, capabilities/inverse_results.md,
general_inverse_uq.py, general_solver/inverse_uq_metrics.json.

### E23. Unknown initial-profile inverse with regularization

**Setup:**17unknownFourierICprofilecoefficients,48Burgersmeasurements,known
viscosity. Nativeforward+tangent model, discrepancy/noise-selected H2penalty,
forwardrefinement and observation-consistency checks.

**Other solve:** repeatednativeforward integration and17parameter regularized
nonlinearLS. Model supplies finite-dimensional prior; not arbitraryfieldrecovery.

**Costs:** sameE22formulas withs17; regularization-selection refits multiply
Iinv. Storage/tangentwork linearins before smallGN s²/s³; irreducibleforward
discovery remains. FinalsliceevalO(H).

**Outcome:** earlynoiselessprofile1.09e-12 afterrefinement,latemeasurements9.73e-7.
0.5%noise earlyprofile2.18–4.29%,late10.18–10.69%, though latefutureprediction
.18–.28%. Deliberate inversecrimefit3.05e-20 failedforwardrefinement. No floor
under noise; no representation removes information loss.

**Evidence:** solver/inverse_profile.py, inverse_profile_study.py,
adaptive_followup/inverse_profile_metrics.json.

### E24. Regularized backward heat then explicit encoding

**Setup:** noisyterminalfield, knownheatoperator; FFT spectral inversion,
noise-based discrepancy cutoff, construct recoveredinitialfield network.

**Other solve:** classicalregularized inverse transform; no neuralfit. Not
neuralPINNdiscovery, but an honest inverse baseline/representationpipeline.

**Costs:** O(qlogq) FFT+cutoff plus O(Nk) directprofileencoding at cutoffk,
O(q+Nk) naivearraymemory; O(N) finalmodel/evaluation.

**Outcome:**128measurements,noiseRMS.000851; unregularizederror2.73e173,
cutoff6error.001624,516neurons,neuralconversion4.20e-16. Thus compilerfloor,
not inversefloor; a failed unregularizedbranch and successful noiselimitedone.

**Evidence:** inverse.py, inverse/backward_heat_metrics.json.

## Proposals/rejected ideas that must not be counted as completed methods

1. Directly reapply complex-shift density to the evolving tanh field: invalid
   naïvely because shifts can hit its meromorphic poles. Checkpoint records this
   mathematical rejection; no separate quantitative production benchmark found.
2. Evolve nonlinear PDE directions independently: rejected by Fourier mode
   coupling k=p+q; E02's spoke independence applies to its linear operator.
3. Forced-NS September2026 blowup manuscript: fullforce NOT instantiated.
   Synthetic concentratingGaussian swirl is a known-function moving-geometry
   representationprobe only (1275units~1.4e-15 at100×shrink). Its forcing diverges,
   so it is not reproduction of smooth-forcedNSblowup. Constructionagent may
   catalogue this representationprobe separately.
4. Generic local/hp partition-of-unity spaces, causal slabs, universal operator-
   aware preconditioning: recommendations in theorycheckpoint; not implemented
   evidence in this earlygenericproductbranch. Do not promote plans to tests.

## Interpretation for the parent catalogue

Most scientifically defensible highscores here are specialized numericalmethods,
not compliance with Sam's full requirements. E03 gives the clearest genuinely
solve-free fixedspacetimeMLP but narrowlinear scope. E07/E08/E10 prove actual
neuralphysics can drive coefficientdiscovery without an externaltrajectory.
E11/E12/E16/E17/E18 solve usefulproblems but violate the pureMLP/originalridge
architecture requirement as implemented. E17/E18 deliver the strongest oldNS
accuracy yet retain enormous basiscachememory. E20 and E24 are essential failure
controls against calling tinyresidual or tinyencodingerror a solvedPDE/inverse.
None warrants a blanket 'consistently reaches machinefloor on generalPDEs'.
