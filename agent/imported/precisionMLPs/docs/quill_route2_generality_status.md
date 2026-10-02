# Route 2: generality and dimensional limits

Coordinator: root. Completed this round October 1, 2026. Sam requests a routine genuine
ridge-MLP PDE solver, tests of the practical benefits proposed for PINNs,
and measured dimensional limits. Results must be explained in chat.

Architecture and provenance requirements remain those in
`quill_pure_mlp_status.md`: ordinary tanh ridge forward, actual neural
derivatives in residuals, no separate PDE solve then encoding. Linearized
numerical coefficient solves are explicit. A collection of local MLPs is a
different deployment and must be labeled, not passed off as one global MLP.

Actual hardware: this workspace is a 16 GB Apple M4 Mac mini, not laptop
timing. Capacity projections must state degree, accuracy/workload and memory
budget. Context sync failed transport; local context check passed.

Ownership:
- root: `solver/route2.py`, declarative integration, parameterized/weak-form
  probes if useful, literature, integration and in-chat synthesis.
- quill_allocation_history: `route2_burgers_*`, forward nonlinear accuracy,
  representation/conditioning diagnosis, dedicated tests/results.
- native_quill_theory: `route2_inverse_*`, `route2_constraints_*`, sparse data,
  noisy/identifiable inverse problems, irregular domains, dedicated tests.
- route2_dimensions: `route2_scaling_*`, actual hardware scaling and honest
  generic versus explicitly structured high-dimensional examples.

Independent workers use single threads and bounded arrays, preserve shared
edits and coordinate core-engine modifications. Artifacts live under the
existing F19 result folder in `route2_generality/`. Previous product-feature
results do not count as route2 evidence. Previously failed box moment
preflights were corrected using high-precision construction in the last
round; their stale reports are not current results.

Evaluation priorities: nonlinear forward accuracy; value/derivative/raw
export agreement; sparse anchors; inverse identifiability and noise;
variable coefficients and boundary types; complex domains; conservation
and nonsmooth limitations; parametric reuse; dimension/cost scaling;
residual-to-solution stability and empirical validation. No claim of
universal convergence, uncertainty calibration, turbulence performance or
classical-solver superiority follows from a small test suite.

## Completed round

Root added `solver/route2.py` and `GeneralProblem.solve_ridge()`: explicit
residual declarations, analytic ordinary-tanh construction, numerical
Gauss–Newton/readout-coordinate solves, neural continuation, fresh sampled
checks, separate standard Torch export/AD validation, resource preflight,
and best-model retention. Low-degree sparse angular maps are available
through auto/tensor/sparse selection. No reference solution callback exists.
Validation is sampled, not a continuum, uniqueness or identifiability proof.

Measurements (ordinary exported float64 model):
- Burgers ν=.1, smooth unforced IVP: degree24/28/32/36 gives relative errors
  1.06e-7 / 6.13e-9 / 2.99e-10 / 1.68e-11. Best final-time error4.44e-9,
  raw AD interior PDE RMS5.75e-11; residual tolerance unmet, line_search_failed.
  N/λ9-point known-profile calibration plus N513 representation-only control
  show center encoding is not the dominant remaining error. Additional face
  PDE constraints exhausted the bounded solve and worsened field accuracy.
  Four time-slab networks are separately labeled, not one global MLP.
- Nineteen annulus/inverse/constraint controls, same p10,N257,R17,λ.2:
  forward variable diffusion+cubic reaction/mixed BC error2.03e-15;
  Neumann+integral gauge1.95e-15; exact12-anchor inverse3.15e-15 and κ1.7.
  No-anchor inverse is exactly unidentifiable; noisy fits retain nonzero
  residual/line_search_failed with separate sampled stationarity diagnostics.
- One (x,μ) MLP for -μu_xx+u=1, zero endpoints, μ∈[1,2]: field1.77e-12,
  parameter derivative4.83e-10, ordinary PDE RMS6.93e-12, ~1.5s. Exact cosh
  solution is validation only; no solution labels or extrapolation claim.
- Sparse angular rules: p2 M=d², p3 M=d+2C(d,2)+4C(d,3). Established
  Stroud/Smolyak rules (Hinrichs–Novak math/0509101 eq24/29); signed weights
  enter basis construction only. 20D quadratic control61,200 neurons and
  231 coordinates:6.92e-15 error. 10D cubic88,740neurons/286coords:8.53e-15.
  These manufactured all-face low-degree controls test architecture and
  conditioning; boundary traces determine the ideal polynomial field, so
  they do not establish difficult high-dimensional interior dynamics.
- Shared derivative profile evaluation reduced the same20D control11.01s
  to2.64s, bitwise-identical coefficients, ~445MB process peak. Implemented
  `BallRidgeFeatures.evaluate_many`, with optional residual-engine support.
- Shock diagnostic is NOT a PDE solve: smooth tanh shock approximations
  converge in L1 as1/γ while strong Burgers residual grows√γ; entropy
  distinguishes compressive/expansive jumps with equal flux balance.

The Sobol face-index alias was fixed in new harnesses: independently sample
all faces. Final high-dimensional and parametric results use the corrected
coverage. Older tensor cost controls remain marked legacy_interleaved.

Independent review findings resolved: nonfinite raw values cannot pass;
resource refusal precedes potentially large training sampling; constructor
monomials avoid M×powers×d temporaries; known disk encoding-gate failures
preserve a prior solved model; fallback selection includes raw export score.
Targeted regression tests cover each substantive numerical guard.

Current limits: full tensor angular construction becomes costly for high
resolution around4–6 inputs on16GB; sparse degree2/3 is much cheaper.
General high-order sparse cubature, scalable streamed feature caches,
automatic anisotropy/local refinement, entropy-aware shock solving,
calibrated UQ, difficult3D Navier–Stokes and matched classical/PINN benchmarks
are not completed. No universal PINN solver or superiority claim is made.

All reproducible scripts, configurations in script defaults, figures,
ordinary model weights and JSON measurements are under the existing
F19 experiment and `route2_generality` result folders. Final answer explains
the findings in chat; Sam is not expected to read this status file.

Final relevant suite:98 tests passed (4 mpmath deprecation warnings).

## Active follow-up: asymptotics and evaluation-cost corrections

Sam asks for global/current costs in d,N and a concrete theory route beyond
solver-bound scaling. Root coordinates: route2_cost_audit (read-only code
complexity), native_quill_theory (operator/frame theorem and primary sources),
quill_allocation_history (isolated directional profile forward/adjoint
prototype, no complete PDE-solver claim). No shared-engine rewrite this round.
Primary criterion: cheap profile analysis/synthesis PLUS iteration counts
controlled with resolution; avoiding a dense solve alone does not meet it.

### Follow-up results (2026-10-01)

Cost audit found the old general route's `T[M,p+1,P]` map and `Q x P`
jet-cache contraction still cost O(Q M p P), with P=binom(p+d,d).
For fixed d, balanced p~N, M~N^(d-1), Q~P, this is O(N^(3d)) setup;
the subsequent cached iterative solve is O(I J s N^(2d)). Dense global
neuron QR has the same cubic exponent in this regime. Fair matrix-free
global solves already have evaluation-cost matvecs, but iteration counts
and conditioning remain. Increasing N alone at fixed p improves the same
profile encoding, not the represented function space.

New isolated `route2_directional_operator.py` uses direct coefficients
b[M,p], with the shared QUILL bank W[N+2R,p], R=ceil(sqrt(N)); no T map
or Q-by-P cache. One jet costs O(M N p + Q M (N+d)). It exports an ordinary
Linear/Tanh/Linear network, including ordinary automatic derivatives.
Benchmark Q2048,p8,N129: M8/16/32/64 forward value/dx/dxx takes
.056/.111/.225/.452s. Tracked transient allocation stays1.85MiB. A small
cached dense operator is much faster to reuse, so this is an asymptotic
storage/operator result, not a matched full-solver speed advantage.

New `route2_frame_iteration.py` tests an actual no-LS PDE iteration:
L u + tanh(u)=f on the unit disk, L=-div[(I-xx^T)grad]+2.
Natural zero flux comes from boundary-degenerate diffusion; this is NOT
ordinary Poisson or a general boundary-value solver. Start all readouts at
zero. Compute the actual tanh residual; analyze it with known Gegenbauer
test profiles; divide degree-n coefficients by 2+n(n+2); subtract the
resulting neural correction. No fitted reference field, Krylov solve,
global LS, or global coordinate map occurs. Manufactured forcing comes
from u*=.1+.25 exp(.4x)cos(1.2y)+.05xy; reference values are validation only.

The ball-frame identity S A=P_p and L C_n=[2+n(n+d)]C_n give ideal
contraction <=1/2 independent of degree. Actual encoding/quadrature errors
must preserve this contraction. The coefficient statement holds on the
canonical frame range, not arbitrary redundant coordinates. Direct update
cost O(MNp+QM(N+d)+QMp), comparable to one whole residual evaluation when
p<=N,Q. Balanced direct cost O(N^(2d)) per correction, not O(N^d).

Independent ordinary Torch/AD checks: p4/8/12/16, N257, lambda .2,
R17, all40 updates; relative field errors6.40e-4/7.21e-8/1.48e-12/1.78e-14.
Corresponding PDE RMS7.79e-3/2.34e-6/9.35e-11/1.69e-14.
Quadrupling residual points changes no conclusion. Tightening stopping
to2e-16 yields46 updates, sampled field error1.26e-16 and PDE RMS1.49e-15;
these are roundoff-level sampled checks, not continuum error certificates.
N129 and N513 checks test independent encoding resolutions. All results,
weights and the figure are in route2_generality/frame_iteration/. Defaults
of the script reproduce the baseline and supplementary checks.

Final new tests:17 passed, covering frame reconstruction of all monomials
through degree12, eigenidentities in dimensions2/3/5, no-reference/no-global-
solve iteration, ordinary AD export, adjoints, and scalar-ridge vs full-
Hessian operator evaluation. No shared general solver was replaced.

Next mathematical barrier: a boundary-compatible inverse for ordinary
operators and nonlinear contraction/preconditioning that stays stable with
resolution. Next computational barrier: fast analysis and synthesis that
avoid visiting every neuron at every point. Established adaptive ridgelet
complexity results require both conditioning AND compressibility.

Primary sources: Xu arXiv0705.1984 (ball-frame constants); Pinar/Xu
arXiv0712.3091 (ball operator eigenvalues); Grohs/Obermeier arXiv1409.1881
(optimal adaptive ridgelet solver conditions). The first prototype is a
matched special case, not evidence of a universal PINN or NS solver.

### Priority clarified by Sam: memory before runtime

Sam explicitly accepts longer runs and additional passes when needed for
accuracy. The immediate success criterion is linear-size working memory
in the directional coefficients/constraints, with bounded temporary tiles,
not a proof of resolution-independent iteration counts before proceeding.
Keep deterministic full-constraint evaluations and high-accuracy stopping;
streaming/recomputation must not quietly become stochastic subsampling or
discard approximation modes. The old general `_Engine` still caches full
feature jets; the direct operator and matched ball prototype do not yet
constitute a streamed general forward/inverse engine. The next integration
target is that engine with recomputed local derivatives and bounded-block
preconditioning, while preserving ordinary-MLP and independent residual
validation. Distinguish coefficient/constraint vectors from Q-by-S feature
matrices, and include construction/export/preconditioner peak memory.

### Active implementation follow-through

Root coordinates native streamed general-engine integration. Ownership:
streamed_engine: general_residual.py, streamed_residual.py and engine tests;
root: directional operator prepared-state/selected-column protocol;
streamed_experiments: route2_streamed_study.py and measured results;
streamed_audit: read-only correctness/memory review.

Implemented selectable `execution='streamed'`, point batch_size, row-aware
measurement callbacks, exact full residual/Jv/JTv with no feature caches,
bounded-column preconditioning, and streamed predictions. Retain O(Q) local
callback sensitivities and residual/Krylov vectors; supplied aggregation
matrices still consume their original memory. Prepared neuron readouts and
adjoint accumulators add O(B) storage, avoiding repeated O(MNp) encoding
for every point batch. The shared QUILL bank is no longer duplicated.

Initial actual nonlinear diffusion-reaction/ordinary Dirichlet checks:
p2,N257,M3 zero-start forward error3.4e-16 and raw PDE RMS6.6e-15;
jointinverse with8anchors recovers diffusivity1.7 to roundoff, field2.2e-16,
raw PDE RMS6.5e-15. No exterior PDE solve. These are small manufactured
cases; independently exported ordinary Torch/AD validation is used.
N129 had an accurate field but held-out PDE RMS3.7e-12 despite sampled
convergence; it is not called a PDE-floor result.

Larger nonpolynomial p12 reaches field4.8e-11, raw PDE2.8e-10. The first
p16 run was under-solved at1200 capped Krylov steps/timebudget, so resolution
must not be blamed from that run. A longer prepared run and an optional
analytic disk angular coordinate representation are being tested. The
latter removes redundant directional coefficients with known trigonometric
maps by degree; it must not construct the old global coordinate-map tensor.
Final memory sweeps and full regression results pending.

### Streamed nonlinear solve and analytical disk coordinates: verified results

The longer p16 direct-coordinate run reaches held-out ordinary-Torch field
relative L2 1.4702e-15 and raw PDE RMS 6.8621e-14 in 247.4 s / 2256 Krylov
iterations. The earlier 131.8 s capped result (field 1.3073e-8) was under-solved;
it must not be described as a representation floor.

New `DiskProfileOperator` uses the exact half-circle angular identity to
represent degree-n profiles with their n+1 cosine/sine modes. It stores an
M-by-(p+1) trigonometric table and O(p^2) labels, not an M-by-p-by-P tensor.
The shared one-dimensional QUILL bank maps these coordinates to actual tanh
readouts. No target fit or matrix factorization constructs the map. Ideal
disk modes are orthonormal for normalized area; actual tanh profiles only
approximate them, and restricting the encoded dictionary also removes its
additional tiny departures from the ideal span.

At p16, M17, N257, R17, lambda .2, the same 4947-neuron family solves from
zero with 153 independent coordinates instead of 273 redundant coordinates.
Held-out ordinary-Torch field relative L2 is 4.3105e-16, raw PDE RMS
4.1504e-15, PDE max 4.0301e-14, boundary RMS 4.2930e-16. Runtime 59.2 s,
3 nonlinear steps / 532 Krylov iterations. This is the ordinary nonlinear
diffusion-reaction Dirichlet problem recorded above, with the nonpolynomial
manufactured target, not the boundary-degenerate matched-frame prototype.
Direct-long vs disk readout L1 norms are 10.5681 vs .521770, consistent with
less cancellation, not a proof that cancellation alone explains improvement.

The isolated memory sweep includes feature construction, residual
linearization, one Jv/JTv pair, and full bounded-block preconditioner setup.
At Q8192/M32/p8/N65, cached vs streamed: feature cache 50.594 MB vs zero;
postconstructor peak traced increment 52.642 MB vs 1.073 MB; total traced
peak including construction 54.504 MB vs 6.402 MB; full process peak RSS
326.353 MB vs 275.857 MB, with roughly 260 MB runtime/import baseline.
These are decimal MB, not total memory claims for the traced allocations.
Stage runtime 1.023 s vs 4.798 s. Full solves have additional live vectors;
this stage benchmark is not a full-trajectory peak-memory measurement.

For fixed field count, derivative channels, point-batch size and coefficient
block size, streaming removes Q-by-coordinate caches and global squared
coordinate matrices. It retains O(Q) constraints/sensitivities, O(B) prepared
neuron state, the O(Np) shared encoding bank, and bounded block factors.
Here B=M*(N+2*ceil(sqrt(N))). A prepared directional product costs
O(MNp + QM(N+d)); disk angular conversion additionally costs O(Mp^2).
Iterations and preconditioner setup still cost time and are not proved
independent of resolution. The disk coordinate improvement is specifically
2D; the streamed engine/direct directional operator support d>=2, but this
is not yet a general nonredundant spherical-harmonic map in higher dimensions.

High-order nonpolynomial inverse case also passes: same p16/N257 disk
coordinates, eight noiseless interior observations, zero field and initial
diffusivity .9. Recovered diffusivity 1.6999999999999988 (absolute error
1.1102e-15), held-out field relative L2 3.83e-16, ordinary AD PDE RMS
4.14e-15, boundary RMS 4.24e-16. Runtime 204.6 s / 1806 Krylov iterations /
9 nonlinear steps. These exact-data tests do not establish identifiability
or noise robustness for arbitrary inverse problems.

Existing `solve_route2` now accepts coordinates='disk', execution='streamed',
batch_size=... . It uses the analytical disk coordinates, preserves mode
identities in degree continuation, batches independent checks and ordinary
AD export, and supports multiple coupled fields. Captured per-point data
use the row-aware batch_function callback. Cached defaults are unchanged;
streamed box use is rejected explicitly rather than falling back to dense
arrays. Generic ResidualProblem streaming is available with any compatible
directional operator. Resource estimates are named arrays, not hard RSS caps.

Full relevant regression suite: 169 passed, four existing mpmath deprecation
warnings, 7.24 s. Independent driver/disk/streamed audit: 34 passed, no
blocking findings. The diagonal-only control also reaches the floor:
field relative L2 3.6834e-16, held-out ordinary AD PDE RMS 4.1758e-15, max
4.0856e-14, 185.6 s / 1701 Krylov iterations / three accepted nonlinear
updates. It constructs no block preconditioners. The last inner solve crossed
the 180 s soft budget; its accepted step satisfies the sampled residual
tolerance (maximum scaled RMS 2.2407e-15), but the existing termination order
returns budget_exhausted before rechecking convergence. Preserve that status;
the saved independent audit records final_sampled_tolerance_met=true.

This control demonstrates the lower-setup-cost option rather than merely
assuming it reaches the floor. Fixed-count diagonal probes and the iterative
products avoid the block setup's conservative O(Q*B*P) time cost. Total solve
time still multiplies evaluation-scale products by a resolution-dependent
iteration count. Block setup costs only .188 s in the current forward case,
so it is faster overall here (59.2 s); it may be less attractive at much
larger coordinate counts. No resolution-independent convergence proof exists.

All current tasks complete. Results/weights/overview figure are under
results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/
streamed_solver/. Next scope: higher-dimensional analytical angular maps
without dense tensors, derivative-aware adaptive allocation, and harder
coupled PDE/independent-validation benchmarks. These results establish the
memory/accuracy trade on smooth nonlinear forward and exact-data inverse
disk problems, not a universal PDE or Navier--Stokes floor claim.
