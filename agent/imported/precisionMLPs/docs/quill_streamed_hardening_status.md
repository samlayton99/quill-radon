# Streamed native ridge solver hardening

**Current status, October 1, 2026: experiments paused at the user's request while results are organized.** The current source of orientation is the [results library](radon_results_library/README.md), especially [rankings by problem class](radon_results_library/rankings.md) and [pause records](radon_results_library/STATUS.md). Descriptions of active/next work below are historical snapshots, not a current execution queue.

Coordinator: root. Started October 1, 2026. User requests generalizing the
strongest native ordinary-MLP solver toward machine precision on difficult
PDEs, including Navier–Stokes. Memory remains the primary resource constraint.

Starting point: E10/E11 in `radon_method_catalogue.md`: analytical independent
disk coordinates, streamed actual-tanh residual products, numerical native
Gauss–Newton/Krylov coefficient solves, independent ordinary Torch export.

Ownership:
- streamed_box: shared solver safety/performance review, resource preflight,
  ideal correction Jacobians and generic native preset; focused tests.
- hard_ns_suite: coupled NS declarations, independent validation, experiments.
- catalogue_early_pde: inverse NS, fused neural jets, boundary-driven physical
  flow benchmarks and independent audits.
- root: shared solver hardening, driver integration, other nonlinear probes,
  integration tests, experimental interpretation and in-chat results.

Rules: no outside PDE trajectory encoded as the answer; no hidden polynomial
or product evaluation in the deployed model; exact manufactured solutions are
forcing/allowed boundary data/validation only, not fitted interior labels.
Report representation, nonlinear convergence, ordinary arithmetic, and
held-out PDE residuals separately. Do not equate a single easy control with
general Navier–Stokes precision. Preserve unrelated work; no commits.

Local context check passed; cross-machine context sync failed transport.
Work uses the existing machine/workspace (16 GB M4 Mac mini), with the working
repository `.venv` and single-thread numerical libraries. These are not laptop timings.

Implemented and checked:
- Factored analytical box-to-directional conversion for d>=2; no retained
  direction-by-degree-by-coefficient map and no dense point-by-coordinate cache.
- Shared forward and transpose evaluation across multiple physical fields.
- Bounded selected-column panels for block preconditioning.
- Optional equation-derived field parity grouping; it is a preconditioner
  grouping, not a restriction on the solution's symmetry.
- Explicit linearized-stationarity exit above tolerance, retaining false
  convergence flags and allowing degree refinement instead of futile retries.
- Encoder-workspace resource preflight, exact antipodal angular-rule counts,
  native checkpoint continuation, per-resolution Krylov work accounting, and
  full-size independent ordinary-network validation sets.
- Same-degree resumption cannot satisfy a richer-resolution check; incompatible
  cached/streamed disk bases are rejected clearly.

Measured native results (ordinary float64 Linear/Tanh/Linear export):

| Case | Neurons | Initialization | Velocity relative L2 | Pressure relative L2 | Momentum RMS |
|---|---:|---|---:|---:|---:|
| Smooth nonpolynomial 2D no-slip NS, p14 | 4,365 | Native p6->p10 continuation | 3.722e-15 | 1.257e-14 | 1.972e-14 |
| Same 2D case, p14 | 4,365 | All readouts zero | 3.465e-15 | 9.286e-15 | 1.961e-14 |
| Same 2D flow, tenfold lower viscosity 0.03 | 4,365 | All readouts zero | 3.477e-15 | 1.160e-14 | 1.066e-14 |
| Joint inverse 2D flow and viscosity, five velocity locations | 4,365 | Zero readouts, viscosity 0.15 | 3.490e-15 | 1.708e-14 | 1.951e-14 |
| 3D quadratic NS control, p2 | 1,377 | All readouts zero | 4.284e-15 | 6.723e-15 | 2.999e-16 |
| 3D interacting trigonometric NS, p8, first diagonal run | 23,671 | Native p4 continuation | 5.83e-7 | 4.11e-6 | 4.49e-7 |
| 3D interacting trigonometric NS, p16 | 84,099 | Native PDE continuation only | 7.95e-15 | 1.57e-14 | 1.06e-15 |

The cold 2D run required four Newton corrections and 733 total Krylov
iterations, about 89.6 seconds. It used prescribed body force, zero wall
velocities and a pressure gauge, with no interior solution labels. Its held-out
check uses 2,048 fresh points. The 3D quadratic case is a deliberately easy
control; the first trigonometric run exhausted its time budget. None of these
results establishes a general turbulent or time-dependent Navier--Stokes floor.

Further completed improvements and ongoing work (updated after the user's
request to keep pushing useful higher-dimensional problems):
- Optional ideal-coordinate panels for preconditioning, and a separate
  optional ideal correction Jacobian. The latter is an approximate-Jacobian
  Newton correction, evaluated at actual neural physics sensitivities. Actual
  residuals, objective gradients, line searches, stopping and ordinary export
  remain neural. No polynomial PDE solution initializes the model.
- The larger 3D native p16 solve reached ordinary-network velocity 7.95e-15,
  pressure 1.57e-14, momentum 1.06e-15 and divergence 1.30e-16. The final
  correction took 211 s / 300 inexpensive Krylov iterations, with only three
  actual forward products and one actual adjoint. Ordinary model 5.38 MB;
  solve peak RSS 450.1 MB. The final single-degree resumed wrapper reports
  resolution_limit, correctly retaining no successive-refinement certificate.
- Joint viscosity recovery succeeded with five explicitly declared velocity
  observation locations (ten scalar values): fitted viscosity has the same
  float64 value as prescribed 0.3. Six Newton steps, 850 Krylov iterations,
  107.6 seconds, 325.5 MiB process RSS. No identifiability theorem is claimed.
- Reflection-balanced sampling did not improve total 2D runtime (~90 seconds),
  despite reducing the first Stokes-like correction cost; retain as optional.
- Optional inexact Newton tolerance scheduling has guarded tight retries and
  unchanged actual residual stopping; its controlled regression uses 77 rather
  than 110 Krylov iterations at the same roundoff-scale residual. This is not
  yet a Navier--Stokes speed result.
- Fused shared stable jet evaluation now measures 2.2–2.3x faster per product
  on fixed multi-field 3D workloads. Bounded panels remain; this is not a
  universal full-solve speed claim.
- Disk analytical correction jets now use stable Cartesian Jacobi/solid
  harmonic formulas. Degree12/24/40 panels are 3.8/5.7/7.5x faster than the
  independent angular identity, with 0.89 MB traced panel allocation. Actual
  tanh evaluation is unchanged; derivatives through second order tested.
- A genuine four-input transient 3D NS solve uses zero/native continuation
  only, with active time derivative and prescribed initial data. Its p12/N129
  ordinary errors are velocity1.90e-11, pressure1.46e-9, momentum5.74e-12,
  divergence4.58e-12. This is not a floor success. The p16/N257 correction is
  completed one100-Krylov correction in2,038.8s: velocity6.28e-13,
  pressure4.04e-11, momentum3.68e-13, divergence3.52e-13. It remains abovegoal.
  This1,429,683-neuron/19,380-unknown run predates tensor acceleration. A
  smaller N129/lambda0.25 model is supported by target-free native-coefficient
  audits, and a stronger full-parity correction is next. Separate analytical
  capacity checks never initialize the PDE solve and are not PDE successes.
- Boundary-driven physical cases have no manufactured interior truth and
  identically zero forcing. Disk stirring at viscosity0.1 improves momentum
  RMS from6.67e-3 (p8) to3.93e-15 (p40,11,931 neurons). A p48/N513 check gives
  RMS4.25e-15 and relative velocity change6.11e-15 with no additional fitting.
  At viscosity0.03, the same wall conditions yield momentum RMS6.71e-15 at
  p48/N257 (14,259 neurons), with maximum observed residual1.97e-13. A
  p56/N513 check changes velocity6.75e-15 relative and retains RMS7.53e-15.
  The lower-viscosity solve peaks at392.7MiB including audit. These are sampled
  residual/refinement results, not known true field errors or uniform bounds.
- Square cavity remains unsuccessful. At Re20, p16/24/32 momentum RMS is
  4.21e-4/1.21e-4/9.40e-5; corner residuals become39–57 times bulk RMS. The
  p24/32 runs are iteration limited, not certified optimal. At p32, box
  encoding itself changes the PDE by5.22e-6 RMS, so both corner treatment
  and stable high-degree conversion need work. Re66.7 p16 reaches1.83e-3
  and passes local actual linearized stationarity, not a global certificate.
  A separate enclosing-disk-coordinate cold/native sequence reduces p32
  encoding discrepancy to1.47e-9 while leaving physical heldout RMS1.14e-4
  and corner RMS5.42e-3. This isolates conversion from the unresolved physical
  approximation/sampling problem. All four degrees reach their iteration cap.
- The generic native preset solves -mu*u_xx+u=1 on x in[-1,1], mu in[1,2]
  with homogeneous boundary values and no interior labels. Final p26/N257
  model has7,857 neurons, relative value error1.63e-15, PDE RMS3.31e-14,
  max PDE1.25e-12 and relative parameter derivative error3.90e-13. Declared
  tolerance is1e-13. Total solve57.3s, audit peak348.2MB; richer p26 check
  needs no additional correction. Analytic solution is audit-only.
- Bounded ideal box tensor products give4.0–4.4x forward and1.6–1.9x adjoint
  speedups at four-input p12/p16, compared with reused-coordinate panels.
  They permit at most4MiB per packed coefficient tensor and16MiB planned
  workspace, then fall back to panels. Workspace is counted in preflight;
  actual neural residuals and deployed model are unchanged.
- An accepted but negligible inexact step requests one tight actual-Jacobian
  check. The coarse p6 parametric case now exits at10 rather than20 iterations
  as linearized_stationary, with residual1.57e-3 and convergence still false.
- Bandwidth audits use fixed native physical-flow coefficients, no target
  values and no new solve. At lambda0.28, doubling centers257->513 worsens
  ordinary momentum RMS9.89e-13->2.00e-12 while values remain near3e-15.
  Lambda0.2 retains4.22e-15/3.91e-15. Infinite-lattice analysis predicts
  second-derivative ghosts proportional to1/h at fixed lambda, so more nodes
  alone are not a monotone PDE-accuracy guarantee.
- Resource preflight now distinguishes ideal coordinate panels from actual
  directional profile panels. The former no longer spuriously counts a huge
  directional-by-degree-by-block workspace it never allocates.

Evidence files and figures live in
`results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/`.
Final focused integration: 261 tests pass in8.75s after tensor products,
fused jets, inverse/physical cases and the tight stationary-step guard; five
dependency deprecation warnings. Historical timings span evolving versions.
The generic dense directional representation still scales as(p+1)^(d-1)
directions: at p16/N257, four-field model arrays are103MB in4D,1.94GB in5D,
36.36GB in6D. These last rows are architecture forecasts, not solved cases.

## Broader battle-test campaign, active October 1 evening

User explicitly requests hard relevant problems, noisy inverse recovery,
fixed general setup, no truth leakage or selected-case overfitting, and measured
dimension limits without exploiting sparsity/known active subspaces.

- Coordinator root owns harder physical-flow campaign and overall synthesis.
  `route2_battletest_flows.py` freezes lower-viscosity stirring and prescribed
  counter-rotating wall motion. Zero forcing and no interior reference exist.
  All initial states are zero or prior native PDE states. Same settings across
  cases; degree/viscosity continuation is disclosed. Results under
  `route2_battletest/flows/`; retain every budget and iteration exit.
- catalogue_construction owns `route2_battletest_forward.py` and tests/results:
  published Helmholtz n2/n4, standard Allen–Cahn and a higher-diffusion control,
  and explicit manufactured variable-diffusion contrasts10/1000. Two seeds,
  twelve fixed-protocol cases. Formula/reference fields are audit-only.
- streamed_box owns `route2_battletest_inverse.py` and tests/results:
  nonlinear Burgers viscosities0.1/0.02,32 sensors,noise0/0.001/0.01,
  two seeds/starts plus crossed starts. Cole–Hopf quadrature generates declared
  observations and independent audits only. True viscosity is not stored in
  the solver declaration. Frozen baseline has14 cases. Uniform tight-control
  case is saved separately and did not improve the equal-time result.
- hard_ns_suite finishes four-input NS polish and expanded independent audit.
  First full-group correction gives ordinary velocity1.887e-15 and pressure
  6.957e-14 relative, momentum5.52e-16; second bounded correction active.

Shared core/preset is frozen during the baseline campaigns. Source hashes and
pre-run protocols are retained. Their60-second solve budgets are screening
budgets, not evidence of an accuracy ceiling; extensions must be separately
declared and apply the same policy across cases. Root flow stages use120s soft
budgets. All limits are soft around inner solves, and actual time is reported.

Emerging issues, not conclusions: coarse inverse stages hit outer iteration
caps, but tightening all inner solves spends more time at nearly identical
coarse residual minima and worsens equal-time field accuracy. Finite equal
PDE/data penalty weights also generally let noisy data pull the fitted field
off the physical solution manifold. Per-block discrepancy thresholds do not
mathematically turn that objective into a constrained inverse solve. No core
fix or universal inverse-recovery claim has been made on this evidence.

### Accuracy extensions and generic controls, October 1 late evening

- Four-input NS is now finished, not merely pending: expanded independent
  audit gives velocity1.845e-15, pressure6.369e-14 relative; interior momentum
  RMS9.597e-16, but corner field max1.318e-13 and momentum max1.463e-13.
  Final model751,689neurons/54.12MB; peakRSS1.081GB; final native polish1999.6s.
  Final status time_budget. No uniform/all-field1e-14 or turbulence claim.
- Forward V1: all12 cases complete. Helmholtzn4 is badly unresolved; Allen–Cahn
  field errors89–94%, and both fitted initial curves erased the true positive
  bump. Diffusion contrast10 versus1000 keeps the SAME target and final model
  space, yet errors change~4e-8 to~.0046. That separates solving difficulty
  from model-space capacity. All were60s screening budgets, not accuracy limits.
- Forward V2: all6 equation cases, seed0, uniformly600s and60outer iterations;
  other settings unchanged. Two single-thread workers, wall times not isolated.
  First Helmholtzn2 result improves2.67e-6 ->3.57e-10 field error, PDE6.02e-3
  ->1.69e-7. Actual733s illustrates the soft budget's Krylov overshoot.
- Inverse V1: all14 Burgers cases retained. ν=.1 parameter errors.34–1.15%;
  ν=.02 errors207–220% (viscosity inflation). No case meets physical tolerance.
  V3 uniformly gives600s,60outer iterations, and degrees through64 to four
  preselected seed0 cases (two viscosities x noise0/.001). Smoother noiseless
  ν=.1 improves to.00546% parameter error and1.95e-5 field error; still unresolved.
- streamed_box owns NEW `solver/route2_inverse.py`: separate physics-only
  validated native forward solves from the noisy outer sensor objective.
  Unresolved forwards are never sensor-scored; parameters frozen and removed
  from inner unknowns. This is reduced PDE-constrained optimization implemented
  with native neural forwards, not a new external solver or new conceptual idea.
  Easy precision control N257: noisy optimal parameter recovered within9e-14,
  actual PDE RMS2.61e-14, max8.97e-14, conditional field error9.99e-16.
  True parameter differs by5.80e-5 due to the fixed noise realization.
  Each control19validated forwards/~5s/~310MB. Not a hard-inverse success.
  Blind physical disk-NS ν=.05 forward gate is running before any sensor use.
- root owns NEW `route2_initial_plane_profiles.py`, tests, and study:
  analytically multiply disk coordinates by(z_y-z_initial), then pair actual
  tanh neurons F(x,t)-F(x,t_initial). Add ONLY an analytical encoding of given
  IC. Export remains flat Linear/Tanh/Linear. Four tests and independent
  theory review pass. Uses no-parity block grouping because time parity mixes.
  Two600s Allen–Cahn controls are active/completing. Hard IC alone is not yet a
  success: large readouts amplify ordinary summation error at higher degree.
- root owns NEW `solver/residual_scaling.py` and normalized6case campaign:
  fixed zero-jet differential-operator norm using domain widths/unit field
  scale; PDE rows normalized, BC/IC unchanged. Same600s/60outer protocol asV2.
  Only mathematical change is residual weighting. Original raw PDE checks
  remain separate; normalized stopping must not be called raw convergence.
  Two tests pass. Two workers active. No shared existing core changes.
- hard_ns_suite owns NEW opt-in `route2_disk_recurrence.py`: bounded Jacobi
  family recurrence/tables to avoid repeated polynomial special-function work
  in ideal products. Actual neural residual/export unchanged. Preliminary
  ~2.2x product speedup,13tests; fuller product optimization under review.
- Current physical p64 counterrotation is spending tens of minutes in its
  first inner solve. Process sampling shows ideal Jacobi/power evaluation,
  NOT actual-J fallback; RSS~280MB. This is a time bottleneck. Soft120s budget
  does not interrupt that inner call. No successful p64 result claimed yet.

All new controls use separate result directories beneath `route2_battletest/`.
No frozen earlier campaign is silently overwritten. Existing core stays frozen
while these comparisons run; new modules are opt-in only.

### October 2 05:32 UTC: requirements review and ongoing controls

The user now asks which historical requirements are too strict for a useful
blind PDE/inverse solver. This is a review request, not an instruction to
silently replace the model with an external PDE solution. Keep no-oracle
construction, independent deployed-model checks, bounded memory and honest
failure reporting. Recommend evaluating local/multilevel Radon models, causal
windows, weak/conservative equations, and constrained inverse objectives.
Global flatness and identical penalties were partly our implementation choices,
not categorical user prohibitions. Preserve the existing strict experiments.

- Forward V2 all6 cases finished. Field errors: Helmholtz n2 3.57e-10,n4 .0322;
  diffusion contrast10 6.00e-14,contrast1000 .00432; Allen–Cahn .01/.0001
  .693/.804. Cross-operator evaluation confirms the contrast10 network also
  satisfies contrast1000 PDE to3.92e-11 RMS, with no refit.
- Generic normalization: contrast1000 field8.03e-11/rawPDE1.15e-7; Helmholtz
  n4 field.00443. All6 fits now complete; final evaluation/replot pending.
  Root normalized wrapper was updated to avoid double scaling after common
  benchmark gained its own normalization hook. Original loaded runs preceded
  that integration; future fits scale once and audit original equations.
- Paired normalized precision extension (tol1e-13): both fail to reach floor
  within613/660s, stall atp24; fields3.26e-11/8.03e-11,rawPDE1.15e-7. Tight
  inner work at coarse resolutions prevents timely refinement.
- Five-input baseline manually interrupted at929s, explicitly not normal soft
  budget termination: p6 actual-J correction unfinished,p8 unattempted. Last
  accepted ordinary state field3.42e-7,PDE1.31e-6,cornermax2.43e-5.
  A separate cold control now uses default-OFF `refine_on_ideal_stall=True`.
  This returns `approximate_model_stalled` above tolerance and explores the
  next resolution without claiming accurate stationarity/capacity. Existing
  default fallback unchanged;59ideal/stationarity/driver tests passed.
- Root physical counterrotation p64 finally completed400idealKrylov iterations
  in3748.7s: momentumRMS2.46e-8,div3.77e-7,wall7.50e-10; nofloor. Subsequent
  viscosity continuation hit a runner bug attempting p64->p48; no later cases
  fitted. Fixed by selecting an existing matching native p48 archive rather
  than truncating coordinates;2regression tests+14related tests pass.
- Separate root physical recurrence control is now running (session82248,
  /private/tmp/quill_physical_recurrence.log). Four physical cases,uniformp64,
  N513,lambda.2,600ssoft,150K,max12outer,up to2048-column parity groups within
  1GiB estimated preconditioner allowance, bounded recurrence products.
  Actual tanh model/residual unchanged; all sources native states only.
  First startup unpacking error repaired before any solve; artifact records it.
- streamed_box continues noise0/.001 wall-NS reduced inverse and blind Burgers
  forward gates. NS had acceptednu.0294766 from.05 with validated forwards;
  not yet final. Reference is finer same-method native state, an explicit
  inverse-crime limitation. Burgers nu.04 gatefailed:field8.86%,PDE.0859;
  cannot score inverse candidates when nativeforwardphysicsunresolved.
- hard_ns_suite owns running cold5Drefinementcontrol (session81794) and audit.
- Source provenance fix: future forward fits capture startup disk/loaded-code
  hashes separately from completion hashes. Legacy completion-only hashes
  are labeled honestly; no invented original-source certification.

### User clarification, October 2: hard architectural and scaling requirements

This supersedes the preceding recommendation to relax the output architecture.
The final deployed field MUST be a single hidden-layer MLP, of the form
u(x)=b+sum_j v_j sigma(w_j dot x + beta_j). No window multiplication,
piecewise routing, stitched local outputs, hidden polynomial evaluator, or
extra nonlinear layers. Additive neuron banks and affine input transformations
can be compiled exactly into that form; test/export the actual final model.
Solver-only block decomposition, multilevel preconditioning, staged/causal
sampling and adaptive directions/centers/bandwidths remain candidates only if
the final function and its global validation preserve the hard architecture.

Numerical fitting is permitted. Its memory and compute must not add avoidable
exponential dependence on input dimension beyond the unavoidable representation
cost. Measure setup, per-product cost, derivative count and iteration growth;
matrix-free storage alone does not establish dimension-robust solving.
The public interface should be routine PDE/domain/BC/IC/observations/tolerance/
memory-budget declarations; adaptation must use those inputs and native error
checks, not hidden solution structure or hand-picked per-equation recipes.
Noise-aware inverse objectives and uncertainty limitations are accepted.

New completed five-input control: p4/p6/p8, full tensor directions, no sparse
axis choice, ordinary one-hidden-layer MLP. Field errors3.32e-5/3.42e-7/2.73e-9;
p8 has1,003,833neurons,1287coordinates,rawPDE1.73e-8,cornermax5.37e-7,
917s solve,476MBpeakRSS. It is a smooth well-posed scaling control, not a hard
five-dimensional floor result. p8 used17idealKrylov iterations andoneaccepted
correction; no actual-Jfallback. Whole stage619s; component fractions unmeasured.

Noiseless physical wall-NS reduced inverse complete at1e-8 requestedphysical
RMS tolerance:nu=.0299999996305 versusreference.03,relativeparametererror1.23e-8,
velocitydifference6.34e-10,ordinarymomentumRMS2.82e-9,max1.09e-7;15validated
forwards/1889s. Same-method finer reference remains an inverse-crime limitation.
Originalpeak1.38GBincludesavoidableunbatchedreferenceaudit; separatebatchaudit
uses466MBandmatchesnumbers. Noisycase remainsrunning. BlindBurgersforwardgates
failed fornu.04and.02; reducedinverse cannot bypass thatfailure.
