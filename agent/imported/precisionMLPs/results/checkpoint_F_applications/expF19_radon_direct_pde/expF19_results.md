# expF19: native streamed Radon MLP hardening — draft, measured results

## TL;DR

- A flat tanh MLP solves the smooth, nonpolynomial, no-slip 2D Navier–Stokes benchmark from zero readouts to relative velocity error $3.5\times10^{-15}$. Reducing viscosity tenfold retains that accuracy.
- The same solver jointly recovers viscosity and the three fields from five velocity observation locations, with no external PDE solve or target-based initialization. This is one noise-free manufactured inverse problem.
- A harder interacting, nonpolynomial 3D Navier–Stokes test now reaches velocity relative error $7.95\times10^{-15}$ and pressure relative error $1.57\times10^{-14}$ in an ordinary 84,099-neuron tanh MLP. An inexpensive approximate correction Jacobian removes most repeated full-network work; the residual being solved remains the actual neural PDE residual.
- A boundary-driven circular flow with identically zero body force reaches ordinary-network momentum RMS $3.93\times10^{-15}$ with 11,931 neurons. Refining to 27,391 neurons changes velocity by $6.11\times10^{-15}$ relative and retains momentum RMS $4.25\times10^{-15}$. This case has no known interior solution, no interior labels and no external PDE solve. Residual and refinement agreement are evidence, not an exact field-error certificate.
- With the same wall motion and zero forcing, lowering the physical flow's viscosity to $0.03$ reaches momentum RMS $6.71\times10^{-15}$ with 14,259 neurons. A richer 31,863-neuron check changes velocity by $6.75\times10^{-15}$ relative. This changes the unknown physical solution, unlike the manufactured lower-viscosity control.
- The generic solver preset also solves a parameterized diffusion family with relative field error $1.63\times10^{-15}$. Its PDE RMS is $3.31\times10^{-14}$ and maximum observed residual $1.25\times10^{-12}$; these metrics must not be conflated.
- A four-input transient 3D flow reaches relative velocity error $1.84\times10^{-15}$ and pressure error $6.37\times10^{-14}$ in a 751,689-neuron ordinary MLP, with 1.08 GB peak RSS. Its interior momentum RMS is $9.60\times10^{-16}$, but corner errors are around $10^{-13}$. This is a smooth manufactured problem on a short time interval, not a turbulent-flow result or an all-domain $10^{-14}$ certificate.
- The broader fixed-protocol tests expose substantial failures: sharp Burgers inverse problems inflate viscosity by roughly 200%, Allen–Cahn loses a small positive initial phase, and higher-frequency Helmholtz is badly unresolved under the initial 60-second screening budget. Uniform accuracy extensions are running; these screening failures are not established accuracy ceilings. The square cavity remains unsuccessful at the requested precision.

## Question / hypothesis

Can the strongest native ridge method retain ordinary-MLP accuracy while becoming useful for coupled nonlinear PDEs, sparse observations, unknown physical parameters, and higher-dimensional inputs under a bounded-memory solve?

## Experiment design

The deployed field is an ordinary one-hidden-layer network. At points $x\in\mathbb R^d$, its $F$ output fields have the form $u(x;c)=\Phi_{\theta}(x)A c+b(c)$, where $\Phi_{\theta}$ contains prescribed tanh ridge features and $c\in\mathbb R^{P\times F}$ contains independent readout coordinates. The analytical map $A$ converts those coordinates to neuronal readouts. It is implemented as factored operations rather than a retained dense matrix. These coordinates restrict the redundant readout space; they are not a bijective reparameterization of every possible neuronal readout.

For a polynomial reference resolution $p$, $P=\binom{p+d}{d}$. The positive tensor angular rule uses $M=(p+1)^{d-1}$ directions, including $M=p+1$ in two dimensions. The number of interior centers is $N$, the halo on each side is $R=\lceil\sqrt N\rceil$, and the actual neuron count is $B=M(N+2R)$. The runs below use $\lambda=0.2$ and $\gamma=\lambda/[2/(N-1)]$. Earlier known-profile controls sweep bandwidth and center count; these controls establish encoding capacity and do not provide initialization for the native PDE runs.

Let $r(c,\eta)$ be the stacked, scaled residual of the equations, boundary conditions, pressure gauge, and any explicitly declared observations; $\eta$ contains unknown physical parameters when present. Every nonlinear residual uses the actual tanh field. By default its correction Jacobian $J$ also uses actual neural derivatives. Gauss–Newton updates are computed by LSMR without assembling a global Jacobian. A bounded block right preconditioner $R_c$ changes the inner coordinates: $\min_z\|J R_c z+r\|_2$, with update $\delta=R_c z$. This remains a numerical native physics solve.

There are two distinct optional uses of the analytical reference coordinates. First, the ideal-coordinate **preconditioner** uses their derivatives while preparing small Gram blocks. The local physics sensitivities are evaluated at the current neural state. Since the regularized triangular factors are invertible, this changes conditioning and the damping metric, not the undamped actual-Jacobian linearized least-squares objective.

Second, the new ideal-coordinate **correction Jacobian** replaces the neural coordinate derivatives in the inner linear solve. Write $A_{\mathrm{nn}}$ for the map from coefficient perturbations to all neural field derivatives required by the equations, $A_0$ for the corresponding analytical-coordinate map, and $S$ for the local equation sensitivities at the current neural state. Then the actual Jacobian is $J=S A_{\mathrm{nn}}$, while the cheaper correction matrix is $J_0=S A_0$. Unknown physical-parameter columns remain actual. Both maps are streamed; neither is stored as a global point-by-coordinate matrix. The solver minimizes $\|J_0\delta+r_{\mathrm{nn}}\|$, then evaluates and accepts the candidate using the **actual** neural residual $r_{\mathrm{nn}}$. Its objective gradients, line searches, stopping checks, and ordinary export remain actual. Rejected, invalid, or negligible approximate steps retry the actual Jacobian before stationary/precision-limit conclusions. This option is an approximate-Jacobian Newton method, not an exact Gauss–Newton step and not a previously solved polynomial field encoded afterward.

The distinction matters for attainable accuracy. Suppose $c_*$ is a consistent root, $r_{\mathrm{nn}}(c_*)=0$, and $J_0$ has full column rank nearby. Put $e=c-c_*$ and $E=J_0-J$. Taylor expansion gives $r_{\mathrm{nn}}(c)=Je+O(\|e\|^2)$. An exact inner correction gives

$$e_{\mathrm{next}}=e-J_0^\dagger Je+O(\|e\|^2)
=J_0^\dagger(J_0-J)e+O(\|e\|^2)
=J_0^\dagger E e+O(\|e\|^2).$$

Here $J_0^\dagger J_0=I$ uses full column rank. Therefore $\|J_0^\dagger E\|<1$ gives local contraction, with an additional $J_0^\dagger$ times the inner-solve defect for an inexact correction. An approximate tangent does not inherently impose a fixed residual floor because the residual itself remains actual. This is a conditional local result, not an established bound for arbitrary PDEs. If the resolution leaves a nonzero best-fit residual, the analogous argument has an additional bias term; ideal-Jacobian stationarity is not actual least-squares stationarity.

For steady incompressible Navier–Stokes, the declared equations are

$$ (v\cdot\nabla)v-\nu\Delta v+\nabla p=g,\qquad \nabla\cdot v=0. $$

The two-dimensional reference is specified by $\psi=(1-x^2-y^2)^2e^{0.35x-0.2y}$, $v=(\partial_y\psi,-\partial_x\psi)$ and $p=0.2\sin(0.7x-0.4y)$ on the unit disk. It has identically zero wall velocity. The known expression is used to prescribe the body force and evaluate held-out error; forward solves receive no interior solution labels. Separate cold runs use $\nu=0.3$ and $\nu=0.03$. Both use $p=14$, $N=257$, 4,365 shared tanh neurons, 360 unknown readout coordinates, 512 interior physics points, 128 wall points, and the gauge $p(0)=0$.

The inverse variant fixes the body force prescribed at $\nu=0.3$ and starts the unknown viscosity at $0.15$, constrained to $[0.05,1]$. Exactly five interior locations supply both velocity components, giving ten scalar observations. All readouts initially vanish. Held-out auditing keeps the original forcing fixed while inserting the recovered viscosity on the equation's left side. Regenerating forcing at the fitted viscosity would be circular and is explicitly avoided.

Independent audits export unmodified Torch `Linear/Tanh/Linear` layers, evaluate on 2,048 fresh interior points, differentiate using ordinary Torch autograd, and check a separately rotated wall grid. Relative field error is $\|u-u_*\|_2/\|u_*\|_2$. Momentum RMS is the unscaled root mean square over both momentum components and all audit points. Small sampled residuals are not a continuum error or uniqueness certificate.

**Code & data.** Solver: `experiments/expF19_radon_direct_pde/solver/{general_residual,streamed_residual,route2}.py`, including the opt-in `solve_route2_native` preset. Analytical coordinate operators: `route2_{directional,disk,affine_disk,box}_profiles.py` in the experiment folder. Reproduction scripts: `route2_hardening_disk.py`, `route2_hardening_ns.py`, `route2_hardening_inverse_ns.py`, `route2_physical_flow.py`, `route2_ns_extended_audit.py`, `route2_ideal_panel_benchmark.py`, `route2_hardening_figures.py`, `route2_physical_figures.py`, `route2_physical_encoding_sweep.py`, `route2_hardening_parametric_native.py`. Native arrays, exported Torch weights, machine-readable configurations, histories and checks: `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_hardening/{disk_ns,ns,inverse_ns,physical_flow,parametric_native}/`. Main figures: `route2_hardening/native_ns_disk.png`, `native_ns_extensions.png`, `native_ns_resolution.png`, `ns/native_3d_accuracy_ladder.png`, `physical_flow/disk_compact.png`. Exact architecture forecasts, not solved high-dimensional cases: `route2_hardening/dimension_footprint.json`. Coordination and precise current limitations: `docs/quill_streamed_hardening_status.md`.

## Results

All three successful 2D cases below start from zero neuronal readouts. Timings are single-thread numerical runs on a 16 GB M4 Mac mini, with other research processes active; these are not controlled laptop speed benchmarks.

| Native case | Velocity relative L2 | Pressure relative L2 | Momentum RMS | Newton / Krylov iterations | Solve time |
|---|---:|---:|---:|---:|---:|
| Forward, $\nu=0.3$ | $3.465\times10^{-15}$ | $9.286\times10^{-15}$ | $1.961\times10^{-14}$ | 4 / 733 | 89.6 s |
| Forward, $\nu=0.03$ | $3.477\times10^{-15}$ | $1.160\times10^{-14}$ | $1.066\times10^{-14}$ | 6 / 796 | 100.3 s |
| Inverse, unknown $\nu$ | $3.490\times10^{-15}$ | $1.708\times10^{-14}$ | $1.951\times10^{-14}$ | 6 / 850 | 107.6 s |

The recovered viscosity has the same float64 value as the prescribed $0.3$. That is a floating-point comparison, not exact mathematical parameter recovery or a global identifiability theorem. The inverse run retains zero feature-cache bytes and has a process peak RSS of 325.5 MiB. The forward runs use approximately the same process memory.

The saved inverse control with actual-neuron Gram blocks reused across nonlinear iterations exhausted its 240-second soft budget after approximately 247 seconds. It recovered viscosity only to absolute error $4.9\times10^{-6}$. Refreshing ideal Gram blocks at every Newton step produced the successful result above. Both basis choice and refresh changed, so this comparison does not isolate their individual contributions.

Reflection-balanced sampling reduced the initial Stokes-like correction cost but did not reduce the full 2D solve: both balanced and unbalanced cold runs took about 90 seconds and reached the same accuracy range. It remains an optional conditioning control, not a general speed claim.

The 3D quadratic control with 1,377 neurons reaches relative velocity error $4.284\times10^{-15}$, pressure error $6.723\times10^{-15}$, and ordinary-AD momentum RMS $2.999\times10^{-16}$. Its driver correctly remains unvalidated by successive resolution because only one degree was run. This is an easy representation control, not evidence for hard or turbulent 3D flow.

The harder 3D interacting trigonometric problem was solved by native continuation through degrees four, eight, twelve and sixteen. At degree twelve the ordinary-network velocity and pressure errors were $4.63\times10^{-12}$ and $4.19\times10^{-11}$. An actual-Jacobian degree-sixteen polish spent 685 seconds and 50 Krylov iterations, reaching velocity error $1.95\times10^{-12}$, pressure $1.31\times10^{-11}$, momentum RMS $1.11\times10^{-12}$ and divergence RMS $6.35\times10^{-12}$: this attempt did not reach the floor.

Resuming that **native** degree-sixteen state with the cheaper ideal correction Jacobian required one accepted correction and 300 inner iterations, taking 211 seconds. The exported 84,099-neuron network then reached velocity relative error $7.95\times10^{-15}$, pressure relative error $1.57\times10^{-14}$, momentum RMS $1.06\times10^{-15}$ and divergence RMS $1.30\times10^{-16}$. The model arrays occupy 5.38 MB and solve peak RSS was 450.1 MB. Only three actual forward calls and one actual adjoint call were needed, versus 301/306 inexpensive ideal products. The wrapper reports `resolution_limit` because this resumed run contains only one degree and cannot independently establish successive-refinement agreement; the native nonlinear residual criterion did pass. No exact readout or outside PDE trajectory initialized this solve. A separate known-profile degree-sixteen control is retained solely as encoding-capacity evidence.

A larger independent audit of this same saved network uses 2,048 value locations, 1,024 derivative locations, all eight corners and 1,024 boundary locations, without updating any coefficient. Velocity/pressure relative errors are $7.934\times10^{-15}$ and $1.557\times10^{-14}$; momentum RMS is $8.945\times10^{-16}$ and divergence RMS $1.034\times10^{-16}$. The largest observed interior momentum residual is $1.195\times10^{-14}$; corner momentum RMS is $3.449\times10^{-15}$. These distinguish RMS accuracy from maximum observed residuals.

The transient three-spatial-dimensional test has four inputs $(x,y,z,t)$, an active time derivative, prescribed initial/wall data and a time-dependent pressure gauge. Its final native degree-sixteen, $N=129$, $\lambda=0.25$ model contains 751,689 neurons and 19,380 independent field coordinates. Two final corrections, starting exclusively from preceding native PDE states, took 1,999.6 seconds and 196 inner iterations; peak RSS was 1.081 GB, and model arrays occupy 54.12 MB. A larger independent audit uses 2,048 value points, 1,024 derivative points, all 16 space-time corners, 1,024 wall points and 1,024 initial points. Velocity and pressure relative errors are $1.845\times10^{-15}$ and $6.369\times10^{-14}$; interior momentum/divergence RMS are $9.597\times10^{-16}$ and $1.174\times10^{-16}$. The largest corner field error is $1.318\times10^{-13}$ and corner momentum maximum is $1.463\times10^{-13}$. Thus the result reaches the velocity interior floor but not a uniform all-field $10^{-14}$ target. The final status remains `time_budget`, with its last accepted state preserved. Separate known-profile capacity checks did not initialize any PDE solve.

The non-manufactured circular-flow problem prescribes the tangential wall velocity

$$v(\cos\theta,\sin\theta)=\bigl(1+0.25\cos(2\theta)+0.15\sin(3\theta)\bigr)(-\sin\theta,\cos\theta),$$

with viscosity $0.1$, zero body force, incompressibility and pressure gauge $p(0)=0$. Interior readouts begin at zero and subsequent resolutions receive only preceding native PDE coefficients. On the diameter-two domain the boundary-speed-based Reynolds-number upper bound is 28. At degree 32 the advection, viscous and pressure-gradient RMS magnitudes are approximately $0.551$, $0.285$ and $0.581$: the tiny net residual arises from a nontrivial nonlinear balance.

| Circular physical flow | Neurons | Ordinary momentum RMS | Wall velocity RMS | Observation |
|---|---:|---:|---:|---|
| $p=8$, $N=257$ | 2,619 | $6.67\times10^{-3}$ | $8.35\times10^{-4}$ | Coarse resolution |
| $p=12$, $N=257$ | 3,783 | $2.19\times10^{-4}$ | $1.03\times10^{-5}$ | Native continuation |
| $p=16$, $N=257$ | 4,947 | $5.61\times10^{-6}$ | — | Native continuation |
| $p=20$, $N=257$ | 6,111 | $1.06\times10^{-7}$ | $1.99\times10^{-9}$ | Time limited |
| $p=24$, $N=257$ | 7,275 | $1.91\times10^{-9}$ | $2.32\times10^{-11}$ | Inner iteration cap reached |
| $p=32$, $N=257$ | 9,603 | $5.79\times10^{-13}$ | $3.66\times10^{-15}$ | Above requested residual tolerance |
| $p=40$, $N=257$ | 11,931 | $3.93\times10^{-15}$ | $2.01\times10^{-15}$ | Actual sampled solve converged |
| $p=48$, $N=513$ | 27,391 | $4.25\times10^{-15}$ | $3.52\times10^{-15}$ | Richer encoding/constraint check passed |

The degree-forty audit has 4,096 fresh points; its maximum observed momentum residual is $3.11\times10^{-14}$, divergence RMS $3.56\times10^{-16}$ and net boundary mass flux $-1.62\times10^{-15}$. Its final correction took 295 seconds. The degree-forty-eight transfer already meets its denser constraints without further optimization. Relative field changes from degree forty are $6.11\times10^{-15}$ in velocity and $1.99\times10^{-14}$ in pressure. That is a refinement check, not an independent cold start or an exact-solution comparison. Intermediate timings use evolving implementations, so the sequence does not isolate an asymptotic rate or controlled speedup.

Keeping the circular wall motion and zero forcing fixed while lowering viscosity to $0.03$ changes the velocity by 5.43% and pressure by 18.02% relative to the higher-viscosity solution. The boundary-speed Reynolds upper bound becomes 93.3. Native continuation to degree 48 with $N=257$ (14,259 neurons, 3,675 unknowns) reaches momentum RMS $6.708\times10^{-15}$ on 4,096 fresh points, divergence RMS $1.285\times10^{-15}$ and wall RMS $3.624\times10^{-15}$. The maximum observed momentum residual is $1.967\times10^{-13}$, so this is not a uniform $10^{-14}$ residual bound. Its final correction takes 551 seconds and process peak RSS including audit is 392.7 MiB. A degree-56, $N=513$ check (31,863 neurons, 9,918 interior locations and 29,754 scalar PDE residuals) already passes without another optimization step: momentum RMS $7.528\times10^{-15}$, relative velocity change $6.752\times10^{-15}$ and pressure change $2.734\times10^{-14}$. Both physical cases remain unknown-solution tests: refinement agreement is not a rigorous continuum error estimate.

The square lid-driven cavity remains a difficult negative control. With zero forcing, $v_x=(1-x^2)^2$ on the lid and stationary other walls, the refined degree-sixteen run gives momentum RMS $4.209\times10^{-4}$ at Reynolds number 20. Degrees 24 and 32 improve this only to $1.205\times10^{-4}$ and $9.401\times10^{-5}$; their corner RMS values are respectively 39 and 57 times the bulk RMS. These latter runs reach their outer iteration caps, not verified minima. Degree-sixteen at Reynolds number 66.7 gives $1.831\times10^{-3}$ and passes the actual linearized-stationarity check, which still does not certify global optimality. A continuously matched lid does not imply an analytic corner solution.

Fixed-coordinate actual-versus-ideal checks separate an additional problem: degree-24 box encoding changes the PDE RMS by only $7.61\times10^{-10}$, negligible beside its residual, whereas degree-32 encoding discrepancy reaches $5.22\times10^{-6}$ (about 5.6% of bulk residual). The corresponding degree-32 second-derivative encoding discrepancy is $4.18\times10^{-5}$. Thus corner concentration is not explained away by the encoding, but simply increasing the box degree also begins to expose numerical conversion error. None of these cavity cases satisfies the accuracy target.

A separate cold/native sequence uses stable disk coordinates on the square's enclosing circle of radius $\sqrt2$, preserving the physical square, equations and wall data. Its degree-32 encoding PDE discrepancy falls to $1.47\times10^{-9}$, over 3,000 times smaller than the box-coordinate run, but held-out momentum RMS remains $1.14\times10^{-4}$ and near-corner RMS $5.42\times10^{-3}$. The training physics RMS is about $6.47\times10^{-6}$, illustrating why independent checks matter. Degrees 8, 16, 24 and 32 use ten outer iterations each; these are not certified best fits. This comparison supports the conclusion that conversion arithmetic is a separable problem, not the main explanation for the cavity error at this resolution. It does not establish that changing coordinates alone improves physical accuracy.

The generic `solve_route2_native` preset was tested on the two-input family $-\mu u_{xx}+u=1$, $u(\pm1,\mu)=0$, $\mu\in[1,2]$. It receives no interior solution values and no equation-specific solver overrides. Native degrees 6, 10, 14, 18, 22 and 26 with $N=257$ produce a final 7,857-neuron model. The analytic hyperbolic-cosine solution is used only afterward for auditing. Ordinary relative field error is $1.63\times10^{-15}$ and maximum field error $8.44\times10^{-15}$; PDE RMS on 2,049 fresh points is $3.31\times10^{-14}$ with maximum $1.25\times10^{-12}$. The declared tolerance was $10^{-13}$, not $10^{-14}$. Parameter-sensitivity relative error is $3.90\times10^{-13}$. The richer degree-26 check requires no additional Newton correction; total solve time is 57.3 seconds with solve peak RSS 331.4 MB and batched audit peak 348.2 MB. The later coarse-step guard reduces the underresolved degree-six solve from twenty to ten iterations; the full 57.3-second timing predates that change.

### Figures

- **Native disk solution:** left shows speed and streamlines from the ordinary exported MLP; middle shows absolute velocity-vector error; right shows native residual against Newton correction from zero readouts. The spatial errors are at the arithmetic scale rather than a missed spatial feature.
- **Forward and inverse extensions:** left follows the recovered viscosity; right compares native residual histories for the original viscosity, tenfold lower viscosity, and joint inverse problem. Only the inverse case includes interior observations.
- **Resolution sequence:** held-out velocity, pressure and relative momentum errors against neuron count for native degree-six, ten and fourteen continuation. The first two runs were time-limited, so this is an observed refinement sequence rather than an estimated asymptotic rate.

## Additional details

The box conversion uses exact rational scalar identities rounded only after finite cancellation, axiswise Legendre-to-monomial transforms, sparse Laplacian powers, and blocked angular evaluation. Retained conversion/work storage scales as $O(pP+Mp+Md+sP)$ for bounded angular block $s$, rather than $O(MpP)$. A conservative conversion-work bound is $O(dp^2P+M(d+p)P)$. Polynomial coefficient cancellation still limits arbitrary high-degree stability; the memory reduction is not a proof of arbitrary precision.

Prepared neural readouts use $O(BF)$ storage. Residuals and local physics sensitivities use $O(Q)$ storage for a fixed system and derivative list, with $Q$ collocation points. A fixed preconditioner block width $b$ requires $O(PFb)$ factor storage and bounded row/column panels. Neither a dense $Q\times P$ feature cache nor the global neuronal Jacobian is constructed. Callback workspaces, supplied dense integral aggregation and process-level library memory are separate costs and are not magically capped by this accounting.

With the actual correction Jacobian, each forward/transpose pass costs approximately the analytical coefficient conversion plus $O(QB)$ neural work for a fixed equation system. The optional ideal correction replaces that repeated inner work with bounded analytical coordinate evaluation, approximately $O(QP)$ times the cost of evaluating one coordinate's required derivatives. Actual $O(QB)$ evaluations and coefficient conversions remain necessary for nonlinear residuals, gradients and trial acceptance. The total still depends on Krylov iterations, outer iterations and preconditioner setup; construction is not proved to equal one inference pass. The angular count also remains exponential in dimension at fixed degree: $M=(p+1)^{d-1}$.

Shared stable tanh jets reduced measured three-dimensional multi-field forward and adjoint pass times by about 2.2–2.3 times on fixed derivative workloads. This is a kernel timing, not a full solve speedup. The disk's analytical correction jets now use the Cartesian identity $(x+iy)^k P_{(n-k)/2}^{(0,k)}(2(x^2+y^2)-1)$, with explicit first and second derivatives. It avoids polar singularities, monomial expansion and angular quadrature for each ideal panel. At 64 points and 64 coordinates, degrees 12/24/40 measured 3.8/5.7/7.5 times faster than the independent angular identity, with normalized differences from about $10^{-15}$ to $4\times10^{-14}$ and 0.89 MB traced panel memory. Actual tanh evaluation is unchanged.

Field parity grouping comes from constant-coefficient local derivative terms. If an equation couples $D^a u_f$ and $D^b u_g$, the grouping imposes $\sigma_f\oplus\sigma_g=(a\bmod2)\oplus(b\bmod2)$. Inconsistent coordinate bits are disabled. This is a heuristic for block preparation; it does not constrain the solution and is not a proof of PDE symmetry. Reflection-invariant sampling is also needed for exact parity orthogonality of a constant-coefficient operator.

The driver now distinguishes residual-evaluation exhaustion, time limits, resource refusal, actual convergence, and a verified stationary linearized model above tolerance. Stationarity cannot be inferred merely from hitting the Krylov iteration cap. Native continuation preserves named coordinates, rejects incompatible disk bases, and requires a genuinely richer degree before a successive-resolution agreement can pass.

The opt-in native preset combines streamed evaluation, ideal corrections, refreshed ideal block preconditioning, equation-inferred parity grouping and guarded inexact Newton. Its local block width is capped and shrinks before allocation when the named-array budget requires it. Its default degree ladder is 4, 8, 12, 16, 24, 32, 40 and 48; the named-array budget determines admission rather than an arbitrary default neuron cap. Caller-specified caps remain supported. No PDE name controls these choices. Invalid ideal panels fall back to an actual-Jacobian correction with diagonal scaling, rather than unexpectedly allocating a large actual directional Gram panel. Factor refresh releases previous factors before building replacements. Native resumption rejects changed affine charts. An accepted but negligible coarse inexact step triggers one tight actual-Jacobian check; actual stationarity and convergence thresholds are unchanged. The final focused integration run passes 261 tests, with five dependency deprecation warnings.

The box's ideal products now optionally use bounded tensor sum-factorization with exact reverse adjoints. A total-degree coefficient vector is temporarily packed into a tensor of side $p+1$ only when individual tensor storage is at most 4 MiB and planned total workspace at most 16 MiB; otherwise it falls back to selected-coordinate panels. The planner accounts for this workspace. At 128 points, four-input degrees 12 and 16 measure forward speedups 4.37 and 4.04, and adjoint speedups 1.93 and 1.63, relative to the already table-reusing panel implementation. Relative product differences are at most $6.3\times10^{-16}$. Actual neural residuals and inference are unchanged. These are product benchmarks, not full PDE solve timings.

The solver-memory improvement does not remove representation growth. At $p=16$, $N=257$ and four output fields, the tensor angular construction has the following exact architecture counts. Rows beyond four inputs are forecasts only; they are not reported as solved PDEs.

| Input dimension | Neurons | Ordinary model arrays | Independent coupled unknowns |
|---|---:|---:|---:|
| 3 | 84,099 | 5.38 MB | 3,876 |
| 4 | 1,429,683 | 102.94 MB | 19,380 |
| 5 | 24,304,611 | 1.94 GB | 81,396 |
| 6 | 413,178,387 | 36.36 GB | 298,452 |

These model bytes exclude temporary evaluations, solver state and runtime libraries. The generic degree-sixteen six-input model alone exceeds the current machine's 16 GB RAM. Low-degree sparse angular rules or declared low-dimensional structure can change this count, but the dense smooth four-input demonstration does not establish arbitrary-dimensional efficiency.

The physical solution also exposes why a PDE-specific derivative audit is necessary when choosing bandwidth. Keeping the saved degree-48 circular-flow coefficients fixed, ten analytical re-encodings vary only $N\in\{257,513\}$ and $\lambda\in\{0.12,0.16,0.20,0.24,0.28\}$. No new fit or exact target values are used. At $\lambda=0.20$, ordinary momentum RMS is $4.22\times10^{-15}$ and $3.91\times10^{-15}$ for the two widths. At $\lambda=0.28$, it worsens to $9.89\times10^{-13}$ and $2.00\times10^{-12}$, although value encoding discrepancies remain around $3\times10^{-15}$. The corresponding second-derivative encoding discrepancies approximately double. These are diagnostic re-encodings of one solved field, not ten independent PDE solves.

There is a concrete lattice explanation for this trend. Let $h$ be center spacing, $\gamma=\lambda/h$, and $K_\gamma(x)=\gamma\operatorname{sech}^2(\gamma x)$, the derivative of a tanh feature. Use the Fourier convention $\widehat K(\xi)=\int K(x)e^{-i\xi x}\,dx$. Then

$$\widehat K_\gamma(\xi)=\frac{\pi\xi/\gamma}{\sinh(\pi\xi/(2\gamma))},\qquad \widehat K_\gamma(0)=2.$$

For one nonzero tone $f(x)=e^{i\omega x}$ below the lattice Nyquist frequency, choose readouts $v_j=h\,i\omega e^{i\omega jh}/\widehat K_\gamma(\omega)$. Poisson summation of the absolutely convergent derivative-kernel sum gives

$$u_h'(x)=i\omega\sum_{m\in\mathbb Z}
\frac{\widehat K_\gamma(\omega+2\pi m/h)}{\widehat K_\gamma(\omega)}
e^{i(\omega+2\pi m/h)x}.$$

The $m=0$ term is exactly $f'(x)$. Differentiating another $r-1$ times multiplies the $m$th ghost by $(i(\omega+2\pi m/h))^{r-1}$. Thus for $r\ge1$, fixed $\omega,\lambda$, and small $h$, the magnitude of each leading ghost in the $r$th derivative is approximately

$$|\omega|\left(\frac{2\pi}{h}\right)^{r-1}
\frac{2\pi^2}{\lambda}e^{-\pi^2/\lambda}.$$

For second derivatives this grows like $1/h$ at fixed bandwidth, while the corresponding value ghost decreases like $h$. This predicts the observed direction of change: twice the center density can double a diffusion residual even while values remain accurate. The calculation is an infinite uniform-lattice tone model, not an exact formula for the finite multidirectional nonlinear test; the measured near-doubling is supporting evidence. A derivative-aware bandwidth choice must control $h^{1-r}(\pi^2/\lambda)e^{-\pi^2/\lambda}$, alongside finite-width encoding and arithmetic error. Merely increasing neuron count at fixed $\lambda$ is not a monotone PDE-accuracy guarantee.

## Conclusions

The measured evidence establishes native ordinary-MLP roundoff-scale field accuracy for the nonlinear no-slip 2D flow, its lower-viscosity variant, one sparse-observation inverse version, and the interacting nonpolynomial steady 3D flow. Those cases use manufactured forcing and independent exact checks. Two boundary-driven circular flows additionally reach roundoff-scale sampled momentum RMS and agree with richer networks without any known interior solution or body forcing. A parameterized diffusion family transfers to the generic preset. The method substantially reduces solver storage, but it still uses a numerical native physics solve and its dense directional representation still grows rapidly with dimension. The cavity failures, finite sampled checks and box conversion limits prevent any claim of general machine-precision Navier–Stokes, turbulence, shocks, or arbitrary inverse-problem recovery.

## Open questions

- How far does the successful inexpensive correction carry into time-dependent 3D flow, higher Reynolds numbers and less regular boundaries?
- How should resolution and equation scaling be selected automatically for unknown solutions with narrow layers or limited smoothness?
- Which problems retain a bounded preconditioned iteration count as degree, dimension, Reynolds number and observation sparsity increase?
