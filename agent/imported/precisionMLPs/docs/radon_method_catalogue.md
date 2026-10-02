# Catalogue of Radon, QUILL and related PDE methods tried

**Current organization:** [Radon results library](radon_results_library/README.md) and [top methods by problem class](radon_results_library/rankings.md). This 60-row catalogue is the preserved historical record. Its numerical ratings predate the latest strict single-hidden-layer requirement and later stress/inverse results; use the new class-specific rankings for current recommendations. Costs and detailed original evidence remain below.

Evidence cut: October 1, 2026. **60 named rows**, including failed variants, architecture departures, and related precursors. This is a historical research catalogue, not 60 independent new inventions. Hyperparameter changes stay within their method; changes to the construction, coefficient solve, architecture, or state-evolution scheme receive separate rows. A means known-function construction/geometry; B native or hybrid flat-field dynamics; C product/non-MLP and weak-form branches; D inverse wrappers; E pure-MLP repairs and memory-efficient native solves.

The latest conversation is the organizing scope. Earlier checkpoint E/H and F18 methods are retained because they underpin the Radon claims; the learned-chart experiment is an earlier related branch. Classical solvers followed by neural encoding and exact-polynomial controls remain visible. Their limitations are not hidden by calling them Radon successes. All numbers below come from saved code/results and the three evidence audits; no new benchmark was run for this catalogue.

## Reading the ratings and floor claims

**Req / PINN / Merit** are subjective scores out of 10, not benchmark-derived statistics. Req measures agreement with the current brief: a genuine tanh/GELU MLP, native physics-based coefficient discovery, high precision, tractable memory, and routine forward/inverse setup. PINN measures the breadth of objectives demonstrated, not advertised applicability. Merit evaluates scientific/practical quality within the method's own scope. A narrow analytical construction can therefore have high Merit and low PINN breadth. No score certifies a universal PDE solver.

“Floor” here means approximately 1e-14 in the quantity explicitly reported. Function error, derivative/PDE residual, encoding error relative to a numerical state, inverse-parameter error, and conservation error are different quantities. Historical reports sometimes called 1e-12 the floating-point floor; this catalogue retains the actual numbers. A method has not reached a PDE floor merely because its neural compiler differs from an external solver by 1e-15. Failure labels distinguish rejected designs, failed individual configurations, and unresolved accuracy goals.

## Symbols and shared cost definitions

All costs are asymptotic upper-order models of the implemented path or explicitly stated representation, not measured wall-time predictions. Independent profiles/functions can have substantial oracle-access costs. Exact constants, hardware, special-function evaluation, derivative channels, precision, and iteration counts matter.

| Symbol | Meaning |
|---|---|
| d | Input dimension, including time when time is an input |
| M | Directional lines/spokes |
| N, R, H | Interior centers, halo per side, total per spoke H=N+2R; current R=ceil(sqrt(N)) |
| B | Actual flat tanh neurons, MH; older H experiments report B=MN with their collar already counted |
| p, P | Profile degree and independent coordinates; full total degree P=binom(d+p,p) |
| Q, E, b | Physics/fitting points or residual rows, inference points, bounded point batch |
| J, F | Required derivative channels and physical fields; fixed when omitted from a formula |
| I, G, K | Total Krylov/update products, nonlinear iterations, or trial/outer iterations as locally stated; none assumed resolution-independent |
| L, T | Time stages, or Taylor degree when explicitly stated; A10/A11 use L for scalar profile samples |
| b_c | Bounded coefficient preconditioner block size, distinct from point batch b |
| A, L_rho | Retained Fourier atoms and distinct radial frequencies |
| q, Q=q^d | Spectral grid size per axis and full tensor point count in spectral branches |
| H p, M p | Shared profile-bank entries and redundant directional coefficients |
| s, k | Number of inverse parameters, sparse directions, active coordinates, or retained Fourier terms, defined per row |

For an overdetermined global dense solve, define

- `T_LS(Q,B)=O(Q B min(Q,B))`; for Q≥B this is O(QB²).
- `S_LS(Q,B)=O(QB+min(Q,B)²)` plus inputs and model; with Q proportional to B the storage is quadratic and compute cubic.
- `T_flat=O(E(Md+B))` for shared-direction evaluation. An ordinary unfactored dense MLP instead uses O(EdB).
- `S_flat=O(Md+B+H+bH)` for shared-direction model plus bounded scratch. Ordinary dense export stores O(dB), with O(bB) activations. Source descriptions indicate when actual implementation uses the latter. Storing all Q inputs contributes O(Qd); retaining all E predictions contributes O(E).

Let `C_enc(s,N)` and `S_enc(s,N)` be corrected QUILL construction of s known profiles. They include the contour correction, not just sampling N values. With m≈R/2, q_h contour nodes, profile-call arithmetic `C_prof`, and profile workspace `S_prof`, a conservative current implementation model is

`C_enc(s,N)=O(q_h³ + C_prof + Hs + m q_h s + m³ + m²s)`

`S_enc(s,N)=O(q_h² + S_prof + Hs + m q_h + q_h s + ms)`.

The q_h³/q_h² terms allow the uncached dense quadrature-node setup; current partial fractions rebuild elementary symmetric polynomials for each pole, producing the m³ term. Known profiles must be analytic on the contour used. In simple vectorized polynomial banks, profile calls themselves cost O((H+q_h)s). The default q_h can grow with N. A shared p+1-profile bank is substantially different from separately encoding M full profiles. These are not hidden linear systems or least-squares fits, but neither are they automatically O(MN) arithmetic. The argument s_enc, if used in a row, is the profile count in that same row’s encoder call. Construction peak is the maximum of encoder peak and solve/evaluation-setup peak even when those phases run sequentially. For older long-halo formulas `C_profile` is the cost of one supplied analytical profile evaluation; deriving that profile from arbitrary f is an additional cost.

Other named costs used in rows:

- `T_sphere=O(d n_ang³+Md)` and `S_sphere=O(n_ang²+Md)` conservatively allow dense univariate quadrature setup and explicit direction enumeration in recursive tensor rules; implementation-specific tridiagonal eigensolvers can reduce that setup. Circle/QMC rules instead enumerate in O(Md). Tensor rules have M≈(p+1)^(d−1) or n_ang^(d−1). Known covariance preprocessing adds O(d³+Md²).
- `T_profile_encoder`, `S_profile_encoder`: explicitly selected A10 scalar LS or A14/A20 analytical encoder; the dependence is a genuine choice, not an omitted free solve. `T_profiles` is O(MH C_profile) for supplied closed-form samples, O(MH q_Abel C_profile) for the tested Abel quadrature, or the Fourier-atom sampling cost stated in A09/A11. A scalar quadrature node count is independent of neuron count unless a scaling rule is explicitly given.
- `T_prec=O(JQP b_c+P b_c²)` per cached block-preconditioner build, `O(Pb_c)` factors.
- `T_product(P)=C_enc(p+1,N)+O(JQ(dHp+dP)+I JQP)+T_prec`; `S_product(P)=S_enc(p+1,N)+O(JQP+Pb_c+Hp)`.
- `G_map(d,p,M)`, `S_map`: known polynomial-to-ridge moment-map construction. Implemented arithmetic includes O(MP²) dense contraction plus moment expansion/high-precision work; retained tensor storage is O(MpP), with moment workspaces up to O(P²). Keeping the additional algebra explicit avoids pretending this is an O(B) constructor.
- `T_E03`, `S_E03`: full E03 formulas, with any substituted active dimension, direction count or coordinate count stated in the referring row.
- `T_reduced_setup=C_enc(P,N)+O(HPQ)`; native reduced dynamics retain O(QP+HP) arrays. `T_B10_setup=C_enc(P,N)+O(JQHP)`.
- `T_adaptive`: shared encoding plus neural-table construction O(QHP_a), actual RK work O(L QP_a), and candidate discovery O(W_win QP_c) plus sorting. P_a active modes, P_c full candidate dictionary; W_win windows; replay increases actual L. `S_adaptive=O(QP_c+QP_a+HP_a+B)` plus inputs/bank construction. Discovery is not sparse merely because the final model is sparse.
- `T_dir=O(MHp+QM(H+d))` for one prepared directional forward/transpose product; once angular disk modes are used, add O(Mp²) coordinate conversion.
- `S_stream(S)=O(Qd+Q+S+B+Md+Hp+S b_c+bH)` for fixed fields/channels; it includes constraint/sensitivity/Krylov vectors, prepared neuron state, bank and bounded tiles. Arbitrary caller-supplied dense aggregation matrices add their own storage. There is no Q×S feature cache.
- `T_stream_prec(S)`: bounded selected-column setup; a conservative current bound is O(QBS+QS b_c+S b_c²), plus selected-column profile/angular transforms. Batching bounds storage, not this arithmetic. Diagonal E11 replaces it by a fixed number r_probe of operator probes.
- For the dense autoencoder, `C_encoder=O(W_encoder)`. With K_train full-dataset-equivalent gradient/line-search evaluations, `T_encoder_train=O(K_train Q W_encoder)`; with optimizer history r_opt and a training batch b_train, `S_encoder_train=O((1+r_opt)W_encoder+b_train A_encoder+Qd)`, where A_encoder is the number of saved hidden activations per input. These extra costs are not included in the ridge head's neuron count.
- `T_forward`: the explicitly identified forward model in the same inverse row, including readout construction for each changed parameter/time; inverse iterations multiply this cost. For D01 its direct construction/sensor cost is C_enc(M,N)+O(Q_obs(Md+B)), with times and parameter sensitivities as stated. For C07, if U candidate multiindices are processed, `T_support_selection` includes O(dU) downward-closure enumeration plus O(U log U) ranking and the additional trial-feature evaluations/solves; U can grow combinatorially. No fixed-cost sparse-discovery claim is made.

In a balanced dense construction with p~N, M~N^(d−1), P~N^d and Q~P, the old cached ball-map setup can cost O(N^(3d)) and retain O(N^(2d)) maps/caches. Streaming removes those maps/caches from its operator path but still requires repeated O(N^(2d))-scale full residual products under the same Q scaling. It does not prove a bounded iteration count or remove approximation's curse of dimensionality.

The safe error bookkeeping is `E_total <= E_angular + E_profile + E_solve/time + E_arithmetic`, with target-transform error added if profiles are themselves numerically estimated. Fixed lambda can leave aliasing bias; naive square-root halos can give root-exponential tails. The conditional model exp(-c B^(1/d)) requires appropriate spectral angular and scalar convergence and adequate boundary correction; it is not a universal measured law.

## Compact reference table

Construction includes solving/evolution where needed. Evaluation means the obtained model; query-time reconstruction is separately recorded. Full detail and sources follow the table. Ratings are Req/PINN/Merit.


| ID / method | Setup / representation | Construction time | Construction RAM | Evaluation time / RAM | Required solves or access | Floor evidence / rate | Status | Ratings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A01 **Uniform Radon grid + global least squares** | Known sampled functions; mostly d=1–4; Flat tanh MLP | O(QdB)+T_LS(Q,B) | S_LS(Q,B)+O(dB) | T_flat; S_flat | Global SVD/readout fit; known function samples | Some smooth 2D cases ~1e-14; many historical floors ~1e-12; sharp/non-smooth and 4D cases unresolved; H05 angular exp(-a M^q), q≈0.9–2.1; tested center curves ~N^-8.7 to N^-12.9 | Successful baseline; memory-limited | 4/2/6 |
| A02 **Smooth monitor-adapted ridge centers** | Known samples with localized features; Flat tanh MLP | O(K T_LS(Q,B)+QdB+MQ log Q) | S_LS(Q,B)+O(Qd) | T_flat; S_flat | Pilot fit then final global LS; oracle-gradient controls separate | Spikes ~2e-14 at 128 versus 512 uniform neurons; broad consistency not established; 4–8x width savings in selected sharp cases; no universal exponent | Mixed; initial rough-monitor version failed | 4/2/6 |
| A03 **Gradient-energy angular-density placement** | Finding ridge directions from gradient second moments; Flat tanh MLP | O(K T_LS(Q,B)+QdB+Qd^2+d^3) | S_LS(Q,B)+O(Qd+d^2) | T_flat; S_flat | Pilot or oracle gradients; global LS | No useful directional gain; no independent floor success; No beneficial directional convergence established | Failed directional heuristic | 2/1/2 |
| A04 **Active-subspace ridge placement** | Embedded low-dimensional known targets; Flat tanh MLP on subspace plus background | O(K T_LS(Q,B)+QdB+Qd^2+d^3) | S_LS(Q,B)+O(Qd+d^2) | T_flat; S_flat | Pilot global fits; eigenspace; oracle gradients only in control | Oracle ~1e-12; estimated d5 examples ~5e-6 or 5e-11; no consistent strict floor; Iteration improved one earlier run then stalled in rerun | Mixed specialization | 4/2/5 |
| A05 **Stagewise projection-pursuit ridge atoms** | Sparse sums of unknown ridges; Flat tanh MLP | O(s A_pool QH^2)+trial T_LS(Q,sH) | O(QH+Qd+sH) | T_flat; S_flat | Repeated scalar fits and target residuals | Alone leaves ~1e-2–1e-1 errors before joint polish; Each atom retained 20–80% of residual; stalls | Insufficient standalone method | 3/1/4 |
| A06 **Joint direction VarPro / Gauss–Newton** | Sparse ridges and directional mesh refinement; Flat tanh MLP | O(K[T_LS(Q,B)+QBT+QT^2+T^3]), T=M(d-1) | S_LS(Q,B)+O(QT+T^2) | T_flat; S_flat | Global readout re-solve every geometry trial | 24/24 hidden-ridge tests ~2e-13–1e-12; not consistent 1e-14; Near exact sparse solutions GN locally quadratic; measured diffuse gains 5–125x | Successful specialization | 5/2/7 |
| A07 **Greedy hierarchical ridge mesh** | Mixed sparse ridges and diffuse smooth content; Flat tanh MLP | sum_t T_LS(Q,B_t)+atom search+joint-polish costs | max_t S_LS(Q,B_t)+search workspace | T_flat; S_flat | Many global trial fits and validation selection | Selected sparse 3D cases 2e-14–8e-14 at 160–3420 neurons; diffuse 4D not at floor; Strong sparse-budget gains; no single general exponent | Successful specialization; dense fit remains | 5/2/7 |
| A08 **Learned latent chart + Radon LS head** | Noiseless 24D data on a 3D manifold; Tanh encoder plus fixed tanh ridge head | T_encoder_train+T_LS(Q,B) | S_encoder_train+S_LS(Q,B) | E C_encoder+T_flat(s=3); W_encoder+S_flat(s=3) | Input-only Adam/L-BFGS encoder training; global head LS | Two encoder seeds: 1.51e-4–1.48e-3; oracle chart 3.12e-7–1.70e-5; No floor or neuron-rate established | Useful hybrid; precision unresolved | 4/2/6 |
| A09 **Nearest-direction Fourier snapping** | Known analytic Fourier expansion; Flat tanh MLP after profile conversion | O(AMd+AH)+T_profile_encoder | O(A+Md+MH)+S_profile_encoder | T_flat; S_flat | Supplied Fourier atoms; chosen scalar encoder may use LS | Composition 6.98e-4 before tanh encoding; Worst phase bound ~M^(-1/(d-1)); not optimal angular convergence | Failed precision variant at tested M | 3/1/3 |
| A10 **Prescribed Radon profiles + shared scalar LS** | Known Radon/Abel profiles; Flat tanh MLP | O(LH^2+LHM)+T_profiles | O(LH+LM+HM+H^2) | T_flat; S_flat | One shared 1D SVD; no joint multivariate fit | Four radial targets at 4096 neurons: 2.45e-15–1.37e-14; Scalar floor reached in tested H64/128; angular error remains | Successful small-solve construction | 5/1/8 |
| A11 **Angular interpolation + scalar profile LS** | Known Fourier composition target; Flat tanh MLP | O(MA+MHL_rho+LH^2+LHM) | O(MA+LH+LM+MH) | T_flat; S_flat | Analytical signed angular interpolation; shared scalar SVD | One composition: 4.10e-15 at 2048; 3.41e-14 at 4096; Repairs snapped 6.98e-4; no universal fitted exponent | Successful angular repair | 5/1/8 |
| A12 **Leading derivative-sample readouts** | Known filtered Radon profile derivatives; Flat tanh MLP | O(MH C_profile+Md) | O(MH+Md) | T_flat; S_flat | Known q and q-prime; no solve | 18–36% errors for tested broad kernels; Smoothing bias remains as M increases | Failed floor construction at tested bandwidth | 2/1/3 |
| A13 **Complex-shift readouts with short halo** | Known analytic Radon profiles; Flat tanh MLP | O(MH C_profile+Md) | O(MH+Md) | T_flat; S_flat | Complex analytic profiles; no fit | Four radial errors ~7e-8–4e-6; Finite-band tail error dominates | Failed boundary variant | 3/1/4 |
| A14 **Complex-shift readouts with long halo** | Known analytic 2D/3D functions; Flat tanh MLP | O(MH C_profile+Md) | O(MH+Md) | T_flat; S_flat | Known analytic profiles; no fitting or PDE solve | 3/3 Gaussian-family 3D targets <1e-14 at 463104 neurons; Anisotropic: 4e-2 at 3216 to 6.45e-15 at 463104; angular-limited | Successful no-fit construction | 5/1/9 |
| A15 **General-d tensor sphere construction** | Known full-rank Gaussian functions; Flat tanh MLP | O(d^3+Md^2+MH C_profile)+T_sphere | O(MH+Md)+S_sphere | T_flat; S_flat | Analytic hypergeometric profiles and Gauss–Jacobi cubature; no fit | 4D 8.15e-15 at 6.59M neurons; 5D 6.45e-11 at 26.9M; Conditional spectral angular model exp(-a M^(1/(d-1))); not universal | Successful 4D; angular curse in 5D | 5/1/8 |
| A16 **Sobol sphere construction** | Known high-dimensional Gaussian functions; Flat tanh MLP | O(d^3+Md^2+MH C_profile) | O(MH+Md) | T_flat; S_flat | Known analytic profiles; Sobol-normal sphere directions | At 3.29M neurons: d4 0.063% to d32 2.34%; no floor; No reliable empirical QMC exponent established | Useful approximation; failed floor goal | 4/1/5 |
| A17 **Known-metric adapted sphere construction** | Known anisotropic Gaussian metric; Flat tanh MLP | O(d^3+Md^2+MH C_profile) | O(MH+Md) | T_flat; S_flat | Supplied precision matrix and analytic profiles; no fit | d32 0.44–0.61%; d64 0.46–0.57%; no floor; Two scrambles at one main budget; no general rate | Useful geometry control | 4/1/6 |
| A18 **Sparse known Fourier-ridge construction** | Known sparse directional spectrum; Flat tanh MLP | O(sH+sd) | O(sH+sd) | O(E(sd+sH)); O(sd+sH+bH) | Directions, amplitudes and frequencies supplied; no solve | 804 neurons, d256: 5.89e-16; lower-d controls also floor; No angular tax once exact support is supplied | Successful specialized construction | 5/1/8 |
| A19 **Naive square-root halo truncation** | Known analytic profiles; Flat tanh MLP | O(MH C_profile+Md) | O(MH+Md) | T_flat; S_flat | Complex-shift coefficients only; no solve | Sine ~1e-4; 3D Gaussian ~3.96e-5; Naive exp(-cR) tail is root-exponential when R=sqrt(N) | Failed boundary shortcut | 2/1/2 |
| A20 **Corrected finite-contour QUILL Radon** | Known analytic profiles; reusable profile-bank construction; Flat tanh MLP | C_enc(M,N)+O(Md) | S_enc(M,N)+O(Md) | T_flat; S_flat | Contour integrals and explicit partial fractions; no linear solve | 3D value 6.40e-16 / Laplacian 7.87e-15 at 244800 neurons; Boundary bound h exp(-lambda R^2/4); total includes angular/alias/roundoff terms | Successful no-fit construction | 6/2/9 |
| A21 **Analytic Gegenbauer ball-frame reconstruction** | Known polynomial ridge; d2–4 capacity audit; Flat ordinary tanh MLP | C_enc(1,N)+O(M(d+p)+dMH) | S_enc(1,N)+O(dMH+Md) | T_flat; S_flat | Known ridge identity and cubature; no fit or PDE solve | N257: all 3 degree4 fields <1e-14; none of 3 degree6 fields <1e-14; N129→257 improves encoding; angular underresolution gives 0.817 error | Successful capacity audit; incomplete floor consistency | 5/2/7 |
| B01 **Dense spacetime ridge Gauss–Newton** | Unsteady 2D incompressible Navier–Stokes; Flat tanh MLP; 3 shared field readouts | O(K[T_LS(Q,S)+QJdB]), S=F(B+1) | O(QJB+QS+S^2) | T_flat; S_flat | Native nonlinear residual solves; same-method cascade; oracle-informed split selection | 4096 neurons/field: velocity 1.1e-11, pressure 8.6e-10; not floor; Roughly 2.5 decades/width doubling except stalled rung | Successful native PDE baseline; memory wall | 5/4/6 |
| B02 **Analytic linear-IVP Radon propagation** | Advection, Airy, heat; prescribed initial profiles; Flat tanh network; heat readouts depend on time | O(M C_enc(1,N))+T_analytic_profiles | O(Md+B)+S_enc(1,N) | T_flat; S_flat | Supported analytical propagator; no coefficient solve | Smooth values near roundoff; Airy residual may remain 1e-12–1e-10; Angular and scalar errors separated; no universal rate | Successful specialized linear method | 5/3/7 |
| B03 **FFT-evolved Radon heat profiles** | 3D constant-coefficient heat IVP; Flat tanh snapshots | O(Mq log q)+M C_enc(1,N) | O(Mq+B+Md)+S_enc(1,N) | T_flat; S_flat | Independent 1D spectral PDE solution per direction | Heat ~2e-15; tested 1048576-neuron evaluation; Linear profile propagation only; angular IC error remains | Valid hybrid; rejected native-general-PINN claim | 3/3/5 |
| B04 **Fixed spacetime wave construction** | 3D-space linear wave IVP, zero initial velocity; Four-input ordinary tanh MLP | O(M C_enc(1,N)+Md+B) | O(Md+2B)+S_enc(1,N) | O(E(Md+2B)); O(Md+2B+bH) | D’Alembert profile propagation; no production solve | ~1e-14 solution; ordinary AD PDE RMS 4.59e-15; One supported linear family; no universal rate | Successful solve-free specialized MLP | 6/3/8 |
| B05 **Cole–Hopf Burgers construction** | Periodic viscous Burgers with supported IC; Flat tanh snapshots | O(qk+q log q+Nk) | O(q+Nk) | O(EN); O(N+bN) | Analytic nonlinear-to-heat transform plus spectral profile evaluation | 309 neurons: value 5.83e-15; PDE RMS 2.25e-13; h .16 failure 1.29e4; .08 ~1.5e-12; .04 ~5.8e-15 | Successful specialized nonlinear formula | 4/3/6 |
| B06 **External Fourier evolution then Radon compilation** | 3D NS and selected high-D reaction–diffusion; Ordinary tanh snapshots after classical PDE solve | O(L Q log Q+HP)+C_enc(P,N) | O(Q+B+Md+HP)+S_enc(P,N) | T_flat; S_flat | Complete external classical Fourier/RK PDE solution | NS conversion ~1e-15; PDE discretization ~3.66e-11 at top tested grid; Grid16^3→48^3: 9.78e-5→3.66e-11; not PDE floor | Rejected native-PINN claim; legitimate compiler | 2/3/4 |
| B07 **Native compact-spline coefficient dynamics** | Periodic viscous Burgers; Compact cubic splines; representable by ReLU-cubed | O(rN+LrN) | O(N+r) | O(E) local; O(EN) literal dense network; O(N) model; O(1) local scratch | Native RK4 and fixed inverse stencil; no external trajectory or LS | N256 error 1.97e-5; literal ReLU-cubed cancellation at N64 ~4.72e-11; Second-order spatial error ~N^-2 | Native but poor precision scaling | 3/3/5 |
| B08 **Native full tanh-readout dynamics** | Periodic viscous Burgers; Evolving full tanh readouts | C_enc(P,N)+O(HPQ+LQH) | max(S_enc(P,N), O(QH+HP+QP)) | O(EH); O(H+bH) | Native RK4; explicit trigonometric analysis; no external trajectory | Best refined run ~2.52e-14; no broad floor demonstration; 153→1091 neurons: 4.77e-3→2.50e-13; then timestep floor | Successful native mechanism; larger storage | 5/4/6 |
| B09 **Native reduced tanh-coordinate dynamics** | Periodic viscous Burgers; Reduced physical coordinates; export flat tanh | C_enc(P,N)+O(HPQ+LQP) | max(S_enc(P,N), O(QP+HP+QH)) | O(HP) export + O(EH); O(HP+H+bH) | Native ODE/DOP853 in reduced API; RK4 in inverse wrappers; fixed analysis; no external trajectory | Independent reduced linear control 1.49e-15; Burgers 1.14e-12 / PDE 3.42e-12; not consistent floor; Separate reduced API endpoints only; B08 ladder is equivalent-reference evidence, not a reduced sweep | Successful native mechanism | 6/4/7 |
| B10 **Native flat-ridge Navier–Stokes evolution** | Periodic interacting 3D NS; Actual flat tanh derivative fields; reduced state | C_enc(P,N)+O(JQHP+L[JQP+Q log Q]) | max(S_enc(P,N), O(JQP+HP+B+QH)) | T_flat; S_flat | Native RK4, FFT analysis and Leray pressure projection | K5: 1.94e-9 PDE solution error; same-cutoff match ~1e-15 is not truth error; K2/3/5: 2.18e-4/3.98e-6/1.94e-9 | Successful native PDE mechanism; cache bottleneck | 5/4/6 |
| B11 **Native NS Taylor-in-time recurrence** | Same periodic NS projected ODE; Flat tanh snapshots from time-polynomial state | T_B10_setup+O(T JQP+TQ log Q+T^2Q) | max(S_enc(P,N), O(JQP+TP+TQ+HP+B)) | O(TP+HP)+T_flat; O(TP+HP)+S_flat | Numerical Taylor recurrence; no production timesteps/LS | Time error ~2.60e-15; total spatial error ~1.97e-9; Taylor order improves temporal error only | Successful time integrator; not PDE floor | 5/3/6 |
| B12 **Flat-ridge shifted elliptic residual iteration** | Nonlinear cubic elliptic Dirichlet PDE; Flat tanh ridge field | C_enc(O(K),N)+O(QPH+GQP), P=K^2 | max(S_enc(O(K),N), O(QP+HP+QH)) | T_flat; S_flat | Native fixed-point iteration; explicit reference-Laplacian inverse | K12: 6.37e-8; off-grid PDE 3.03e-6; K4/8/12: 4.08e-3/1.60e-5/6.37e-8 | Successful special operator iteration; not floor | 6/4/7 |
| B13 **Adaptive ridge dynamics without replay** | Nonlinear periodic 2D reaction–diffusion; Adaptive flat tanh snapshots | T_adaptive | S_adaptive | T_flat; S_flat | Native RK stages; dense candidate analysis; no external trajectory | 7.82e-5; adding modes too late loses earlier dynamics; No reliable floor/rate; worse than replay | Failed adaptive variant | 3/3/3 |
| B14 **Adaptive ridge dynamics with rejected-window replay** | Same nonlinear reaction–diffusion; Adaptive flat tanh snapshots | T_adaptive with actual replay stages | S_adaptive | T_flat; S_flat | Native RK; reject window, restore state, expand directions and replay | 2.18e-7 at 19098 neurons; fixed 80056-neuron control 1.53e-7; Fewer final neurons; actual 260 vs 120 nominal RK steps | Successful adaptive specialization; not floor | 6/4/7 |
| C01 **Shared-product elliptic box solver** | Cubic elliptic Dirichlet PDE; QUILL 1D banks + exact product gates | C_enc(K,N)+O(Q[dHK+P]+GQP), P=K^2 | max(S_enc(K,N), O(QP+HK)+O(QH)) | O(E[dHK+P]); O(HK+P+bP) | Native scalar-preconditioned fixed-point solve | K20,P400: 1.17e-12; not strict floor; K12→20: 6.38e-8→1.17e-12 | Useful solver; not pure MLP | 4/4/6 |
| C02 **Mapped annulus Galerkin / Newton–CG** | Variable diffusion, nonlinear reaction, mixed annulus BC; Mapped QUILL products and coordinate features | C_enc(p,N)+O(QP^2+P^3+I[QP+P^2])+O(QHp+QP) | max(S_enc(p,N), O(QP+P^2+Hp)+O(QH)) | O(E[Hp+P]); O(Hp+P+bP) | Native Newton–CG; dense diffusion/Robin Cholesky preconditioner | P190: 2.19e-12; contrast10000: 3.91e-12; P28/66/120/190: 1.11e-2/1.85e-5/9.25e-9/2.19e-12 | Useful domain-specific solver; not pure MLP | 4/5/7 |
| C03 **Stiff QUILL reaction–diffusion with BDF** | Allen–Cahn and quadratic blowup reaction; Reduced trigonometric/QUILL field | T_reduced_setup+O(LQP+F_fac[QP^2+P^3]) | max(S_enc(P,N), O(QP+P^2+HP)) | O(HP) export + O(EH); O(HP+H+bH) | Native implicit BDF with dense state Newton solves | K128 Allen–Cahn 2.82e-5; K256 reference uncertainty dominates; Refinement improves; no validated floor law | Useful stiff specialization; not floor | 4/4/6 |
| C04 **Finite-volume Burgers with ReLU primitive** | Inviscid shocks and rarefactions; Cell field; derivative of ReLU primitive | O(N+LN) | O(N) | O(E) uniform-cell lookup; O(EN) literal primitive; O(N) model; O(1) lookup scratch | Classical Rusanov flux + SSPRK3 is the solver | Conservation ~1e-15; solution not floor; Approximately first-order L1; max jump error remains large | Valid shock baseline; not smooth Radon PINN | 2/3/5 |
| C05 **Shared-product native NS and animation** | Periodic 3D NS over longer times; Shared QUILL trigonometric banks + product tensor | C_enc(K,N)+O(dqHK+L[dKQ+Q log Q]) | max(S_enc(K,N), O(P+Q+dqK+HK), P=(2K+1)^d+O(qH)) | O(E[dHK+dP]); O(HK+P+bP) | Native Galerkin/RK, FFT and Leray projection | T1 differences ~2.4e-6–7.2e-6; animation T4 ~3.7e-6; Time halving ~1e-12; spatial truncation dominates | Useful native solver/visualization; not pure MLP | 4/4/6 |
| C06 **Generic cached QUILL-product residual engine** | General declared smooth forward/inverse residuals; Total-degree QUILL products; exact multiplication | T_product(P) | S_product(P) | O(E[dHp+dP]); O(Hp+P+EP) actual materialized query features | Native GN/LSMR; no external trajectory; global coordinate solve | Selected 4D NS control ~1e-15; 13 default tasks ~1e-8–1e-13; Conditional spectral degree refinement; defaults/conditioning matter | Broad useful prototype; not pure MLP | 5/7/7 |
| C07 **Sparse-support precision product solver** | High-precision unforced 4D NS/control PDEs; Selected Legendre/QUILL products | T_product(S)+T_support_selection | S_product(S)+support workspace | O(E[dHp+dS]); O(Hp+S+ES) actual query cache | Native iterative solve; support from previous PDE iterates | Actual QUILL best velocity 5.74e-16, pressure 1.50e-14; endpoint pressure 1.22e-13; p10→14 helps; no uniform width-only law | Useful precision refinement; not consistently floor | 5/6/7 |
| C08 **Exact-polynomial residual control** | Same declared PDEs as QUILL-product engine; Classical Legendre products; no neural model | O(JQ[dp+dP]+I JQP+QPb_c+Pb_c^2) | O(JQP+Pb_c) | O(E[dp+dP]); O(P+EP) | Native GN/LSMR; no QUILL approximation | Matched controls often equal QUILL accuracy; some ~1e-15; Conditional spectral in degree; solve/conditioning remain | Successful control; not Radon/MLP | 2/6/6 |
| C09 **Restricted weak/entropy front-family solve** | Burgers shock speeds and rarefactions; 2-parameter tanh front or 3-parameter ReLU ramp | O(G[Qs^2+s^3]), s=2 or 3 | O(Qs+s^2) | O(E); O(1) model and point scratch | Tiny nonlinear LS; supplied front family and entropy constraints | Shock speed ~2.7e-12; rarefaction L1 2.5e-6; Conservation-only expansion shock residual 9.6e-17 but error 0.2 | Useful small weak-form test; generic discovery absent | 4/3/6 |
| C10 **Scaled Gaussian concentrating-core probe** | Known shrinking divergence-free synthetic swirl; Shared QUILL profiles + exact product gates | C_enc(2,N)+O(dH); O(d) rescale per time | S_enc(2,N)+O(dH) | O(EdH); O(dH+bH) | Known field and prescribed scaling; no PDE solve or paper forcing | 1275 tanh units ~1.4e-15 through 100x core shrink; Accuracy stays near floor under supplied coordinate rescaling | Successful representation probe only | 2/1/5 |
| D01 **Analytical heat-tensor inverse wrapper** | Six diffusion-tensor entries, 36 observations; Analytical heat Radon forward model | O(I_inv[(s+1)T_forward+Q_obs s^2+s^3]), s=6 | O(Q_obs B+Q_obs s+s^2)+model | T_flat; S_flat | Small physical inverse LS; supported analytic heat forward | Noiseless tensor 2.8e-9; 1% noise gives 1.2–5.3%; No parameter-floor or neuron-rate established | Useful specialized inverse wrapper | 4/4/6 |
| D02 **Native Burgers parameter inverse with tangents** | Viscosity + two IC amplitudes; 24 observations; Native reduced QUILL forward and tangent ODEs | T_reduced_setup+O(I_inv[LQP(s+1)+Q_obs s^2+s^3]), s=3 | max(S_enc(P,N), O(QP+P(s+1)+HP+Q_obs s)+O(QH)) | O(HP) snapshot export + O(EH); O(HP+H+bH) | Repeated native forward integration and small LS | Refined viscosity rel 1.14e-11; withheld field 2.81e-13; Forward resolution controls bias; noisy parameter errors persist | Successful specialized inverse/UQ | 5/5/7 |
| D03 **Unknown initial-profile inverse with regularization** | 17 unknown IC Fourier coefficients; Burgers observations; Native QUILL forward plus finite-dimensional IC prior | T_reduced_setup+O(I_inv[LQP(s+1)+Q_obs s^2+s^3]), s=17 | max(S_enc(P,N), O(QP+P(s+1)+HP+Q_obs s+s^2)+O(QH)) | O(HP) export + O(EH); O(HP+H+bH) | Repeated forward/tangent integration and regularized LS | Noiseless early profile 1.09e-12; late 9.73e-7; noisy 2–11%; Information loss and noise dominate; no floor claim | Successful restricted inverse experiment | 5/5/7 |
| D04 **Regularized backward-heat compilation** | Noisy terminal field; known heat operator; Recovered initial field encoded as tanh | O(q log q+Nk) | O(q+Nk) | O(EN); O(N+bN) | Classical spectral inverse with discrepancy cutoff | Inverse error 0.001624; conversion 4.20e-16; Unregularized error 2.73e173; cutoff stabilizes noise | Valid inverse baseline; not native neural discovery | 2/3/5 |
| E01 **Deep tanh polarization compiler + native residual solve** | Nonlinear elliptic PDE; architecture repair route 1; Ordinary deep tanh MLP | C_enc(p+1,N)+O(W+JQW+I JQP)+T_prec | max(S_enc(p+1,N), O(W+JQP+Pb_c+bH[d+Pa])) | O(EW); O(W+bH[d+Pa]) | Native PDE coordinate GN/LSMR; no external product gates | One cold PDE: field 6.85e-16; AD PDE RMS 3.56e-15; One main PDE; no broad neuron-error exponent | Successful pure-MLP repair; dense layers expensive | 7/4/7 |
| E02 **Cached analytical disk ridge coordinates** | Smooth disk PDEs and exact-data inverse constraints; Flat ordinary tanh MLP | C_enc(p+1,N)+O(Mp^3+JQM[Hp+P+d]+I JQP)+T_prec | max(S_enc(p+1,N), O(MP+Hp+JQP+Pb_c+dB)) | T_flat; S_flat | Explicit geometry map; native GN/LSMR; no target-fit encoder | p16/p18 disk errors 8.76e-16/4.05e-16; mixed constraints ~2e-15; p4/8/12/16: 9e-3/2.8e-6/8.7e-11/8.8e-16 | Successful cached disk method | 7/5/8 |
| E03 **Cached enclosing-ball / box ridge coordinates** | General d-input box PDEs including 4D NS and Burgers; Flat ordinary tanh MLP | C_enc(p+1,N)+G_map(d,p,M)+O(JQM[Hp+pP+d]+I JQP)+T_prec | max(S_enc(p+1,N), O(MpP+Hp+JQP+Pb_c+dB)+S_map) | T_flat; S_flat | Algebraic moment map; native PDE solve; high-precision map only when needed | 4D NS ordinary velocity 7.68e-14; Burgers best 1.68e-11; not consistent floor; Burgers p24→36 about 16–21x improvement per 1164 neurons | Useful pure-MLP route; constructor/cache bottlenecks | 6/6/7 |
| E04 **Low-degree sparse angular cubature** | Controlled quadratic/cubic high-dimensional PDEs; Flat ordinary tanh MLP | T_E03 with M=d^2 (p2) or d+2C(d,2)+4C(d,3) (p3) | S_E03 with sparse M | T_flat; S_flat | Known sparse cubature and native PDE solve | 20D quadratic 6.92e-15; 10D cubic 8.53e-15; raw PDE ~1e-15; Polynomial d-scaling only at fixed degree 2/3 | Successful low-degree specialization | 7/4/8 |
| E05 **Declared-active-axis ridge construction** | High ambient d with k supplied active coordinates; Flat tanh MLP with zero inactive columns | T_E03(k)+O(Qd+dB) | S_E03(k)+O(Qd+dB) | T_flat; S_flat | Known active axes; native readout solve | 10D ambient/2 active: best 2.79e-12; PDE 3.81e-11; N129→257 improves; no floor or learned compression | Useful disclosed prior; not manifold discovery | 5/3/6 |
| E06 **Sequential time-slab ridge networks** | Smooth unforced Burgers over multiple time windows; Collection of flat ordinary tanh MLPs | sum_l T_E03(l) | max_l S_E03(l)+O(LdB) | One slab T_flat+O(log L) selection; O(LdB)+one-slab workspace | Previous native neural terminal trace supplies next IC; no external marcher | Four slabs: spacetime 2.29e-9; final 8.88e-9; Improves global p24; no matched floor/cost advantage | Useful continuation; incomplete solves | 6/4/6 |
| E07 **Direct directional-profile operator** | Generic d>=2 neural forward/adjoint building block; Flat tanh MLP; redundant b[M,p] coordinates | C_enc(p+1,N)+O(Md+Mp); matvec T_dir | S_enc(p+1,N)+O(Md+Mp+B+Qd+bH) | T_flat; S_flat | No coefficient/PDE solve in this standalone benchmark | Operator identity/export verified; no standalone PDE floor result; M8/16/32/64 time .056/.111/.225/.452s; transient ~1.85MiB | Implemented operator; not complete solver by itself | 6/2/7 |
| E08 **Matched ball-frame residual iteration** | Special nonlinear disk PDE with boundary-degenerate diffusion; Actual flat tanh MLP | C_enc(p+1,N)+O(I[T_dir+QMp]) | S_enc(p+1,N)+O(B+Mp+Md+Qd+Q+bH) | T_flat; S_flat | Explicit frame analysis and scalar eigenvalue division; no LS/Krylov | Tightened p16: field 1.26e-16, PDE RMS 1.49e-15; p4/8/12/16: 6.4e-4/7.2e-8/1.5e-12/1.8e-14 | Successful matched no-LS native solve | 7/3/8 |
| E09 **Streamed general solve with redundant profiles** | Ordinary nonlinear Dirichlet diffusion–reaction; inverse controls; Flat ordinary tanh MLP | C_enc(p+1,N)+O(I T_dir)+T_stream_prec(S), S=1+Mp | S_enc(p+1,N)+S_stream(S) | T_flat; S_flat | Native GN/LSMR; bounded-block preconditioning; no exterior solve | p16 longer run field 1.47e-15 but PDE 6.86e-14; tiny inverse at floor; p12 ~4.8e-11; initial p16 1.3e-8 was under-solved, not capacity floor | Successful memory trade; conditioning still costly | 7/5/7 |
| E10 **Streamed analytical disk modes with block preconditioning** | Ordinary nonlinear disk forward and joint inverse problem; Flat ordinary tanh MLP; independent angular modes | C_enc(p+1,N)+O(I[T_dir+Mp^2])+T_stream_prec(P) | S_enc(p+1,N)+S_stream(P)+O(Mp) | T_flat; S_flat | Native GN/LSMR; analytic angular conversion, small block solves | Forward field 4.31e-16/PDE 4.15e-15; inverse field 3.83e-16/PDE 4.14e-15; Same 4947 neurons, 273→153 coordinates; 247s→59s forward; no universal I bound | Strongest current smooth disk route | 8/6/8 |
| E11 **Streamed analytical disk modes with diagonal scaling** | Same nonlinear disk forward control; Same flat tanh MLP; independent angular modes | C_enc(p+1,N)+O((I+r_probe)[T_dir+Mp^2]) | S_enc(p+1,N)+S_stream(P; b_c=1)+O(Mp) | T_flat; S_flat | Native GN/LSMR; fixed-count diagonal probes; no block factorizations | Field 3.68e-16/PDE 4.18e-15; returned budget_exhausted after final accepted step; 185.6s/1701 Krylov vs block 59.2s/532; no universal rate | Floor attained in saved model; soft-budget status preserved | 8/6/8 |

## Method descriptions and evidence

### A01. Uniform Radon grid + global least squares

Tensor and interlaced layouts are variants of this same global solve.

**Status:** Successful baseline; memory-limited. **Ratings (Req/PINN/Merit):** 4/2/6.

**Setup/status.** Implemented and heavily tested fixed-geometry approximation from known function samples. Ordinary one-hidden-layer tanh MLP. Tensor angle×offset and alternate-direction half-offset/interlaced variants were tested in E01; H01/H05/H06 use recentered directions, uniform per-spoke grids, gamma h=.25, a 25% collar, and one truncated SVD. Tensor versus interlaced changes placement, not solve asymptotics. Random ridges and radial-tangent point meshes are controls, not explicit Radon inversion.

**Cost.** Geometry O(dB), feature assembly O(QdB) in flattened code, plus G(Q,B), memory O(QB+B²+dB). Shared-spoke assembly can reduce dB to Md+B but cannot remove dense solve scaling. Evaluation follows common costs. No external PDE solution; does require samples and global fitting. Rank truncation changes the selected coefficient representation.

**Accuracy.** E01 Gaussian reaches about 1e-14 by 576–1024 units; final 8192 tensor result max-error about 5.9e-14; Runge remains resolution-limited about 8e-9 under even its best geometry. H05 local 2D known-function split results: at B=256,512,1024,2048,4096, radial Runge best errors 8.2e-9,8.5e-11,8.0e-12,7.4e-15,7.5e-15; composition 2.3e-8,6.0e-10,4.2e-11,1.5e-13,1.3e-14; spatial packet 7.4e-7,1.5e-10,4.3e-11,3.0e-13,1.9e-14. Thus not monotone at arithmetic floor. H01 4096-feature smooth 1D/2D suite generally 3e-14–2e-12, but narrow spikes/product peaks unresolved and nonsmooth targets not at floor. H06 d3 end-state Gaussian1.3e-14, composition2.7e-14, product sine5.9e-14 at 12288 features; fast waves3e-13, packet3.3e-12 even after geometry polish.

**Rate.** Empirical two-axis bottleneck e≈max(e_M,e_N); H05 direct fits give angular exp(-a M^q), q≈.9–2.1, while tested center leg roughly N^-8.7 to N^-12.9 with shoulders. Do not replace these measured finite-grid curves by a universal exponential theorem. Allocation/halo/width change the regimes.

**Evidence.** `results/checkpoint_E_2d/expE01_geometry_zoo_2d/expE01_results.md`; `results/checkpoint_H_highdim/expH01_highdim_suite/expH01_results.md`; `results/checkpoint_H_highdim/expH05_direction_cliff_2d/expH05_results.md`; `results/checkpoint_H_highdim/expH06_ridge_hierarchy/expH06_results.md`; `docs/ridge_quadrature_theory.md`.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A02. Smooth monitor-adapted ridge centers

Data-density, slope, curvature, residual and frequency monitors are variants; smoothing removed much of the floor penalty.

**Status:** Mixed; initial rough-monitor version failed. **Ratings (Req/PINN/Merit):** 4/2/6.

**Setup/status.** H04 precursor, implemented/tested. First even fitted surrogate estimates per-direction projected gradient/curvature energy or residuals. Construct a smooth center-density map, set local gamma h=.25, then fit readouts again. Variants: data density, true-gradient oracle, surrogate gradient/curvature, residual, frequency monitor. These are placement variants, not independent construction equations.

**Cost.** Two or several G(Q,B) solves plus O(QdB+QMd) monitoring/binning/sorting (sorting projected samples may be O(MQ log Q)). Peak still dense G storage, not doubled if sequential. Inference common. Surrogate versions require target samples but no true gradients; oracle variants explicitly use them.

**Result.** Smooth monitor fixed the initial rough-mesh floor penalty. Sharp 1D spikes reached2e-14 with128 features versus uniform needing512; 2D hotspot packet at4096 improves1e-11→4e-13. Initial 1.5-gap monitor smoothing and 2%-gap jitter were failures at high precision (up to1e-11 versus1e-15). Revised ~6-gap smoothing improves. No global jump/shock resolution: step dense-region accuracy can be tiny where the jump is not, while whole-domain error remains around1e-2.

**Evidence.** H04 `expH04_results.md`, `mesh.py`, `floor_price.py`, `mesh_map_scale.py`; initial run results retained under `bw1.5/`.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A03. Gradient-energy angular-density placement

A broad quadratic angular energy cannot locate sparse delta-like ridge support.

**Status:** Failed directional heuristic. **Ratings (Req/PINN/Merit):** 2/1/2.

**Setup/status.** H04 tested density m(theta)∝[v(theta)^T C v(theta)]^(1/3), C=E[gradient f gradient f^T], alongside estimated versions/joint center+angle allocation. Same global LS, same complexity as A02.

**Result.** No useful directional improvement on isotropic suite or known ridge. The quadratic angular energy is broad and cannot locate a delta-like ridge direction. Mark the directional inference mechanism failed, not the final model unexecutable. Evidence H04 results and `known_answer_2d.py`.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A04. Active-subspace ridge placement

Supplied/estimated linear subspace; not a general nonlinear manifold learner.

**Status:** Mixed specialization. **Ratings (Req/PINN/Merit):** 4/2/5.

**Setup/status.** H04 tested covariance eigenspace, allocate80% of features to s-dimensional active subspace and20% ambient background. Oracle and pilot-surrogate variants; repeated fit→covariance→fit tried. Not a general nonlinear manifold method.

**Cost.** K dense fits, surrogate gradient O(QdB), covariance O(Qd²), eigensolve O(d³), plus head on s dimensions; memory dense head plus O(Qd+d²). Eval active+background common; no external PDE solution. Oracle uses actual gradients.

**Result.** Embedded 2D composition at ambient d5,B4096: ambient8e-4→5e-6 estimated subspace,1e-12 oracle; noisy sheet1e-6→5e-11. Iteration once reached1e-9 but stalled in revised run. Pure ridge+background failed: covariance direction biased .01 degrees, inadequate for high precision. No consistent floor.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A05. Stagewise projection-pursuit ridge atoms

Useful proposal mechanism inside A06/A07.

**Status:** Insufficient standalone method. **Ratings (Req/PINN/Merit):** 3/1/4.

**Setup/status.** H06 candidate pool600 directions d3 or1500 d4; fit a32-offset scalar block to residual for each candidate, direction-polish best candidates, append best atom. A precursor stage to A06/A07.

**Cost.** For A_pool candidates and s installed atoms, scalar candidate scans O(s A_pool QH²) in overdetermined dense LS; memory O(QH+Qd) for sequential scan plus installed model. Candidate/trial joint refits add G(Q,sH). No no-solve claim. Evaluation common with M=s.

**Result.** Alone retains20–80% residual per atom and typically stalls at1e-2–1e-1 before joint polish. Treat as useful proposal/growth mechanism but failed standalone floor method.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A06. Joint direction VarPro / Gauss–Newton

Some stuck directions required extra atoms; iteration counts not bounded independently of resolution.

**Status:** Successful specialization. **Ratings (Req/PINN/Merit):** 5/2/7.

**Setup/status.** H06 readout-eliminated joint direction optimization following A05, or applied to all directions of an even dictionary. Re-solve readout at every trial; Kaufman projected Jacobian for s(d−1) direction unknowns. This learns geometry, not merely fixed construction.

**Cost.** K[G(Q,B)+O(Q B T)+O(QT²+T³)] conservative direct algebra, T=M(d−1), plus line-search trial refits. The projection term uses retained readout SVD factors; no uniformly bounded K. Memory O(QB+B²+QT+T²). Evaluation common. Fit depends on target samples; no external PDE solver.

**Result.** All24 hidden-ridge tests (1,2,4,8 ridges,d3/d4,3seeds)2e-13–1e-12; some seeds require extra atoms after one direction sticks. Excellent high accuracy but not strict1e-14 consistent. All-direction polish helps direction-bound cells5–125× and does little in center-bound cells. d3 Radial Runge9.6e-14 after polish, packet3.3e-12.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A07. Greedy hierarchical ridge mesh

Nested background, discovered atoms, and opening/refining directions chosen by held-out improvement.

**Status:** Successful specialization; dense fit remains. **Ratings (Req/PINN/Merit):** 5/2/7.

**Setup/status.** H06 repeatedly trial-fits nested even-direction additions, discovered atoms, background refinement, atom refinement; selects validation-log-error reduction per added neuron. Atoms polished jointly. One underlying global LS remains.

**Cost.** Sum of A05/A06/global LS costs over trial dictionaries: O(sum_t G(Q,B_t)+atom-search/polish), dense peak at max B. No justified simple near-linear construction bound. Evaluation common.

**Result.** d3 ridge2:2e-14 at160 units; ridge4:4e-14 at992; product sine2e-14 at1088; ridge4+bump8e-14 at3420. At4096 d4 ridge4+bump9e-12, Gaussian7e-11, composition1.6e-9, Runge1.5e-7. Strong sparse-structure savings; diffuse high-dimensional examples remain unresolved. Single-seed hierarchy evidence.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A08. Learned latent chart + Radon LS head

PCA 3D control folds and fails; the final head is fitted, not analytical Radon construction.

**Status:** Useful hybrid; precision unresolved. **Ratings (Req/PINN/Merit):** 4/2/6.

**Setup/status.** I04 24D noiselesscurvedgraph manifold withtrueintrinsic3Dchart. Input-onlytanh autoencoder24→64→64→3,6000Adamsteps+LBFGS; frozen2048-featureFibonaccisphereRadonhead128directions×16centers, globalLS; per-targetbandwidth/cutoffselectedbyvalidation. Controlsoracletrue3D,PCA3D,ambient24D. ThisheadisLS,notanalyticalRadonreadouts.

**Cost.** Encodertraining K_A Q C_encoder +LBFGS(architecture-dependentmemory), plusG(Q,B), thenencoder+common3Dheadinference. NooutsidePDEsolver, but learnedcharttraining andglobalfit bothrequired. Modelincludesencoder; equalheadwidthdoesnotequalend-to-endcost.

**Result.** Learned2seeds/3targets1.51e-4–1.48e-3; oraclechart3.12e-7–1.70e-5; ambient24D2.40e-4–4.08e-3; PCA.107–.415andactualfoldingcollision. No floor. Usefulcontrolledchart+approximationtest, notcompletegeometricprecisionmethod.

**Evidence.** `results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/manifold/report.md`.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A09. Nearest-direction Fourier snapping

Preserving radial frequency while snapping angle introduces a real approximation error.

**Status:** Failed precision variant at tested M. **Ratings (Req/PINN/Merit):** 3/1/3.

**Setup/status.** Whole-space Fourier polar decomposition; send every known Fourier atom to nearest available line, preserving radial frequency; group profiles on M spokes. Implemented for2D analytic composition. Tanh conversion via A10 or K11. Fourier access is given analytically here, not discovered from samples.

**Cost.** Assign A atoms O(AMd), stored atom lists O(A+Md), profile samples O(AH) if sparse assignment used, plus chosen scalar encoder. No multivariate fit or external PDE solve, but obtaining spectrum is an unpriced input requirement. Network evaluation common.

**Result.** Composition atM32 has6.98e-4 error already in continuous ridge sum; scalar fit reproduces that sum to3.41e-14 but cannot remove angular error. Bound O(r * integral |xi|theta(xi,V)d|mu|), worst covering rate M^(-1/(d−1)), not an optimal exact representation theorem. Failed as floor construction at thisM; corrected by A11.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A10. Prescribed Radon profiles + shared scalar LS

Uses one factorization for all spokes; different per-spoke geometry would lose that reuse.

**Status:** Successful small-solve construction. **Ratings (Req/PINN/Merit):** 5/1/8.

**Setup/status.** H05 known target→analytic/Abel ridge profiles. Convert profiles to tanh independently; common geometry permits one shared scalar SVD and many RHS. Never fits multidimensional target values. Four radial targets and composition tested. Abel quadrature order96 supplies profiles where closed forms unavailable.

**Cost.** Shared scalar factorization O(LH²), apply to M profiles O(LHM); peak O(LH+LM+HM+H²) in actual all-RHS batch. L≈8H→O(H³+MH²) and O(H²+MH). With different geometry per direction would become O(MLH²), but tested implementation explicitly reuses one factorization. Add profile-generation cost: closed form O(MH); Abel O(MHq_Abel); Fourier atoms as A09/A11. Evaluation common.

**Result.** B4096,M32: Gaussian4.00e-15,Runge2.45e-15,waves1.37e-14,packet1.16e-14. Snapped composition6.98e-4. Shows removal of global solve wall while still using small scalar LS. No uniform asymptotic error law measured; scalar error floor reached at tested H64/128 for profiles.

**Evidence.** H05 `spoke_profiles/radon_prediction/forward_check/README.md`, `theory_forward_check.py` (one SVD for160 RHS).

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A11. Angular interpolation + scalar profile LS

More neurons slightly worsened roundoff; it does not indicate lost approximation capacity.

**Status:** Successful angular repair. **Ratings (Req/PINN/Merit):** 5/1/8.

**Setup/status.** H05 repair of snapping: trigonometric cardinal interpolation of the plane-wave dependence on angle; signed analytical weights on64 oriented nodes /32 projective lines. Group Fourier atoms into real profile frequency sums; then A10 scalar conversion. No joint2D fit.

**Cost.** Angular weights O(MA), storage O(MA) in current dense table; profile generation O(MHL_rho) where L_rho distinct radial frequencies≤A; plus A10 shared SVD O(H³+MH²). These A/L_rho factors prevent a blanket O(B) claim. Evaluation common.

**Result.** Composition continuous3.37e-16; tanh2048features4.10e-15 and4096features3.41e-14. Floor-level on this one target; larger dictionary slightly worse due finite precision. Corrects snapped6.98e-4 without changing directions.

**Evidence.** `composition_angular_interpolation.py`; same forward README; `angular_interpolation.json`.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A12. Leading derivative-sample readouts

The familiar derivative readout is a leading approximation, not exact deconvolution.

**Status:** Failed floor construction at tested bandwidth. **Ratings (Req/PINN/Merit):** 2/1/3.

**Setup/status.** a_mj=(h w_m/2)q'_m(c_j). No smoothing inverse; anchor bias analytically. In3D q=−(Rf)''/(2pi), so readout proportional(Rf)'''. No fit.

**Cost.** O(MH C_profile), O(MH+Md) model storage, eval common. Derivative availability/profile-construction cost separate.

**Result.** Same geometry as corrected I04 Gaussian tests gives relative errors.293,.356,.326 (isotropic,anisotropic,shifted mixture); F19 anisotropic2D18.2%,3D22.7%. Valid leading small-width approximation, not floor method at tested gamma. Error saturates as only M rises because smoothing bias unchanged.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A13. Complex-shift readouts with short halo

Continuum inverse smoothing does not remove finite-band truncation.

**Status:** Failed boundary variant. **Ratings (Req/PINN/Merit):** 3/1/4.

**Setup/status.** Exact continuum sech² smoothing inverse density rho(c)=Im q(c+i a)/a,a=pi/(2gamma); readout h w_m rho/2. Truncate on original H05 band with no finite-boundary correction. Profiles analytic in needed complex strip.

**Cost.** Same as A12 with complex evaluation cost. No fit. No additional PDE solution. Not a discrete cardinal inverse; lattice/contour/truncation errors remain.

**Result.** H05 B4096 radial targets:2.24e-7,7.03e-8,3.79e-6,2.02e-7. Failed floor because omitted tails, even though interior coefficient interpretation accurate.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A14. Complex-shift readouts with long halo

Longer halo fixed short-band failure; known whole-space profile access is substantial information.

**Status:** Successful no-fit construction. **Ratings (Req/PINN/Merit):** 5/1/9.

**Setup/status.** Same coefficients, much longer center interval. H05 add80 centers each end; I04/F19 use201 centers over[-4,4] while evaluation on small cube/unit ball. No scalar/global LS.

**Cost.** O(MH C_profile +Md), model O(MH+Md), eval common. No fit. Requires known analytic profiles and adequate angular quadrature.

**Result.** H05 radial B9216:6.22e-15,3.32e-15,5.38e-14,1.91e-15. Extra halos necessary at that spacing. I04 three3D Gaussian-family targets with weighted sphere quadrature, H201: directions16/64/144/400/1024/2304; neurons3216/12864/28944/80400/205824/463104. Anisotropic errors3.99e-2/1.10e-3/2.40e-5/1.18e-8/5.28e-14/6.45e-15. Final isotropic4.24e-15,shifted mixture8.78e-15. Three cases reach1e-14 at adequateM; not broad-class guarantee.

**Evidence.** I04 `radon/report.md`, `radon.py`; H05 forward README; F19 `scaling/report.md`.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A15. General-d tensor sphere construction

High-D profile evaluation cost is not a uniform constant; high neuron counts belong beside floor claim.

**Status:** Successful 4D; angular curse in 5D. **Ratings (Req/PINN/Merit):** 5/1/8.

**Setup/status.** F19 formula q_v(t)=det(B)^(-1/2)(v^T B^-1 v)^(-d/2) 1F1(d/2;1/2;−t²/(v^T B^-1 v)). Handles even d correctly (nonlocal filtered Radon); use A14 tanh encoding. Circle quadrature2D, GL×azimuth3D, recursive Gauss–Jacobi d4/d5. Number of antipodally reduced directions M=n_ang^(d−1). Dense/full-rank rotated Gaussian, not secretly low-dimensional. Known analytic target access.

**Cost.** O(d³+Md²+MH C_hypergeometric) plus quadrature-node setup and O(Md) enumeration; model O(MH+Md), eval common. C_hypergeometric may depend on d/arguments/precision, so not true uniform O(1). No fit/external solve. Untied model O(dMH) memory.

**Result.** d4,H201,M32768,B6586368 gives8.15e-15 on1024 fresh points; H121,B3964928 gave6.91e-15 on128-point diagnostic. d5,M331776,H81,B26873856 gives6.45e-11, H201,B66686976 still6.43e-11 angular-limited. d5 cheaperB13172736 independently validated1.19e-7. Largest d5 peak~2197.8MiB whole process, shared model521.4MiB,target; untied3561.5MiB. No5D strictfloor.

**Rate.** Conditional analytic/spectral-cubature model e_ang≈exp(-a M^(1/(d−1))), e_center≈exp(-bH); balanced B gives M∝B^((d−1)/d), H∝B^(1/d), error exp(-c B^(1/d)), with regularity-dependent constants. Not a theorem for arbitrary functions, QMC, or noisy profiles. Explicit angular curse remains.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A16. Sobol sphere construction

Center conversion at roundoff; angular error dominates.

**Status:** Useful approximation; failed floor goal. **Ratings (Req/PINN/Merit):** 4/1/5.

**Setup/status.** Same profile/encoding as A15, normalized inverse-normal Sobol samples instead of tensor cubature, two scrambles. Does not use learned geometry. M enumeration O(Md); same analytic preprocessing and neuron/eval costs.

**Result.** M16384,H201,B3293184 anisotropic medianerrors d4:.063%,d5:.079%,d8:.265%,d16:1.17%,d32:2.34%. Center conversion still1e-15–1e-14. No floor and no empirical asymptotic exponent established. Do not apply tensor-spectral rate toQMC.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A17. Known-metric adapted sphere construction

Improvement assumes the desired metric is already known, not learned.

**Status:** Useful geometry control. **Ratings (Req/PINN/Merit):** 4/1/6.

**Setup/status.** Take known Gaussian precision root B^(1/2), transform uniform direction u→B^(1/2)u/|B^(1/2)u|, change profile scale accordingly. Pure analytical geometry adaptation; not learning an unknown manifold. Same M/H asK13.

**Cost.** O(d³+Md²+MH C_profile), model/eval common. No fit; supplied target precision matrix is essential information.

**Result.** d32 .44–.61%, d64 .46–.57%, two scrambles, B3293184. Improves rawQMC but not precision-floor solution. No measured neuron/error exponent.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A18. Sparse known Fourier-ridge construction

Does not discover unknown sparse directions or solve an unknown PDE.

**Status:** Successful specialized construction. **Ratings (Req/PINN/Merit):** 5/1/8.

**Setup/status.** Four supplied ridge directions and frequencies, exact Fourier inverse of sech² multiplier, H201 long-band centers. No angular quadrature or direction search. General-d ordinary tanh network.

**Cost.** O(sH+sd) construction from prescribed amplitudes/frequencies, modelO(sH+sd), evalO(E(sd+sH)); s4. No fitting, no PDE solve, requires actual sparse directional decomposition. Not a generic d-dimensional black-box method.

**Result.** s4,B804 achieves5.89e-16 at d256; d3,16,64 also tested floor. Demonstrates ambientd alone does not determine cost; does not demonstrate discovery of unknown directions.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A19. Naive square-root halo truncation

Square-root halo count without its rational correction is not the newer QUILL method.

**Status:** Failed boundary shortcut. **Ratings (Req/PINN/Merit):** 2/1/2.

**Setup/status.** F19 applied current halo count R=ceil(sqrtN) to complex-shift coefficients without implementing newer rational boundary correction. No fit. Cost asK11b.

**Result.** Sine2pi N128cells maxerror1.14e-4; N256cells3.15e-5. 3DGaussian withM1600,N129interior,R12 plainvalue error3.96e-5 versus corrected6.40e-16. A genuine failed method at machine-precision requirement. R alone is not boundary algorithm.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A20. Corrected finite-contour QUILL Radon

Uses existing halo neurons; contour setup prevents a strict O(MN) claim for current code.

**Status:** Successful no-fit construction. **Ratings (Req/PINN/Merit):** 6/2/9.

**Setup/status.** Actual newer construction: complex-shift density, contour moments mu−nu at endpoints, explicit stable partial fractions, modify outer existing readouts, reanchor bias. No LS/SVD/matrix solve, no added neurons. Requires valid analytic tube; entire-profile default is valid only for entire profiles. Ninterior caller passesn_cells=N−1 and R=ceil(sqrtN); scalar diagnostic file also contains historical Ncells convention, explicitly documented.

**Cost (current code, not aspirational linear claim).** Let m≈R/2, q contour nodes, L_profiles profiles encoded together. Shared `leggauss(q)` current NumPy path conservatively O(q³) setup/O(q²) memory; base coefficients O(L_profiles H C_profile); contour projection O(L_profiles m q); stable fractions O(m³+L_profiles m²), since current code rebuilds elementary symmetric polynomials for everypole. Peak O(L_profiles H+L_profiles q+mq+L_profiles m+q²), plus modeldirections. Node/table setup can theoretically be cached but current encode call does it. For known M profiles L_profiles=M; shared polynomialbank encodes p+1 profiles once, independent ofM. Default dynamic entiretube at asymptoticN has k~sqrtN and q~N, so cannot label current construction strictly O(MN). Evaluation remains ordinarytanh common costs.

**Result.** Sine2pi at128cells correctedmax4.44e-16,256cells5.55e-16; highfrequency sin12pi at128cells relative2.19e-14,256cells2.11e-15; Gaussian128cells1.10e-16. Current3D anisotropicGaussian,M1600,N129interior,R12,B244800: heldoutvalue6.395e-16,gradient1.487e-15,Laplacian7.866e-15, versus naivehalo3.96e-5value. Budget ladder optimized: B6480value3.83e-5;18000value2.29e-8;46080value2.88e-10;84992value2.09e-13;244800value6.40e-16. At84992 derivative errors still~1e-12–1e-11. Valuefloor and PDEderivativefloor must remain separate.

**Rate.** Exact-arithmetic boundary component bound h exp(-lambda R²/4), not total network bound. Balancing angular/center errors tested; no universalbestlambda: scalar N32best.40, largerN.25; 3DLaplacianN257besttested.18. Fixedcountchoices can fail despite asymptotic theory.

**Evidence.** `quill_boundary.py`; `quill_review/boundary_method.md`, `boundary_metrics.json`, `boundary_validation.json`; `quill_sweep.py`, `allocation_optimized_selected.json`, `allocation_meta.json`.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### A21. Analytic Gegenbauer ball-frame reconstruction

M=(p+1)^(d-1); current constructor explicitly stores ordinary dense layer.

**Status:** Successful capacity audit; incomplete floor consistency. **Ratings (Req/PINN/Merit):** 5/2/7.

**Setup/status.** Implemented representation/derivative audit in d=2,3,4, degrees p=4,6. A supplied normalized Gegenbauer ridge along one direction is distributed to a common angular frame using the exact sphere reproducing identity. Positive cubature is exact for the required degree in exact arithmetic; the one scalar polynomial profile is encoded by corrected QUILL (A20). Export is a literal ordinary tanh MLP, with no polynomial evaluator or product gates in its forward call. This is capacity evidence, not an unknown PDE solve.

**Cost.** Product sphere rule M=(p+1)^(d−1). One profile bank costs C_enc(1,N,q), then angular weights O(Md+Mp), and ordinary explicit export O(dMH) time and memory. A shared representation can retain O(Md+MH). Generic construction of all degree≤p polynomial features is a different, larger map; do not price this single-ridge audit as the complete general polynomial operator. Evaluation common above. No fitting or external solve; known ridge formula is supplied.

**Result.** At N=257 interior centers, R=17: d2,p4,B1455 max value error4.44e-15; d3,p4,B7275 max value3.55e-15; d4,p4,B36375 max value1.86e-15. For p6 the corresponding values are1.22e-13 (B2037),1.51e-14 (B14259),5.86e-14 (B99813). Thus all three degree4 fields reach strict1e-14, none of the three degree6 fields do at that budget. Largest scaled derivative errors are approximately6.2e-15 to1.2e-13 across these N257 cases. N129 has values1e-13–1e-12 and sometimes derivative errors2e-11. The intentionally underresolved d4,p6 rule uses M64 rather than343 and gives0.817 max value error and3.37 worst scaled derivative error. The angular identity alone does not forgive an inadequate rule or finite scalar encoding error.

**Evidence.** `experiments/expF19_radon_direct_pde/ridge_frame_dimension_study.py`; `results/checkpoint_F_applications/expF19_radon_direct_pde/pure_mlp_repair/ridge_frame_dimensions.json`.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B01. Dense spacetime ridge Gauss–Newton

QR-before-SVD reduced peak from ~12GB to ~7GB without changing asymptotic order.

**Status:** Successful native PDE baseline; memory wall. **Ratings (Req/PINN/Merit):** 5/4/6.

**Setup/status.** F18 direct precursor,3inputs(x,y,t), 2D unsteady incompressibleNS primitiveu,v,p, fixedgeometry, jointly solvedreadouts. NativePDE residual, no outsidePDErun. Derivatives actualtanh. DenseSVD everyGNstep; above6000columns augmentedQR thenSVDofR, same asymptotic solve but lowerconstants. Same-method coarse-to-fine warmstarts fit priornetwork; initialcoarserungzero. **Oracle exactsolution fit did select N/M geometry split**; this is oracle-informed experimentaldesign even though no oracleweights entered dynamicrun. Not fully autonomous setup.

**Cost.** Ffields3,S=F(B+1). O(K[QD dB+Q S²+S³]) with D derivativechannels fixedhere; memoryO(QDB+QS+S²). WithQ∝S: cubiccompute/quadraticRAM. Model/eval sharedgeometry common plusFreadouts. AugmentedQR removesstoredQfactor but notquadratic scaling.

**Result.** Bperfield256/512/1024/2048/4096: velocity1.8e-3/5.7e-6/8.2e-9/2.3e-9/1.1e-11; pressuretop8.6e-10; maxdiv6e-10. Bestsplit2048gives1.2e-10 ratherthan2.3e-9;4096N16gives9.1e-12. Neverstrict1e-14 onwholeclosedcube; localmidinterior1e-12–1e-13. ~2.5ordersperdoubling onmost ladder, pressurelags. Top~20minutes,~7GB,8192estimated25GBnotrun. SuccessfulnativePDE method butmemorywall andboundary-derivativefloor.

**Evidence.** `results/checkpoint_F_applications/expF18_ns_spacetime/expF18_results.md`; `experiments/expF18_ns_spacetime/{ridge3d.py,run.py}`. Oldreport says “no training,” meaning noAdam; catalogue should callGNfitting whatitis.

Source audit: [radon_catalogue_construction_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_construction_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B02. Analytic linear-IVP Radon propagation

Heat output time can require fresh readout construction; not a fixed general spacetime solver.

**Status:** Successful specialized linear method. **Ratings (Req/PINN/Merit):** 5/3/7.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B03. FFT-evolved Radon heat profiles

No neural LS, but an external spectral solution supplies each profile.

**Status:** Valid hybrid; rejected native-general-PINN claim. **Ratings (Req/PINN/Merit):** 3/3/5.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B04. Fixed spacetime wave construction

Two spacetime directions per spatial spoke; no query-time coefficient solve.

**Status:** Successful solve-free specialized MLP. **Ratings (Req/PINN/Merit):** 6/3/8.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B05. Cole–Hopf Burgers construction

Unresolved high modes destabilized inverse smoothing; not general nonlinear PDE construction.

**Status:** Successful specialized nonlinear formula. **Ratings (Req/PINN/Merit):** 4/3/6.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B06. External Fourier evolution then Radon compilation

Mode pruning/corrected halo are compiler refinements, not independent PDE discovery methods.

**Status:** Rejected native-PINN claim; legitimate compiler. **Ratings (Req/PINN/Merit):** 2/3/4.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B07. Native compact-spline coefficient dynamics

Production uses compact spline evaluation, not an executed ordinary ReLU-cubed MLP.

**Status:** Native but poor precision scaling. **Ratings (Req/PINN/Merit):** 3/3/5.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B08. Native full tanh-readout dynamics

Caches full analysis-to-readout map; algebraically equivalent field to reduced form.

**Status:** Successful native mechanism; larger storage. **Ratings (Req/PINN/Merit):** 5/4/6.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B09. Native reduced tanh-coordinate dynamics

Independent reduced API: Burgers B1091/P129/Q388, setup .05583s and solve .004781s; the stronger B08 timestep-refinement floor belongs to the full formulation.

**Status:** Successful native mechanism. **Ratings (Req/PINN/Merit):** 6/4/7.

**Benchmark provenance correction:** The numerical forward refinement ladder below is the full-readout B08 run. Independent reduced-API checks in `general_solver/api_metrics.json` give heat/advection error 1.49456e-15 with B291/P33, and Burgers error 1.14138e-12 with B1091/P129/Q388, raw off-grid PDE 3.41969e-12, setup .05583s and solve .004781s. The generic API uses DOP853, while inverse wrappers use RK4. No matching reduced timestep-refinement sweep establishes the full formulation’s 2.52e-14 result for this implementation. Current reduced factories initially materialize full Q×H neural feature arrays, so peak memory contains QH rather than an assumed bounded bH scratch. Sources include `solver/base.py` and `native_inverse.py`.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B10. Native flat-ridge Navier–Stokes evolution

Q~P~K^d makes cached state-to-fields storage ~K^(2d).

**Status:** Successful native PDE mechanism; cache bottleneck. **Ratings (Req/PINN/Merit):** 5/4/6.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B11. Native NS Taylor-in-time recurrence

Local time series does not eliminate spatial error or finite convergence radius.

**Status:** Successful time integrator; not PDE floor. **Ratings (Req/PINN/Merit):** 5/3/6.

**Setup:** same native projected NS ODE b'=Lb+B(b,b), generate time coefficients
(n+1)b[n+1]=L b[n]+sum_j B(b[j],b[n-j]) directly, then Horner evaluation.

**Other solve:** no production time steps/LS, but this IS a numerical truncated
Taylor solution of a finite spatial ODE. Not a universal closed form.

**Costs:** B10 spatial cache; degree T recurrence stores O(TP+TQ+JQP), work
O(T JQP + T QlogQ + T^2 Q) with cached coefficient fields and fixed d; Horner
state O(TP), then O(HP) export and ordinary flat eval. Local convergence radius
requires control; longer times need restart/slabs, not demonstrated here.

**Outcome:** degree8 discrepancy2.60e-15 against same spatial cutoff, but total
error1.97e-9 versus refined spatial reference. Degree12 construction .098s after
15.2s spatial cache; exact-mode Taylor .027s. Thus time floor only.

**Evidence:** `native_taylor.py`, `native_dynamics/taylor_metrics.json`.

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B12. Flat-ridge shifted elliptic residual iteration

Shift 0 diverged; shift 5 contracted.

**Status:** Successful special operator iteration; not floor. **Ratings (Req/PINN/Merit):** 6/4/7.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B13. Adaptive ridge dynamics without replay

Sparse final geometry does not retroactively correct the path.

**Status:** Failed adaptive variant. **Ratings (Req/PINN/Merit):** 3/3/3.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### B14. Adaptive ridge dynamics with rejected-window replay

Candidate discovery still stores dense Q×P_c arrays.

**Status:** Successful adaptive specialization; not floor. **Ratings (Req/PINN/Merit):** 6/4/7.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### C01. Shared-product elliptic box solver

Counts 714 tanh units but 400 product coefficients; tanh width alone is misleading.

**Status:** Useful solver; not pure MLP. **Ratings (Req/PINN/Merit):** 4/4/6.

**Setup:** same smooth cubic elliptic PDE; two shared 1D QUILL sine banks,
explicit product gates and affine boundary corrections, K² physical readouts.
THIS IS NOT A PURE TANH MLP or original flat Radon network.

**Other solve:** same scalar-preconditioned fixed-point iteration as B12; no
external trajectory/global readout LS.

**Costs:** Cenc(K,N)+O(Q(dHK+P)) basis setup, O(QP+HK) cache, O(GQP) solve.
One-point eval O(dHK+P), model O(HK+P), tanh activations dH. Product gates and P
coefficients must be counted. K-specific changes do not have a neurons-only rate.

**Outcome:** K12 P144 B442 error6.38e-8 in.0159s; K20 P400 B714 error1.17e-12
in.0884s. Spectral-like decrease, stopped above1e-14 in tested range. Matched
classical sine control same accuracy in.0326s. Baseline PINNs 2.45–3.72e-5 in
83–281s; this one comparison does not prove general superiority.

**Evidence:** `solver/tensor_box.py`, `general_solver/comparison_metrics.json`.

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### C02. Mapped annulus Galerkin / Newton–CG

Diagonal-only 30-CG budget failed; problem-aware factorization succeeded.

**Status:** Useful domain-specific solver; not pure MLP. **Ratings (Req/PINN/Merit):** 4/5/7.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### C03. Stiff QUILL reaction–diffusion with BDF

A large threshold event is not a certified singularity time.

**Status:** Useful stiff specialization; not floor. **Ratings (Req/PINN/Merit):** 4/4/6.

**Setup:** native reduced trigonometric/QUILL field for stiff Allen–Cahn and
quadratic blowup reaction, actual neural derivatives. Implicit BDF with dense
P×P Newton matrices, physical invariant and underresolution guards.

**Other solve:** YES internal dense implicit state solves; no external trajectory.
Different problem-specific backend, not general pure-MLP spacetime solution.

**Costs:** B08/B09 reduced setup plus O(P²) matrices, O(QP+P²+HP) RAM. Across L
implicit stages and F factorizations, O(LQP + F(QP²+P³)) conservative cost.
Snapshots collapse to H-unit tanh readout, O(H) inference. State trajectories
and reporting add storage if retained.

**Outcome:** K32/64/128 Allen–Cahn errors .0263/.00275/2.82e-5; K256 difference
5.62e-7 lies below reference uncertainty1.68e-6. Not floor. Blowup threshold
10000 near.838906 is an event, not true singularity time. Analytical bracket
5/6≤T*≤1 independent of solver. Coarse invariant violations were real failures.

**Evidence:** `solver/reaction.py`, `general_hard.py`, hard_results.md.

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### C04. Finite-volume Burgers with ReLU primitive

Conservation/entropy accuracy is not pointwise solution accuracy.

**Status:** Valid shock baseline; not smooth Radon PINN. **Ratings (Req/PINN/Merit):** 2/3/5.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### C05. Shared-product native NS and animation

867 tanh units still require 36501–59049 velocity coefficients.

**Status:** Useful native solver/visualization; not pure MLP. **Ratings (Req/PINN/Merit):** 4/4/6.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### C06. Generic cached QUILL-product residual engine

Matrix-free Jacobian still keeps Q×P derivative feature caches.

**Status:** Broad useful prototype; not pure MLP. **Ratings (Req/PINN/Merit):** 5/7/7.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### C07. Sparse-support precision product solver

Oracle capacity fits were diagnostics, not solution initialization.

**Status:** Useful precision refinement; not consistently floor. **Ratings (Req/PINN/Merit):** 5/6/7.

**Setup:** C06 with selected Legendre multiindices, downward-closed expansion,
pruning/refinement based on earlier PDE-solved coefficients; anchored halo
evaluation, tighter stopping/defect refinement and trace-sized BC sampling.
Known-solution capacity diagnostics inform research resolution choices but
reference coefficients do not initialize the solve or choose support.

**Other solve:** same global iterative native PDE solve. Sparse support removes
some product dimensions, not underlying d-dependence or feature-cache issue.

**Costs:** replace fullP by selectedS in C06: O(JQS) cache and Krylov products;
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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### C08. Exact-polynomial residual control

Separates solver-coordinate gains from any intrinsic advantage of tanh.

**Status:** Successful control; not Radon/MLP. **Ratings (Req/PINN/Merit):** 2/6/6.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### C09. Restricted weak/entropy front-family solve

Entropy constraints reject a genuine residual-only false solution.

**Status:** Useful small weak-form test; generic discovery absent. **Ratings (Req/PINN/Merit):** 4/3/6.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### C10. Scaled Gaussian concentrating-core probe

Fixed global grids failed; required synthetic forcing diverges; not reproduced Navier–Stokes blowup.

**Status:** Successful representation probe only. **Ratings (Req/PINN/Merit):** 2/1/5.

The synthetic known field is a Gaussian swirl whose radial and axial widths are prescribed functions of the time-to-concentration parameter. Two known 1D profiles, Gaussian and coordinate-times-Gaussian, are encoded once on normalized coordinates; exact products construct the 3D field. At N=385 interior centers, three coordinate banks contain 1,275 tanh units. Rescaling the input geometry preserves roughly 1.4e-15 relative core error as the core shrinks by 100×, whereas fixed global grids can become inaccurate or nonfinite. No Navier–Stokes residual is solved. The exact forcing required by this synthetic swirl diverges, and the cited manuscript’s smooth forcing, axial flow and correction profiles were not instantiated. This is evidence for coordinate adaptation in representation, not reproduced Navier–Stokes blowup. Sources: `experiments/expF19_radon_direct_pde/moonshot_geometry.py`, `results/checkpoint_F_applications/expF19_radon_direct_pde/adaptive_followup/moonshot_geometry_metrics.json`.

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### D01. Analytical heat-tensor inverse wrapper

Known forward formula does not make general inverse field discovery free.

**Status:** Useful specialized inverse wrapper. **Ratings (Req/PINN/Merit):** 4/4/6.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### D02. Native Burgers parameter inverse with tangents

IC-only measurements give zero viscosity sensitivity; tiny residual does not imply identification.

**Status:** Successful specialized inverse/UQ. **Ratings (Req/PINN/Merit):** 5/5/7.

**Peak-memory correction:** The present `native_inverse.QuillForward` retains its full neural model, including Q×H value/derivative and analysis arrays. Reduced-coordinate state alone therefore does not describe peak or retained RAM; the cost row includes QH.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### D03. Unknown initial-profile inverse with regularization

Deliberate inverse-crime fit 3.05e-20 failed forward refinement.

**Status:** Successful restricted inverse experiment. **Ratings (Req/PINN/Merit):** 5/5/7.

**Peak-memory correction:** `QuillBasis1D` forms full Q×H neural arrays during construction and then releases them. The cost row therefore includes QH in constructor peak, without claiming those arrays remain resident for the later tangent solve.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### D04. Regularized backward-heat compilation

Compiler floor is not inverse-solution floor.

**Status:** Valid inverse baseline; not native neural discovery. **Ratings (Req/PINN/Merit):** 2/3/5.

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

Source audit: [radon_catalogue_early_pde_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_early_pde_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### E01. Deep tanh polarization compiler + native residual solve

W=O(H[d^2+Pad+P^2a^2]), a=O(min(d,p)); structured zeros still stored densely.

**Status:** Successful pure-MLP repair; dense layers expensive. **Ratings (Req/PINN/Merit):** 7/4/7.

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

Source audit: [radon_catalogue_pure_routes_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_pure_routes_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### E02. Cached analytical disk ridge coordinates

Disk coordinates on a square became severely ill-conditioned; box repair is E03.

**Status:** Successful cached disk method. **Ratings (Req/PINN/Merit):** 7/5/8.

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

Source audit: [radon_catalogue_pure_routes_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_pure_routes_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### E03. Cached enclosing-ball / box ridge coordinates

Retains M×p×P tensor; more centers sometimes worsen ordinary cancellation.

**Status:** Useful pure-MLP route; constructor/cache bottlenecks. **Ratings (Req/PINN/Merit):** 6/6/7.

`BallRidgeFeatures` maps a physical box into its enclosing unit ball,
constructs the coefficients of known Legendre-coordinate features from
algebraic ball moments, applies tensor spherical cubature, and explicitly
encodes Gegenbauer directional profiles with QUILL. Ordinary forward and
actual residuals contain no products/polynomials; polynomial expressions
are construction coordinates only. Full angular rule has M=(p+1)^(d-1).

Unlike method E02 it retains `profile_map[M,p+1,P]`: O(MpP) storage. Cached
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

Source audit: [radon_catalogue_pure_routes_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_pure_routes_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### E04. Low-degree sparse angular cubature

All-face traces make controls easier; no arbitrary-degree sparse rule established.

**Status:** Successful low-degree specialization. **Ratings (Req/PINN/Merit):** 7/4/8.

Same ball-coordinate constructor and PDE solve, different angular rule:
degree<=2 uses M=d^2; degree3 uses
M=d+2*binom(d,2)+4*binom(d,3). Established signed Stroud/Smolyak-type
rules preserve required even angular moments. Signed weights enter map
construction, not a signed PDE loss. This changes exponential-in-d
direction cost to polynomial-in-d at those fixed low degrees.

Use method E03's compute/memory formulas with the smaller M. p2 has
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

Source audit: [radon_catalogue_pure_routes_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_pure_routes_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### E05. Declared-active-axis ridge construction

Actual dense export still stores and executes dB, even with zero columns.

**Status:** Useful disclosed prior; not manifold discovery. **Ratings (Req/PINN/Merit):** 5/3/6.

`ActiveAxisRidgeFeatures` applies method E03 only to k explicitly supplied
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

Source audit: [radon_catalogue_pure_routes_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_pure_routes_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### E06. Sequential time-slab ridge networks

First three slabs line_search_failed; not one global spacetime MLP.

**Status:** Useful continuation; incomplete solves. **Ratings (Req/PINN/Merit):** 6/4/6.

Uses method E03 on each of L prescribed time slabs; the previous network's
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

Source audit: [radon_catalogue_pure_routes_audit.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/radon_catalogue_pure_routes_audit.md). The source paths cited above are relative to this repository unless otherwise identified.

### E07. Direct directional-profile operator

Eliminates old global map and Q×P feature cache.

**Status:** Implemented operator; not complete solver by itself. **Ratings (Req/PINN/Merit):** 6/2/7.

The direct operator replaces the old M×(p+1)×P geometry tensor by coefficients b[M,p], a shared QUILL bank, and direct angular directions. Prepared forward state forms neuron readouts once before point batches; the adjoint accumulates neuron gradients and converts them back once. No coefficient solve is implied by a successful forward/adjoint identity. A Q=2048, p=8, N=129 benchmark scales roughly linearly in M=8,16,32,64, with .056/.111/.225/.452 seconds and about 1.85 MiB tracked transient allocation. This is an operator/storage result; cached small matrices were faster to reuse. Sources: `experiments/expF19_radon_direct_pde/route2_directional_operator.py`, `route2_directional_benchmark.py`, and the status record.

Source audit: [quill_route2_generality_status.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/quill_route2_generality_status.md). The source paths cited above are relative to this repository unless otherwise identified.

### E08. Matched ball-frame residual iteration

Ideal contraction <=1/2; operator is not ordinary Poisson or arbitrary Dirichlet PDE.

**Status:** Successful matched no-LS native solve. **Ratings (Req/PINN/Merit):** 7/3/8.

This is the genuinely no-least-squares PDE iteration tested on L u+tanh(u)=f, L=-div[(I-xx^T)grad]+2, on the unit disk. The boundary-degenerate diffusion supplies natural zero flux; it is not an ordinary Dirichlet Poisson problem. Actual tanh residuals are analyzed with known Gegenbauer profiles, divided by degree eigenvalues 2+n(n+2), and synthesized as neural corrections. Ideal frame analysis/synthesis gives an at-most-one-half contraction on the canonical frame range. All coefficients start at zero. At p=4/8/12/16 and N=257, the baseline 40-update field errors were 6.40e-4/7.21e-8/1.48e-12/1.78e-14. Tightening stopping gave 46 updates, ordinary sampled field error 1.26e-16 and PDE RMS 1.49e-15. These are sampled validations, not continuum certificates. Sources: `route2_frame_iteration.py` and `results/checkpoint_F_applications/expF19_radon_direct_pde/route2_generality/frame_iteration/`.

Source audit: [quill_route2_generality_status.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/quill_route2_generality_status.md). The source paths cited above are relative to this repository unless otherwise identified.

### E09. Streamed general solve with redundant profiles

Deterministic full constraints; no stochastic subsampling.

**Status:** Successful memory trade; conditioning still costly. **Ratings (Req/PINN/Merit):** 7/5/7.

The streamed residual engine supports general declared local residuals, coupled fields, measurement constraints and inverse parameters. It recomputes neural jets in deterministic point batches; full residual vectors, local callback sensitivities and Krylov vectors remain. No Q×coordinate feature cache or complete Jacobian is stored. Bounded coefficient panels construct the preconditioner. The shared 1D bank and prepared neuron state remain resident. A p=2 manufactured forward problem reached field 3.4e-16 and raw PDE 6.6e-15; its eight-anchor inverse also reached roundoff. The nonpolynomial p=16 run initially stopped at field 1.3e-8, then reached 1.47e-15 with more iterations: that first run was under-solved, not representation-limited. Its PDE RMS remained 6.86e-14. Sources: `solver/streamed_residual.py`, `route2_streamed_study.py`, and the `streamed_solver` result directory.

Source audit: [quill_route2_generality_status.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/quill_route2_generality_status.md). The source paths cited above are relative to this repository unless otherwise identified.

### E10. Streamed analytical disk modes with block preconditioning

No global map tensor, feature cache or full Jacobian; 2D angular simplification only.

**Status:** Strongest current smooth disk route. **Ratings (Req/PINN/Merit):** 8/6/8.

The disk adapter removes redundant angular coefficients using an exact half-circle trigonometric identity degree by degree. It stores M×(p+1) trigonometric values and O(p²) labels, never the old M×p×P map. Actual tanh profiles approximate the ideal orthonormal disk modes. At p=16, M=17, N=257, R=17, the 4,947-neuron network has 153 coordinates instead of 273 redundant ones. From zero, the ordinary Dirichlet diffusion–reaction solve reached field 4.3105e-16, raw AD PDE RMS 4.1504e-15 and maximum PDE residual 4.0301e-14 in 59.2 seconds, 532 Krylov products and three nonlinear steps. The exact-data inverse with eight anchors recovered diffusivity 1.6999999999999988, with field 3.83e-16 and PDE RMS 4.14e-15. These are two closely related smooth manufactured cases, not universal consistency. The direct-long readout L1 norm was 10.5681 versus .521770 here; less cancellation is consistent with improved derivative accuracy, but not a complete causal proof. Sources: `route2_disk_profiles.py`, `solver/streamed_residual.py`, `route2_streamed_study.py` and the status record.

Source audit: [quill_route2_generality_status.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/quill_route2_generality_status.md). The source paths cited above are relative to this repository unless otherwise identified.

### E11. Streamed analytical disk modes with diagonal scaling

Lower setup option, slower in this case; independent audit confirms sampled tolerance and held-out errors.

**Status:** Floor attained in saved model; soft-budget status preserved. **Ratings (Req/PINN/Merit):** 8/6/8.

This is the same analytical disk model and native general residual engine as E10, using fixed-count diagonal scaling probes instead of constructing small Gram blocks. It reached field relative L2 3.6834e-16, raw AD PDE RMS 4.1758e-15 and PDE max 4.0856e-14 in 185.6 seconds and 1,701 Krylov iterations. The final accepted step crossed the 180-second soft budget; the saved termination status remains budget_exhausted, while the independent audit confirms that the returned weights satisfy the final sampled tolerance and held-out errors. This is evidence for a lower-setup-cost option, not a successful run-status claim. On this benchmark the block version was faster because block setup cost only .188 seconds. At larger coordinate counts that tradeoff can change; no resolution-independent iteration bound has been proved. Sources: `route2_streamed_study.py`, `solver/streamed_residual.py`, and `docs/quill_route2_generality_status.md`.

Source audit: [quill_route2_generality_status.md](/Users/sam/my-repos/research/collaborations/precisionMLPs/docs/quill_route2_generality_status.md). The source paths cited above are relative to this repository unless otherwise identified.


## Proposals, rejected ideas and controls that are not extra completed methods

1. **Squared-ReLU multiplication/compiler.** The user allowed a separate RePU result; the exact square/multiplication identity is valid, but no completed RePU PDE implementation/benchmark was found in the audited route. This is a proposal beside E01, not a tested 61st method. GELU was also allowed as a main activation but the implemented repair uses tanh; no GELU PDE performance or measured cost is claimed.
2. **Independent nonlinear spokes.** Independent profile evolution works for certain linear operators (B03/B04). Nonlinear products couple Fourier directions through k=p+q. A generic nonlinear PDE cannot be solved by simply evolving every prescribed spoke independently; no valid general algorithm or successful experiment was found.
3. **Complex-shift self-encoding of an evolving tanh field.** Naively shifting the current tanh approximation can encounter its meromorphic poles. The analytic-profile construction does not justify this operation on arbitrary iterates. It was rejected mathematically, not validated as a production method.
4. **Published forced-Navier–Stokes blowup reproduction.** The manuscript's full forcing and corrected velocity were not instantiated. C10 is a separate synthetic known-field representation probe; its required force diverges. It must not be presented as reproducing a smooth-forced NS singularity.
5. **General higher-dimensional factorized angular maps.** E10/E11's nonredundant trigonometric conversion is specifically 2D. General spherical-harmonic analogues without the old dense tensor, scalable arbitrary-degree sparse cubature, and uniformly effective preconditioners remain next work. E04 only proves low-degree sparse rules in its tested scope.
6. **Automatic local tangent patches / hp geometry / learned manifolds.** Local patches, zero-sum ridge profiles, a global background, and automatic local anisotropy were proposed in H04 and later theory. A08 and A04 are limited learned/linear-chart experiments; they do not implement the proposed universal local geometry solver. E06 supplies explicit prescribed time slabs, not automatic general patch discovery.
7. **Same-family failure controls.** Keep these attached to their numbered methods: rough monitors (A02), angular underresolution (A21/E02), initial short/plain halos (A13/A19), timestep limits (B08–B11), no-replay adaptation (B13), conservation-only expansion shocks (C09), inverse crimes/noise and unidentified parameters (D01–D04), disk coordinates used on a square (E02), extra PDE face constraints and high-degree arithmetic failures (E03), initial under-solved p16 (E09), and the soft-budget status despite floor-valued returned weights (E11). They are evidence about limits, not hidden successes.
8. **Coefficient-space projection diagnostics.** H05 projected a constructed readout into the original global SVD's retained subspace. It explained why different coefficients represent nearly identical functions, but costs the global SVD and is not a faster construction/solver. It is not counted as an additional method.

## What the comparison does and does not establish

The most successful known-function constructions use analytical target information to prescribe readouts. Their floor accuracy does not solve coefficient discovery for an unknown nonlinear PDE. Native methods remove the external-reference trajectory, but still perform a numerical solve or a mathematically supported special-case iteration. Product-coordinate solvers proved useful precision and broad residual setup before the ordinary-MLP repair; their architecture difference remains explicit here. The latest streamed disk results establish an encouraging accuracy/memory trade on smooth nonlinear forward and exact-data inverse benchmarks. They do not yet establish generic Navier–Stokes floor accuracy, broad noise robustness, automatic shock handling, or uniform high-dimensional convergence.

The source audit is exhaustive over the discovered attempt families within this conversation and their relevant repository precursors, not a claim that every unrelated experiment in the repository is a Radon method. Repeated targets, seeds, bandwidths, halo sizes, cutoffs and implementation micro-optimizations are summarized as variants. All reported successes remain tied to their problem, metric and resolution.
