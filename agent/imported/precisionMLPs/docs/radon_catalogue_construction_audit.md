# Radon catalogue: known-function constructions and precursors

Read-only source/results audit, 2026-10-01. This file supplies evidence to the task coordinator, who owns the full catalogue and ratings. No new experiment was run. Methods below are grouped by substantive algorithm; a bandwidth/halo allocation sweep is recorded as a variant rather than falsely advertised as a new PDE solver. Earlier H/E/F18 experiments are related repo precursors; I04 and F19 direct constructions are explicitly discussed in this conversation. No Claude interactive implementations were inspected.

## Common costs and conventions

- d: input dimension. M: directional lines. N: interior centers where the current convention applies. R: halo centers per side. H=N+2R, B=MH actual tanh neurons. Older H experiments use N as total offsets, so their B=MN already includes their collar; do not add another halo to reported neuron counts.
- Q: fitting/collocation samples (or residual rows up to a fixed number of equations). E: evaluation samples. b: bounded evaluation batch. L: scalar samples per prescribed 1D profile. A: Fourier atoms retained. s: active/latent dimension or number of discovered ridge atoms, as stated locally. K: outer iterations/trial solves; not assumed bounded as resolution rises.
- G(Q,B)=O(QB min(Q,B)) dense SVD work and O(QB+min(Q,B)^2) peak algebraic storage, in addition to input/model memory. For the usual Q≥B, G=O(QB²), storage O(QB+B²). If Q∝B these are cubic time/quadratic memory. Stacking a fixed number of fields/equations changes constants; an unfixed field count must appear explicitly.
- Once an M-spoke network is constructed, shared-direction inference costs O(E(Md+B)); model memory O(Md+B+H); working temporary memory O(bH) if one direction and one point batch are processed at a time. The ordinary untied MLP implementation costs O(EdB) and stores O(dB). Both describe the same function. Existing flattened H/E/F18 implementations often pay the latter cost; F19 construction evaluators exploit shared directions and bounded direction chunks. Derivative channels multiply costs by their count, not automatically by d² if only a Laplacian is required.
- All bare O(MH) analytical construction claims require O(1)-cost prescribed profile evaluations and exclude deriving those profiles from arbitrary f. Analytic Gaussian preprocessing costs O(d³+Md²) for covariance inverse/eigendecomposition and direction quadratic forms. A generic numerical Radon transform has its own target-access/quadrature cost, not evaluated by these analytic-family experiments.
- The strict target in the current conversation is around 1e-14 relative solution error, separately from PDE derivative/residual error. Old reports loosely called 1e-12 a double-precision floor; the catalogue should preserve numbers rather than relabel these strict-floor successes.

## K1. Uniform QI–Radon dictionary + one global dense least-squares solve

**Setup/status.** Implemented and heavily tested fixed-geometry approximation from known function samples. Ordinary one-hidden-layer tanh MLP. Tensor angle×offset and alternate-direction half-offset/interlaced variants were tested in E01; H01/H05/H06 use recentered directions, uniform per-spoke grids, gamma h=.25, a 25% collar, and one truncated SVD. Tensor versus interlaced changes placement, not solve asymptotics. Random ridges and radial-tangent point meshes are controls, not explicit Radon inversion.

**Cost.** Geometry O(dB), feature assembly O(QdB) in flattened code, plus G(Q,B), memory O(QB+B²+dB). Shared-spoke assembly can reduce dB to Md+B but cannot remove dense solve scaling. Evaluation follows common costs. No external PDE solution; does require samples and global fitting. Rank truncation changes the selected coefficient representation.

**Accuracy.** E01 Gaussian reaches about 1e-14 by 576–1024 units; final 8192 tensor result max-error about 5.9e-14; Runge remains resolution-limited about 8e-9 under even its best geometry. H05 local 2D known-function split results: at B=256,512,1024,2048,4096, radial Runge best errors 8.2e-9,8.5e-11,8.0e-12,7.4e-15,7.5e-15; composition 2.3e-8,6.0e-10,4.2e-11,1.5e-13,1.3e-14; spatial packet 7.4e-7,1.5e-10,4.3e-11,3.0e-13,1.9e-14. Thus not monotone at arithmetic floor. H01 4096-feature smooth 1D/2D suite generally 3e-14–2e-12, but narrow spikes/product peaks unresolved and nonsmooth targets not at floor. H06 d3 end-state Gaussian1.3e-14, composition2.7e-14, product sine5.9e-14 at 12288 features; fast waves3e-13, packet3.3e-12 even after geometry polish.

**Rate.** Empirical two-axis bottleneck e≈max(e_M,e_N); H05 direct fits give angular exp(-a M^q), q≈.9–2.1, while tested center leg roughly N^-8.7 to N^-12.9 with shoulders. Do not replace these measured finite-grid curves by a universal exponential theorem. Allocation/halo/width change the regimes.

**Evidence.** `results/checkpoint_E_2d/expE01_geometry_zoo_2d/expE01_results.md`; `results/checkpoint_H_highdim/expH01_highdim_suite/expH01_results.md`; `results/checkpoint_H_highdim/expH05_direction_cliff_2d/expH05_results.md`; `results/checkpoint_H_highdim/expH06_ridge_hierarchy/expH06_results.md`; `docs/ridge_quadrature_theory.md`.

## K2. Smooth monitor-adapted centers + pilot/final global solves

**Setup/status.** H04 precursor, implemented/tested. First even fitted surrogate estimates per-direction projected gradient/curvature energy or residuals. Construct a smooth center-density map, set local gamma h=.25, then fit readouts again. Variants: data density, true-gradient oracle, surrogate gradient/curvature, residual, frequency monitor. These are placement variants, not independent construction equations.

**Cost.** Two or several G(Q,B) solves plus O(QdB+QMd) monitoring/binning/sorting (sorting projected samples may be O(MQ log Q)). Peak still dense G storage, not doubled if sequential. Inference common. Surrogate versions require target samples but no true gradients; oracle variants explicitly use them.

**Result.** Smooth monitor fixed the initial rough-mesh floor penalty. Sharp 1D spikes reached2e-14 with128 features versus uniform needing512; 2D hotspot packet at4096 improves1e-11→4e-13. Initial 1.5-gap monitor smoothing and 2%-gap jitter were failures at high precision (up to1e-11 versus1e-15). Revised ~6-gap smoothing improves. No global jump/shock resolution: step dense-region accuracy can be tiny where the jump is not, while whole-domain error remains around1e-2.

**Evidence.** H04 `expH04_results.md`, `mesh.py`, `floor_price.py`, `mesh_map_scale.py`; initial run results retained under `bw1.5/`.

## K3. Gradient-energy angle-density allocation (failed directional idea)

**Setup/status.** H04 tested density m(theta)∝[v(theta)^T C v(theta)]^(1/3), C=E[gradient f gradient f^T], alongside estimated versions/joint center+angle allocation. Same global LS, same complexity as K2.

**Result.** No useful directional improvement on isotropic suite or known ridge. The quadratic angular energy is broad and cannot locate a delta-like ridge direction. Mark the directional inference mechanism failed, not the final model unexecutable. Evidence H04 results and `known_answer_2d.py`.

## K4. Active-subspace reduction + global ridge LS (mixed)

**Setup/status.** H04 tested covariance eigenspace, allocate80% of features to s-dimensional active subspace and20% ambient background. Oracle and pilot-surrogate variants; repeated fit→covariance→fit tried. Not a general nonlinear manifold method.

**Cost.** K dense fits, surrogate gradient O(QdB), covariance O(Qd²), eigensolve O(d³), plus head on s dimensions; memory dense head plus O(Qd+d²). Eval active+background common; no external PDE solution. Oracle uses actual gradients.

**Result.** Embedded 2D composition at ambient d5,B4096: ambient8e-4→5e-6 estimated subspace,1e-12 oracle; noisy sheet1e-6→5e-11. Iteration once reached1e-9 but stalled in revised run. Pure ridge+background failed: covariance direction biased .01 degrees, inadequate for high precision. No consistent floor.

## K5. Stagewise projection-pursuit ridge atoms without joint polish (insufficient alone)

**Setup/status.** H06 candidate pool600 directions d3 or1500 d4; fit a32-offset scalar block to residual for each candidate, direction-polish best candidates, append best atom. A precursor stage to K6/K7.

**Cost.** For A_pool candidates and s installed atoms, scalar candidate scans O(s A_pool QH²) in overdetermined dense LS; memory O(QH+Qd) for sequential scan plus installed model. Candidate/trial joint refits add G(Q,sH). No no-solve claim. Evaluation common with M=s.

**Result.** Alone retains20–80% residual per atom and typically stalls at1e-2–1e-1 before joint polish. Treat as useful proposal/growth mechanism but failed standalone floor method.

## K6. Joint variable-projection direction Gauss–Newton (successful sparse ridge specialization)

**Setup/status.** H06 readout-eliminated joint direction optimization following K5, or applied to all directions of an even dictionary. Re-solve readout at every trial; Kaufman projected Jacobian for s(d−1) direction unknowns. This learns geometry, not merely fixed construction.

**Cost.** K[G(Q,B)+O(Q B T)+O(QT²+T³)] conservative direct algebra, T=M(d−1), plus line-search trial refits. The projection term uses retained readout SVD factors; no uniformly bounded K. Memory O(QB+B²+QT+T²). Evaluation common. Fit depends on target samples; no external PDE solver.

**Result.** All24 hidden-ridge tests (1,2,4,8 ridges,d3/d4,3seeds)2e-13–1e-12; some seeds require extra atoms after one direction sticks. Excellent high accuracy but not strict1e-14 consistent. All-direction polish helps direction-bound cells5–125× and does little in center-bound cells. d3 Radial Runge9.6e-14 after polish, packet3.3e-12.

## K7. Greedy hierarchical ridge mesh (nested background + atoms + refine/open decisions)

**Setup/status.** H06 repeatedly trial-fits nested even-direction additions, discovered atoms, background refinement, atom refinement; selects validation-log-error reduction per added neuron. Atoms polished jointly. One underlying global LS remains.

**Cost.** Sum of K5/K6/global LS costs over trial dictionaries: O(sum_t G(Q,B_t)+atom-search/polish), dense peak at max B. No justified simple near-linear construction bound. Evaluation common.

**Result.** d3 ridge2:2e-14 at160 units; ridge4:4e-14 at992; product sine2e-14 at1088; ridge4+bump8e-14 at3420. At4096 d4 ridge4+bump9e-12, Gaussian7e-11, composition1.6e-9, Runge1.5e-7. Strong sparse-structure savings; diffuse high-dimensional examples remain unresolved. Single-seed hierarchy evidence.

## K8. Fourier nearest-direction snapping (constructive certificate, failed precision example)

**Setup/status.** Whole-space Fourier polar decomposition; send every known Fourier atom to nearest available line, preserving radial frequency; group profiles on M spokes. Implemented for2D analytic composition. Tanh conversion via K9 or K11. Fourier access is given analytically here, not discovered from samples.

**Cost.** Assign A atoms O(AMd), stored atom lists O(A+Md), profile samples O(AH) if sparse assignment used, plus chosen scalar encoder. No multivariate fit or external PDE solve, but obtaining spectrum is an unpriced input requirement. Network evaluation common.

**Result.** Composition atM32 has6.98e-4 error already in continuous ridge sum; scalar fit reproduces that sum to3.41e-14 but cannot remove angular error. Bound O(r * integral |xi|theta(xi,V)d|mu|), worst covering rate M^(-1/(d−1)), not an optimal exact representation theorem. Failed as floor construction at thisM; corrected by K10.

## K9. Prescribed Radon/Abel profiles + independent scalar least squares

**Setup/status.** H05 known target→analytic/Abel ridge profiles. Convert profiles to tanh independently; common geometry permits one shared scalar SVD and many RHS. Never fits multidimensional target values. Four radial targets and composition tested. Abel quadrature order96 supplies profiles where closed forms unavailable.

**Cost.** Shared scalar factorization O(LH²), apply to M profiles O(LHM); peak O(LH+LM+HM+H²) in actual all-RHS batch. L≈8H→O(H³+MH²) and O(H²+MH). With different geometry per direction would become O(MLH²), but tested implementation explicitly reuses one factorization. Add profile-generation cost: closed form O(MH); Abel O(MHq_Abel); Fourier atoms as K8/K10. Evaluation common.

**Result.** B4096,M32: Gaussian4.00e-15,Runge2.45e-15,waves1.37e-14,packet1.16e-14. Snapped composition6.98e-4. Shows removal of global solve wall while still using small scalar LS. No uniform asymptotic error law measured; scalar error floor reached at tested H64/128 for profiles.

**Evidence.** H05 `spoke_profiles/radon_prediction/forward_check/README.md`, `theory_forward_check.py` (one SVD for160 RHS).

## K10. Plane-wave angular interpolation + prescribed-profile scalar conversion

**Setup/status.** H05 repair of snapping: trigonometric cardinal interpolation of the plane-wave dependence on angle; signed analytical weights on64 oriented nodes /32 projective lines. Group Fourier atoms into real profile frequency sums; then K9 scalar conversion. No joint2D fit.

**Cost.** Angular weights O(MA), storage O(MA) in current dense table; profile generation O(MHL_rho) where L_rho distinct radial frequencies≤A; plus K9 shared SVD O(H³+MH²). These A/L_rho factors prevent a blanket O(B) claim. Evaluation common.

**Result.** Composition continuous3.37e-16; tanh2048features4.10e-15 and4096features3.41e-14. Floor-level on this one target; larger dictionary slightly worse due finite precision. Corrects snapped6.98e-4 without changing directions.

**Evidence.** `composition_angular_interpolation.py`; same forward README; `angular_interpolation.json`.

## K11a. Leading derivative QI coefficient samples (failed at broad width)

**Setup/status.** a_mj=(h w_m/2)q'_m(c_j). No smoothing inverse; anchor bias analytically. In3D q=−(Rf)''/(2pi), so readout proportional(Rf)'''. No fit.

**Cost.** O(MH C_profile), O(MH+Md) model storage, eval common. Derivative availability/profile-construction cost separate.

**Result.** Same geometry as corrected I04 Gaussian tests gives relative errors.293,.356,.326 (isotropic,anisotropic,shifted mixture); F19 anisotropic2D18.2%,3D22.7%. Valid leading small-width approximation, not floor method at tested gamma. Error saturates as only M rises because smoothing bias unchanged.

## K11b. Continuum complex-shift inverse + short finite center band (failed halo variant)

**Setup/status.** Exact continuum sech² smoothing inverse density rho(c)=Im q(c+i a)/a,a=pi/(2gamma); readout h w_m rho/2. Truncate on original H05 band with no finite-boundary correction. Profiles analytic in needed complex strip.

**Cost.** Same as K11a with complex evaluation cost. No fit. No additional PDE solution. Not a discrete cardinal inverse; lattice/contour/truncation errors remain.

**Result.** H05 B4096 radial targets:2.24e-7,7.03e-8,3.79e-6,2.02e-7. Failed floor because omitted tails, even though interior coefficient interpretation accurate.

## K11c. Complex-shift inverse + extended plain halo (successful direct construction)

**Setup/status.** Same coefficients, much longer center interval. H05 add80 centers each end; I04/F19 use201 centers over[-4,4] while evaluation on small cube/unit ball. No scalar/global LS.

**Cost.** O(MH C_profile +Md), model O(MH+Md), eval common. No fit. Requires known analytic profiles and adequate angular quadrature.

**Result.** H05 radial B9216:6.22e-15,3.32e-15,5.38e-14,1.91e-15. Extra halos necessary at that spacing. I04 three3D Gaussian-family targets with weighted sphere quadrature, H201: directions16/64/144/400/1024/2304; neurons3216/12864/28944/80400/205824/463104. Anisotropic errors3.99e-2/1.10e-3/2.40e-5/1.18e-8/5.28e-14/6.45e-15. Final isotropic4.24e-15,shifted mixture8.78e-15. Three cases reach1e-14 at adequateM; not broad-class guarantee.

**Evidence.** I04 `radon/report.md`, `radon.py`; H05 forward README; F19 `scaling/report.md`.

## K12. General-dimensional explicit Gaussian Radon profiles + tensor sphere cubature

**Setup/status.** F19 formula q_v(t)=det(B)^(-1/2)(v^T B^-1 v)^(-d/2) 1F1(d/2;1/2;−t²/(v^T B^-1 v)). Handles even d correctly (nonlocal filtered Radon); use K11c tanh encoding. Circle quadrature2D, GL×azimuth3D, recursive Gauss–Jacobi d4/d5. Number of antipodally reduced directions M=n_ang^(d−1). Dense/full-rank rotated Gaussian, not secretly low-dimensional. Known analytic target access.

**Cost.** O(d³+Md²+MH C_hypergeometric) plus quadrature-node setup and O(Md) enumeration; model O(MH+Md), eval common. C_hypergeometric may depend on d/arguments/precision, so not true uniform O(1). No fit/external solve. Untied model O(dMH) memory.

**Result.** d4,H201,M32768,B6586368 gives8.15e-15 on1024 fresh points; H121,B3964928 gave6.91e-15 on128-point diagnostic. d5,M331776,H81,B26873856 gives6.45e-11, H201,B66686976 still6.43e-11 angular-limited. d5 cheaperB13172736 independently validated1.19e-7. Largest d5 peak~2197.8MiB whole process, shared model521.4MiB,target; untied3561.5MiB. No5D strictfloor.

**Rate.** Conditional analytic/spectral-cubature model e_ang≈exp(-a M^(1/(d−1))), e_center≈exp(-bH); balanced B gives M∝B^((d−1)/d), H∝B^(1/d), error exp(-c B^(1/d)), with regularity-dependent constants. Not a theorem for arbitrary functions, QMC, or noisy profiles. Explicit angular curse remains.

## K13. Explicit profiles + Sobol-normal sphere directions (high-d, not high precision)

**Setup/status.** Same profile/encoding as K12, normalized inverse-normal Sobol samples instead of tensor cubature, two scrambles. Does not use learned geometry. M enumeration O(Md); same analytic preprocessing and neuron/eval costs.

**Result.** M16384,H201,B3293184 anisotropic medianerrors d4:.063%,d5:.079%,d8:.265%,d16:1.17%,d32:2.34%. Center conversion still1e-15–1e-14. No floor and no empirical asymptotic exponent established. Do not apply tensor-spectral rate toQMC.

## K14. Known-metric adapted directions + explicit profiles (high-d geometry control)

**Setup/status.** Take known Gaussian precision root B^(1/2), transform uniform direction u→B^(1/2)u/|B^(1/2)u|, change profile scale accordingly. Pure analytical geometry adaptation; not learning an unknown manifold. Same M/H asK13.

**Cost.** O(d³+Md²+MH C_profile), model/eval common. No fit; supplied target precision matrix is essential information.

**Result.** d32 .44–.61%, d64 .46–.57%, two scrambles, B3293184. Improves rawQMC but not precision-floor solution. No measured neuron/error exponent.

## K15. Known sparse Fourier ridge support + analytical tanh encoding

**Setup/status.** Four supplied ridge directions and frequencies, exact Fourier inverse of sech² multiplier, H201 long-band centers. No angular quadrature or direction search. General-d ordinary tanh network.

**Cost.** O(sH+sd) construction from prescribed amplitudes/frequencies, modelO(sH+sd), evalO(E(sd+sH)); s4. No fitting, no PDE solve, requires actual sparse directional decomposition. Not a generic d-dimensional black-box method.

**Result.** s4,B804 achieves5.89e-16 at d256; d3,16,64 also tested floor. Demonstrates ambientd alone does not determine cost; does not demonstrate discovery of unknown directions.

## K16a. Naive sqrt(N) halo truncation (failed boundary shortcut)

**Setup/status.** F19 applied current halo count R=ceil(sqrtN) to complex-shift coefficients without implementing newer rational boundary correction. No fit. Cost asK11b.

**Result.** Sine2pi N128cells maxerror1.14e-4; N256cells3.15e-5. 3DGaussian withM1600,N129interior,R12 plainvalue error3.96e-5 versus corrected6.40e-16. A genuine failed method at machine-precision requirement. R alone is not boundary algorithm.

## K16b. Finite-contour QUILL + rational correction on existing sqrt(N) halo

**Setup/status.** Actual newer construction: complex-shift density, contour moments mu−nu at endpoints, explicit stable partial fractions, modify outer existing readouts, reanchor bias. No LS/SVD/matrix solve, no added neurons. Requires valid analytic tube; entire-profile default is valid only for entire profiles. Ninterior caller passesn_cells=N−1 and R=ceil(sqrtN); scalar diagnostic file also contains historical Ncells convention, explicitly documented.

**Cost (current code, not aspirational linear claim).** Let m≈R/2, q contour nodes, L_profiles profiles encoded together. Shared `leggauss(q)` current NumPy path conservatively O(q³) setup/O(q²) memory; base coefficients O(L_profiles H C_profile); contour projection O(L_profiles m q); stable fractions O(m³+L_profiles m²), since current code rebuilds elementary symmetric polynomials for everypole. Peak O(L_profiles H+L_profiles q+mq+L_profiles m+q²), plus modeldirections. Node/table setup can theoretically be cached but current encode call does it. For known M profiles L_profiles=M; shared polynomialbank encodes p+1 profiles once, independent ofM. Default dynamic entiretube at asymptoticN has k~sqrtN and q~N, so cannot label current construction strictly O(MN). Evaluation remains ordinarytanh common costs.

**Result.** Sine2pi at128cells correctedmax4.44e-16,256cells5.55e-16; highfrequency sin12pi at128cells relative2.19e-14,256cells2.11e-15; Gaussian128cells1.10e-16. Current3D anisotropicGaussian,M1600,N129interior,R12,B244800: heldoutvalue6.395e-16,gradient1.487e-15,Laplacian7.866e-15, versus naivehalo3.96e-5value. Budget ladder optimized: B6480value3.83e-5;18000value2.29e-8;46080value2.88e-10;84992value2.09e-13;244800value6.40e-16. At84992 derivative errors still~1e-12–1e-11. Valuefloor and PDEderivativefloor must remain separate.

**Rate.** Exact-arithmetic boundary component bound h exp(-lambda R²/4), not total network bound. Balancing angular/center errors tested; no universalbestlambda: scalar N32best.40, largerN.25; 3DLaplacianN257besttested.18. Fixedcountchoices can fail despite asymptotic theory.

**Evidence.** `quill_boundary.py`; `quill_review/boundary_method.md`, `boundary_metrics.json`, `boundary_validation.json`; `quill_sweep.py`, `allocation_optimized_selected.json`, `allocation_meta.json`.

## P1. Frozen space-time QI–Radon MLP + dense collocation Gauss–Newton

**Setup/status.** F18 direct precursor,3inputs(x,y,t), 2D unsteady incompressibleNS primitiveu,v,p, fixedgeometry, jointly solvedreadouts. NativePDE residual, no outsidePDErun. Derivatives actualtanh. DenseSVD everyGNstep; above6000columns augmentedQR thenSVDofR, same asymptotic solve but lowerconstants. Same-method coarse-to-fine warmstarts fit priornetwork; initialcoarserungzero. **Oracle exactsolution fit did select N/M geometry split**; this is oracle-informed experimentaldesign even though no oracleweights entered dynamicrun. Not fully autonomous setup.

**Cost.** Ffields3,S=F(B+1). O(K[QD dB+Q S²+S³]) with D derivativechannels fixedhere; memoryO(QDB+QS+S²). WithQ∝S: cubiccompute/quadraticRAM. Model/eval sharedgeometry common plusFreadouts. AugmentedQR removesstoredQfactor but notquadratic scaling.

**Result.** Bperfield256/512/1024/2048/4096: velocity1.8e-3/5.7e-6/8.2e-9/2.3e-9/1.1e-11; pressuretop8.6e-10; maxdiv6e-10. Bestsplit2048gives1.2e-10 ratherthan2.3e-9;4096N16gives9.1e-12. Neverstrict1e-14 onwholeclosedcube; localmidinterior1e-12–1e-13. ~2.5ordersperdoubling onmost ladder, pressurelags. Top~20minutes,~7GB,8192estimated25GBnotrun. SuccessfulnativePDE method butmemorywall andboundary-derivativefloor.

**Evidence.** `results/checkpoint_F_applications/expF18_ns_spacetime/expF18_results.md`; `experiments/expF18_ns_spacetime/{ridge3d.py,run.py}`. Oldreport says “no training,” meaning noAdam; catalogue should callGNfitting whatitis.

## P2. Learned latent chart + global Radon head (related manifold precursor)

**Setup/status.** I04 24D noiselesscurvedgraph manifold withtrueintrinsic3Dchart. Input-onlytanh autoencoder24→64→64→3,6000Adamsteps+LBFGS; frozen2048-featureFibonaccisphereRadonhead128directions×16centers, globalLS; per-targetbandwidth/cutoffselectedbyvalidation. Controlsoracletrue3D,PCA3D,ambient24D. ThisheadisLS,notanalyticalRadonreadouts.

**Cost.** Encodertraining K_A Q C_encoder +LBFGS(architecture-dependentmemory), plusG(Q,B), thenencoder+common3Dheadinference. NooutsidePDEsolver, but learnedcharttraining andglobalfit bothrequired. Modelincludesencoder; equalheadwidthdoesnotequalend-to-endcost.

**Result.** Learned2seeds/3targets1.51e-4–1.48e-3; oraclechart3.12e-7–1.70e-5; ambient24D2.40e-4–4.08e-3; PCA.107–.415andactualfoldingcollision. No floor. Usefulcontrolledchart+approximationtest, notcompletegeometricprecisionmethod.

**Evidence.** `results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/manifold/report.md`.

## Additional distinctions that should not be lost

- Analytical Fourier/Radon existence argument does not by itself give computable coefficients for an unknown target/PDE. Htheory explicitly says construction was initially an existencecertificate while actualreadouts camefromglobalLS.
- Target-dependentdataorigin, M/Nallocation, localgammah rule, smoothermeshmaps, rcondchange1e-13→1e-14, 25%collar, andsqrtNcorrectedhalos are substantivevariants but notstandalonePDEsolvers. Include notes undercorrectfamilies.
- The H05 SVD projection of a theoreticallyconstructedreadout into the joint retained coefficientspace is a diagnostic, not a moreefficient construction. It costs the originalglobalSVD and changesfloor-level functionlittle. Composition99.9913% squaredcoefficientdiscrepancy discarded, confirmingnonuniqueness. Donotrateitasseparatepracticalsolver.
- A proposed localpatch/tangent-space system +zero-sumlocalizedridgeprofiles +coarseglobalmesh was described inH04 but **not built there**. Do not reporttheoreticalintrinsicdimension scaling asmeasuredgeneralmanifoldalgorithm.
- No direct known-function construction here proves arbitraryprecision for noisydata, nonsmoothtargets, analyticprofileswithoutvalidcomplexextension, or arbitraryunknownPDEs. The floor successscope is the testsreported.

## K17. Analytic Gegenbauer ball-frame reconstruction of known ridges

**Setup/status.** Implemented representation/derivative audit in d=2,3,4, degrees p=4,6. A supplied normalized Gegenbauer ridge along one direction is distributed to a common angular frame using the exact sphere reproducing identity. Positive cubature is exact for the required degree in exact arithmetic; the one scalar polynomial profile is encoded by corrected QUILL (K16b). Export is a literal ordinary tanh MLP, with no polynomial evaluator or product gates in its forward call. This is capacity evidence, not an unknown PDE solve.

**Cost.** Product sphere rule M=(p+1)^(d−1). One profile bank costs C_enc(1,N,q), then angular weights O(Md+Mp), and ordinary explicit export O(dMH) time and memory. A shared representation can retain O(Md+MH). Generic construction of all degree≤p polynomial features is a different, larger map; do not price this single-ridge audit as the complete general polynomial operator. Evaluation common above. No fitting or external solve; known ridge formula is supplied.

**Result.** At N=257 interior centers, R=17: d2,p4,B1455 max value error4.44e-15; d3,p4,B7275 max value3.55e-15; d4,p4,B36375 max value1.86e-15. For p6 the corresponding values are1.22e-13 (B2037),1.51e-14 (B14259),5.86e-14 (B99813). Thus all three degree4 fields reach strict1e-14, none of the three degree6 fields do at that budget. Largest scaled derivative errors are approximately6.2e-15 to1.2e-13 across these N257 cases. N129 has values1e-13–1e-12 and sometimes derivative errors2e-11. The intentionally underresolved d4,p6 rule uses M64 rather than343 and gives0.817 max value error and3.37 worst scaled derivative error. The angular identity alone does not forgive an inadequate rule or finite scalar encoding error.

**Evidence.** `experiments/expF19_radon_direct_pde/ridge_frame_dimension_study.py`; `results/checkpoint_F_applications/expF19_radon_direct_pde/pure_mlp_repair/ridge_frame_dimensions.json`.

## Error-rate qualification for the coordinator

The universally safe bookkeeping here is E≤E_angular(M)+E_profile(N,lambda,R)+E_arithmetic, with additional target-transform error if the supplied profiles are themselves numerical. Fixed lambda can leave an aliasing floor. Naive halos often contribute exp(−cR), which becomes root-exponential exp(−c sqrt(N)) when R=sqrt(N); the corrected finite-contour construction has a different boundary bound. The exp(−cB^(1/d)) allocation model in K12 is explicitly conditional on spectral angular convergence and an exponentially convergent, adequately corrected scalar leg in the applicable pre-roundoff range. It is not a blanket description of every known-function method in this audit.
