# Shocks, stiff reaction, and finite-time blow-up — tested backend extension; general accuracy not certified

## TL;DR

- A conservative finite-volume backend handles Burgers shocks and rarefactions. Refinement reduces integrated error and preserves the mass balance to approximately machine precision. Pointwise error at a jump stays large.
- An actual QUILL reaction–diffusion backend uses compressed neural coordinates and implicit BDF time integration. A stiff Allen–Cahn test improves from $2.63\%$ relative error at $K=32$ to $2.82\times10^{-5}$ at $K=128$. The finest comparison is limited by the independent reference.
- The quadratic-source equation develops a genuine finite-time singularity. The solver returns finite states with explicit amplitude-limit or underresolution status. Threshold times converge under bandwidth and time-tolerance refinement, but are not labeled as the singularity time.
- `reached_t_end` means the time integrator finished, not that the PDE solution is accurate. Coarse runs without an invariant guard violate the maximum principle or positivity. The added optional invariant guard stops these runs at the requested tolerance.

## Question / hypothesis

Which numerical mechanisms are required when the smooth periodic pilot is extended to discontinuities, thin reaction layers, and unbounded growth? The implementation uses a weak conservative backend where shocks require one, and keeps the actual QUILL construction in the stiff smooth branch.

## Experiment design

### Conservative shock backend

The finite-volume state consists of cell values $v_j$ on a uniform one-dimensional mesh. For Burgers flux $f(v)=v^2/2$, the Rusanov interface flux is

$$\widehat f(v_L,v_R)=\frac{f(v_L)+f(v_R)}2-\frac{\max(|v_L|,|v_R|)}2(v_R-v_L).$$

The coefficient evolution is conservative:

$$v_j'=-\frac{\widehat f_{j+1/2}-\widehat f_{j-1/2}}h.$$

Three-stage SSPRK3 advances this system with CFL $0.4$. Periodic and outflow boundary conditions are supported. Mass diagnostics subtract the integrated boundary flux; entropy diagnostics use $\eta(v)=v^2/2$, entropy flux $q(v)=v^3/3$, and the associated local Lax–Friedrichs numerical entropy flux. Boundary integrals use the same Runge–Kutta stage weights as the update.

This is a conventional finite-volume entropy method. It is not a tanh solver, and it does not differentiate a smooth neural field to propagate a shock. The reconstruction is piecewise constant. Its exact primitive is a finite ReLU network,

$$M(x)=\sum_j v_j\left[(x-e_j)_+-(x-e_{j+1})_+\right],$$

where $e_j$ are cell edges. Almost everywhere, $M'(x)$ is the represented field. The API returns these knots and coefficients directly from the evolving state. This algebraic neural realization does not make the numerical method novel.

Three problems on $[-1,1]$ use $64,128,256,512,1024$ cells:

- **Moving shock:** initial left/right states $1,0$, outflow boundaries, $T=0.4$. The exact entropy shock moves at speed $(1+0)/2=1/2$, reaching $x=0.2$.
- **Rarefaction:** initial states $-1,1$, outflow boundaries, $T=0.4$. The exact solution is $u(x,t)=\operatorname{clip}(x/t,-1,1)$.
- **Shock formation from smooth data:** $u_0(x)=-\sin(\pi x)$, periodic boundaries, $T=0.5$. The first shock forms at $1/\pi$. For $x>0$, the entropy characteristic comes from the root $y\in[x,1]$ of $x=y-t\sin(\pi y)$, giving $u=-\sin(\pi y)$; the negative branch follows by symmetry. This selects the entropy solution after the stationary shock forms at zero.

Errors use 16,384 off-grid points and additional probes immediately beside each shock. The reported $L_1$ is the domain integral of absolute error, not the mean. The pointwise probes ensure that shrinking transition width cannot hide the remaining maximum error.

### Actual QUILL stiff backend

The existing corrected QUILL construction encodes fixed entire trigonometric profiles with $\lambda=0.25$ and square-root halo scaling. If $E$ maps reduced coefficients $b$ to network readouts $\theta=Eb$, the cached matrices are built from actual tanh features:

$$U=\Phi E,\qquad U_{xx}=\Phi_{xx}E.$$

For reaction $R(u)$, the forward system and its analytic Jacobian are

$$b'=A\left[\nu U_{xx}b+R(Ub)\right],\qquad J(b)=A\left[\nu U_{xx}+\operatorname{diag}(R'(Ub))U\right].$$

Here $A$ is explicit real trigonometric quadrature with $Q=4K+4$ points. This supports the cubic reaction at the retained ideal trigonometric bandwidth; the actual neural tails introduce the same small additional projection effect as in the earlier native pilot. Matrix reassociation compresses the invariant neural subspace. It does not replace the tanh feature products by ideal trigonometric derivatives.

SciPy BDF advances the stiff state. Its implicit Newton systems have dimension $2K+1$; these physical state solves are explicitly present. There is no least-squares neural readout fit. Their dense storage and factorization cost matter at larger $K$.

The backend monitors the maximum absolute value on its quadrature grid and the relative coefficient norm in the highest quarter of retained frequencies. It stops at an amplitude limit or when that tail norm reaches $5\%$. The tail condition is a heuristic warning of underresolution, not a certified error bound. Returned diagnostics also record a denser-grid minimum and Allen–Cahn maximum-principle excess. Those diagnostics can expose an invalid coarse approximation even when the tail threshold has not fired.

An optional `invariant_tolerance` adds a terminal physical-invariant check on the union of quadrature and output points. Allen–Cahn bounds are $[\min(-1,\min u_0),\max(1,\max u_0)]$; nonnegative initial data for the quadratic-source equation have lower bound zero. This option defaults to `None`, leaving the original tail-only runs reproducible as an ablation. The guarded demonstrations set it to $10^{-3}$ in the units of $u$. Applications should choose their tolerated invariant violation explicitly. Monitoring finitely many points is not a certificate between them.

**Stiff layer test.** Allen–Cahn uses

$$u_t=0.1u_{xx}+100(u-u^3),\qquad u(x,0)=0.3\sin x+0.05\cos(2x),\quad T=0.1.$$

The bandwidth ladder is $K=16,32,64,128,256$, with $16K+1$ interior centers before adding halo neurons. BDF tolerances are $10^{-8}$ relative and $10^{-10}$ absolute. A same-bandwidth time check at $K=128$ tightens both tolerances by 100 and halves the maximum step. An independent classical reference uses periodic finite-difference heat evolution, diagonalized by FFT, and exact pointwise reaction in Strang splitting. Its two resolutions are 8,192 points with step $5\times10^{-5}$ and 16,384 points with step $2.5\times10^{-5}$. These references are generated after all native runs and are not inputs to the native solver.

**Finite-time blow-up.** The native backend also solves

$$u_t=0.05u_{xx}+u^2.$$

For the homogeneous initial value $u_0=1$, the exact solution is $u=1/(1-t)$, with singularity time one. The threshold $u=1000$ is reached at exactly $t=0.999$; this checks event detection under time-tolerance refinement. The nonconstant test uses $u_0=1+0.2\cos x$, bandwidths $16,32,64,128$, and amplitude thresholds $100,1000,10000$. The requested end time is $1.05$, past any possible classical solution of this positive-data problem, so correct behavior requires stopping. A tighter time check at $K=128$ and threshold 10,000 uses tolerances $10^{-11},10^{-13}$ and maximum step $0.005$.

**Code & data**

```text
Run: .venv/bin/python experiments/expF19_radon_direct_pde/general_hard.py
Conservative API: experiments/expF19_radon_direct_pde/solver/hyperbolic.py
Native stiff API: experiments/expF19_radon_direct_pde/solver/reaction.py
Metrics: results/checkpoint_F_applications/expF19_radon_direct_pde/general_solver/hard_metrics.json
Reaction fields: same directory, hard_reaction_fields.npz
Figures: same directory, hard_hyperbolic_profiles.png,
         hard_hyperbolic_refinement.png, hard_reaction_and_blowup.png,
         hard_guard_comparison.png
Guard demonstration metrics: same directory, hard_invariant_guards.json
```

Both callable APIs return `x`, `values`, `time`, `times`, `solution`, `history`, `status`, and `metrics`. The stiff solver distinguishes `reached_t_end`, `amplitude_limit`, `underresolved`, `invariant_violation`, and `integrator_failure`. It returns the last finite state and its actual time; it does not fill later requested times with fabricated values.

## Results

### Shocks and rarefactions

| Case | $L_1$, 64 cells | $L_1$, 1024 cells | Maximum error, 1024 cells |
|---|---:|---:|---:|
| Moving shock | $0.0293$ | $0.00185$ | $0.513$ |
| Rarefaction | $0.0721$ | $0.0104$ | $0.0500$ |
| Smooth data after shock formation | $0.0642$ | $0.00405$ | $0.425$ |

The two shock cases show approximately first-order integrated convergence. Maximum errors next to jumps remain large as their numerical transition regions narrow. Rarefaction also converges, more slowly at its corners. Mass-balance residuals stay below $1.7\times10^{-15}$, and all measured entropy-balance defects are nonpositive. In the moving-shock case the defect approaches the exact dissipative value $-T/12=-1/30$; entropy is not supposed to remain constant across that shock.

### Stiff Allen–Cahn: tail-only ablation

| Retained frequency $K$ | Outcome | Relative difference from independent reference | Maximum $|u|$ |
|---:|---|---:|---:|
| 16 | Underresolved at $t=0.0486$ | Not scored at $T$ | $1.078$ at stopping time |
| 32 | Reached $T$ | $2.63\times10^{-2}$ | $1.035$ |
| 64 | Reached $T$ | $2.75\times10^{-3}$ | $1.00373$ |
| 128 | Reached $T$ | $2.82\times10^{-5}$ | $1.000029$ |
| 256 | Reached $T$ | $5.62\times10^{-7}$ | $0.999999993$ |

The two independent reference resolutions differ by $1.68\times10^{-6}$, so the finest row is reference-limited; it is not a certified $5.62\times10^{-7}$ error against the exact PDE. Tightening the BDF tolerances at $K=128$ changes the field by only $1.29\times10^{-9}$, separating temporal error from the much larger spatial error there.

Because the initial field lies inside $[-1,1]$, the exact Allen–Cahn solution remains inside that interval. The coarse overshoots are genuine approximation defects. In particular, $K=32$ finishes with a $3.5\%$ overshoot even though its coefficient tail remains below the configured threshold. A completion status and a heuristic tail test cannot replace resolution checks or physical invariant diagnostics.

Native Allen–Cahn setup and evolution range from a few milliseconds at the coarsest resolution to approximately $0.5$ seconds combined at $K=256$ on one CPU thread. BDF takes roughly 230–260 accepted steps on the completed runs and performs 44–48 implicit factorizations, with state dimensions from 65 to 513. These timings describe the implementation; they establish no advantage over a classical solver.

### Blow-up stopping and refinement: tail-only ablation

The exact homogeneous threshold-time errors are $4.04\times10^{-5}$, $1.24\times10^{-6}$, and $1.87\times10^{-8}$ as BDF relative tolerance is reduced through $10^{-6},10^{-8},10^{-10}$. All runs stop at amplitude 1000 with finite returned states.

For nonconstant positive data, representative stopping results are:

| $K$ | Requested amplitude limit | Outcome | Returned time |
|---:|---:|---|---:|
| 16 | 1000 | Underresolved near amplitude 302 | $0.835771$ |
| 32 | 1000 | Amplitude limit | $0.838018$ |
| 64 | 1000 | Amplitude limit | $0.8380002$ |
| 128 | 1000 | Amplitude limit | $0.83799999$ |
| 32 | 10000 | Underresolved near amplitude 1216 | $0.838202$ |
| 64 | 10000 | Underresolved near amplitude 4958 | $0.838809$ |
| 128 | 10000 | Amplitude limit | $0.83890571$ |

At $K=128$, tightening the time tolerances changes the last threshold time by $2.89\times10^{-8}$. Raising the amplitude threshold moves the stopping time closer to the singularity, but none of these times is the exact singularity time. The optional local-reaction proxy $t+1/\max u$ is recorded as a diagnostic only; diffusion and spatial error prevent treating it as a certificate.

The $K=64$ run targeting amplitude 10,000 has a minimum value approximately $-0.128$ when the tail guard fires, although the exact solution stays positive. This is an additional failure of the coarse spatial representation. The returned `underresolved` status warns that this state should not be used as a valid continued PDE solution. The $K=128$ run at that threshold remains positive on the scoring grid.

### Enabling the physical-invariant guard

With absolute invariant tolerance $10^{-3}$, the $K=32$ Allen–Cahn run stops at $t=0.0399328$ with `invariant_violation`, before the unguarded run accumulates a $3.5\%$ overshoot. The $K=64$ quadratic-source run targeting amplitude 10,000 stops at $t=0.83880433$, before the unguarded tail detector permits a minimum of $-0.128$. Both monitored violations equal the requested $10^{-3}$ tolerance at stopping. This converts the observed defects into usable stopping behavior; it does not make either coarse approximation accurate.

### Figures

- **Hyperbolic profiles:** exact entropy solutions and three mesh resolutions are shown for a moving shock, rarefaction, and a shock formed from smooth data. Refinement narrows the artificial transition regions.
- **Hyperbolic refinement:** integrated errors, maximum errors, and entropy-balance defects are plotted against cell count. The distinction between weak convergence and pointwise accuracy at jumps is visible.
- **Reaction and blow-up:** the left panel shows the sharp Allen–Cahn layers, the middle panel compares bandwidth refinement with the independent-reference floor, and the right panel zooms into inverse peak amplitude near blow-up. Endpoint markers distinguish underresolution stops from amplitude stops.
- **Invariant guards:** two panels compare output-grid bound violations at the returned states with and without the additional physical-invariant guard. The dotted line is the configured tolerance. Guarded and unguarded runs stop at different times because early termination is the intended behavior.

## Additional details

**A rigorous finite-time bracket for the nonconstant PDE.** While the positive periodic classical solution exists, its spatial mean $m(t)$ satisfies

$$m'(t)=\operatorname{mean}(u^2)\geq m(t)^2.$$

The Laplacian integrates to zero, and Jensen's inequality gives the final step. Since $m(0)=1$, comparison with $m'=m^2$ gives the upper bound $T_*\leq1$. Conversely, the spatially constant solution starting from $\max u_0=1.2$ is a supersolution until its pole, so the classical PDE solution cannot blow up before $1/1.2$. Thus

$$\frac{1}{1.2}\leq T_*\leq1.$$

This analytic bracket establishes genuine finite-time blow-up independently of the numerical threshold detector. It does not identify $T_*$ to the many digits displayed for numerical stopping times.

**Limits of the extension.** The shock backend changes the discretization and representation. The stiff backend keeps actual QUILL features but still uses a trigonometric analysis map and dense implicit state solves. Neither path is a universal solver for arbitrary dimensions, geometry, discontinuous coefficients, or coupled systems. A nonlinear PDE may lose classical regularity, admit multiple weak continuations, or blow up; the API must expose those distinctions rather than promise a smooth network for every requested future time.

## Conclusions

The implementation can handle these harder cases by matching the numerical backend to the equation and returning explicit stopping information. The tests support conservative weak evolution for shocks and refined QUILL evolution for stiff smooth fields; they also expose accuracy and physical-invariant failures that a successful time-integration status alone would miss.

## Open questions

The remaining general-solver problem is reliable dispatch and validation: which equations admit the available backends, what physical invariants should be checked, and when refinement or a change of representation is required. A heuristic tail cutoff is insufficient as that policy by itself.
