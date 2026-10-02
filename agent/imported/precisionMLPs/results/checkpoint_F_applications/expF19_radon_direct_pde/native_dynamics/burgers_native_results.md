# Native cubic-kernel Burgers pilot — data-obvious numerical result; research implications pending Sam

## TL;DR

- A kernel coefficient vector evolves from the prescribed initial condition and the nonlinear PDE. Every RK4 stage evaluates the current kernel expansion and its derivatives; no previous solution, Fourier state, training loss, or readout solve enters that path.
- The method converges at approximately second order on this smooth viscous Burgers problem. The relative solution error reaches $1.97\times10^{-5}$ at 256 coefficients.
- This is cubic-spline collocation with an explicit local coefficient map. Each kernel has a finite ReLU$^3$ realization. It changes the activation from tanh and does not establish the requested higher-dimensional QUILL/Radon construction.
- The classical Fourier baseline is substantially more accurate at comparable solve cost. This pilot establishes native nonlinear evolution without a global readout solve, not a numerical advantage.

## Question / hypothesis

Can an explicitly constructed kernel representation remain the evolving PDE state, with a fixed local analysis map returning the nonlinear right-hand side to coefficients, rather than encoding a solution produced by another solver?

## Experiment design

We solve the unforced periodic initial-value problem

$$u_t+uu_x=\nu u_{xx},\qquad x\in[-\pi,\pi],\quad \nu=0.05,\quad u(x,0)=\sin x+0.2\cos(2x),$$

through $T=0.5$. The only problem data given to the native solver are this initial condition, viscosity, time horizon, periodic boundary condition, and numerical resolution.

**1. Fix the kernel geometry.** With $h=2\pi/N$ and $x_j=-\pi+jh$, use the centered cubic B-spline

$$\beta_3(s)=\begin{cases}(4-6|s|^2+3|s|^3)/6,& |s|<1,\\(2-|s|)^3/6,&1\leq |s|<2,\\0,&|s|\geq2.\end{cases}$$

The periodic kernel state is

$$u_a(x)=\sum_{j=0}^{N-1}a_j\sum_{\ell\in\mathbb Z}\beta_3\!\left(\frac{x-x_j}{h}-\ell N\right).$$

Only four terms contribute at a generic evaluation point. The $N$ coefficients $a_j$ are the evolving state. Geometry is fixed throughout the integration.

**2. Specify the analysis map explicitly.** At the nodes, $u_a(x_j)=(a_{j-1}+4a_j+a_{j+1})/6$. Write this three-point convolution as $Ba$. Its infinite-lattice inverse has coefficients

$$c_r=\sqrt3\rho^{|r|},\qquad \rho=-(2-\sqrt3).$$

This formula is verified directly: for $r\ne0$, $c_{r-1}+4c_r+c_{r+1}=0$ because $\rho^2+4\rho+1=0$; at $r=0$, $(c_{-1}+4c_0+c_1)/6=1$. Consequently $Bc$ is the unit impulse. Applying the same coefficients with wrapped indices gives the periodic inverse without constructing or inverting a matrix.

The implementation truncates at radius $R=28$:

$$[C_R f]_j=\sqrt3\sum_{r=-R}^{R}\rho^{|r|} f_{(j-r)\bmod N}.$$

The omitted coefficient mass is bounded by

$$\sum_{|r|>R}|c_r|=\frac{2\sqrt3|\rho|^{R+1}}{1-|\rho|}=1.23\times10^{-16}.$$

This is a fixed 57-tap convolution, implemented as a direct spatial convolution. Its coefficients depend on the kernel, not on the target values or the current solution. No dense matrix, factorization, SVD, least-squares solve, iteration to solve a readout system, or FFT is used in the native path. For coarse grids the stencil wraps around more than once; its size remains fixed as $N$ grows.

**3. Initialize from the initial condition.** Set $a(0)=C_R[u_0(x_j)]$. This applies the explicit analysis formula to prescribed initial samples. At off-grid points the resulting cubic expansion approximates $u_0$; it does not receive future-time solution values.

**4. Evaluate the current representation and differentiate it.** At each RK stage, compute the exact kernel derivatives at the nodes:

$$u_j=\frac{a_{j-1}+4a_j+a_{j+1}}6,\qquad (u_x)_j=\frac{a_{j+1}-a_{j-1}}{2h},\qquad (u_{xx})_j=\frac{a_{j-1}-2a_j+a_{j+1}}{h^2}.$$

These are derivatives of the actual spline expansion, not derivatives of an external grid solution. Form $g_j=-u_j(u_x)_j+\nu(u_{xx})_j$ and evolve

$$a'(t)=C_R g(a(t)).$$

Thus $B a'=g$ up to the explicit convolution truncation and floating-point roundoff. The nonlinear product is recomputed from the stage coefficient vector, including the intermediate RK states.

**5. Integrate and evaluate.** Classical RK4 advances the coefficients. The spatial ladder is $N\in\{16,32,64,128,256\}$; the time-step rule is $\Delta t\leq\min(10^{-3},0.12h^2/\nu)$. For these resolutions every run uses $\Delta t=10^{-3}$. Solution errors and physical PDE residuals are evaluated on 2,048 uniformly spaced points shifted off the collocation nodes. The relative error is $\|u_a-u_{\rm ref}\|_2/\|u_{\rm ref}\|_2$. The off-grid residual is $u_{a'}+u_a\partial_xu_a-\nu\partial_{xx}u_a$, using the coefficient ODE to evaluate $u_{a'}$.

**6. Construct the independent error meter after the native runs.** A separate Fourier pseudospectral RK4 code pads the spectrum to $2N$ before multiplying, removes the Nyquist mode, and truncates the product back to the retained modes. Reference runs at $N=512,\Delta t=10^{-4}$ and $N=768,\Delta t=5\times10^{-5}$ differ by $2.51\times10^{-15}$ in relative $L_2$. This is a numerical refinement check, not an exact-error certificate. The same Fourier algorithm at each native resolution and time step supplies the cost baseline. It is never called by the native initializer, RHS, time integrator, or evaluator.

The native solve and baseline timings include setup and evolution, use one CPU thread, and report medians of three repetitions. Native runs also collect 21 lightweight trajectory records. Reference construction and plotting are excluded. Inference timings use ten repetitions for local-kernel and FFT-grid evaluation and five for direct Fourier sums. These tiny CPU timings measure this implementation and have no GPU or asymptotic performance implication.

**Code & data**

```text
Run: .venv/bin/python experiments/expF19_radon_direct_pde/native_burgers.py
Focused checks: same command with --validate-only
Code: experiments/expF19_radon_direct_pde/native_burgers.py
Metrics: results/checkpoint_F_applications/expF19_radon_direct_pde/native_dynamics/burgers_native_metrics.json
Rows: results/checkpoint_F_applications/expF19_radon_direct_pde/native_dynamics/burgers_native_metrics.jsonl
Saved coefficient states: same directory, burgers_native_n{16,32,64,128,256}.npz
Predictions: same directory, burgers_native_predictions.npz
Figures: same directory, burgers_native_summary.png and burgers_native_checks.png
```

## Results

| Coefficients $N$ | Native relative $L_2$ | Off-grid residual RMS | Fourier relative $L_2$ | Native solve (ms) | Fourier solve (ms) |
|---:|---:|---:|---:|---:|---:|
| 16 | $9.89\times10^{-3}$ | $5.20\times10^{-2}$ | $7.93\times10^{-3}$ | 25.1 | 19.7 |
| 32 | $1.35\times10^{-3}$ | $9.69\times10^{-3}$ | $1.62\times10^{-4}$ | 24.8 | 20.6 |
| 64 | $3.19\times10^{-4}$ | $2.22\times10^{-3}$ | $9.08\times10^{-8}$ | 26.0 | 21.3 |
| 128 | $7.91\times10^{-5}$ | $5.43\times10^{-4}$ | $2.52\times10^{-13}$ | 28.2 | 24.8 |
| 256 | $1.97\times10^{-5}$ | $1.35\times10^{-4}$ | $2.50\times10^{-13}$ | 30.7 | 32.4 |

Beyond the coarsest resolutions, doubling $N$ reduces the native error by about four. The Fourier method reaches far lower error at comparable cost. Matching only the number of stored spatial coefficients therefore does not imply matching approximation power.

At $N=64$, halving the temporal step through approximately $0.04,0.02,0.01,0.005$ reduces the difference from the $0.0025$ trajectory by approximately factors of 14–17, consistent with fourth-order RK4. The production $0.001$ step is well below the temporal error scale observed in this check. This does not remove the cubic spatial error.

The explicit inverse matters: at $N=64$, retaining only its central tap gives relative error $0.775$; radius 2 gives $0.0548$; radius 6 reaches $3.53\times10^{-4}$; radii 12 and 28 both reach the spatial floor of approximately $3.19\times10^{-4}$. Radius zero is a deliberately incomplete inverse, not a meaningful alternative PDE method.

Evaluating the compact cubic expansion at 2,048 points takes approximately $0.14$–$0.15$ ms across this ladder. The direct Fourier sum costs $0.34$–$4.90$ ms, but a shifted uniform grid supports FFT evaluation in $0.015$–$0.018$ ms. Reporting only direct Fourier sums would misrepresent the baseline. Local support permits the native evaluator to use four active kernels per point; a literal dense neural evaluation would not share this cost.

### Figures

- **Native summary:** upper left compares the initial condition, refined reference, and native $N=64$ solution; upper right shows the error profiles for four native resolutions; lower left shows error against coefficient count with a second-order guide; lower right shows error against measured setup-plus-evolution time. The Fourier comparison makes the accuracy limitation visible.
- **Native checks:** the left panel shows the energy trajectories for the spatial ladder; the middle panel shows temporal refinement at fixed $N=64$; the right panel shows the explicit inverse-stencil ablation. Legends are outside the axes.

## Additional details

**Neural realization and its limitation.** With $\operatorname{ReLU}^3(s)=\max(s,0)^3$,

$$\beta_3(s)=\frac16\left[\operatorname{ReLU}^3(s+2)-4\operatorname{ReLU}^3(s+1)+6\operatorname{ReLU}^3(s)-4\operatorname{ReLU}^3(s-1)+\operatorname{ReLU}^3(s-2)\right].$$

On the bounded periodic domain, only finitely many image kernels are needed, so this expansion admits a finite one-hidden-layer ReLU$^3$ realization with tied linear combinations of coefficients. The production implementation evaluates the compact piecewise polynomial to avoid cancellation between large cubic terms. An independent literal ReLU$^3$ evaluator at $N=64$ agrees to $4.72\times10^{-11}$ on random points; that gap reflects cancellation in the literal form. This pilot is a native kernel solver with an exact algebraic neural realization, not a timing claim for a dense ReLU$^3$ implementation and not an experiment with the tanh activation.

**Implementation checks.** A random coefficient vector passes analysis/synthesis roundtrip at $8.88\times10^{-16}$. Node derivative formulas agree with arbitrary-point analytic-kernel evaluation to $4.27\times10^{-12}$ in absolute error; large random-vector second derivatives make this the largest absolute representation check. Finite-difference probes agree with analytic derivatives to $9.67\times10^{-9}$ relative error. The coefficient RHS reproduces the physical nodal RHS to $3.89\times10^{-16}$. Periodicity holds to $2.78\times10^{-14}$. Initialization, time evolution, and inference still pass when NumPy FFT, inverse, solve, pseudoinverse, least-squares, and SVD entry points are replaced by functions that raise immediately.

**Cost and scope.** With fixed radius $R$, each stage costs $O(RN)$ and stores $O(N+R)$ data; no $N\times N$ readout matrix exists. This favorable coefficient map follows from the specific uniform periodic cubic kernel, whose nodal convolution is uniformly well conditioned. It does not generalize automatically to ill-conditioned broad tanh features or irregular higher-dimensional Radon geometries. The spline's second derivative has only second-order consistency, which explains why an accurate coefficient map does not yield machine-precision PDE accuracy at modest width.

## Conclusions

For this one-dimensional nonlinear initial-value problem, an explicitly initialized kernel coefficient state can be evolved and evaluated directly without solution labels or a global readout solve. The measured accuracy and timing favor the classical Fourier baseline on this smooth periodic test.

## Open questions

Whether the tanh/QUILL analysis map supports similarly stable native evolution, and whether an analogous practical map exists for the intended higher-dimensional Radon representation, remain separate questions. This cubic pilot does not settle either one.
