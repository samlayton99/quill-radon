# Sparse-observation QUILL inverse pilot — data-obvious numerical result; general replacement claim unsupported

## TL;DR

- The actual QUILL forward model recovers viscosity and two initial-condition amplitudes from 24 sparse velocity observations. A refined noiseless run reaches $1.14\times10^{-11}$ relative viscosity error and $2.81\times10^{-13}$ relative error at withheld spatial points.
- Measurement noise affects parameter recovery more strongly than field reconstruction. Across five seeds, $0.1\%$ noise gives median viscosity error $1.21\%$ and withheld field error $0.0328\%$; $1\%$ noise gives $12.1\%$ viscosity error and $0.324\%$ field error.
- Initial-condition observations alone contain exactly no viscosity information. Three different initial viscosity guesses remain unchanged while fitting the observations to tiny residuals, and later predictions disagree.
- This is a three-parameter PDE-constrained inverse problem with a prescribed initial-condition family. It does not test arbitrary missing fields, unknown forcing functions, unknown geometry, or general PINN replacement.

## Question / hypothesis

Can sparse measurement anchors identify a few physical unknowns using the existing QUILL evolution, while avoiding a large neural readout fit? The test separates fitting the measured values from predicting unobserved locations and times, and includes data that make one parameter unidentifiable.

## Experiment design

The forward equation is periodic viscous Burgers on $[-\pi,\pi]$:

$$u_t+uu_x=\nu u_{xx},\qquad u(x,0)=\alpha\sin x+\beta\cos(2x).$$

The three unknowns are $p=(\nu,\alpha,\beta)$. Synthetic truth is $(0.05,1,0.2)$, and every identifiable fit starts from $(0.09,0.8,0.05)$. The initial-condition family and PDE form are assumed known. Bounds are $\nu\in[0.005,0.2]$, $\alpha\in[0.4,1.6]$, and $\beta\in[-0.5,0.7]$; there is no penalty or statistical prior. The bounds keep the tested forward trajectories in a smooth, numerically stable regime.

**Actual QUILL forward model.** The existing construction supplies an explicit encoding matrix $E$, mapping retained real trigonometric coordinates $b$ to tanh network readouts $\theta=Eb$. It uses $\lambda=0.25$, the rational boundary correction, and square-root halo scaling. If $\Phi$, $\Phi_x$, and $\Phi_{xx}$ are the actual tanh features and their analytic derivatives on the quadrature grid, define

$$U=\Phi E,\qquad U_x=\Phi_x E,\qquad U_{xx}=\Phi_{xx}E.$$

These products are computed from the neural feature matrices. They are not replaced by ideal sine/cosine tables. With real trigonometric analysis matrix $A$, the compressed forward equation is

$$b'=A\left[-(Ub)\odot(U_xb)+\nu U_{xx}b\right].$$

This is the same neural dynamics as $\theta'=EA F(\theta)$, restricted to its invariant image of $E$ and evaluated by matrix reassociation. It remains a modified spectral method in QUILL coordinates. Compression reduces redundant state; it does not remove the trigonometric analysis map or create a general neural PDE solver.

The main runs use maximum frequency $K=32$, 513 interior centers, 559 tanh neurons including halos, 65 reduced coordinates, and 132 quadrature points. RK4 uses a maximum step $0.002$. A noiseless refinement uses $K=64$, 1,025 interior centers, 1,091 neurons, 129 reduced coordinates, and step $0.001$. The shared construction's example initial coefficient vector is discarded; each inverse trajectory initializes only from its current $(\alpha,\beta)$.

**Observations and withheld data.** Eight fixed spatial locations are

$$x_j=-\pi+2\pi(j+0.37)/8,\qquad j=0,\ldots,7.$$

Each location is observed at $t=0.15,0.30,0.50$, giving 24 scalar measurements. The data generator is a separate dealiased Fourier RK4 solver with 384 grid points and step $10^{-4}$. It is used to simulate measurements, not to provide the inverse solver with a solution field. A second reference with 512 points and half the step differs by $1.26\times10^{-15}$ relative $L_2$ at $t=0.7$.

The fitting function accepts only the sensor positions, observation times, observed scalar values, and the fixed QUILL forward model. Independent dense reference fields are used after fitting returns. Scores use 1,024 off-grid spatial points at the observation times and a separately withheld future time, $t=0.7$. These withheld points are absent from the objective.

**Noise.** Measurements are either exact or receive independent zero-mean Gaussian noise with standard deviation $0.001$ or $0.01$ times the RMS of the clean 24 observations. Each noisy level has five deterministic seeds. The reported medians and ranges are descriptive across these five draws, not confidence intervals.

**Only three physical parameters are optimized.** The objective is

$$\min_{p}\ \frac12\sum_{i,j}\left[u_p(x_j,t_i)-y_{ij}\right]^2.$$

SciPy's bounded trust-region least-squares optimizer receives a $24\times3$ sensitivity matrix. Its small parameter solve is allowed here: it is not a neural readout solve. There are no trainable geometry parameters or free per-neuron readouts. Function and Jacobian evaluations at the same parameter vector share one cached forward/tangent integration; at most 40 such optimizer evaluations are allowed.

**Tangent sensitivities.** For a parameter $p_r$, write $z_r=\partial b/\partial p_r$. The integrated tangent equation is

$$z_r'=A\left[-(Uz_r)\odot(U_xb)-(Ub)\odot(U_xz_r)+\nu U_{xx}z_r+\mathbf1_{p_r=\nu}U_{xx}b\right].$$

The viscosity tangent starts at zero; the amplitude tangents start at the analyzed $\sin x$ and $\cos(2x)$ profiles. RK4 advances the state and all three tangent vectors together. Thus the main augmented state has shape $65\times4$, and the refined state has shape $129\times4$.

**Unidentifiable control.** Eight noiseless measurements at $t=0$ replace the three later observation times. Such observations depend on $\alpha$ and $\beta$ but not on $\nu$. Three fits start with viscosity $0.02$, $0.09$, and $0.17$, keeping the same initial amplitude guesses. The singular values of the small physical observation Jacobian diagnose the missing information.

**Code & data**

```text
Run: .venv/bin/python experiments/expF19_radon_direct_pde/native_inverse.py
Code: experiments/expF19_radon_direct_pde/native_inverse.py
Results: results/checkpoint_F_applications/expF19_radon_direct_pde/capabilities/inverse_metrics.json
Withheld reference and refined predictions: same directory, inverse_withheld.npz
Figure: same directory, inverse_summary.png
```

## Results

| Measurement noise | Median relative viscosity error | Median withheld spatial field error | Median future-time field error |
|---|---:|---:|---:|
| None, $K=32$ | $2.06\times10^{-6}$ | $4.71\times10^{-8}$ | $6.62\times10^{-6}$ |
| $0.1\%$ RMS, five seeds | $1.21\%$ | $0.0328\%$ | $0.0708\%$ |
| $1\%$ RMS, five seeds | $12.1\%$ | $0.324\%$ | $0.717\%$ |
| None, refined $K=64$ | $1.14\times10^{-11}$ | $2.81\times10^{-13}$ | $1.97\times10^{-10}$ |

The refined noiseless parameter estimate is approximately $(0.050000000000572,1.000000000000138,0.200000000000118)$. The improvement over $K=32$ shows that the coarser noiseless inverse was limited by forward discretization, rather than by the three-parameter optimizer. The weaker future-time accuracy reflects stronger later spatial structure; that time was absent from fitting.

At $0.1\%$ noise, relative viscosity errors range from $0.122\%$ to $2.39\%$ across five draws. At $1\%$ noise, they range from $2.29\%$ to $23.6\%$. The two initial amplitudes are more accurately recovered: at $1\%$ noise their median relative errors are $0.110\%$ and $1.36\%$. Small field error therefore does not imply equally accurate physical parameters.

The principal fits require 8–14 forward-plus-tangent trajectories. Geometry and compressed-operator setup takes approximately 4 ms, followed by 0.15–0.26 s for a fit including sensor evaluation setup. The refined noiseless case takes approximately 11 ms setup and 0.52 s fitting, using eight forward-plus-tangent trajectories and 16,000 augmented RHS evaluations. These are single-thread CPU measurements, with setup charged separately; observation generation, finite-difference checks, withheld scoring, and plotting are excluded from fit times. No classical inverse solver or PINN timing baseline was run, so no performance superiority follows.

For initial-condition-only data, the physical Jacobian has singular values $(2,2,0)$ and rank two. Each fitted viscosity stays at its initial guess. Nevertheless, sensor residual RMS is below $3\times10^{-11}$, and two of the three optimizers report success. One reaches the 40-evaluation cap. At $t=0.7$, the three predicted fields have relative errors of $4.56\%$, $4.90\%$, and $12.6\%$. More optimization cannot recover information absent from these observations.

### Figure

- **Inverse summary:** the left panel shows all parameter-error draws versus measurement-noise level, with color identifying the physical parameter; the middle panel compares withheld spatial and future-time field errors; the right panel compares observation-Jacobian singular values for the three-time and initial-only designs. The zero singular value is displayed at the plotting floor and labeled as exactly zero. Noiseless points in the first two panels use the main $K=32$ model; the refinement is reported in the table.

## Additional details

The integrated sensitivity columns agree with central finite differences to relative errors $1.01\times10^{-10}$, $2.02\times10^{-11}$, and $4.46\times10^{-11}$. A random-state check compares the compressed RHS against the original full tanh-feature RHS and finds a $3.45\times10^{-16}$ relative difference. These checks verify tangent implementation and the claimed algebraic compression. They do not certify global inverse uniqueness.

The three-time physical Jacobian has singular values approximately $(3.62,3.37,0.511)$ at the refined solution. Its nonzero smallest singular value supports local identifiability for this sensor design and parameterization. Singular values depend on parameter units; the direct zero-viscosity column in the initial-only experiment is the stronger structural result.

The test assumes the correct PDE, a known periodic domain, known boundary conditions, and a two-dimensional family for the unknown initial field. A real inverse problem may violate any of those assumptions. Unknown forcing, nonparametric initial conditions, model mismatch, missing velocity components, irregular geometry, and long-horizon instability were not tested. The fit still needs a differentiable numerical forward solve and an optimizer over the physical unknowns. The absence of a large neural readout fit does not remove inverse ill-posedness.

## Conclusions

Sparse measurements can anchor this QUILL forward model well enough to recover three locally identifiable physical parameters, with noise-dependent errors verified at unobserved points and a future time. The deliberately unidentifiable case demonstrates that the representation provides no general cure for missing physical information.

## Open questions

The next consequential choice is which inverse problem class matters: a few physical parameters, an unknown initial field, an unknown forcing function, or incomplete boundary data. This experiment supports only the first class under a correct model and an informative sensor layout.
