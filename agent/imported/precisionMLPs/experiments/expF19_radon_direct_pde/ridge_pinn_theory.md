# Physics-informed readout discovery in an actual flat ridge MLP

The construction now gives a useful PDE parameterization, not a formula for
the unknown PDE solution. It builds a fixed, explicit map from a small set of
meaningful function coordinates to ordinary tanh readouts. We then find those
coordinates by imposing the PDE on the actual neural network. No polynomial
PDE solution is computed first, and no hidden product gates are present.

## 1. A complete function space made from one-dimensional profiles

Let B be the unit disk in R², with ordinary area measure. Let U_n be the
degree-n Chebyshev polynomial of the second kind:

U_0(t)=1, U_1(t)=2t, U_(n+1)(t)=2t U_n(t)-U_(n-1)(t).

For each degree n, choose unit vectors
v_(n,k)=(cos(kπ/(n+1)),sin(kπ/(n+1))), k=0,...,n. The scalar functions

    ψ_(n,k)(x)=U_n(v_(n,k)·x)/√π

form an orthonormal basis of the degree-n polynomials orthogonal to all lower
degrees. Consequently, all pairs 0≤k≤n≤p give an orthonormal basis of the
polynomials of total degree at most p. Its dimension is
P=(p+1)(p+2)/2. These are the classical Logan–Shepp ridge polynomials; this is
not a new polynomial-spanning claim. See
[Waldron's account](https://www.math.auckland.ac.nz/~waldron/Preprints/Disc-Polys/disc-polys.html)
and [Petrushev's ridge/Radon construction](https://people.math.sc.edu/pencho/Publications/PP-Ridge-func-1998.pdf).

The identity behind the normalization is

    ∫_B U_n(v·x) U_n(w·x) dx = π U_n(v·w)/(n+1).

At v=w, U_n(1)=n+1, so the norm squared is π. At distinct selected directions,
U_n(cos((k−l)π/(n+1)))=0, so the cross terms vanish. Different degrees are
orthogonal. One way to see the latter is to rotate v to the x-axis and integrate
a lower-degree monomial x^a y^b across the disk. Odd b gives zero; b=2j gives
a constant times the univariate integral of U_n(x) against
sqrt(1−x²) x^a(1−x²)^j. This polynomial factor has degree a+2j<n, so
Chebyshev orthogonality makes the integral zero.

## 2. Put every basis function on one common direction grid

Choose M directions ω_m=(cos θ_m,sin θ_m), θ_m=mπ/M. They are lines modulo
sign, so the interval is [0,π), not the full circle. If M≥p+1, then

    U_n(v·x) = (1/M) Σ_m U_n(ω_m·v) U_n(ω_m·x),   0≤n≤p.

Here is why the finite formula is exact. Write v=(cos φ,sin φ). The kernel
U_n(cos(θ−φ)) is the finite Fourier sum

    Σ_(j=0)^n exp(i(n−2j)(θ−φ)).

The function θ↦U_n(ω_θ·x) has only those same frequencies. Multiplication and
integration over θ therefore reproduces its value at φ. Their product has
only even frequencies 2q with |q|≤n. On the stated M-point grid,

    (1/M) Σ_(m=0)^(M−1) exp(2π i q m/M) = 0,  0<|q|≤n<M.

Thus this quadrature reproduces the integral exactly. This proof also tells us
what insufficient angular resolution loses: degree-n angular content has
n+1 independent components, while M<n+1 directions cannot span them all.

Define the known M-by-(n+1) matrix

    C_n[m,k]=U_n(ω_m·v_(n,k))/(M√π).

For an unknown coefficient vector a∈R^P, the known angular map turns its
coordinates into M unknown univariate profiles:

    g_m(t)=Σ_(n,k) C_n[m,k] a_(n,k) U_n(t).

The corresponding polynomial function is Σ_m g_m(ω_m·x). We never solve the
PDE using that exact polynomial expression in the QUILL experiments.

## 3. Turn these coordinates into ordinary tanh readouts explicitly

For each KNOWN profile U_n, the corrected QUILL formula constructs real
numbers β_n and W[j,n] such that

    Uhat_n(t)=β_n+Σ_j W[j,n] tanh(γ(t−c_j)).

The construction is linear in the supplied profile. There are N interior
centers, h=2/(N−1), γ=λ/h, and R=ceil(sqrt(N)) halo centers per side. Thus each
direction has L=N+2R neurons. With τ=π/(2γ), the uncorrected interior readout
is explicitly

    W_base[j,n]=(h/2) Re{[U_n(c_j+iτ)−U_n(c_j−iτ)]/(2iτ)}.

The implemented contour-moment corrections modify the halo readouts, and an
anchor determines β_n. No least-squares fit is used to construct W. These
polynomials are entire, so complex profile evaluations do not require querying
an unknown solution or crossing a solution's unknown singularities.

Define E∈R^(ML×P) by

    E[(m,j),(n,k)] = W[j,n] C_n[m,k].

Then w=Ea gives the ML actual readouts. The output bias is the corresponding
linear combination of β_n. The n=0 coordinate is implemented directly as an
output bias. For every value of a, the model is exactly of the form

    u_a(x)=b(a)+Σ_(m,j) w_(m,j)(a) tanh(γ(ω_m·x−c_j)).

This is Linear→Tanh→Linear. There are no products at evaluation time. The
unknowns are P coordinates a, rather than ML unconstrained readouts. That
restriction is deliberate: it selects combinations of neurons that implement
a complete, controlled low-degree approximation space.

On a physical disk of radius r and midpoint x0, replace x by (x−x0)/r. Every
order-|α| derivative is evaluated as

    Σ_(m,j) w_(m,j) (γ/r)^|α| ω_m^α
             tanh^(|α|)(γ(ω_m·(x−x0)/r−c_j)).

The code caches these actual neural basis jets; it does not substitute exact
polynomial derivatives. Stable anchored evaluation subtracts cancelling tanh
constants before summation. A separately exported ordinary PyTorch Sequential
model, with standard dense layers and Tanh, is checked using automatic
differentiation, including independent boundary points.

## 4. How the PDE determines the previously unknown readout

For the first test the data are a forcing s on B and a boundary value g on
the circle. The equation is

    −Δu+u³=s,   u|_(∂B)=g.

At interior sample points the residual is r(a)=−Δu_a+u_a³−s. Boundary rows are
u_a−g. We initialize a=0. A Gauss–Newton correction δa is found from the
linearized residual equations. For the interior rows the Jacobian is

    J(a)[q,l]=−Δψhat_l(x_q)+3 u_a(x_q)² ψhat_l(x_q).

The generic solver approximately minimizes ||J(a)δa+r(a)||², takes a checked
step, and repeats. It uses matrix-free LSMR and bounded block preconditioning.
All ψhat and derivative values here come from the actual tanh construction.
This is a global numerical nonlinear solve. It is NOT a closed-form solution
of a nonlinear PDE and it does use linearized least-squares steps. The explicit
part is the geometry/readout-coordinate map E, which removes the need to
optimize or independently solve thousands of redundant neuron weights.

Fixed-hidden-layer PDE solves with nonlinear least-squares/Newton readout
updates already exist in the ELM literature; see
[Dong and collaborators](https://www.math.purdue.edu/~sdong/PDF/ConcELM_JSC2023.pdf).
The distinction tested here is deterministic, analytically constructed ridge
geometry and a known readout map, rather than random hidden features. We do
not claim the generic readout-solving strategy is new.

The continuum semilinear problem is uniquely solvable under the usual
regularity/existence assumptions. If u and v have the same boundary data,
subtracting their equations and testing with e=u−v gives

    ∫|∇e|² + ∫e²(u²+uv+v²) = 0,

so e=0. More generally, for an approximate v with exact boundary data, the
same calculation bounds ||∇(v−u)|| by the H^−1 norm of its continuum residual.
This does not turn finitely sampled residuals into an automatic certificate,
nor does uniqueness prove global convergence of discrete Gauss–Newton.

## 5. Predictions and measured evidence

The construction makes three separate predictions: increasing profile degree
enriches the function space; insufficient M loses angular content regardless
of N; insufficient N corrupts neural values/derivatives regardless of M.
The tests sweep these separately and check fresh residual points.

For the smooth nonlinear disk problem, the prescribed reference used only to
manufacture forcing/boundary data and validate is

    u*(x,y)=exp(0.3x)cos(1.3y)+0.1xy.

Degree 4,8,12,16,18 produced QUILL relative function errors approximately
9.0e−3,2.8e−6,8.7e−11,8.7e−16,2.7e−16. The degree-18 ordinary exported MLP
has relative error 4.0e−16. It has M=19 directions, N=257 interior centers,
R=17 halos per side: 5529 tanh neurons, but only P=190 solved coordinates.
The generic solver took five nonlinear steps from zero. The ordinary Torch
autodifferentiated PDE residual is about 5.3e−15 on fresh interior points and
5.5e−15 on fresh boundary points. These errors are float64 measurements,
not certified exact-arithmetic error bounds.

At degree12 and N257, M=2,4,8,13 gives errors 4.4e−2,1.3e−3,1.5e−7,8.7e−11.
At degree12 and M13, N=33,65,129,257 gives errors 6.2e−7,1.2e−9,8.7e−11,8.7e−11.
The low-degree or insufficient-resolution cases do not satisfy the requested
1e−12 residual tolerance and are not reported as converged.

A separate unforced Burgers IVP supplies only its initial condition and zero
spatial boundary values. That harder test exposes limits: high-degree
space-time residual solves can stagnate or hit iteration limits even when
the same representation solves the elliptic example to roundoff. The exact
polynomial control has the same difficulty at degree20. The JSON results
retain status, full iteration history, cost, heldout function error, final-time
error and raw Torch derivative checks. We do not infer that a complete angular
representation automatically gives a well-conditioned PDE Jacobian.

In the bounded stronger Burgers run, degree28 with 4000 Krylov iterations per
linear solve reached space-time relative L2 2.44e−7 in 60.1 seconds, compared
with 2.20e−6 for the smaller linear-solve budget. It still reached the nonlinear
iteration limit; final-time relative error was 1.19e−5. Ordinary Torch AD gave
fresh interior PDE RMS 4.37e−7, boundary PDE RMS 2.50e−6, and initial-face PDE
RMS 3.19e−6. No convergence claim is made for this case.

A target-free diagnostic isolates a concrete geometry issue. On 4096 sampled
points, the condition number of the SAME known degree-p ridge coordinates
on the full disk versus its inscribed square was:

| Degree p | Disk | Inscribed square |
|---|---:|---:|
| 12 | 1.12 | 3.91e3 |
| 20 | 1.74 | 3.18e6 |
| 28 | 3.98 | 3.09e9 |

These are singular-value diagnostics of the known feature evaluation matrix,
not a PDE solve or a way of constructing the readout. They verify that the
disk's well-conditioned coordinates can become poor coordinates for the
space-time square. They do not by themselves measure the PDE Jacobian or
prove that domain restriction explains every part of the observed error.

## 6. Boundaries of this result

* This is a genuine flat ridge MLP and an unknown-solution physics solve, but
  the smooth elliptic test is a small manufactured problem, not turbulence.
* The method learns only the constrained readout coordinates a. It neither
  learns directions nor optimizes independent readouts, centers or gammas.
* M≥p+1 is a complete polynomial-spanning guarantee in 2D, not an optimal
  universal direction budget. One-dimensional encoding error must be checked
  in all derivative orders used by the PDE.
* These disk-orthogonal coordinates need not remain well-conditioned on a
  subdomain such as a space-time rectangle, or under a differential operator.
* Higher-dimensional ball constructions use Gegenbauer ridge profiles and
  spherical quadrature, but their counts grow with dimension and degree.
  This prototype does not remove the curse of dimensionality.

Files: `solver/ridge_pinn_features.py`, `ridge_pinn_study.py`,
`ridge_pinn_burgers.py`, `tests/test_ridge_pinn_features.py`. Saved ordinary
Torch state dictionaries are named `*_torch_state.pt`; reconstruct with
`Sequential(Linear(2,L),Tanh(),Linear(L,1)).double()` and load the state dict.
