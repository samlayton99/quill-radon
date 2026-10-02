# Explanation: the math, `qi_mpmath.py`, and how it all maps to the paper

This is a conceptual walkthrough. The goal is that by the end you can read
`src/construction/qi_mpmath.py` and see the paper's equations staring back at you,
and you understand *why* each piece exists — including the two things that are
easy to conflate (the **Toeplitz solve** and the **least-squares / Φ solve**) and
the thing that sounds contradictory (using **mpmath** when "the point is fp64").

Paper: `papers/QIs_workshop.pdf` ("Constructing Machine-Precision Neural Networks
with Quasi-Interpolants"). Section 3 in that PDF is stale; the current construction
is `papers/Section_3_Rewrite.pdf`, and the fp64/mpmath details are in
`papers/practical_implementation.tex`.

---

## 0. What you're probably actually confused about

Before the details, let me name the five conceptual knots, because every specific
question you asked is a symptom of one of them.

1. **You think there is "a solve," singular.** There are *two completely different*
   linear-algebra problems in this codebase, and they answer different questions:
   - **Toeplitz solve** → the *cardinal coefficients* `c_j`. This is part of the
     **construction**. It is target-independent. It defines *how to interpolate at all*.
   - **Least-squares solve with the Φ matrix** → the *outer (readout) weights*.
     This is an *alternative* to the construction's convolution formula, and it's
     also exactly the object the paper uses to diagnose trained networks (§4.3).

   Once you see these as two separate machines, half the confusion dissolves.

2. **You don't see how a matrix solve "is" the math.** The math says: "find a
   cardinal function `L_h` that equals 1 at its own node and 0 at every other
   node." That sentence *is* a system of linear equations. Writing the requirement
   down at the grid points produces a matrix, and because the grid is uniform and
   the kernel only cares about *distance*, that matrix is Toeplitz. Toeplitz isn't
   a different method; it's the linear-algebra shadow of the interpolation
   condition.

3. **You think "fp64" means "every computation must be fp64."** It doesn't. It
   means the *representation* — the model's weights, its forward pass, its
   gradients, its evaluation — lives in fp64. The construction computes a handful
   of *constants* (the weights of a specific MLP). You can compute constants as
   accurately as you like and then round them to fp64, exactly like `numpy.pi`.
   mpmath is how we compute the *best possible fp64 constants*.

4. **You don't see where Φ / least squares connects to the paper.** It's not a
   side-quest. It is literally the paper's §4.2 ("fix the geometry, fit only the
   output coefficients") and §4.3 ("freeze the hidden layer, form `Φa=b`, ask how
   compressible Φ is"). The repo's `readout.py` is that experiment's engine.

5. **You think the number of sample points sets the number of neurons.** In the QI
   *construction* they do move together — but the causality runs the *other* way:
   the **architecture dictates the sampling**. Choosing the grid resolution `N`
   (plus halo `R`, stencil `K_c`) fixes both the `W` neurons *and* the exact set of
   points where `g'` must be evaluated — you don't get to choose a sample count. The
   least-squares readout breaks this link: same neurons, but you pick the fit points
   freely. (§8.3 lists every point set; the trap is conflating "neurons," the
   "construction sample set," and the "lstsq fit points" — they are three things.)

Keep these five in mind; the rest of the document earns them.

---

## 1. The object we're building

A one-hidden-layer `tanh` MLP from `ℝ → ℝ` is just a weighted sum of shifted,
scaled `tanh`s plus a bias:

$$ \tilde f(x) \;=\; b \;+\; \sum_{m} w_m \,\tanh\!\big(\gamma\,(x - x_m)\big). $$

- `x_m` — the **centers** (where each `tanh` sits).
- `γ` — the **bandwidth** / steepness (how sharp each `tanh` is). One shared `γ`
  in the construction.
- `w_m` — the **outer / readout weights**.
- `b` — a single **bias**.

In code this is `QIMlp`: `readout(tanh(inner_layer(x)))`, where `inner_layer`
computes `γ·(x − center)` (`src/models/layers.py`, `GammaLinear`) and `readout` is
`nn.Linear(width, 1)` holding `w_m` and `b`.

The research question: can we *choose* `(γ, x_m, w_m, b)` so that `tilde f`
matches a target `f` to ~10⁻¹⁵ (machine epsilon)? The paper's Section 3 answers
**yes, explicitly** — that explicit recipe is the QI construction, and
`qi_mpmath.py` is its implementation. (The harder, open question — can *gradient
training* find such weights — is Section 4 and the rest of the repo.)

Everything below is: *how do we pick those numbers?*

---

## 2. Why `tanh`, and why we secretly work with the derivative

Quasi-interpolation (next section) is a theory about **kernels** — bump-like
functions `K` you place at each sample and add up. The natural kernel here is

$$ K(u) = \operatorname{sech}^2(u), $$

a symmetric bump. The magic fact (paper Eq. `kernel-derivative`):

$$ \operatorname{sech}^2 = \tanh', \qquad\text{i.e. } K = \psi' \text{ with } \psi=\tanh. $$

So a sum of **bumps** is the derivative of a sum of **tanhs**. If we build a great
kernel approximation of `f'` using `sech²` bumps,

$$ f'(x) \approx \sum_m a_m \, K_\gamma(x - x_m), \qquad K_\gamma(x):=\gamma K(\gamma x), $$

then integrating once turns each `sech²` bump back into a `tanh`, and the bump
weights `a_m` become the MLP's outer weights (paper Eq. `chain-rule-realization`,
`mlp-def`):

$$ f(x) \approx b + \sum_m a_m \,\tanh(\gamma(x - x_m)). $$

The constant of integration is the bias `b`. (For `tanh`/`sech²` the "integration
polynomial" is just this one constant; for GELU/Swish it would be a linear term —
paper Eq. `integration-poly`.)

**This is why the construction convolves with `g'`, not `g`.** The kernel lives at
the derivative level; the `tanh` MLP is its antiderivative. Hold onto that — it
explains a line of code that otherwise looks arbitrary.

The scaled kernel `K_γ(x) = γ·K(γx)` is a bump of width `~1/γ` and unit mass.
Larger `γ` = narrower, taller bump. The grid spacing is `h`. The single most
important quantity in the whole theory is the **dimensionless bandwidth**

$$ \boxed{\;\lambda := \gamma h\;} \qquad\text{(paper Eq. \texttt{lambda-def})} $$

which measures "how many grid points fit under one bump." It is the knob that
trades off two competing errors (Section 6).

---

## 3. Quasi-interpolation: reconstructing `f` from samples

> **Index scales** (full table in §5.1): `m` labels grid nodes, which are also the
> kernel centers (`x_m = −1 + mh`), `m ∈ [−R, N+R]`. Separately, `r` and `j` are
> *local stencil offsets* in `[−K_c, K_c]` used to build the single cardinal
> function `L_h`; offset `0` means "at that kernel's own peak," not the middle of
> the domain. (The code calls the center index `n` and the stencil index `k`.)

Classical interpolation finds one global polynomial through all samples — it
blows up (Runge phenomenon). Quasi-interpolation instead reconstructs `f` as a
**translation-invariant** sum: put the *same* local cardinal function `L_h` at
every node and weight it by the sample there (paper Eq. `qh-infinite`):

$$ (Q_h f)(x) \;=\; \sum_{m\in\mathbb{Z}} f(x_m)\, L_h(x - x_m), \qquad x_m = -1 + mh. $$

For this to actually interpolate — to reproduce the samples and converge nicely —
`L_h` must be a **cardinal function**: it equals 1 at its own peak (zero
displacement) and 0 at every other grid displacement:

$$ L_h(0) = 1, \qquad L_h(rh) = 0 \ \text{ for integer } r\neq 0, \qquad\text{i.e. } L_h(rh)=\delta_{r,0}. $$

That property is what makes `Q_h f` reproduce data and have clean error behavior.
**The entire job of the Toeplitz solve is to build this `L_h`.** That's the link
you were missing: "Toeplitz" is not a separate idea — it's how we *enforce the
cardinal property*.

### 3.1 Building `L_h` out of kernels

We don't have `L_h` in closed form. We *build* it as a combination of our kernel
bumps placed on the grid, with unknown coefficients `c_j` (paper Eq.
`truncated-cardinal`):

$$ L_h(x) \;=\; \sum_{j} c_j \, K_\gamma(x - jh). $$

Now impose the cardinal property, evaluating at each displacement `x = rh`:

$$ L_h(rh) = \sum_j c_j\, K_\gamma\big((r-j)h\big) \;=\; \delta_{r,0}. $$

This is a **linear system in the unknowns `c_j`**: one equation per displacement
`r`, with `K_γ((r−j)h)` multiplying `c_j`.

### 3.2 Why it's Toeplitz (and what Toeplitz means)

A **Toeplitz matrix** is constant along each diagonal: entry `(r, j)` depends only
on the difference `r − j`. Our entry `K_γ((r−j)h)` does exactly that — because the
kernel only cares about *distance* and the grid is uniform — so the matrix is
Toeplitz automatically.

Imposing the condition for `|r| ≤ K_c` against the `2K_c+1` unknowns `c_j`
(`|j| ≤ K_c`) gives a square `(2K_c+1)×(2K_c+1)` system with `T_{r,j} = h·K_γ((r−j)h)`:

$$ \boxed{\,T\,c = h\,e_0\,} \qquad\text{(code uses RHS } b[K_c]=h). $$

`e_0` has its single `1` at the `r = 0` row — the kernel's own peak, where
`L_h = 1`. Since `r` runs symmetrically `−K_c … K_c`, that row is the **middle** of
the vector, which is why the code sets `b[Kc] = h` (`K_c` is the midpoint of a
length-`2K_c+1` array).

Solve this linear system → you get the coefficients `c_j` → you have `L_h` → you
can quasi-interpolate. The `c_j` are called **cardinal coefficients**. Two facts
make them special:

- **They do not depend on the target `f`.** They depend only on `(λ, K_c)` — the
  shape of the kernel relative to the grid. So you solve the Toeplitz system once
  and reuse `c_j` for every target. (This is exactly why the code caches them to
  disk; see §7.)
- **They decay but oscillate.** `|c_0| ≈ 300+` and they alternate in sign as `|j|`
  grows. This alternation is the seed of the fp64 precision problem (§8).

> There's also a Fourier picture (paper Eq. `fourier-character`):
> `Ĉ_h(ω) = h / D_h(ω)` where `D_h` sums the kernel's Fourier transform over all
> aliases, and the `c_j` are the Fourier coefficients of that ratio. "Normalize the
> kernel's spectrum so the sum-of-shifts is flat" is the frequency-domain way of
> saying "make `L_h` cardinal." The Toeplitz solve is the real-space equivalent.
> Same `c_j`, two derivations.

### 3.3 Why `K_c` (truncation) and `halo R`

Two practical truncations turn the infinite theory into a finite computation:

- **`K_c` (stencil half-width):** we only keep `c_j` for `|j| ≤ K_c` (the sum in
  §3.1 is finite). Since `c_j` decays, this is fine if `K_c` is big enough.
  `K_c = 160` is the working value. (`K_c = 12` is famously *not* enough — the
  coefficients haven't decayed yet.)
- **`halo R`:** near the domain edges `x = ±1`, an interpolation node needs
  neighbors that fall *outside* `[−1,1]`. The halo adds `R` "ghost" centers on each
  side so edge nodes still have a full stencil. The MLP keeps those ghost `tanh`s —
  their centers sit outside `[−1,1]`. (Strikingly, the paper's §4.2 reports that
  *trained* networks also push some centers outside the domain — they rediscover
  the halo.)

Total hidden width: `W = N + 2R + 1` (interior nodes plus both halos).

---

## 4. From cardinal coefficients to the MLP weights

Now we have `c_j` (the interpolation machinery). To approximate a *specific*
target we do two cheap, target-dependent steps. (`m` is still the center index,
now ranging over the finite, halo-padded grid `m ∈ [−R, N+R]`; `j` the stencil.)

**Step A — convolve to get outer weights** (paper Eq. `single-kernel-sum`).
Re-indexing the quasi-interpolant into a single sum of kernels gives outer weights

$$ a_m \;=\; \sum_{|j|\le K_c} c_j \, g'(x_{m-j}). $$

This is a **discrete convolution** of the cardinal coefficients with the sampled
target derivative `g'`. (Remember §2: the kernel lives at the derivative level, so
we feed it `g'`.) One `a_m` per center.

**Where the sampling comes from — and why you don't choose it.** Look at the
indices: `m ∈ [−R, N+R]` and `j ∈ [−K_c, K_c]`, so the arguments `x_{m−j}` sweep
exactly `s ∈ [−R−K_c, N+R+K_c]` — the paper's `I_{R,K_c}`. That's `W + 2K_c`
evaluation points, and **it is forced, not chosen**: the moment you fix the grid
resolution `N` (and the halo `R`, stencil `K_c`), you have simultaneously fixed
*both* the `W = N+2R+1` neurons *and* the precise set of points where `g'` is read.
In the construction, "how many points do we sample" is not a knob — it is a
*consequence of the architecture*. (Two corollaries that catch people: the set
reaches `K_c` *outside* `[−1,1]` on each side, and it samples the *derivative* `g'`,
not `f`. Both are fine because `g'` is analytic and queryable anywhere — but note
the construction is using information a data-only learner would not have.) The
least-squares readout in §8 severs this link: it keeps the same neurons but lets you
pick the fit points freely. Keeping these point sets straight is the whole content
of §8.3 — don't conflate neurons, the construction's `I_{R,K_c}`, and lstsq's fit
points.

**Step B — fix the bias from a boundary condition.** We pin the antiderivative so
the MLP matches the target at the left endpoint, `tilde f(−1) = g(−1)`:

$$ b \;=\; g(-1) - \sum_m a_m \,\tanh\!\big(\gamma(-1 - x_m)\big). $$

That's the whole construction. The final MLP is

$$ \tilde f(x) = b + \sum_m a_m \,\tanh(\gamma(x - x_m)), \qquad \gamma = \lambda^*/h,\ \ x_m = -1 + mh. $$

Notice `γ = λ*/h = λ*·N/2`: as you add neurons (`N↑`), `γ` **grows linearly**.
That O(N) growth of the bandwidth is the central structural prediction of the
paper — and the thing trained networks fail to do (§9).

---

## 5. `qi_mpmath.py`, line by line, against the math

Now the file reads like the math above. Here's the map (function → equation):

| Code | Math | Section |
|---|---|---|
| `default_halo(N, λ)` | choose `R` so halo-truncation `e^{-c₃λR}` is below ε | §3.3 |
| `_build_toeplitz_c_f64 / _mpmath` | solve `T c = h·e₀`, `T_{r,j}=h·K_γ((r−j)h)`, `K_γ=γ·sech²(γx)` | §3.1–3.2 |
| `_build_a_f64 / _build_a_mpmath_kahan` | `a_m = Σ_j c_j g'(x_{m−j})` (convolution) | §4 Step A |
| `_compute_c0_f64 / _mpmath` | `b = g(−1) − Σ a_m tanh(γ(−1−x_m))` | §4 Step B |
| `construct_qi(...)` | orchestrates: `c_j` → `a_m` → `b`, returns `QIResult` | §3–4 |
| `evaluate_qi(qi, x)` | `tilde f(x) = b + Σ a_m tanh(γ(x−x_m))` | §4 |

Walking `construct_qi` top to bottom:

1. **Defaults & geometry** (`qi_mpmath.py:431-437`): pick `λ*` (0.30 fp64 / 0.25
   mpmath), `halo`, then `h = 2/N`, `γ = λ*/h`. This is `λ = γh` made concrete.
2. **Cardinal coefficients** (`:439-482`): build/solve the Toeplitz system for
   `c_j` — *with caching*, because `c_j` is target-independent (§7). The matrix is
   `T_{r,j} = h·γ·sech²(γ(r−j)h)`; RHS is `h` at the `r=0` row. This is §3.2.
3. **Convolution** (`:484-511`): sample `g'` on the extended grid and convolve with
   `c_j` to get `a_m`. (`_build_a_f64` does this as one vectorized sliding-window
   matmul; the mpmath path uses compensated summation — see §8.) This is §4-A.
4. **Bias** (`:493-510`): pin `tilde f(−1)=g(−1)` to get `b` (`c0` in code). §4-B.
5. **Package** (`:513-531`): split interior vs halo, return an immutable
   `QIResult` carrying `centers, a_coeffs, c0, γ, λ, halo, K_c`. Pure data; no
   model. `initialize.py` later copies these numbers into a `QIMlp`'s parameters.

That's it. There is no training, no gradient, no optimizer in this file. It is a
deterministic recipe that emits the weights of one specific MLP.

### 5.1 Index conventions (math ↔ code)

Two index families, two sizes. The math here and the code use different letters —
here is the full correspondence:

| Family | Range | Count | This doc | Code | Paper |
|---|---|---|---|---|---|
| **Center / neuron** | `[−halo, N+halo]` | `W = N + 2·halo + 1` | `m` | `n` (`n_idx`) | `k` → `m` |
| **Stencil offset** | `[−K_c, K_c]` | `2K_c+1` (default **321**) | `r` (row), `j` (coeff) | `k` (`k_list`), `r`/`j` (Toeplitz) | `j` |

The one real gotcha: **the code's `k` is the stencil offset** (this doc's / the
paper's `j`), not a neuron. So in `a_n = Σ_k c_k·g'(x_{n−k})`, `n` is the neuron
(`[−halo, N+halo]`; e.g. `N=64`, `halo=59` → `n ∈ [−59, 123]`, `W=183`) and `k` is
the stencil (`[−160, 160]`). Rule of thumb: `K_c`-sized ⇒ stencil; `halo`/`N`-sized
⇒ neurons.

(The code samples `g'` on a wider grid `[−(halo+K_c), N+halo+K_c]` so every neuron
can reach its full `±K_c` stencil — that's why `g'` is sampled past the halo.)

---

## 6. The λ tradeoff (why there's a "right" bandwidth)

Why not just crank `K_c`, `R`, `N` and pick any `λ`? Because of the error bound
(paper Theorem 1, Eq. `main-error-bound`). The total error splits into four
exponentially-decaying pieces:

$$ \|f - \tilde f\|_\infty \;\le\;
\underbrace{C_1(\lambda)e^{-c_1/h}}_{\text{resolution}} +
\underbrace{C_2 e^{-c_2/\lambda}}_{\text{aliasing}} +
\underbrace{C_3(\lambda)e^{-c_3\lambda R}}_{\text{halo}} +
\underbrace{C_4(\lambda)e^{-c_4\lambda K_c}}_{\text{stencil}}. $$

Look at how `λ` appears with opposite signs:

- **Large `λ`** (wide bumps relative to grid): the **aliasing** term `e^{−c₂/λ}`
  is large — fat bumps overlap and can't resolve detail. Bad.
- **Small `λ`** (narrow bumps): aliasing shrinks, *but* the prefactors
  `C₁, C₃, C₄` **blow up** algebraically as `λ→0` — the construction becomes
  ill-conditioned. Bad.

So there's a **U-shaped** error-vs-`λ` curve with a sweet spot around
`λ* ≈ 0.25–0.30`. That's why those exact values are hard-coded, and why
`λ=1.5` (too aliased) and tiny `λ` (too ill-conditioned) both fail. **expC01 is the
experiment that traces this U-curve and confirms the sweet spot** (§10).

The deep consequence (paper §3 end, and the abstract): to *stay* at the good `λ*`
as you refine the grid (`h = 2/N → 0`), you must grow `γ = λ*/h ∝ N`. Holding
`λ` fixed *forces* `γ = O(N)`. Conversely, if `γ` stays `O(1)` while `N` grows
(what optimizers do), then `λ = γh → 0` and you fall off the left side of the U
into the ill-conditioned regime. This is the seed of "violation #1."

---

## 7. The cache (a small but important detail)

`c_j` depends only on `(λ*, K_c, N, precision, mp_dps)` — **not** on the target.
The mpmath Toeplitz solve costs ~30–55 s cold. So `qi_mpmath.py` caches `c_j` to
`results/qi_cache/` keyed by those parameters (fp64 as `.npz`, mpmath as
full-precision text). First call for a given config pays the cost; every later
call (any target, any seed) reloads in ~0.25 s and just redoes the cheap
convolution + bias. This is why running a whole sweep of targets at fixed `(N, λ)`
is fast.

---

## 8. The Φ matrix and least squares — the *other* solve

Everything so far computes the outer weights `a_m` by the **convolution formula**.
But here is a completely different way to get outer weights, and it's the one your
question about "least squares / the Φ matrix" is really about.

**Why even switch, after all that QI work?** Because the construction and the
research question ask different things. The construction *proves a machine-precision
MLP exists* and is realizable in fp64 — but it gets there by privileged means
(analytic `g'`, samples outside the domain, a fixed formula on a rigid grid). The
project's actual question is about **optimization**: can a *learning* procedure,
given only data, find machine-precision weights? The readout is the linear part of
that question, and least-squares is the simplest possible "optimizer" for it (convex,
closed-form). So this section is the first real step from *construct* to *learn* — and
its result (below) is what reframes the whole problem.

**Setup:** suppose you've *already fixed* the geometry — the centers `x_m` and the
bandwidth `γ` (say, from the construction). The only things left to choose are the
outer weights `v_m` and bias. The model is now **linear** in those weights:

$$ \tilde f(x_i) = \sum_m v_m \,\underbrace{\tanh(\gamma(x_i - x_m))}_{\Phi_{i m}} + b. $$

Stack the training points `x_i` as rows and the centers `x_m` as columns. The
matrix

$$ \boxed{\;\Phi_{im} = \tanh\big(\gamma\,(x_i - x_m)\big)\;} $$

is the **feature matrix** (a.k.a. design matrix). Column `m` is "what neuron `m`
outputs across all the data points"; row `i` is "all neuron activations at point
`x_i`." Then fitting the readout is just the linear least-squares problem

$$ \min_{v}\ \|\Phi\,v - y\|_2^2, \qquad y_i = f(x_i). $$

This is `src/construction/readout.py`: `build_phi` builds `Φ`; `solve_readout`
solves the least-squares system (via `lstsq` / `qr` / `svd` / ridge);
`solve_readout_with_bias` adds a ones-column for `b`. (In the paper's §4.3
notation, `Φa=b` with `Φ_ij = tanh(γ_j(x_i−x_j))` and the bias mode removed via
`b_i = f(x_i) − f̄`. Same object.)

### 8.1 Two solves, side by side — do not conflate these

| | **Toeplitz solve** (§3) | **Least-squares / Φ solve** (§8) |
|---|---|---|
| Unknowns | cardinal coefficients `c_j` | outer/readout weights `v_m` |
| Matrix | `T_{r,j}=h K_γ((r−j)h)` — Toeplitz, square | `Φ_{im}=tanh(γ(x_i−x_m))` — rectangular (data × neurons) |
| Depends on target? | **No** (target-independent) | **Yes** (right-hand side is `y=f(x)`) |
| Role | *defines* the interpolation operator `L_h` | *fits* the outer weights given fixed geometry |
| Where used | the QI **construction** (`qi_mpmath.py`) | the **alternative readout** (`readout.py`); paper §4.2, §4.3 |
| Output feeds | convolution → `a_m` | directly the `v_m` |

The QI construction (Toeplitz + convolution) and the least-squares readout are
**two routes to the outer weights of the same fixed-geometry MLP**. The QI route
uses a fixed analytic formula (`a_m = Σ c_j g'(x_{m−j})`); the least-squares route
directly minimizes the residual on the data.

### 8.2 What expA02 compares — and what it does *not* show

expA02 fixes **one** geometry (the QI centers `x_m` and bandwidth `γ`) and fills the
readout four ways: QI-formula (mpmath / fp64) and least-squares (mpmath / fp64).
Same `W` neurons every time. Eval `L∞` on a dense grid in `[−1,1]` (sine, `λ*=0.25`,
from `results/checkpoint_A_numerics/expA02_qi_vs_lstsq/data.json`):

| width | QI (mpmath) | QI (fp64) | lstsq (fp64) | lstsq (mpmath) |
|---:|---:|---:|---:|---:|
| 173 | 1.7e-14 | 8.8e-11 | 1.6e-13 | 1.6e-15 |
| 205 | 3.0e-15 | 1.7e-10 | 6.6e-14 | 8.2e-16 |
| 269 | 2.3e-15 | 1.3e-10 | 1.4e-13 | 8.0e-16 |

**What it shows (verified):**
- Given the QI geometry, a plain least-squares readout reaches the precision floor:
  fp64 lstsq ~1e-13, mpmath lstsq ~1e-15. The readout is an easy convex solve.
- lstsq has lower eval error than the QI formula in all 48 configs.

**How close are the two solutions? (this is what actually justifies the switch.)**
Closer than "a different solution" suggests — but *how* close depends on the
coordinate system:
- **As functions on `[−1,1]`: essentially identical, ~1e-13.** Both are
  machine-precision approximations of `f`, so they approximate *each other* to
  machine precision: `‖f̃_QI − f̃_lstsq‖ ≤ ‖f̃_QI − f‖ + ‖f − f̃_lstsq‖ ≈ 1e-13`.
- **As coefficients, it depends on the basis.** The paper shows (Fig 4b, §4.2) that
  in the *cardinal* basis `f = Σ_j v_j L_h(·−x_j)` — where the target coefficient is
  literally `v_j = f(x_j)` — least squares **recovers** those coefficients: *"the
  coefficient-recovery subproblem is numerically stable when the kernel geometry is
  specified correctly."* There, lstsq and QI coincide coefficient-for-coefficient.
  **That is the real justification for the switch:** with the right geometry, lstsq
  isn't finding an alien readout — it *recovers* the QI solution by a stable convex
  solve.
- **The catch — our repo uses a different basis.** `readout.py`/expA02 fit the *raw
  tanh* basis `f = b + Σ_m a_m tanh(γ(x−x_m))`, where the coefficients are the QI
  convolution weights `a_m` (not `f(x_m)`) and `Φ` is **ill-conditioned** (the
  saturated, redundant features of §4.3). So the *function* is pinned to ~1e-13, but
  the *coefficients* are floppy — many `a_m` give nearly the same function.
  expA02 measured eval error, **not** coefficient distance, so the tanh-basis
  closeness is not verified here. That's the experiment in §13.

**What it does *not* show (so you don't over-read it):**
- **The 48/48 win is empirical, not automatic.** lstsq minimizes the *train-L2*
  residual on the interior fit points; the metric is *eval-L∞* on a different,
  denser set. lstsq doesn't even optimize L∞, so winning on it is a measured
  outcome, not a definitional one.
- It is **not a controlled "same-inputs" comparison** — see §8.4.

**Working conclusion (from expA02 — scoped, not yet proven in general):** the readout
is the *easy* part. Once the geometry `(γ ∝ N, grid-spaced centers)` is correct, even
fp64 least-squares hits ~1e-13, so the open problem becomes *whether an optimizer can
discover that geometry*. The stronger half — that end-to-end *training* actually
fails to find it — is exactly what expD01 (ladder) and exp13 (solution basins) are
meant to test, and exp13 is currently a stub. So treat "geometry is the bottleneck"
as the current **working hypothesis**, not a settled result.

This mirrors the paper (§4.2 fixes the geometry and fits only the output
coefficients to show MLPs *can* reach the fp64 floor; §4.3 reuses the same Φ to
diagnose why trained nets don't), and it matches the reference repo: `continuous-mlps`
solves its readout with `np.linalg.lstsq` on fixed QI geometry
(`reproduce_cpu/linear_solve_reconstruct.py`, `1d_interpolation/sweep_fixed_lambda.py`),
and its sparsity pipeline uses the mean-centered Φ + lstsq/pinv. So switching to a
least-squares readout is the established tool in this line of work, not a departure.

### 8.3 Three point sets — what's tied to the grid and what isn't

The crux that's easy to miss: there are **three** point sets, and only some are
coupled.

| set | what it is | size | chosen how |
|---|---|---|---|
| **centers** `x_m` | where the `tanh`s sit (QI grid + halo) | `W = N + 2R + 1` | fixed by resolution `N` (+ halo `R`) |
| **construction samples** | where `g'` is evaluated, `I_{R,K_c}` | `W + 2K_c` | **forced** by the grid: `[−R−K_c, N+R+K_c]` |
| **lstsq fit points** `x_i` | where the residual is evaluated to fit `v` | `n_train` | **your choice** (expA02: `max(512, 2W)`) |
| **eval points** | where error is measured | 2048 | measurement only — never fits anything |

The coupling that **is** real: *neurons ↔ grid resolution* (`W = N + 2R + 1`). Pick
`N`, you've picked the grid *and* the neurons.

The coupling that does **not** exist in the lstsq route: *neurons ↔ number of fit
points*. `n_train` is a free knob; doubling it adds zero neurons. In the
*construction* there is no such freedom — the sample set is exactly `I_{R,K_c}`, of
size `W + 2K_c`, dictated by the stencil reach. **So sampling and neurons are locked
together inside the construction; the freedom only appears once you switch to fitting
a residual.**

Causality, once: pick resolution/width `→` that sets the `W` neurons on the grid `→`
*then* pick however many fit points you want to pin their readout weights. Sample
count is downstream of (or independent of) neuron count — never upstream.

### 8.4 Why it's not apples-to-apples (and why it's still useful)

The four methods share the **same model** (same centers, `γ`, `W`). They do **not**
share the same **data**:

- **QI** consumes the *analytic derivative* `g'`, sampled on `I_{R,K_c}` — which
  *includes points outside* `[−1,1]` (the halo + stencil overhang).
- **lstsq** consumes *function values* `f`, at interior fit points only.

Different objects (`g'` vs `f`) on different domains (extended vs interior), so this
is **not** a controlled same-information comparison — and a perfectly controlled one
is *impossible*, because QI structurally needs `g'` (its kernel is `tanh'`) while
lstsq fits `f`. So do **not** read "lstsq beats QI" as "lstsq is the better estimator
from equal inputs."

What the comparison *legitimately* isolates is the **geometry**: it's held fixed
while only the readout method varies, judged by a common eval metric. That is what
licenses the one sound takeaway — *for a fixed correct geometry, a data-driven
readout reaches the floor* — which is, if anything, strengthened by lstsq getting
there with *less* information (interior `f` only; no derivative, no out-of-domain
access).

### 8.5 Regular spacing: the construction needs it; lstsq doesn't

A structural asymmetry worth keeping straight:

- The **construction requires the uniform grid.** The cardinal coefficients `c_j`
  and the Toeplitz system exist *only* because of translation invariance on an evenly
  spaced grid; `a_m = Σ_j c_j g'(x_{m−j})` is one shared stencil precisely because
  every node looks like every other. Perturb the spacing and there is no single
  cardinal stencil — the construction has no closed form.
- The **least-squares readout has no spacing requirement.** `Φ` is built from
  whatever fit points `x_i` you pass; uniform, random, or clustered all give a valid
  (if differently-conditioned) system. (The *centers* still sit on the regular grid —
  that's the model; it's the *fit points* that are free.)

So irregular or noisy **sampling points** are compatible with the design-matrix route
but break the construction route. **Whether lstsq actually retains machine precision
under noisy or irregular sampling is now addressed by expB01 (sampling and noise),
which implements exactly this question.** Don't assume graceful degradation; it has
to be measured.

---

## 9. mpmath vs fp64 — "I thought the whole point was fp64"

This is the knot worth slowing down on.

**What "fp64" actually refers to.** The *model* is fp64: its weights are fp64
numbers, its forward pass, gradients, and evaluation all run in fp64. That is the
deliverable and it never changes. The construction's job is to produce the *fp64
numbers* `(γ, x_m, a_m, b)` that make an fp64 MLP accurate. Computing those
numbers is an **offline precomputation** — it is not part of the model.

**The `numpy.pi` analogy (this is the whole point).** `numpy.pi` is a single fp64
constant. Nobody computed it *in* fp64 by an fp64 algorithm; π was computed to
enormous precision by other means and then rounded once to the nearest fp64
double. Using `numpy.pi` does not "violate fp64." The QI coefficients are the same
kind of thing: constants for a specific MLP. Computing them in 30-digit mpmath and
then rounding to fp64 gives you the *best fp64 constant set* — it does not make the
model anything other than fp64.

**Why fp64 *computation* of the constants isn't good enough.** Two fp64 hazards in
the construction:

1. *Toeplitz conditioning.* As `λ` shrinks toward the high-precision regime, the
   Toeplitz matrix `T` becomes ill-conditioned; an fp64 solve loses digits. (This
   is why the fp64 path backs off to `λ=0.30` while mpmath can use `λ=0.25`.)
2. *Catastrophic cancellation in the convolution (the dominant one).* Recall the
   `c_j` reach `|c_0| ≈ 338` and **alternate in sign**. The convolution
   `a_m = Σ_j c_j g'(x_{m−j})` adds ~321 terms of magnitude ~hundreds whose true
   sum is `O(1)`. Adding big numbers of alternating sign to get a small answer is
   the textbook setup for **catastrophic cancellation**: you lose roughly
   `log10(300) ≈ 2.5` digits. fp64 has ~16 digits, so you floor out around
   `10^{-12}` — not `10^{-15}`.

So the **fp64 path is cancellation-limited at ~1e-12**, by arithmetic, not by the
method. The **mpmath path** does the Toeplitz solve and the convolution in 30-digit
arithmetic (so cancellation costs 2.5 of 30 digits, irrelevant), then rounds the
*final* `a_m, b` to fp64 — landing at true machine epsilon ~`2–3e-15`.

You can see both regimes directly in `results/checkpoint_A_numerics/expA01_numerics_sanity/summary.txt`
(target=sine):

| N | fp64 L∞ | mpmath L∞ |
|---:|---:|---:|
| 32 | 9.2e-12 | 1.7e-14 |
| 64 | 5.4e-12 | 2.8e-15 |
| 128 | 5.5e-12 | 2.0e-15 |
| 256 | 2.8e-12 | 1.4e-15 |

The fp64 column is stuck at the cancellation floor; the mpmath column rides down to
machine epsilon. **Same construction, same final fp64 model — only the precision of
the offline coefficient computation differs.**

**Practical rule** (encoded in the experiments): use **mpmath** when the QI
solution is a *fixed reference you compare against* (you want it exact — expD01
ladder, exp13 solution basins). Use **fp64** when you're *training, sweeping,
or initializing* (you want speed and 1e-12 is plenty). The repo even mitigates the
fp64 floor with compensated summation (Kahan in the mpmath convolution; `math.fsum`
in the fp64 bias) so it gets as close to the floor as fp64 allows.

---

## 10. Current results, in depth

The implemented/run experiments are expA01 numerics_sanity, expA02 qi_vs_lstsq,
expA03 coeff_nullspace, expA04 activation_conditioning, expA05 weight_blowup,
expB01 sampling_and_noise, expB02 scaling_laws, expC01 lambda_tradeoff, expC02
lambda_vs_frequency, expC03 lambda_basin, expC04 center_geometry, expC05
geometry_interpolation, expC06 soft_neuron_interp, expD01 geometry_ladder
(Phase 1), expD02 adam_geometry, and expE01 geometry_zoo_2d — plus the
`results/setup/` convergence probes; the remaining stubs are expD03
reparameterization, expD04 varpro, and the deprioritized exp13 solution_basins.
Together they establish the *foundations* the paper's Section 4 rests on. Here's what
each actually shows.

### 10.1 `results/setup/` — the construction works and scales

The convergence probes confirm the headline claim of Section 3: as width grows,
the QI construction's error falls geometrically and bottoms out at the precision
floor of its arithmetic (mpmath → ~1e-15; fp64 → ~1e-12). This is the empirical
version of "machine-precision interpolation is achievable," and it's verified in
the test suite too (`tests/test_construction.py::TestPrecisionFlag::
test_mpmath_path_reaches_machine_eps`).

### 10.2 `expA01` — it's not a numerics bug

Before blaming optimization for the training gap, rule out that the *floor itself*
is a numerical artifact. expA01 does this exhaustively (`results/checkpoint_A_numerics/expA01_numerics_sanity/`):

- **Construction baseline** (table in §9): fp64 floors at ~1e-12, mpmath at ~1e-15,
  cleanly and reproducibly. The floor is real and well-understood (cancellation),
  not a bug.
- **Readout solvers** (`summary.txt` section [2]): on the QI geometry,
  least-squares and SVD recover the outer weights to ~1e-13 with small weights
  (`v_max ≈ 6–10`), while naive QR can blow up (`v_max ≈ 2700`) and ridge with any
  nonzero penalty destroys precision. **Lesson: the readout solve is easy but you
  must use a stable, truncating solver** — which is exactly why `readout.py`'s SVD
  path now truncates tiny singular values.
- **Geometry matters enormously:** the same solve on the *interior-only* geometry
  (no halo) degrades to ~1e-4. The halo is not optional.
- **Conditioning, tanh stability, density sweeps:** all confirm the eval metric is
  trustworthy and `cond(Φ)` vs `cond(ΦᵀΦ)` behave as expected (squaring the
  condition number — why you never form the normal equations when you can avoid it).

Takeaway: **the precision floor is arithmetic, not a solver/eval artifact.** The
training gap is therefore a real optimization phenomenon, not a measurement bug.

### 10.3 `expC01` — the λ U-curve is real

expC01 sweeps `λ` across widths and targets, comparing full QI vs least-squares on
the same geometry (`results/checkpoint_C_geometry/expC01_lambda_tradeoff/`, plots
`consolidated_linf.png`). It confirms the Theorem-1 tradeoff empirically:

- A clear **U-shaped** error-vs-`λ` curve, minimized around **`λ ≈ 0.23–0.26`**.
- Below ~0.15: ill-conditioning (the diverging prefactors) dominates.
- Above ~0.5: aliasing dominates.
- The optimum is shared across targets and widths — i.e. `λ*` is a property of the
  *method*, not the function. This is what licenses hard-coding `λ* ≈ 0.25–0.30`.

Why it matters for the big question: unconstrained training lets `λ` drift toward 0
(because `γ` stays `O(1)` while `h→0`). expC01 shows that's precisely the
ill-conditioned side of the U — quantifying *why* drifting `λ` can't reach high
precision.

### 10.4 `expA02` — the readout is easy given the geometry

Covered in detail in §8.2 (including what it does *not* show). The full four-way
grid — **QI vs lstsq × mpmath vs fp64**, same geometry, across widths (target=sine,
`λ*=0.25`, eval `L∞`, from `results/checkpoint_A_numerics/expA02_qi_vs_lstsq/data.json`):

| N | W | QI mpmath | QI fp64 | lstsq fp64 | lstsq mpmath |
|---:|---:|---:|---:|---:|---:|
| 32 | 173 | 1.7e-14 | 8.8e-11 | 1.6e-13 | 1.6e-15 |
| 64 | 205 | 3.0e-15 | 1.7e-10 | 6.6e-14 | 8.2e-16 |
| 96 | 237 | 2.7e-15 | 3.3e-11 | 2.8e-13 | 1.0e-15 |
| 128 | 269 | 2.3e-15 | 1.3e-10 | 1.4e-13 | 8.0e-16 |

Reading the columns: both **mpmath** columns ride at machine epsilon (~1e-15); the
**fp64** columns sit at their arithmetic floors. Note QI fp64 here is ~1e-10, *worse*
than the ~1e-12 in §9 — because expA02 uses `λ*=0.25` (the mpmath-optimal value, so
both methods share one geometry), and at that smaller `λ` the fp64 Toeplitz +
convolution are more ill-conditioned (§6 U-curve). lstsq fp64 dodges that
cancellation (it fits the residual directly) and stays ~1e-13. So precision is
gated by *arithmetic* per column, not by the method — and given the geometry, the
readout reaches the floor either way.

**Working hypothesis the repo currently operates under** (scoped, not yet proven —
see §8.2): since the readout is easy, the difficulty must lie in *discovering the
geometry* (`γ ∝ N`, grid-spaced centers). That hypothesis is what points the
geometry experiments (expD01 ladder, exp13 solution basins) and the
reparameterization/VarPro stubs (expD03/expD04) at the geometry — they are what would
actually test it.

### 10.5 What is *not* done yet

The remaining stubs are exp13 (solution basins), expD03 (reparameterization), and
expD04 (varpro) — scaffolded but unimplemented (docstring + `# TODO`). They are the
planned diagnostics that would fill in the paper's Section 4.3 story (the solution
landscape near the optimum: Hessian spectrum + basin/perturbation/recovery + path
interpolation; reparameterization; VarPro). See `docs/future_experiments.md`.

---

## 11. How this maps to Section 4 (Experiments) of the paper

Now the payoff. The paper's `QIs_workshop.pdf` Section 4 is organized exactly
around the construction-vs-training gap, and every part of it has a counterpart in
this repo. (Note: the *figures* in the PDF were produced from a fuller pipeline;
the repo's implemented experiments cover the foundations and some of §4.2, with
§4.3's remaining diagnostics scaffolded as exp13, expD03, and expD04.)

**The paper's framing (abstract + Fig. 1).** "Optimization, not expressivity, is
the bottleneck." Fig. 1 shows three panels that *are* the thesis: (left) QI kernels
with halo nodes outside `[−1,1]`; (middle) relative L₂ vs width — the explicit QI
interpolant rides down to the fp64 floor while trained MLPs plateau ~3 orders
higher; (right) mean `λ` vs width — QI plateaus at `λ*`, trained networks drive
`λ → 0`. Your `results/setup/` and expC01 plots are the repo's versions of the
middle and right panels.

**§4.1 — "Direct training fails to reach machine precision" (Fig. 3).** Across
optimizers (Adam, BFGS, LBFGS, LM, SSBroyden, …), activations (GELU/tanh/sech/…),
and targets, training shows geometric gains at small width then **saturates far
above fp64**, with widening giving diminishing/non-monotone returns.
- *Repo counterpart:* not yet implemented as a training experiment — this is the
  motivation the diagnostic experiments serve. The infrastructure exists
  (`src/training/`, the multi-stage Adam→LBFGS loop, the metric schema), so this is
  the most direct "fill in Section 4.1" task. expA01 establishes that the gap these
  curves show is genuinely optimization, not numerics.

**§4.2 — "MLPs can realize quasi-interpolants" (Fig. 4).** Fix the grid and
bandwidth `γ`, so training reduces to fitting only the **output coefficients** —
i.e. a least-squares solve on the Φ features. Result: the learned coefficients
match the samples at grid points, and the interpolant converges geometrically to
the fp64 floor (for Gaussian and sinc kernels). Trained nets also place centers
outside the domain (the halo).
- *Repo counterpart:* **this is exactly expA02 and the `readout.py` Φ machinery.**
  "Fix geometry, fit output coefficients by least squares" is `build_phi` +
  `solve_readout`. expA02's finding (fp64 lstsq ≈ 1e-13 on correct geometry) is the
  repo's version of Fig. 4c. The §8.1 "two solves" distinction is precisely the
  §3-vs-§4.2 distinction in the paper.

**§4.3 — "Empirical diagnosis" (Figs. 5–7).** Three sub-findings, the three
"violations":
- *Weight-magnitude scaling (Fig. 5).* Explicit QI grows `γ` with width (keeping
  `λ=γh` constant) and keeps outer weights `O(1)`; trained nets keep `γ=O(1)` (so
  `λ→0`) and use much larger outer weights — relying on cancellation among
  overlapping features. → **violation #1 (γ scaling) and #2 (weight blowup).**
  - *Repo counterpart:* the metric schema already logs `γ`, `λ=γh`, and outer-weight
    norms at every eval step (`src/training/metrics.py`), so this plot drops out of
    any training run; expD01 (ladder) and exp13 (solution basins) are designed to produce it.
- *Rank saturation / node utilization (Fig. 6).* Freeze the hidden layer, form the
  feature system `Φa=b` with `Φ_ij = tanh(γ_j(x_i−x_j))`, and use OMP to count how
  many neurons you can delete while keeping error below a threshold. Trained nets
  are **much more compressible** (a few neurons do the work — rank saturation); QI
  nets spread the work across all neurons. → **violation #3 (rank saturation).**
  - *Repo counterpart:* `Φ` is `readout.build_phi`; the metric schema already
    computes `feature_rank` and `feature_stable_rank` from the SVD of `Φ`. The OMP
    pruning curve is the planned expD01 (ladder) training analysis (feature-matrix
    conditioning itself is covered in expA03/expA04/expC04). This is the *same Φ* you
    asked about — here used as a diagnostic rather than a solver.
- *Endpoint Hessian curvature alone doesn't explain it (Fig. 7).* QI solutions
  don't have systematically larger top-Hessian curvature than trained ones, so the
  gap isn't simple endpoint conditioning — it's more consistent with a
  *representational* mismatch (how width is used). → motivates exp13 (solution basins)
  and the reduced-coordinate view (expD04 VarPro).

**§Discussion.** "The first explicit MLP construction achieving machine precision
with `log(1/ε)` parameter scaling, realizable in fp64; optimization, not
expressivity, is the bottleneck." Everything in this repo is in service of that one
sentence: Section 3 (the construction, = `qi_mpmath.py`) proves the representation
*exists* in fp64; Section 4 (the experiments, = `experiments/` + `readout.py`'s Φ
machinery) is the campaign to show training can't find it *yet* — and to figure out
how to make it.

---

## 12. One-paragraph summary

`qi_mpmath.py` is the paper's **Section 3 construction**: it builds a specific
fp64 `tanh` MLP that interpolates a target to machine precision. It does so by (1)
solving a **Toeplitz** system for target-independent **cardinal coefficients**
`c_j` — that solve *is* the cardinal/interpolation condition written at grid
points, Toeplitz because the uniform grid + distance-only kernel make it
shift-invariant; (2) **convolving** `c_j` with the target's derivative to get outer
weights `a_m` (derivative because `sech² = tanh'`); (3) pinning a **bias** at the
boundary. The **Φ matrix / least-squares** solve is a *separate* tool — it fits the
outer weights of a *fixed-geometry* MLP directly, and it's both an alternative to
the convolution (expA02) and the paper's diagnostic for trained networks (§4.2,
§4.3). **mpmath vs fp64** is not a contradiction: the model is always fp64; mpmath
just computes the offline coefficient *constants* accurately enough to dodge the
catastrophic cancellation that floors the fp64 computation at ~1e-12, yielding fp64
constants good to ~1e-15 — the `numpy.pi` move. And the whole repo is the empirical
build-out of the paper's thesis: the machine-precision MLP *exists* (Section 3 ✓);
the open problem is teaching an optimizer to *find its geometry* (Section 4, in
progress).

---

## 13. A small experiment worth running: *how close* are lstsq and QI?

**The gap it fills.** expA02 established that lstsq and QI agree *as functions*
(~1e-13 eval error). It never measured whether they agree *as coefficients* — which
is the paper's actual closeness claim (Fig 4b: fitted `v_j ≈ f(x_j)`). So the one
thing that would turn "lstsq also hits the floor" into "lstsq *recovers* the QI
solution" — the nail in the coffin for the switch — has not been checked in this
repo. This experiment checks it, directly, in this repo's coordinates.

**What to run** (small — reuses existing functions, no new machinery):

1. Fix a QI geometry: `qi = construct_qi(target.fn_numpy, target.deriv_numpy, N, …)`
   → gives the QI weights `a_QI = qi.a_coeffs` and `centers, gamma`.
2. Fit the readout by least squares on the *same* geometry:
   `Phi = build_phi(x_train, gamma_vec, centers)` (`readout.py`),
   `a_LS, b, info = solve_readout_with_bias(Phi, y_train, method="svd")`.
3. Report, per width `N ∈ {32,64,128,256}`:
   - **coefficient distance** `‖a_LS − a_QI‖ / ‖a_QI‖` (the number expA02 never logged),
   - the **conditioning** `info["cond"]` of `Φ`,
   - alongside the eval `L∞` already measured.
4. Repeat in the **cardinal basis** (`L_h` features) to reproduce the paper's Fig 4b,
   and once with a **degraded geometry** (`γ = O(1)`, or randomized centers) as a
   control.

**Why it's useful.**
- It directly tests the justification for using lstsq: if `a_LS ≈ a_QI` (or
  `v_LS ≈ f(x_j)` in the cardinal basis), lstsq is *recovering* QI, not replacing it.
- It makes the basis/conditioning story concrete instead of asserted. Prediction
  (stated as a hypothesis, to be confirmed): in the well-conditioned **cardinal**
  basis the coefficient distance is tiny and flat in `N`; in the ill-conditioned
  **tanh** basis the function still matches but the coefficient distance grows with
  `cond(Φ)` — i.e. the gap *is* rank saturation (§4.3), viewed from the weights.
- The degraded-geometry control turns "geometry is the bottleneck" from working
  hypothesis (§8.2) toward evidence: recovery should hold on the QI grid and fail
  off it.

This is essentially the first rung of expD01 (geometry ladder) plus the paper's
Fig 4b, scoped down to a single, cheap, high-information plot.

---

## 14. Appendix: paper ↔ code map

One table to translate from a spot in the paper to the spot in the repo. (Doc §
column points back into this file for the intuition.)

| Paper | Symbol / eq | This repo (file · symbol) | Doc |
|---|---|---|---|
| One-hidden-layer tanh MLP | `g_MLP`, eq. `mlp-def` | `models/mlp.py · QIMlp`; `models/layers.py · GammaLinear` | §1 |
| Kernel = activation derivative | `K = ψ' = sech²`, eq. `kernel-derivative` | `qi_mpmath.py` (`Kd = γ·sech²(γx)`) | §2 |
| Dimensionless bandwidth | `λ = γh`, eq. `lambda-def` | `construct_qi`: `gamma = lambda_star / h` | §2, §6 |
| Quasi-interpolant | `Q_h f`, eq. `qh-infinite` | (conceptual; realized by steps below) | §3 |
| Cardinal coeffs (Toeplitz) | `c_j`, eq. `fourier-character` / Alg. 1 | `qi_mpmath.py · _build_toeplitz_c_f64 / _mpmath` | §3.1–3.2 |
| Halo `R`, stencil `K_c`, sample set | `I_{R,K_c}` | `default_halo(...)`; `Kc=` arg; sample range in `_build_a_*` | §3.3, §4 |
| Outer weights (convolution) | `a[m]`, eq. `single-kernel-sum` | `qi_mpmath.py · _build_a_f64 / _build_a_mpmath_kahan` | §4 |
| Bias / integration poly | `p_{r-1}` → bias, eq. `integration-poly` | `qi_mpmath.py · _compute_c0_f64 / _mpmath` | §4 |
| Construction → model params | — | `construction/initialize.py · initialize_from_construction` | §5 |
| Error bound / λ tradeoff | Thm. 1, eq. `main-error-bound` | expC01 (`experiments/expC01_lambda_tradeoff/`) | §6 |
| fp64 vs extended precision | App. / `practical_implementation.tex` | `construct_qi(precision=...)`; `precision` config | §9 |
| §4.2 fix geometry, fit coeffs | `f = Σ_j v_j L_h(·−x_j)`, Fig 4 | `construction/readout.py · build_phi`, `solve_readout` | §8, §8.2 |
| §4.3 rank saturation | `Φa=b`, `Φ_ij = tanh(γ_j(x_i−x_j))` | `readout.py · build_phi`; `training/metrics.py` (`feature_rank`) | §8.5, §4.3-ref |
| §4.3 weight-scaling mismatch | `γ`, `‖a‖` vs width | `training/metrics.py` (gamma/lambda/readout stats) | §9-ref |

If you only remember one row: **the construction = `qi_mpmath.py` (Toeplitz → convolution → bias); the readout/diagnostic = `readout.py`'s `Φ`.** Everything else hangs off those two.
