# Structured quasi-interpolation networks: the compositional ridge-QI theory (Version 2, consolidated)

**Status:** theory of record for checkpoint I (2026-09-01). Sam's Version 2 with the record's corrections folded in. Each claim is tagged **[theorem]**, **[exact in 2-D]**, **[measured]** (with the experiment), **[derived]** (follows from the record, not yet measured) or **[open]**. Sources: the workshop paper (Theorem 14, Appendix C), the Section 3 rewrite (Theorem 1), `docs/ridge_quadrature_theory.md`, checkpoints A, B, C, H, and `~/Downloads/SQIN_corrections.md`.

**The point in one paragraph.** One-dimensional QI reaches machine precision with $O(\log 1/\varepsilon)$ units on analytic targets; the single-hidden-layer ridge lift pays a direction tax $(kr)^{d-1}$ that walled off 4-D at 25 to 50k units. The compositional block trades that tax for scalar resolution: a few channels, each a short ridge sum of 1-D QI profiles, each followed by a 1-D QI outer function. If the target admits a short factorization whose scalar pieces are quantitatively analytic, the total error is a weighted sum of 1-D QI errors and decays exponentially in the scalar resolutions. The structure contains 1-D QI, the Radon ridge model and the Kolmogorov-Arnold form as special cases; KAT supplies universality and nothing about rates. What is open is whether useful targets factor and whether the factorization can be found from data.

Notation: $d$ input dimension; $M$ directions; $N_1$ offsets per direction; $K$ channels; $N_2$ outer offsets per channel; $\mathfrak r$ the data radius about the origin $x_0$; $\lambda=\gamma h$.

---

## 1. Two claims

**Unconditional [theorem].** The untied block of Section 5 is dense in $C(K)$. This already follows from the one-hidden-layer ridge-QI model it contains (universal approximation of tanh networks); the Kolmogorov-Arnold containment (Section 4) is a second proof and adds nothing quantitative.

**Conditional [theorem, Section 6].** If $f=b_0+\sum_{k}g_k\big(b_k+\sum_r p_{kr}(v_r^\top(x-x_0))\big)$ with $K$ and the number of active profiles moderate, the $p_{kr}$ and $g_k$ quantitatively analytic, and the directions accurate, then the block's error is bounded by a weighted sum of 1-D QI errors and decays exponentially in $N_1$ and $N_2$ until the numerical floors.

Roles: 1-D QI gives scalar precision; ridge theory gives directional scalar coordinates; composition gives short nonlinear factorizations; KAT gives universality only; structured linear algebra makes it an ordinary tanh MLP.

---

## 2. The one-dimensional construction and its numbers

### 2.1 The cardinal construction [theorem]

Grid $x_k=a+kh$, $\gamma=\lambda/h$, kernel $K=\psi'=\mathrm{sech}^2$ for tanh, $K_\gamma(t)=\gamma K(\gamma t)$. The cardinal function $L_h(x)=\sum_jc_jK_\gamma(x-jh)$ with $L_h(kh)=\delta_{k0}$; coefficients from the Toeplitz system $\sum_{|j|\le K_c}c_j\,hK_\gamma((k-j)h)=\delta_{k0}$, equivalently $\widehat C_h=h/D_h$ with $D_h(\omega)=\sum_m\widehat K_\gamma(\omega+2\pi m/h)$. The operator is $(Q_hf)(x)=\sum_kf(x_k)L_h(x-x_k)$, applied to $f'$ and integrated (derivative trick):

$$\widetilde f(x)=b+\sum_{m=-R}^{N+R}w_m\tanh(\gamma(x-x_m)),\qquad w_m=\sum_{|j|\le K_c}c_jf'(x_{m-j}).$$

Parameters of record: $K_c=160$, halo $R=\max(\lceil35/(2\lambda)\rceil,\lceil0.4N\rceil)$, $|c_0|\approx338$ at $\lambda=0.30$ with alternating signs. Not the naive kernel sum $\sum_kf(x_k)K_\gamma(x-x_k)$, whose error is first order.

### 2.2 The error theorem [theorem]

For $f$ analytic in a strip (mapped Bernstein ellipse, assumption T1) and the normalizer bounded away from zero on a strip,

$$\|f-Q^{(K_c)}_{h,R}f\|_\infty\le C_1(\lambda)e^{-c_1/h}+C_2e^{-c_2/\lambda^p}+C_3(\lambda)e^{-c_3\lambda R}+C_4(\lambda)e^{-c_4\lambda K_c},\qquad p=1\ (\mathrm{sech}^2),\ p=2\ (\text{Gaussian type}).$$

The prefactors $C_1,C_3,C_4$ diverge as $\lambda\to0$ (the conditioning wall). The resolution term is exponential only because the target is analytic. **[measured]** Analytic targets reach the $5\times10^{-14}$ floor by $N=64$ to $256$ (expB02); $|x|^3$ ($C^2$) is still $2.4\times10^{-11}$ at $N=1024$ for all six activations (expA07); kinks $10^{-5}$ and steps $10^{-2}$ (expH01); relu basis $N^{-2}$ (expB02).

### 2.3 What sets $\lambda$ [measured, approved: expC07]

$A(\lambda)=|\widehat K(2\pi/\lambda)|/|\widehat K(0)|$, the first Poisson alias; for $\mathrm{sech}^2$, $A(\lambda)=\frac{\pi^2/\lambda}{\sinh(\pi^2/\lambda)}$, and $A(0.25)=4\pi^2/\sinh(4\pi^2)=5.7\times10^{-16}$. The rule anchored at tanh predicts sigmoid $0.500$ (exact identity), gaussian $0.530$, gelu $0.707$ (measured $0.716$), swish $0.455$ (measured $0.437$); what matters is the kernel's analyticity class, not the derivative order or any energy. Two walls: aliasing on the right, cardinal conditioning on the left (fp64 construction at $0.30$; machine precision at $0.25$ with 30-digit cardinal coefficients rounded to fp64). With a least-squares readout the basin is wide, $[0.14,0.27]$, width independent (expC03). $\lambda=0.25$ is the fp64 alias-limited operating point for this normalization, not a universal tanh constant.

### 2.4 The working solve [measured, approved]

QI chooses the geometry; a truncated-SVD least squares on $[\Phi,\mathbf 1]$ chooses the coefficients: at least as accurate as the convolution in 48 of 48 cells and $10^{-13}$ against $10^{-10}$ in fp64 (expA02); normal equations lose seven decades (expA01); the null space is $\approx108$-dimensional at every width and data does not shrink it (expA03); the halo (1-D) or the 25% collar (ridge) is necessary (expA01, expH01); Adam on the frozen geometry stalls at $10^{-3}$ where the solve gives $10^{-13}$ (expD01). The min-norm readout is, to leading order, the sampled derivative $v_k\approx\tfrac h2f'(c_k)$ with $\|v_{\rm inner}\|=\|f'\|_{L_2}/\sqrt{2N}$ (expA06, expA07); the right initial readout magnitude is $1/\sqrt N$, not $O(1)$. rcond $10^{-13}$ in 1-D, $10^{-14}$ on wide ridge dictionaries (expH06).

### 2.5 The center geometry [measured]

Of the $2N$ inner parameters, one continuous knob survives (the shared $\lambda$) plus a deterministic grid: centers uniform or a smooth deformation of it, biases derived $b=-\gamma c$, signs irrelevant, uniform weights only after uniform centers (one-way coupling; expC03 to C06). Smooth non-uniform spacing with $\gamma_jh_j=\lambda$, $h_j=(c_{j+1}-c_{j-1})/2$, costs nothing (a $3\times$ density change reaches the floor at every width, expH02); the widest gap decides the width; a jump that does not shrink stalls the error. Gap-scale jitter of 2% costs $10^{-15}\to7\times10^{-11}$; the mesh is a coordinate map $c_j=\Phi(jh_0)$ and must be band-limited above $L^\star\approx12$ gaps, which fixes any monitor smoothing at $\sigma\ge5.8$ gaps (expH04). Status: uniform mesh [theorem]; smooth mesh map [measured, structurally motivated]; independent jitter outside the theory. A variable-mesh theorem needs a uniform inverse bound for the non-Toeplitz system **[open]**.

### 2.6 Data [measured, approved]

Centers decide precision, not samples; $x$-jitter harmless; $y$-noise floor $\sigma n^{-1/2}$ (expB01), in the ridge setting $\|w\|\varepsilon_y/\sqrt n$; $n\ge8$ to $16$ times the unit count for the floor (expH02).

---

## 3. The ridge extension

### 3.1 Directional coordinates

$F_V(x)=\sum_{r=1}^{M}p_r(v_r^\top(x-x_0))$, each profile 1-D and QI-approximable.

### 3.2 The finite-direction certificate [theorem]

With $F(x_0+z)=\int e^{i\xi^\top z}d\mu(\xi)$ and $\int\|\xi\|d|\mu|<\infty$, snap every frequency to its nearest line in $V$ preserving $\|\xi\|$. The snapped function is an exact finite ridge sum and

$$\|F-F_V\|_{L^\infty(B_{\mathfrak r}(x_0))}\le\mathfrak r\int\|\xi\|\,\theta(\hat\xi,V)\,d|\mu|(\xi)\le\mathfrak r\,\Theta_V\int\|\xi\|\,d|\mu|,\qquad\Theta_V\sim c_dM^{-1/(d-1)}.$$

The directional scale is radius times frequency times angular error. A target on a few spectral lines costs nothing once those lines are in $V$ (the atoms theorem), and a direction error $\delta$ still costs $k\mathfrak r\delta$. The profiles carry $F$'s radial spectrum on a cone (Fourier slice; the $|\omega|^{d-1}$ of filtered backprojection is the polar Jacobian), so the 1-D leg costs $\|\mu\|Ce^{-\alpha N}$ with no factor of $M$ or $d$ **[measured: $e_N$ flat in $M$, identical in 2, 3, 4-D]**. Adding the profile errors: $\|F-\widehat F\|\le\mathfrak r\int\|\xi\|\theta\,d|\mu|+\sum_rA_r\delta_r$ with $A_r$ the profile masses.

### 3.3 Rates and allocation [measured]

- Angular error is a cliff, not a Hölder power: plateau, then 5 to 9 orders over two or three steps of $M$, then the floor (expH05, nine targets); fits $e_M\sim e^{-aM^q}$ with $q=0.9$ to $2.1$; exact in 2-D by Jacobi-Anger **[exact in 2-D]**; 2.5 orders per doubling of $M$ in 3-D, 1.3 to 1.5 in 4-D (expH06).
- Direction count: $M\approx\max\big(\binom{p+d-1}{d-1},c(k\mathfrak r)^{d-1}\big)$, $p\approx12$; 2-D: 12 at $k\mathfrak r\approx1$, 24 at $7.5$, against 64 domain-centered; 3-D: 64, 128, 256 at $k\mathfrak r\approx1,2,4$ (expH04). The origin is free and the data radius, not the domain, sets the count.
- Two floors: $e(M,N_1)=\max(e_M,e_N)$ with the bracket $\max\le e\le\sqrt2\max$; predicts every above-floor cell within 21% in 2-D, 46 of 48 within 10% in 3-D, all 4-D cells at 0.97 to 1.00. $e_N\sim N^{-10}$ in 2-D **[open why]**. Optimal split $M^\star\propto B^\alpha$, $\alpha=0.27$ to $0.44$. 3-D floor at $(192,48)$, 9216 units; 4-D crossing near 25 to 50k units: **4-D needs packing, not scale**.
- Readout norm tracks the error over 24 orders: a reference-free diagnostic.
- Far field: off the data, saturated walls give $O(1)$ error ($3\times10^{-14}$ on a sheet, $5\times10^{-1}$ off it, expH01).

### 3.4 Learning directions [measured]

A rotation with $M=d$ buys nothing. What pays: the active subspace (eigenvectors of $\mathbb E[\nabla F\nabla F^\top]$) when the active dimension is below $d$ (composition in $d=3$: $4\times10^{-7}\to2\times10^{-11}$; sheet in $d=5$: $10^{-6}\to5\times10^{-11}$, expH04), and exact ridge atoms by projection pursuit plus variable-projection Gauss-Newton (sums of 1 to 8 hidden ridges recovered to $10^{-13}$ with directions exact to fp64 in $d=3,4$, expH06). Smooth angular densities cannot find a ridge (a delta in angle). A $0.01^\circ$ tilt on a 7.5-period ridge costs $3\times10^{-3}$. Sequential re-aiming at the residual is essential; residual-graded offsets did not pay at uniform data.

---

## 4. The role of Kolmogorov-Arnold [theorem, negative]

Strong form: $f=\sum_{k=1}^{2d+1}g\big(\sum_i\alpha_i\phi_k(x_i)\big)$ (Lorentz: one $g$; Sprecher: shifted copies of one $\phi$). If the scalar functions are approximated by dense 1-D QI spaces, universality follows. But the universal inner functions can be Lipschitz, not $C^1$ (Fridman; Vitushkin 1954, Vitushkin-Henkin 1967), the constructive ones are Cantor-type staircases, and $g$ is only continuous even for analytic $f$. Such factors are outside Theorem 1's hypothesis; a Lipschitz scalar factor has an algebraic rate, and by the naive first-order bound reaching $10^{-13}$ would need $\sim10^{13}$ centers per channel. KAT controls the number of channels, not the scalar resolution inside them; the angular complexity is encoded into rough scalar functions, not removed. (Girosi-Poggio 1989; Kůrková 1991, 1992.)

The rank-1 tie $C_{rjk}=a_r\Phi_{jk}$ gives $P_k=\sum_ra_r\phi_k(v_r^\top x)$: one profile shape per channel on every direction with a shared weight vector. It cannot build $p_1(v_1^\top x)+p_2(v_2^\top x)$ with unrelated $p_1,p_2$, and splitting into channels does not help for $g(P)$ (the mixed derivative of $e^{s+t}$ is nonzero, that of $u(s)+v(t)$ is zero). On the record's own composition target ($\exp$ of three ridges with two profile shapes) it fails with smooth pieces. Keep it as the universality subfamily and as the most rigid point of a rank sweep.

---

## 5. The primary architecture

$$H_1(x)_{rj}=\tanh\big(\gamma_1[v_r^\top(x-x_0)-c_j]\big),\qquad P_k(x)=b_k+\sum_{r,j}C_{rjk}H_1(x)_{rj},\qquad C\in\mathbb R^{M\times N_1\times K},$$
$$\hat f(x)=b_0+\sum_{k=1}^{K}\hat g_k(P_k(x)),\qquad\hat g_k(t)=\sum_{\ell=1}^{N_2}\Psi_{k\ell}\tanh\big(\tilde\gamma_{k}[t-\tilde c_{k\ell}]\big).$$

Submodels: shallow ridge-QI ($K=1$, outer map the identity or absent); strong KAT-QI ($M=d$, $v_r=e_r$, $K=2d+1$, $C_{rjk}=\alpha_r\Phi_{jk}$, $\Psi_{k\ell}=\psi_\ell$); coordinate-transformed KAT (any invertible $d$-frame). The rank family $C_{rjk}=\sum_{s=1}^{S}a^{(s)}_r\Phi^{(s)}_{jk}$ interpolates from KAT ($S=1$) to untied. **[derived]** The natural relaxation is per-channel rank, $C_k=\sum_sa_k^{(s)}\phi_k^{(s)\top}$, where $S$ counts distinct profile shapes in a channel; $S=1$ suffices only when the origin sits on the anchor, a quadratic about an off-origin anchor is rank $d$ under a common mesh, composition is $S=2$, the two-anchor packet needs $S\ge3$ with shared $a$. The tie saves parameters only when $M$ is large, which is exactly when the block has no dividend; its practical value is a smaller search space.

---

## 6. The central conditional theorem

### 6.1 Target class and the error bound [theorem]

$f(x)=b_0+\sum_{k=1}^{K}g_k(P_k(x))$, $P_k(x)=b_k+\sum_{r=1}^{M}p_{kr}(v_r^\top(x-x_0))$, some $p_{kr}$ possibly zero. Let $I_r$ be the projected band, $\varepsilon_{kr}=\|p_{kr}-\hat p_{kr}\|_\infty$ on an expansion of $I_r$, $\delta_r=\|v_r-\hat v_r\|_2$, $L_{kr}=\sup|p_{kr}'|$. Since $K\subset B_{\mathfrak r}(x_0)$, $|v_r^\top(x-x_0)-\hat v_r^\top(x-x_0)|\le\mathfrak r\delta_r$, hence

$$|p_{kr}(v_r^\top(x-x_0))-\hat p_{kr}(\hat v_r^\top(x-x_0))|\le\varepsilon_{kr}+L_{kr}\mathfrak r\delta_r,\qquad\|P_k-\hat P_k\|_\infty\le\eta_k:=\sum_r(\varepsilon_{kr}+L_{kr}\mathfrak r\delta_r).$$

With $\varepsilon_k^{\rm out}=\|g_k-\hat g_k\|_\infty$ and $L_k=\sup|g_k'|$ on an interval containing the $\eta_k$-neighbourhood of $P_k(K)$,

$$\boxed{\|f-\hat f\|_\infty\le\sum_{k=1}^{K}\big[\varepsilon_k^{\rm out}+L_k\eta_k\big].}$$

*Proof.* For $x\in K$: $|g_k(P_k)-\hat g_k(\hat P_k)|\le|g_k(P_k)-g_k(\hat P_k)|+|g_k(\hat P_k)-\hat g_k(\hat P_k)|\le L_k|P_k-\hat P_k|+\varepsilon_k^{\rm out}\le L_k\eta_k+\varepsilon_k^{\rm out}$; sum over $k$ and take the supremum. $\square$

### 6.2 Quantitative analyticity

Define $\mathcal A_\rho(I;B)$: $u$ extends holomorphically to $\Omega_\rho(I)=\{z:\mathrm{dist}(z,I)<\rho\}$ with $\sup_{\Omega_\rho}|u|\le B$. Cauchy gives $\sup_I|u'|\le B/\rho$. So $p_{kr}\in\mathcal A_{\rho_{kr}}(I_r;B_{kr})$ gives $L_{kr}\le B_{kr}/\rho_{kr}$ and $g_k\in\mathcal A_{\tau_k}(J_k;G_k)$ gives $L_k\le G_k/\tau_k$. "Analytic" alone is too weak ($\cos(10^6t)$ is entire); the radii, bounds, interval lengths and effective frequencies are the inputs.

### 6.3 With the QI estimates [theorem]

With $E^{(1)}_{kr}$ the four-term bound of 2.2 for $p_{kr}$ and $E^{(2)}_k$ for $g_k$,

$$\boxed{\|f-\hat f\|_\infty\le\sum_{k=1}^{K}\Big[E^{(2)}_k+\frac{G_k}{\tau_k}\sum_{r=1}^{M}\Big(E^{(1)}_{kr}+\frac{B_{kr}}{\rho_{kr}}\mathfrak r\delta_r\Big)\Big].}$$

Four contributions: first-level scalar QI error, direction error, outer scalar QI error, downstream sensitivity. No multivariate tensor-product error appears.

### 6.4 Exponential-rate corollary [theorem]

If $E^{(1)}_{kr}\le A_1e^{-a_1N_1}$ for the $S$ active profiles, $E^{(2)}_k\le A_2e^{-a_2N_2}$, $L_k\le L$, $\delta_r=0$: $\|f-\hat f\|_\infty\le KA_2e^{-a_2N_2}+LSA_1e^{-a_1N_1}$, so $N_1\ge a_1^{-1}\log(2LSA_1/\varepsilon)$ and $N_2\ge a_2^{-1}\log(2KA_2/\varepsilon)$ suffice above the floors, with $MN_1+KN_2$ units. The theorem covers the class that admits a controlled factorization and says nothing about how large that class is.

### 6.5 What the record adds [derived]

- **The composed fp64 floor.** Roundoff in $P_k$ is amplified by $L_k$: floor $\approx\varepsilon_{\rm mach}(1+\sum_kL_k\|P_k\|_\infty)$, a compositional condition number invariant under rescaling a channel. Fast waves: $L\approx18\pi^2$, so about $10^{-14}$. Measure it first (expI01 E1).
- **Existence versus learnability.** In the shallow model the certificate transfers to the solve because the readout is a projection. Here $C$ sits inside the level-2 tanh, so certificate-to-projection covers only $\Psi$; finding $C,V$ is projection-pursuit regression with nonlinear indices (alternate: solve $g_k$ given $P_k$, move $P_k$ given $g_k$). For $K=1$, $\nabla f=g'(P)\nabla P$ makes $P$ identifiable up to a monotone reparametrization (the level sets of $f$); for $K\ge2$ it is not unique **[open]**.
- **Quadratic channels need no direction learning.** $\|x-a\|^2=\sum_i(x_i-a_i)^2$ in any orthonormal frame; the shift lives in the profile. The radial family is served by the $d$ axis blocks; only ridge-structured channels need learned directions.
- **The dividend on the record's targets.** Every smooth 3-D target of expH06 is $\sum_kg_k(P_k)$ with $P_k$ a quadratic or a short ridge sum and $g_k$ analytic (fast waves: $g(s)=\cos(6\pi\sqrt s)$, entire in $s$; composition: $\exp$ of three ridges; packet: two quadratics). A quadratic channel is three directions at a dozen offsets; an outer $g$ is one block of $64$ to $128$; order $10^2$ units against the shallow model's $9216$ to $12288$. Caveats: unmeasured; the suite's targets are compositional by construction and its separability test only excludes sums of 1-D profiles, so a control with no short factorization is required (random ridges: shallow with exact directions is the shortest form).

---

## 7. What the extra expressiveness buys

Different profiles on different directions (needed for $\tfrac12\sin\pi(x_1+x_2)+\tfrac12\sin\pi(x_1-x_2)+\tfrac12\cos(\pi x_3+1)$ followed by $\exp$); several channels ($K$ is a compositional rank, not $2d+1$; two anchors need two quadratic channels); separate outer functions (no precision reason to share $g$); learned directions only when there is a low active dimension or a few ridge atoms. Every finite tanh network is real analytic on $\mathbb R^d$, which is irrelevant; the useful property is a factorization whose pieces have radii bounded away from zero, controlled magnitude, controlled effective frequency, controlled channel ranges and controlled outer derivatives.

---

## 8. It is an ordinary tanh MLP

First layer: $w^{(1)}_{rj}=\gamma_1v_r$, $b^{(1)}_{rj}=-\gamma_1(c_j+v_r^\top x_0)$: directional rays with QI-prescribed spacing. Channel bottleneck: $P_k=b_k+C_k^\top h_1(x)$. Second layer: $W^{(2)}_{(k\ell),(rj)}=\tilde\gamma_kC_{rjk}$, so $\mathrm{rank}(W^{(2)})\le K$ with $MN_1K$ channel coefficients against $MN_1KN_2$ for a dense map. Output: solved $\Psi$. In the ordinary-MLP reading the second layer's biases must follow the first layer's output scale, which is the layer-2 version of the $\lambda\to0$ violation the paper documents for training.

---

## 9. Numerical realization [the notebook's rules]

- Forward: $S=(X-x_0)V^\top$ with $V$ row-normalized; $H_1=\tanh(\gamma_1(S[:,:,\cdot]-c_1))$, $[B,M,N_1]$; $Z=\mathrm{einsum}(H_1,C)+b$, $[B,K]$; $u_k=(Z_k-m_k)/s_k$; $H_2=\tanh(\gamma_2(u[:,:,\cdot]-c_2))$, $[B,K,N_2]$; $y=H_2\cdot\Psi+b_0$. No loop over directions or channels.
- **Range tracking.** $(m_k,s_k)$ from the training data with a 25% collar, re-read on a fixed schedule and before the final solve; $\gamma_2=\lambda N_2/2$ in $u$. Fixing the level-2 biases while the channel amplitudes drift breaks the QI scale relation.
- **Readout.** QR then SVD of $R$, rcond $10^{-14}$; never normal equations; never a gradient parameter in the solved arms.
- **Nonlinear parameters.** $V$ (tangent updates through normalization), $C$ (or $a,\Phi$), $b$. Arms: Adam on everything with a final solve; VarPro (readout re-solved every step); Gauss-Newton with the Kaufman Jacobian $J=-P^\perp(\partial A_2/\partial\vartheta)\Psi^\star$, readout re-solved at every trial.
- **Constraints in the model definition:** $\|v_r\|=1$; $\gamma_1h_1=\gamma_2h_2=\lambda$; fixed centers; collars; zero function coefficients at init with distinct channel biases (a zero-zero init is a stationary point); readout scale $O(N^{-1/2})$ if initialized at all; optionally a small soft-neuron sub-bank (expC06).

---

## 10. Depth and residual streams [theorem, standard]

Vector-valued blocks compose, $F=B_L\circ\cdots\circ B_1$, and $\|B_L\circ\cdots-\hat B_L\circ\cdots\|\le\sum_\ell\delta_\ell\prod_{j>\ell}L_j$. Residual $R_\ell=z+\eta_\ell B_\ell(z)$ has $\mathrm{Lip}\le1+|\eta_\ell|\mathrm{Lip}(B_\ell)$, so $\prod_\ell\mathrm{Lip}(R_\ell)\le\exp(\sum_\ell|\eta_\ell|\mathrm{Lip}(B_\ell))$: tame composition needs $\sum|\eta_\ell|\mathrm{Lip}(B_\ell)$ controlled. Explicit bounds: $\mathrm{Lip}(P_k)\le\sum_{r,j}|C_{rjk}|\gamma_1$ and $\mathrm{Lip}(\hat f)\le\sum_{k,\ell}|\Psi_{k\ell}|\tilde\gamma_k\mathrm{Lip}(P_k)$, conservative but monitorable.

---

## 11. Domain and far field

Every bound is on $K$ and the projected bands. Off the data the tanh walls saturate; the shallow model's far field is a saturated-wall arrangement with $O(1)$ to $10^9$ error (expH04, expH01). The composed block's level-2 features saturate too, so its far field is at least bounded by $\|\Psi\|_1$; that is a bound, not control. Precision on a data manifold does not imply precision in the surrounding volume **[open]**.

---

## 12. Ledger

**Proved:** 2.1, 2.2, 3.2, 6.1 to 6.4, 8, 10.
**Exact in 2-D:** the angular cliff mechanism.
**Measured:** 2.3 to 2.6, 3.3, 3.4; the composition counterexample to the tie is algebra.
**Derived, unmeasured:** the composed floor, the dividend table, direction-free quadratics, per-channel rank predictions.
**Open:** whether broad classes of analytic targets admit short quantitatively controlled factorizations; whether VarPro or Gauss-Newton finds them from data ($K\ge2$ non-unique); a stability theorem for smooth non-uniform meshes; when depth reduces total scalar resolution rather than rearranging it; far-field control.

The research question is no longer whether QI fits inside KAT. It is whether a useful target class decomposes into a few QI-friendly analytic ridge summaries and analytic outer maps, and whether that decomposition can be recovered. expI01 measures the first half on the record's targets and takes the first pass at the second.
