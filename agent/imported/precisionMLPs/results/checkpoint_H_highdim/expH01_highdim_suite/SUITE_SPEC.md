# Checkpoint H (high-dimensional ridge-QI) -- benchmark suite specification

**Status: Sam's specification. Version 3 (at the bottom of this file) is the one that is built and the one that wins wherever the versions disagree. Version 2 (the 60-task suite) is superseded: its targets were finite sums of one-dimensional profiles along fixed directions, which builds the model's own way of representing functions into the benchmark. Version 1 (the 138-task factorial) is context: it defines the data geometries, the interaction experiments, the metrics, and the four-model comparison the suite exists to serve. Read Versions 1 and 2 for framing, Version 3 for what exists.**

Both versions are reproduced verbatim below (a few obvious typos in the pasted text are left as they were).

---

# Version 2 (SUPERSEDED by Version 3): the 60-task designed suite

Yes. I would make the 60 a **designed experiment suite**, not a reduced factorial. The goal should be that when a curve moves, we know what mechanism caused it.

I would deliberately preserve matched pairs: low vs high frequency with the same geometry; uniform vs hotspot sampling with the same target; hotspot aligned vs anti-aligned with the hard region; clean vs noisy manifold with the same target. That makes the suite much more interpretable than 60 unrelated functions.

There is good precedent for this philosophy. The Genz families vary oscillation, peaks, smoothness, and discontinuities across dimension rather than using one generic random-function family, and manifold benchmarks deliberately vary intrinsic dimension, curvature, and sampling geometry. The suite below is more specifically designed around the ridge-QI theory.

## Common definitions

Use the domain

$$
\Omega_d=[-1,1]^d.
$$

To avoid axis-aligned toy problems, define a deterministic orthogonal basis $u_1,\dots,u_d$ using the DCT-II matrix:

$$
Q_{j0}=\frac1{\sqrt d},
\qquad
Q_{jk}=
\sqrt{\frac2d}
\cos\!\left(\frac{\pi(j+\frac12)k}{d}\right),
\quad k>0.
$$

The columns are $u_k$. Define

$$
y_k(x)=u_k^\top x,
\qquad
z_k(x)=\frac{u_k^\top x}{\|u_k\|_1}\in[-1,1].
$$

For an arbitrary unit direction $v$,

$$
z_v(x)=\frac{v^\top x}{\|v\|_1}.
$$

I would use these reusable target atoms:

$$
S_v(\omega,\phi)
=
\sin(\pi\omega z_v+\phi)
$$

for global frequency,

$$
W_v(\omega,\mu,\tau,\phi)
=
e^{-((z_v-\mu)/\tau)^2}
\sin\!\big(\pi\omega(z_v-\mu)+\phi\big)
$$

for localized high frequency,

$$
R_v(\alpha,\mu)
=
\frac1{1+\alpha^2(z_v-\mu)^2}
$$

for Runge-like near-singular analytic behavior,

$$
J_v(\mu)=\mathbf 1_{\{z_v>\mu\}},
\qquad
C_v(\mu)=|z_v-\mu|,
$$

and

$$
P_m(a,b)
=
\prod_{k=1}^m
\frac1{1+a_k^2(z_k-b_k)^2}
$$

as a genuinely multidimensional product-peak control. Product-peak and discontinuous families are standard stress cases in multidimensional interpolation/integration benchmarks.

Normalize every raw target once using a fixed large deterministic uniform reference set:

$$
F(x)=\frac{\widetilde F(x)-\mu_U}{\sigma_U}.
$$

That prevents hotspot datasets from changing the scale of the loss.

For the data geometries, use:

$$
\mathsf{REG}_d:
$$

equispaced midpoints in 1D; deterministic Halton points using bases $2,3,5,7,11$ in $d>1$, mapped to $[-1,1]^d$.

$$
\mathsf U_d=U([-1,1]^d).
$$

For $d\ge2$, define

$$
\mu_+=0.45(1,\dots,1),\qquad
\mu_-=-0.45(1,\dots,1),
$$

$$
\mu_\perp
=
0.35\,\frac{u_2}{\|u_2\|_\infty}.
$$

Then the standard hotspot distribution is

$$
\boxed{
\mathsf G_d
=
.10U
+.45N_T(\mu_+,.08^2I)
+.25N_T(\mu_-,.12^2I)
+.20N_T(\mu_\perp,.10^2I)
}
$$

with rejection outside the cube. Notice

$$
z_1(\mu_+)=0.45,\qquad
z_1(\mu_-)=-0.45.
$$

For $d=1$, use the analogous mixture at $0.45,-0.45,0$.

Also define an anisotropic hotspot distribution:

$$
\mathsf G^{\rm aniso}_d
=
.10U+
.45N_T(.35\mathbf1,\Sigma)
+
.45N_T(-.35\mathbf1,\Sigma),
$$

where

$$
\Sigma
=
Q\,\mathrm{diag}
(.22^2,.045^2,.08^2,\ldots,.08^2)Q^\top.
$$

That makes variation along $u_2$ much less visible in the data than variation along $u_1$.

For clean flat manifolds, use $y=Q^\top x$. In 2D:

$$
y=(t,0),\qquad t\sim U[-.65,.65].
$$

For $d\ge3$:

$$
y=(s,t,0,\ldots,0),
\qquad
s,t\sim U[-.65,.65].
$$

For the noisy version, replace the zero normal coordinates by

$$
\epsilon_j\sim N(0,.05^2).
$$

For curved manifolds:

$$
d=2:\qquad
y=(.65t,\;.25\sin\pi t).
$$

For $d\ge3$,

$$
y=
\left(
.55s,\;
.55t,\;
.22\sin\pi s,\;
.18\sin\pi t,\;
.15\sin\pi(s+t)
\right)_{1:d},
$$

with $s,t\sim U[-1,1]$, and $x=Qy$.

For noisy curved manifolds, add $N(0,.04^2)$ noise projected into the normal bundle.

With that fixed, here are the 60 experiments I would actually run.

---

# Dimension 1: learn the longitudinal problem perfectly first

There is no meaningful lower-dimensional manifold inside $\mathbb R$, so I would devote all 12 tasks to understanding adaptive center placement.

| ID       | Data               | Exact raw target $\widetilde F$   | What it isolates                                       |
| -------- | ------------------ | ----------------------------------- | ------------------------------------------------------ |
| **1.1**  | $\mathsf{REG}_1$ | $S_1(1,.2)$                       | Absolute easiest QI baseline                           |
| **1.2**  | $\mathsf{REG}_1$ | $S_1(12,.2)$                      | Same target geometry, $12\times$ frequency           |
| **1.3**  | $\mathsf{REG}_1$ | $\sin[\pi(4x+2x^2)]$              | Smooth chirp; continuously varying local frequency     |
| **1.4**  | $\mathsf{REG}_1$ | $R_1(14,.15)$                     | Analytic but small analyticity radius                  |
| **1.5**  | $\mathsf U_1$    | $.55S_1(1,-.3)+W_1(12,.45,.16,0)$ | Low frequency + localized high-frequency packet        |
| **1.6**  | $\mathsf U_1$    | $.35S_1(1,0)+J_1(.13)$            | True discontinuity                                     |
| **1.7**  | $\mathsf G_1$    | **same as 1.5**                     | Density adaptation; hard region coincides with hotspot |
| **1.8**  | $\mathsf G_1$    | $.55S_1(1,-.3)+W_1(12,.80,.12,0)$ | High frequency deliberately placed in sparse region    |
| **1.9**  | $\mathsf G_1$    | $R_1(20,.45)+.15S_1(1,.4)$        | Very sharp analytic feature exactly at hotspot         |
| **1.10** | $\mathsf G_1$    | $.3S_1(1,0)+J_1(.78)$             | Sparse-region discontinuity                            |
| **1.11** | $\mathsf U_1$    | $R_1(4,-.35)+.5R_1(22,.52)$       | Asymmetric broad + narrow smooth scales                |
| **1.12** | $\mathsf U_1$    | $C_1(.23)+.25S_1(2,.1)$           | Continuous but non-$C^1$; softer failure than jump   |

Tasks 1.5, 1.7, and 1.8 are especially important:

$$
\text{same local complexity under uniform data}
\rightarrow
\text{aligned hotspot}
\rightarrow
\text{anti-aligned hotspot}.
$$

If the mesh theory is correct, these should produce visibly different center-density profiles.

---

# Dimension 2: first real test of angular resolution

Now direction selection becomes meaningful and still completely visualizable.

| ID       | Data               | Exact raw target                                             | What it isolates                                           |
| -------- | ------------------ | ------------------------------------------------------------ | ---------------------------------------------------------- |
| **2.1**  | $\mathsf{REG}_2$ | $S_{u_1}(1,.2)$                                            | Single low-frequency oblique ridge                         |
| **2.2**  | $\mathsf{REG}_2$ | $S_{u_1}(12,.2)$                                           | Same exact $v$, high longitudinal resolution             |
| **2.3**  | $\mathsf{REG}_2$ | $S_{u_1}(2,.2)+.65S_{u_2}(9,-.4)$                          | Different kernel budgets should go to different directions |
| **2.4**  | $\mathsf U_2$    | $R_{u_1}(16,.18)+.3S_{u_2}(1,.5)$                          | Sharp analytic ridge + easy ridge                          |
| **2.5**  | $\mathsf U_2$    | $J_{u_1}(.08)-.7J_{u_2}(-.22)+.2S_{u_1}(1,0)$              | Two intersecting discontinuity lines                       |
| **2.6**  | $\mathsf U_2$    | $.45S_{u_1}(1,-.3)+W_{u_1}(12,.45,.15,0)+.35S_{u_2}(2,.4)$ | Local longitudinal adaptation with uniform data            |
| **2.7**  | $\mathsf G_2$    | **same as 2.6**                                              | Same function; density now favors the difficult region     |
| **2.8**  | $\mathsf G_2$    | replace $W_{u_1}(12,.45,.15)$ by $W_{u_1}(12,.80,.12)$   | Hard region in sparse tail                                 |
| **2.9**  | $\mathsf G_2$    | $P_2((6,14),(.30,-.25))$                                   | Smooth asymmetric non-low-ridge-rank control               |
| **2.10** | curved clean       | $\sin(6\pi y_1/.65)+.45\sin(4\pi y_2/.25)$                 | Curved 1D support with two ambient ridge coordinates       |
| **2.11** | flat clean         | $\sin(2\pi y_1/.65)+.8\sin(12\pi y_2/.65)$                 | High-frequency normal component is completely unobservable |
| **2.12** | flat noisy         | **same as 2.11**                                             | Tiny normal noise suddenly makes that component relevant   |

I especially like 2.11 $\rightarrow$ 2.12. The target is identical. Only $P_X$ changes. If the learned angular mesh is really data-aware, it should not waste directions on $u_2$ in 2.11, but should begin resolving it in 2.12. That is a remarkably clean diagnostic.

---

# Dimension 3: separate angular complexity from ridge complexity

Define

$$
v_\pm(\theta)
=
\cos\theta\,u_1
\pm
\sin\theta\,u_2.
$$

For task 3.3 use $\theta=6^\circ$, so the true ridge directions are only $12^\circ$ apart.

| ID       | Data                    | Exact raw target                                                                            | What it isolates                                                            |
| -------- | ----------------------- | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **3.1**  | $\mathsf{REG}_3$      | $S_1(1,.1)+.65S_2(1,-.4)+.4S_3(1,.7)$                                                     | Uniformly easy 3-ridge problem                                              |
| **3.2**  | $\mathsf{REG}_3$      | $S_1(1,0)+.65S_2(4,.2)+.4S_3(10,-.3)$                                                     | Explicit per-direction frequency ladder                                     |
| **3.3**  | $\mathsf{REG}_3$      | $.7S_{v_+}(8,.1)+.7S_{v_-}(8,-.1)$                                                        | Can the angular mesh separate nearby directions?                            |
| **3.4**  | $\mathsf U_3$         | $R_1(18,.15)+.45R_3(8,-.25)+.25S_2(1,0)$                                                  | Two different analyticity radii                                             |
| **3.5**  | $\mathsf U_3$         | $.4S_1(1,0)+W_1(14,.40,.14,0)+.5S_3(3,.2)$                                                | Local high-frequency region on only one ridge                               |
| **3.6**  | $\mathsf U_3$         | $J_2(.12)+.35S_1(2,0)-.25S_3(1,.4)$                                                       | Oblique discontinuity plane                                                 |
| **3.7**  | $\mathsf G_3$         | **same as 3.5**                                                                             | Local hard region aligned with dominant hotspot                             |
| **3.8**  | $\mathsf G_3$         | change packet center $0.40\to0.80$, width $.14\to.11$                                   | Sparse + narrower hard region                                               |
| **3.9**  | $\mathsf G^{aniso}_3$ | $S_2(12,.1)+.3S_1(1,0)$                                                                   | Data-induced angular metric; nominally high-frequency $u_2$ barely varies |
| **3.10** | curved clean            | with $s=y_1/.55,t=y_2/.55$: $\sin2\pi s+.7e^{-((s-.35)/.16)^2}\sin12\pi s+.4\sin5\pi t$ | Intrinsic manifold + localized frequency                                    |
| **3.11** | flat clean              | $\sin(2\pi y_1/.65)+.5\sin(3\pi y_2/.65)+.8\sin(12\pi y_3/.65)$                           | Hidden normal ridge                                                         |
| **3.12** | flat noisy              | **same as 3.11**                                                                             | Online/data-driven emergence of normal resolution                           |

Task 3.9 is very important. Euclidean geometry says $u_1$ and $u_2$ are equivalent orthogonal directions. The data say they are not. That directly tests the proposed $L^2(P_X)$-induced geometry on direction space.

---

# Dimension 4: resource allocation starts becoming genuinely constrained

| ID       | Data               | Exact raw target                                                                                                         | What it isolates                                                      |
| -------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| **4.1**  | $\mathsf{REG}_4$ | $S_1(1)+.7S_2(1)+.5S_3(1)+.35S_4(1)$                                                                                   | Dense but low-frequency ridge representation                          |
| **4.2**  | $\mathsf{REG}_4$ | $S_1(11,.2)+.7S_4(9,-.3)$                                                                                              | Ambient $d=4$, but true ridge rank only 2                           |
| **4.3**  | $\mathsf{REG}_4$ | $S_1(1)+.7S_2(3)+.5S_3(7)+.35S_4(12)$                                                                                  | Extreme per-direction budget allocation                               |
| **4.4**  | $\mathsf U_4$    | $R_1(18,.20)+.5R_3(10,-.20)+.2S_2(1)$                                                                                  | Mixed analyticity scales                                              |
| **4.5**  | $\mathsf U_4$    | $J_1(.10)-.6J_4(-.15)+.3S_2(2)+.2S_3(1)$                                                                               | Multiple discontinuity hyperplanes                                    |
| **4.6**  | $\mathsf U_4$    | $P_4((3,5,8,12),(.35,-.25,.10,-.40))$                                                                                  | Full-dimensional smooth negative control                              |
| **4.7**  | $\mathsf G_4$    | $.35S_1(1)+W_1(12,.45,.14)+.45S_3(3)$                                                                                  | High-frequency packet at dominant hotspot                             |
| **4.8**  | $\mathsf G_4$    | same except packet $W_1(12,.80,.11)$                                                                                   | Same architecture; difficult region starved of data                   |
| **4.9**  | $\mathsf G_4$    | $W_1(12,.45,.14)+.65W_2(9,c_\perp,.16)+.25S_4(2)$                                                                      | Different hotspots require different ridge directions and resolutions |
| **4.10** | curved clean       | $s=y_1/.55,t=y_2/.55$: $\sin2\pi s+.5\sin3\pi t+.7e^{-((s-.30)/.17)^2}\sin11\pi s+.35e^{-((t+.35)/.20)^2}\sin7\pi t$ | Full two-dimensional adaptive manifold problem                        |
| **4.11** | flat clean         | $\sin(2\pi y_1/.65)+.5\sin(3\pi y_2/.65)+.8\sin(12\pi y_4/.65)$                                                        | Normal component should consume essentially zero capacity             |
| **4.12** | flat noisy         | **same as 4.11**                                                                                                         | How quickly does normal resolution turn on?                           |

Here $c_\perp=z_2(\mu_\perp)$ is known exactly from the GMM construction.

Task 4.9 is the first one where I expect the ideal mesh to look genuinely complicated: angular mass near $u_1$; angular mass near $u_2$; more longitudinal kernels along $u_1$ than $u_2$; different center distributions inside those ridge blocks.

---

# Dimension 5: stress-test whether the theory actually scales

Use

$$
v_\pm
=
\cos(4^\circ)u_1
\pm
\sin(4^\circ)u_2,
$$

so task 5.4 has two true directions only $8^\circ$ apart.

| ID       | Data               | Exact raw target                                                                                                | What it isolates                                                             |
| -------- | ------------------ | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **5.1**  | $\mathsf{REG}_5$ | $S_1(1)+.75S_2(1)+.55S_3(1)+.4S_4(1)+.3S_5(1)$                                                                | Dense but smooth 5-direction baseline                                        |
| **5.2**  | $\mathsf{REG}_5$ | $S_1(12,.1)+.7S_5(10,-.3)$                                                                                    | High ambient dimension, ridge rank 2                                         |
| **5.3**  | $\mathsf{REG}_5$ | $S_1(1)+.75S_2(2)+.55S_3(4)+.4S_4(8)+.3S_5(13)$                                                               | Severe heterogeneous longitudinal budgets                                    |
| **5.4**  | $\mathsf U_5$    | $.7S_{v_+}(10,.1)+.7S_{v_-}(10,-.1)+.25S_5(2)$                                                                | Angular super-resolution / nearby ridges                                     |
| **5.5**  | $\mathsf U_5$    | $R_1(20,.18)+.5R_3(12,-.20)+.35R_5(6,.30)$                                                                    | Three distinct analytic length scales                                        |
| **5.6**  | $\mathsf U_5$    | $J_1(.10)-.6J_3(-.18)+.4J_5(.30)+.2S_2(2)$                                                                    | Several oblique jumps                                                        |
| **5.7**  | $\mathsf U_5$    | $P_5((2,3,5,8,12),(.35,-.30,.15,-.40,.25))$                                                                   | Hardest smooth full-dimensional negative control                             |
| **5.8**  | $\mathsf G_5$    | $.35S_1(1)+W_1(14,.45,.13)+.35S_4(4)$                                                                         | Dense hotspot + very high local frequency                                    |
| **5.9**  | $\mathsf G_5$    | same except $W_1(14,.82,.10)$                                                                                 | Extreme low-density/high-frequency conflict                                  |
| **5.10** | $\mathsf G_5$    | $W_1(14,.45,.13)+.55W_1(7,-.45,.18)+.6W_2(10,c_\perp,.15)$                                                    | Three hotspot/frequency combinations; joint allocation test                  |
| **5.11** | curved clean       | $s=y_1/.55,t=y_2/.55,r=y_5/.15$: $\sin2\pi s+.5\sin3\pi t+.75e^{-((s-.35)/.16)^2}\sin10\pi s+.25\sin4\pi r$ | $2$-D intrinsic manifold inside $5$-D with curvature-induced third ridge |
| **5.12** | curved noisy       | **same target and same manifold as 5.11**, plus normal noise $\sigma=.04$                                     | Does mesh complexity track intrinsic geometry or ambient noise?              |

That gives exactly $5\times12=60$ experiments.

## Why I like these 60 much more

They contain several deliberate chains rather than isolated cells.

The **longitudinal-resolution chains** are $1.1\rightarrow1.2$, $2.1\rightarrow2.2$, and then $3.2,\;4.3,\;5.3$. Those should tell you whether the allocation of $N(v)$ follows local ridge bandwidth.

The **density/frequency interaction chains** are $1.5\rightarrow1.7\rightarrow1.8$, $2.6\rightarrow2.7\rightarrow2.8$, and analogous 3D-5D cases. These directly test whether the learned monitor behaves like

$$
\rho(v,t)
\sim
\left[
p_v(t)
|\partial_t^r g(v,t)|^2
\right]^{1/(2r+1)}
$$

rather than simply following $p_v$.

The **angular-resolution tasks** are $3.3$ and $5.4$. Because you know the exact angular separation, you can empirically derive something resembling an angular Nyquist curve: minimum resolvable ridge separation $=f(M,\gamma,\omega,d)$.

The **data-induced angular geometry task** is 3.9.

The **clean/noisy manifold pairs** are $2.11\leftrightarrow2.12$, $3.11\leftrightarrow3.12$, $4.11\leftrightarrow4.12$, $5.11\leftrightarrow5.12$.

And the **negative controls** are $2.9,\ 4.6,\ 5.7$. Those matter because Petrushev-style ridge spaces really can inherit multivariate approximation power from a 1D approximation space plus a direction set, but there is no reason to expect every genuinely full-dimensional function to be highly compressible in a small number of ridge directions.

## What I would record for every experiment

Do not reduce these to a single test MSE. For every run save

$$
E_{P_X}=\mathbb E_{P_X}(F-\hat F)^2,\qquad E_U=\mathbb E_U(F-\hat F)^2,
$$

the actual learned directions $v_i$, number of kernels $N_i$ per direction, centers $c_{ij}$, local spacings $h_{ij}$, and scales $\gamma_{ij}=\lambda^\star/h_{ij}$.

For targets with known true ridges, measure angular recovery $\min_j\arccos|v_i^\top u_j|$.

For localized targets, separately measure error inside and outside the high-frequency region.

For GMM tasks, report error by data-density decile.

And most importantly, where the oracle is available, compare the learned center density against

$$
\rho^\star_v(t)
\propto
\left[
p_v(t)|g_v^{(r)}(t)|^2
\right]^{1/(2r+1)}.
$$

That gives you a way to ask not merely whether the method predicted well, but whether it learned the **geometry predicted by the approximation theory**.

If I were implementing this tomorrow, I would not begin with all 60. I would first do **1.1-1.8, 2.1-2.8, 2.11-2.12, 3.2-3.3, and 3.9**. Those ~20 experiments would tell us very quickly whether the core story about longitudinal resolution, angular resolution, and data-adaptive meshing is actually true before paying for the higher-dimensional campaign.

References: Genz integration test functions (sfu.ca/~ssurjano/integration.html); Burkardt test_interp_nd; Petrushev, "Approximation by Ridge Functions and Neural Networks," SIAM J. Math. Anal.

---

# Version 1 (context): the 138-task factorial and the framing it carries

Yes. I would turn this into a formal benchmark suite now, because the experimental design can answer several of the theoretical questions directly.

The full factorial is $5$ dimensions $\times5$ densities $\times6$ functions $=150$; omitting the 12 meaningless 1D-manifold cells gives a **138-task core suite**. This has precedent in the Genz benchmark families, which deliberately vary oscillation, peaks, smoothness, and discontinuity across arbitrary-dimensional hypercubes; we adapt that philosophy specifically to ridge approximation.

## 1. Global benchmark conventions

$\Omega_d=[-1,1]^d$, $d=1,\ldots,5$. Generate a fixed generic rotation $Q_d$ with a frozen random seed; columns $u_k$; normalized ridge coordinate $z_k(x)=u_k^\top x/\|u_k\|_1\in[-1,1]$ (so a frequency of $8\pi$ means the same thing in 2D and 5D). Use $r_d=\min(3,d)$ true ridge directions for most targets, with $a=(1,\;0.65,\;0.4)$, $\phi=(0.2,-0.7,1.1)$. Normalize every target using the **same uniform reference distribution**, not the training distribution: $F=(\widetilde F-\mathbb E_U\widetilde F)/\sqrt{\operatorname{Var}_U\widetilde F}$. Initially use **zero label noise**; label noise is a later fourth axis.

## 2. The six function families

Five have known ridge structure (oracle $v$'s and oracle longitudinal regularity); the sixth deliberately violates low-ridge-rank structure.

- **L, globally low frequency:** $\widetilde F_L=\sum_{k=1}^{r_d}a_k\sin(\pi z_k+\phi_k)$. Tests basic direction recovery and QI convergence under benign conditions.
- **H, globally high frequency:** $\widetilde F_H=\sum_k a_k\sin(8\pi z_k+\phi_k)$. Same directions as L, $8\times$ the longitudinal resolution; increasing frequency should manifest as $h\downarrow,\gamma\uparrow,N(v)\uparrow$, not a reshuffling of directions.
- **R, Runge / almost-nonanalytic:** $\widetilde F_R=\sum_k a_k/(1+64(z_k-\mu_k)^2)$, $\mu=(0.20,-0.25,0.10)$; poles at distance $1/8$. Later sweep $\alpha\in\{2,4,8,16\}$ as a continuous distance-to-singularity knob.
- **J, jump:** $\widetilde F_J=0.5F_L+0.8\,\mathbf1_{\{z_1>0.12\}}-0.5\,\mathbf1_{\{z_2<-0.22\}}$ (second jump omitted in 1D). Oblique discontinuity hyperplanes; the negative control for smooth QI theory.
- **X, asymmetric spatially mixed frequency:** $\widetilde F_X=F_L+0.75e^{-((z_1-0.45)/0.18)^2}\sin(10\pi z_1+0.3)+\mathbf1_{d\ge2}\,0.45e^{-((z_2+0.35)/0.22)^2}\sin(7\pi z_2-0.2)$. The single most important target for the adaptive-mesh idea: tests *where along $v$* to put kernels and *which $v$'s* deserve more kernels; analytic everywhere.
- **A, asymmetric full-dimensional control:** $\widetilde F_A=\prod_{k=1}^d1/(1+4(z_k-b_k)^2)$, $b=(0.45,-0.30,0.20,-0.40,0.10)$. Genz product-peak; if the method wins L/H/R/J/X but loses its advantage on A, the architecture is exploiting ridge compressibility, not defeating dimensionality -- a good result.

## 3. Five data geometries

- **Q, deterministic quasi-regular coverage:** equispaced in 1D; Sobol + a few Lloyd/CVT steps for $d>1$.
- **U, iid uniform** on the cube. $Q$ vs $U$ measures the cost of finite-sample irregularity.
- **G, hotspot / GMM:** $P_G=0.15U+0.85\sum_{\ell=1}^3\pi_\ell\mathcal N_T(\mu_\ell,\sigma_\ell^2I)$, $\pi=(0.5,0.3,0.2)$, $\sigma=(0.10,0.18,0.26)$, means from a separate frozen seed with good separation. The uniform background gives high density plus low-but-nonzero coverage elsewhere.
- **C, clean manifold:** $d=2$: $m_2(t)=R_2(t,\,0.45\sin\pi t)$, $t\sim U[-1,1]$. $d=3,4,5$: intrinsic dimension fixed at two, $b(z)=(z_1,z_2,0.35\sin\pi z_1,0.35\sin\pi z_2,0.25\sin\pi(z_1+z_2))$ truncated to $d$ coordinates, rotated by $R_d$, rescaled into the cube. Tests whether performance follows ambient or effective dimension.
- **N, noisy manifold:** same manifold plus **normal** noise, $X=m(z)+\sigma_\perp\Pi_\perp(z)\xi$, $\Pi_\perp=I-J(J^\top J)^{-1}J^\top$, $\sigma_\perp=0.05$.

## 4. Naming

`D<dimension>-<density>-<function>`, densities $Q,U,G,C,N$, functions $L,H,R,J,A,X$; 18 tasks in 1D (no manifold rows) and 30 in each of $d=2..5$.

## 5. The most important interaction experiment

The theory says mesh density should depend on data density $\times$ local approximation difficulty, so the two signals must be distinguished. For GMM $\times$ mixed-frequency, add **aligned** (high-frequency packet inside the largest hotspot), **anti-aligned** (packet in a low-density region), and **density-only** (largest hotspot in a low-frequency region). These tell whether the learned mesh discovered $[p(x)|F^{(r)}(x)|^2]^{1/(2r+1)}$ or merely "put neurons where points are."

## 6. Manifold orientation suite

For manifold datasets, a supplemental suite where a true ridge direction is (1) tangent, (2) oblique, (3) normal to the manifold, e.g. $F=\sin(8\pi u^\top x)$: globally identical frequency, but $u^\top X$ varies enormously in the tangent case and almost not at all in the normal case. Tests that the relevant geometry is $v\mapsto f_v(v^\top X)$ under $L^2(P_X)$, not Euclidean direction space.

## 7. What to measure

$E_P=\mathbb E_{P_X}(F-\hat F)^2$ (did we allocate resolution correctly for the data?) and $E_U=\mathbb E_U(F-\hat F)^2$ (what did adaptation sacrifice elsewhere?). GMM: error in top-10%, middle, bottom-10% density bins. X: error in high- vs low-frequency regions. Manifolds: $E_{\mathcal M}$ plus tubes at increasing distance. Ridge targets: direction recovery $\min_j\arccos(|v_i^\top u_j|)$, and learned center density vs the oracle monitor $\rho^\star_v(t)\propto[p_v(t)|g_v^{(r)}(t)|^2]^{1/(2r+1)}$.

## 8. The four models to compare inside every cell

U/U (uniform directions, uniform centers), U/A (uniform directions, adaptive centers), A/U (adaptive directions, uniform centers), A/A (adaptive both). All four use the same number of first-layer neurons, the same activation, the same $\lambda^\star$, and finish with the same least-squares solve. External controls: a matched-width conventionally trained MLP and random tanh features.

## 9. Capacity and sample-efficiency sweeps

$B\in\{64,128,256,512,1024,2048\}$ first-layer kernels; for the uniform Ridge-QI baseline $M\approx B^{(d-1)/d}$ directions, $N_{\rm ridge}\approx B^{1/d}$ centers per direction; sample ratio $N_{\rm train}/B\in\{1.25,2,4,8\}$. Two scaling surfaces, error$(B,N)$.

## Expected winning mechanisms

Q-L uniform construction; Q-H global resolution scaling; Q-R QI regularity / analyticity sensitivity; Q-J failure of smooth convergence near jumps; G-L density adaptation; G-X density + local-frequency adaptation; C-L effective dimensionality; C-H data-induced angular resolution; N-H robustness of manifold adaptation; A limitation of low-ridge-rank bias. First experiment to look at: D2-G-X (every ingredient visualizable), then D3-C-X, then D5-C-X.

---

# Version 3 (built): genuinely multivariate targets, softened data

Version 2's targets were finite sums of one-dimensional profiles along fixed directions, $\sum_i g_i(v_i^\top x)$. That builds the model's own way of representing functions into the benchmark: any question about whether a method finds the right directions is answered in advance by the construction, and "how well does this represent a function" collapses into "how well does this represent a function it was built to represent". Version 3 replaces every target with a function that is not a finite sum of such profiles -- analytic or piecewise, radial, product, composed, or split by a curved surface. Two data changes come with it: the clusters in the hotspot geometries were too tight, and the noise on the sheet geometries was too large. Both are softened. The task count goes from 12 to 16 per dimension, so 80 tasks.

## Common definitions

The domain is $\Omega_d=[-1,1]^d$ and the directions are unchanged: the DCT-II columns $u_1,\dots,u_d$, with $y_k(x)=u_k^\top x$ and the normalized coordinate $z_k(x)=u_k^\top x/\|u_k\|_1\in[-1,1]$. Write $z(x)$ for the whole vector. Every target below is written in $z$.

Distances to a point $a\in[-1,1]^d$ use the **scaled radial distance**

$$\rho_a(x)=\frac{\|z(x)-a\|_2}{\sqrt d},$$

so a corner of the cube is at distance $\lesssim1$ in every dimension and a width written in $\rho$ means the same thing in every $d$. Three anchor points, given in $z$ and truncated to $d$ coordinates:

$$a^{(1)}=(.30,-.20,.25,-.15,.10),\quad a^{(2)}=(-.40,.35,-.10,.20,-.25),\quad a^{(3)}=(.15,.10,-.30,.05,.30).$$

The hotspot means in $z$: since $u_1\propto\mathbf 1$ and every other $u_k\perp u_1$, $\mu_\pm=\pm0.45\mathbf 1$ have $z_1=\pm0.45$ and all other coordinates zero, and $\mu_\perp=0.35\,u_2/\|u_2\|_\infty$ has only $z_2=c_\perp$ nonzero. In $d=1$ the three means are $0.45,-0.45,0$.

Every family exposes `value(X)`, an analytic `grad(X)` of shape `[n, d]` in $x$ coordinates (chain rule through $z=Q^\top x/\|u\|_1$ per coordinate), and a flag `differentiable`, which is `False` for the two step families -- their `grad` returns the gradient of the smooth part and an `interface_mask` marks the points near the step. Every gradient is checked against central finite differences in the tests.

## The function families

1. **Bumps at several widths.** $B=\sum_k A_k\exp(-\rho_{a_k}^2/2\sigma_k^2)$ with $(A,\sigma,a)=\{(1.0,.50,a^{(1)}),(0.7,.25,a^{(2)}),(0.5,.10,a^{(3)})\}$. Also the single wide bump $B_0=\exp(-\rho_{a^{(1)}}^2/0.5)$, used as the smooth background under the packet, step and kink families.
2. **Concentric waves.** $O_\omega=\cos(\pi\omega\rho_{a^{(1)}})$, $\omega\in\{1,6\}$. Cosine, not sine: $\rho$ is a distance, and $\cos(\pi\omega\rho)$ is an analytic function of $\rho^2$, so the waves are smooth at their center (a sine would have a cone point there).
3. **Composition.** $C=\exp(\sin(\pi z_1)\cos(\pi z_2))$ for $d\ge2$; $C=\exp(\sin(\pi z_1))$ for $d=1$.
4. **Polynomial.** $P=\sum_{k=1}^d(z_k^2z_{k+1}-z_kz_{k+1}^3)$ with cyclic index $z_{d+1}=z_1$; in $d=1$ this is $z_1^3-z_1^4$.
5. **Radial spike.** $R_\alpha=1/(1+\alpha^2\rho_{a^{(1)}}^2)$, $\alpha\in\{4,12\}$.
6. **Product peak** (kept from Version 2). $\prod_{k\le d}1/(1+a_k^2(z_k-b_k)^2)$ with $a=(3,5,8,12,6)$, $b=(.35,-.25,.10,-.40,.25)$, truncated to $d$.
7. **Burst of oscillation in a ball.** $W_a=B_0+0.8\,e^{-\rho_a^2/\tau^2}\cos(\pi\omega\rho_a)$ with $\tau=0.18$, $\omega=10$ (cosine carrier, smooth at the center for the same reason). Two placements: on the densest cluster, $a=\mu_+$; and away from every cluster, $a=(0.85)$ in $d=1$ and $a=(0.40,-0.55,0,\dots)$ for $d\ge2$. The away anchor is asserted to sit more than $2\sigma$ from each cluster mean (see the note on $d=1$ below).
8. **Steps.** Sphere step $S=B_0+0.8\cdot\mathbf 1[\rho_{a^{(2)}}<0.35]$. Wavy step ($d\ge2$) $V=B_0+0.8\cdot\mathbf 1[z_2>0.3\sin(\pi z_1)+0.1]$; in $d=1$ the step is at $z_1>0.78$, out in the sparse tail of the hotspot data.
9. **Kinks.** Kink ring $K=|\rho_{a^{(1)}}-0.4|$ (value continuous, slope flips). One-sided kink $K_1=\max(0,\rho_0^2-0.25)$ (value and slope continuous, curvature is not; $\rho_0$ is the distance to the origin).
10. **Piecewise.** $D=\sin(2\pi\rho_{a^{(1)}})$ for $\rho_{a^{(1)}}<0.4$ and $\sin(0.8\pi)+3(\rho_{a^{(1)}}-0.4)^2$ otherwise: continuous, with a jump in the slope.

Scaling is unchanged from Version 2: each raw target is centered and scaled once, $F=(\widetilde F-\text{mean})/\text{sd}$, on a fixed uniform reference set that is the same for every task of a given $d$.

## Data changes

- **Hotspots.** Weights $(0.20\,\text{uniform},\,0.40,\,0.25,\,0.15)$, widths $\sigma=(0.22,0.28,0.25)$, means as before. The $d=1$ analogue keeps means $(.45,-.45,0)$ with the same weights and widths.
- **Stretched hotspots.** $\Sigma=Q\,\mathrm{diag}(.25^2,.083^2,.15^2,\dots)Q^\top$ (a 3:1 stretch), weights $(0.20\,\text{uniform},\,0.40,\,0.40)$, means $\pm0.35\mathbf 1$.
- **Sheets.** Flat parameters uniform on $[-.75,.75]$. Curved, $d=2$: $y=(.75t,.30\sin\pi t)$; $d\ge3$: $y=(.65s,.65t,.25\sin\pi s,.20\sin\pi t,.15\sin\pi(s+t))_{1:d}$. Perpendicular noise $\sigma_\perp=0.015$ for both the flat and the curved thickened variants. The assertion that the sheet stays inside the cube is kept and holds with room to spare (largest $|x|$: $0.55$ in $d=2$, $0.83$ in $d=3$, $0.76$ in $d=4$, $0.68$ in $d=5$ for the curved sheet).
- **Test sets.** Unchanged in structure and in the rules that define them: the densest cluster within one standard deviation, the shrunken cube $[-0.8,0.8]^d$, the inner 80% of the sheet's parameter range.

## The roster: 16 tasks per dimension, ids `d.1` .. `d.16`

For $d\ge2$:

| id | data | target |
|---|---|---|
| .1 | even grid | bumps at three widths $B$ |
| .2 | even grid | $O_1$ |
| .3 | even grid | $O_6$ |
| .4 | even grid | polynomial $P$ |
| .5 | uniform | composition $C$ |
| .6 | uniform | $R_4$ |
| .7 | uniform | $R_{12}$ |
| .8 | uniform | sphere step $S$ |
| .9 | uniform | wavy step $V$ |
| .10 | uniform | kink ring $K$ |
| .11 | hotspots | $O_6$ (the same object as .3) |
| .12 | hotspots | burst on the densest cluster, $W_{\mu_+}$ |
| .13 | hotspots | burst away from the clusters |
| .14 | hotspots | product peak |
| .15 | curved sheet | composition $C$ |
| .16 | curved sheet, thickened | composition $C$ (the same object as .15) |

For $d=1$: .9 is the piecewise target $D$ under uniform data; .10 is the one-sided kink $K_1$ instead of the ring (a ring in one dimension is two kinks; using $K_1$ puts both kinds of kink somewhere in the suite); .15 is the $z_1>0.78$ step plus $B_0$ under hotspot data (a step in the sparse tail); .16 is $R_{12}$ with its anchor moved to $\mu_+$ (a sharp feature sitting exactly on the densest cluster).

Tasks meant to be the same function share the same target object, so their scaled versions agree bit for bit: .3 and .11 (same function, different data), .15 and .16 (same function and sheet, with and without noise). Tasks .12 and .13 are the same burst in different places, and .6 and .7 the same shape at two sharpnesses.

## Metrics

- Angular recovery is removed: there are no true directions any more. The `geometry()` interface on models is kept.
- Burst region: $\rho_a\le2\tau$. Step region: $|\rho_a-r|\le0.05$ (sphere), $|z_2-0.3\sin(\pi z_1)-0.1|\le0.05$ (wavy), $|z_1-0.78|\le0.05$ (the $d=1$ step).
- The predicted center density along a direction $v$ is $\rho^\star_v(t)\propto[p_v(t)D_v(t)]^{1/(2r+1)}$ with $D_v(t)=\mathbb E[|\partial_vF(X)|^2\mid v^\top X=t]$ estimated by binning the projected sample and averaging $|\nabla F(X)\cdot v|^2$ per bin, Gaussian-smoothed. It is undefined for the step families and reports that rather than returning a number.

## One place the specification could not be met exactly

In $d=1$ the away-from-clusters burst anchor $a=(0.85)$ sits $1.82$ widths from the $+0.45$ cluster, not the $2$ the specification asks for: with the softened width $\sigma=0.22$ there is no point of $[-1,1]$ that is $2\sigma$ from that cluster and still far from the domain boundary. For $d\ge2$ the anchor clears every cluster by $2.5$ widths or more. The test asserts $2$ widths for $d\ge2$ and $1.8$ for $d=1$, and prints the table.
