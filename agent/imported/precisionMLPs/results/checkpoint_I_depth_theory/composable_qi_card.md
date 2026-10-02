# Composable QI: the current core (a card of considerations)

**Status:** considerations recorded 2026-09-02 at Sam's request. Not on record as law: Sam does not agree with all of it, and nothing here has been measured beyond expI01's first rows. The theory of record stays `compositional_qi_theory.md`. Claude's annotations are marked at the end.

**Base idea.** Exploit what QI is best at, high-accuracy 1-D approximation, by factoring high-dimensional functions into learned scalar directions, low-rank mixtures, and repeated 1-D QI compositions.

## The layer

For layer $\ell$ with $M_\ell$ scalar input channels and $N_\ell$ QI units per channel,

$$H_\ell(z_\ell)_{rj}=\tanh\big(\gamma_{\ell j}(z_{\ell,r}-c_{\ell j})\big)\in\mathbb R^{M_\ell\times N_\ell},\qquad z_{\ell+1,k}=\sum_{r,j}C^{(\ell)}_{rjk}H_{\ell,rj}.$$

Low-rank factorization:

$$C^{(\ell)}_{rjk}=\sum_{s=1}^{S_\ell}a^{(\ell,s)}_r\Phi^{(\ell,s)}_{jk},\qquad z_{\ell+1}=\sum_{s=1}^{S_\ell}(a^{(\ell,s)})^\top H_\ell(z_\ell)\Phi^{(\ell,s)}.$$

Then $M_{\ell+1}=q_\ell$: the output channels of one block are the inputs of the next. Any extra linear $V_{\ell+1}$ before the next $H$ is redundant and absorbed into $C_\ell$. Only the first projection from raw $x\in\mathbb R^d$ needs an explicit direction matrix $V$. Each row of $H$ is one scalar direction, each column one QI kernel; a coefficient vector $\phi\in\mathbb R^N$ defines one 1-D function along a row, $H_r\phi\approx\phi(v_r^\top x)$.

## Rank hierarchy

$S=1$ is the strong-KAT factorization $C_{rjk}=a_r\Phi_{jk}$: inside channel $k$ every direction carries the same 1-D profile scaled by $a_r$; $\operatorname{rank}(C_k)=1$ with $C_k=a\,\Phi_{:k}^\top$. Increasing $S$ allows several shared direction/profile modes; full $C$ lets each direction/channel pair have its own QI function.

$$\text{KAT rank 1}\subset\text{low-rank compositional QI}\subset\text{full }C.$$

The main experiment is to sweep $S$, not assume full $C$. Rank 1 may be enough; small $S>1$ may capture the needed extra expressiveness. Full $C$ is the upper-capacity control.

## Relation to ridge, Radon, KAT

- QI: high-accuracy scalar approximation.
- Ridge/Radon: directional coordinates $v^\top x$; the shallow version pays heavily for generic direction coverage.
- KAT: finite scalar composition is universally sufficient, but may hide complexity in rough 1-D functions.
- Composable QI: search the middle; learn a modest number of directions, channels and rank so the scalar functions remain QI-friendly.

The trade: high-dimensional directional complexity $\longrightarrow$ scalar compositional complexity. Success means the latter stays simple and analytic.

## Composition

After $z\in\mathbb R^q$ the next QI layer applies $H$ coordinatewise, $H_2(z)\in\mathbb R^{q\times N_2}$, so $q_\ell=M_{\ell+1}$. Generic deep form $z_{\ell+1}=\mathcal C_\ell(H_\ell(z_\ell))$, residual form $z_{\ell+1}=z_\ell+\eta_\ell\mathcal C_\ell(H_\ell(z_\ell))$.

## Equivalent ordinary two-hidden-layer tanh MLP

For $q=M$, $N_1=N_2=N$, both hidden layers have width $h=MN$: the block is an ordinary $d\to MN\to MN\to1$ tanh MLP with constrained weights. First layer $W_1=\gamma\otimes V^\top$ (neurons on $M$ directional rays with QI-prescribed $\gamma,c$). Middle matrix $W_2=GC_\flat$ with $G=I_M\otimes\tilde\gamma$, hence $\operatorname{rank}(W_2)\le M$; for rank $S$,

$$W_2=\sum_{s=1}^{S}(\Phi^{(s)})^\top\otimes\big(\tilde\gamma(a^{(s)})^\top\big),$$

a sum of $S$ Kronecker products: $S$ is a Kronecker (separation) rank, and the unfolded $C$ lies on a determinantal variety. A generic $MN\times MN$ matrix can have rank $MN$.

## Parameter and FLOP laws (middle map, $q=M$)

| model | parameters | MACs per sample |
|---|---|---|
| rank 1 | $\sim MN$ | $\sim2MN$ |
| rank $S$ | $\sim SMN$ | $\sim2SMN$ |
| rank $\sqrt M$ | $\sim NM^{3/2}$ | $\sim2NM^{3/2}$ |
| full $C$ | $M^2N$ | $M^2N$ |
| dense MLP $W_2$ | $M^2N^2$ | $M^2N^2$ |

At equal hidden width the structured block is $N\times$ cheaper even untied, and low rank gives another large reduction. Do not size against the fictitious dense $(MN)^2$ MLP; match the real transformer FFN parameter and MAC budget.

## Transformer sizing (guess)

$M$: directional/channel capacity. $N$: scalar QI resolution and activation-memory cost. $S$: interaction rank, the parameter/compute knob. Depth: compositional capacity. Likely regime: $N$ small to moderate (4 to 32), $S$ moderate, $M$ large enough, depth rather than huge $N$.

## GPU

Each two-QI block performs $2MN$ tanh evaluations per token independent of $S$; as $S$ shrinks the GEMM work falls but the activation work does not, so the kernel becomes activation-bound. Never materialize $H\in\mathbb R^{B\times M\times N}$ to HBM: generate the QI features in registers or shared memory and contract them into the low-rank intermediate immediately; $S$ then also controls arithmetic intensity. GELU is a candidate: Gaussian-type QI kernel with $\lambda\approx0.707$, and an ALU-only fused approximation may be hardware-friendly.

## Central hypothesis

Can depth, modest rank and learned directions keep each scalar QI function simple enough that $N$ stays small? Works if, with scale, $N^\star$ stays bounded and $S^\star=o(M)$, and added depth reduces the rank or resolution needed per layer. Failure looks like $N\uparrow$, $S\to M$, $M\uparrow$ together: the curse has reappeared inside the factorization. The main thing to measure is the required scaling of $(M,S,q,L,N)$, especially whether $S$ and $N$ stay small as $M$ grows.

---

## Claude's annotations (2026-09-02)

- The first-layer bias in the record's convention carries the origin: $b^{(1)}_{rj}=-\gamma_j(c_j+v_r^\top x_0)$.
- $\gamma_{\ell j}$ per unit is the graded-mesh case; the record's blocks share one $\gamma$ per even mesh. Level-2 meshes must be range-tracked (expI01), which the card leaves implicit.
- GELU: the aliasing rule gives $\lambda=0.707$ (expC07), but GELU's feature matrix has an $O(N)$ null space and is 1 to 3 orders worse at the floor than tanh (expA04). Fine for a deep-learning setting, not for the precision setting.
- The "absorbed $V_{\ell+1}$" statement is exactly the gauge argument by which composed Radon equals the general model (chat, 2026-09-02); the intermediate width $q_\ell$ then acts as a rank bottleneck on the channel tensor, a different restriction from the Kronecker rank $S$.
- The dense-MLP row of the table is the fictitious comparison; the notebook's comparisons at matched parameters and matched FLOPs are the ones that count.
- Measured so far (expI01, one row, $d=3$ fast waves, QUICK): block $10^{-6}$ against shallow and MLP at $10^{-2}$ at equal cost; California housing: MLP 0.46 beats block 0.49 to 0.51 and shallow 0.56, and the block trains 10 to 40x slower with per-step solves.
