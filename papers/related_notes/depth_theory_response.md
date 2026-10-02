# Depth for QI networks: what can be true, what the record already says, and what to measure next

A response to the two proposed views ("Pareto between dimensionality and 1-D complexity" and "Compositional results"), written with the repository's record in hand. Every claim below carries a status: **theorem** (proved, with the source), **measured** (in the record, with the experiment), **derived** (follows from the record, not yet measured), or **open**. Notation is the record's: $M$ directions, $N$ offsets per direction ($N_1$, $N_2$ per level), $K$ channels, $\mathfrak r$ the data radius about the origin $x_0$, $k$ the spectral scale, $u=2^{-53}$ the unit roundoff, $\lambda=\gamma h=0.25$.

---

## 0. The verdict

1. **Neither view is the theory; together they contain most of it.** View B is correct and too small: its radial theorem is the E1 oracle arm of expI01 written as a lemma, and its near-radial extension bounds nothing that is not already radial. View A has the right shape and is mostly known: the error ledger is the standard telescoping of deep approximation theory, and the product separation is the classical Waring-rank argument. Its genuinely new content is that the edges are QI blocks with a solvable last readout.

2. **Both avoid the fact that decides the question.** For arbitrary analytic functions of $d$ variables no method escapes the curse of dimensionality; this is a theorem about the class, not about ridges or depth. The shallow ridge-QI model of the record is already within a factor of about four of the class-optimal count in 3-D and two to three in 4-D. Depth cannot improve on that for generic targets. Depth exists to exploit structure, and a theory of depth is a theory of which structures, what they cost, and whether they can be found from data.

3. **The theory that is both true and useful has five parts:** the lower bound; the shallow cost with the tolerance entering logarithmically; the composition ledger with its floating-point floor; a table of structure classes with their costs and their detection procedures; and an honest learnability statement. Two of the five are in neither view.

4. **The next real fact about depth is a measurement.** Three experiments, each buildable in a day with the expI01 machinery, settle the questions the theory cannot: the composed floor against depth, the Waring cliff of the shallow model in fp64, and whether Gauss-Newton finds a two-gate tree from a random start.

---

## 1. The two views, assessed

**View B (four claims).** The discipline is right: import the 1-D theorem, state the shallow certificate, state the composition ledger, then prove one class. The fourth claim, analytic and radial implies $f=g(\lVert x-a\rVert^2)$ with analytic $g$, is correct (rotation invariance kills odd degrees and forces each even homogeneous part to be a multiple of $\lVert x\rVert^{2j}$). It is also the single easiest case: the record's Version 2 theory already lists it under "quadratic channels need no direction learning", and expI01's E1 arm is exactly this construction. The near-radial bound $\eta_{\rm rad}+E_{\rm out}+L_gdE_{\rm in}$ is a bound with an $O(1)$ first term for every target that is not radial; it should be dropped. The demotion of KAT is right. The decision to drop the sharper direction theorem is wrong, for a reason given in Section 3.2: without it the shallow model is charged a fictitious cost, and every claim that depth wins is measured against that fiction.

**View A (Pareto).** The three-resource picture (directions, scalar difficulty, depth) is the right picture. Theorem 1 (the ledger) is correct and standard. Theorem 2 (KAT modulus transfer) is correct and demotable to a remark. Theorem 3 (the product) is correct and is the most useful thing in either document: an explicit target for which depth changes the direction count from $2^{d-1}$ to $O(d)$ and the profile degree from $d$ to $2$, with an exact lower bound. Its Corollary 4 (arithmetic circuits) is the right generalization. Two weaknesses. First, the Pareto class $\mathcal C_{L,B}(\Gamma)$ defined by minimizing over representations is honest but unfalsifiable until characterized by properties of $f$; View A says so, and the class should stay out of the theorem statements until it is. Second, the document never states what a computation graph costs to find, which is the only question that makes the product theorem practical.

**What both miss.** (i) The information-theoretic floor of Section 2. (ii) The measured shallow cost, so that the comparison is honest. (iii) That only the last readout is a least-squares projection; everything upstream is a nonlinear search whose solvability is the actual research question. (iv) The floating-point floor of a composed network.

---

## 2. The wall is the class, not the method

**Theorem (entropy of analytic classes; Kolmogorov and Tikhomirov 1959, Vitushkin).** Let $\mathcal A_\rho^d$ be the functions bounded by one on a product of Bernstein ellipses of parameter $\rho>1$ around $[-1,1]^d$. Its $\varepsilon$-entropy in the uniform norm on the cube is $\asymp(\log1/\varepsilon)^{d+1}$. Consequently any approximation scheme whose parameters carry $O(\log1/\varepsilon)$ bits each, or whose parameter selection is continuous in $f$ (DeVore, Howard and Micchelli 1989), needs

$$n\;\gtrsim\;c_d\,(\log1/\varepsilon)^d$$

parameters to reach uniform error $\varepsilon$ on the whole class. The matching upper bound is the truncated Chebyshev expansion of total degree $p\approx\log(1/\varepsilon)/\log\rho$, which has $\binom{p+d}{d}$ coefficients.

For targets of exponential type $k$ on a ball of radius $\mathfrak r$ (the suite's oscillatory targets) the same count holds with $p$ the degree at which the Taylor tail of $e^{ikt}$ falls below $\varepsilon$, i.e. $(e\,k\mathfrak r/2p)^p\le\varepsilon$. Values of $p$ at $\varepsilon=10^{-13}$:

| $k\mathfrak r$ | 1 | 2 | 4 | 5.3 | 7.5 |
|---|---|---|---|---|---|
| $p$ | 14 | 17 | 22 | 25 | 29 |

The record's fast-waves target $\cos(6\pi\rho)$ with $\rho=\lVert x-a\rVert/\sqrt2$ has $k=13.3$; on the expH06 ball ($\mathfrak r=0.3$) $k\mathfrak r=4.0$, so $p=22$. The monomial counts at $p=22$:

| $d$ | 2 | 3 | 4 | 5 | 8 | 10 |
|---|---|---|---|---|---|---|
| $\binom{p+d}{d}$ | 276 | 2,300 | 14,950 | 80,730 | $5.9\times10^6$ | $6.5\times10^7$ |

**The record's shallow model against this count [measured].** The even nested ridge mesh reaches the fp64 floor in 3-D at $(M,N)=(192,48)=9216$ units (expH06), a factor $4.0$ above the monomial count. In 4-D the floor crossing is estimated at 25k to 50k units, a factor $1.7$ to $3.3$ above. The quadrature certificate of the review (Section 3.2 below) explains why the exponents agree: $M\asymp(c\,k\mathfrak r+O(\log1/\varepsilon))^{d-1}$ and $N\asymp c\,k\mathfrak r+O(\log1/\varepsilon)$, so $MN\asymp p^d$ up to dimension constants.

**Consequence.** The 4-D wall in expH06 ("25 to 50k units, beyond the dense solve") is the class. In 5-D the same target needs of order $10^5$ coefficients by any method, and $10^8$ in 10-D. No architecture, shallow or deep, learned or constructed, does better on generic analytic targets. This answers the question "can we do it for arbitrary higher-dimensional functions in a way that scales like 1-D": no, by theorem, and the theory should say so in its first paragraph. It is the fact that makes the rest coherent: depth is not a way around the curse; it is the mechanism by which targets with structure escape the count $\binom{p+d}{d}$, and the theory of depth is a theory of that escape.

The 1-D recipe scales because in 1-D the class-optimal count is $p=O(\log1/\varepsilon)$ and the construction achieves it with a rule and a solve. The $d$-dimensional analogue of that sentence, for generic targets, is $\binom{p+d}{d}$ with a rule and a solve, and the record already has it. What is left is structure.

---

## 3. The theory I would write

Five claims. Each is one page or less.

### 3.1 The lower bound [theorem, cited]

Section 2. It sets the exponent and it says the shallow tax is the class.

### 3.2 The shallow cost [proposition with proof sketch; exact in 2-D; measured]

The angular-quadrature certificate: for a target whose spectral measure is angularly smooth (spherical-harmonic degree at most $L$ on each sphere $\lVert\xi\rVert=\rho\le k$), a spherical $t$-design with $t\ge L+n$ and $n\gtrsim e\,k\mathfrak r/2+O(\log1/\varepsilon)$ gives a ridge sum with uniform error $\varepsilon$ on $B_{\mathfrak r}(x_0)$, and designs with $M\le c_dt^{d-1}$ points exist (Bondarenko, Radchenko and Viazovska 2013). Hence

$$M\;\asymp\;\Big(L+\tfrac e2k\mathfrak r+O(\log1/\varepsilon)\Big)^{d-1}.$$

The first-order snapping certificate (both views' Theorem 1) gives $M\gtrsim(k\mathfrak rB_0/\varepsilon)^{d-1}$ instead, polynomial in $1/\varepsilon$; at $k\mathfrak r=7.5$, $M=12$ it bounds the error by $0.98$ where the exact 2-D error is $10^{-10}$. View B proposes to keep only the first-order certificate. That charges the shallow model $10^{13(d-1)}$ times its real cost, and every "depth wins" statement is then measured against a number nobody pays. Keep both: snapping for line components (atoms), quadrature for the smooth remainder (background). Their sum is the record's background-plus-atoms design, now as a theorem.

### 3.3 The composition ledger, with its floor [theorem; derived]

View A's Theorem 1 as stated, with two additions. First, the floating-point floor. With unit roundoff $u$ and channel coefficients $C_{\ell q}$, evaluating layer $\ell$ carries an error of order $u\lVert C_{\ell q}\rVert_1$, so

$$\lVert f-\widehat F\rVert_\infty\;\ge\;u\sum_{\ell}\Big(\max_q\lVert C_{\ell q}\rVert_1\Big)\prod_{j>\ell}K_j\qquad\text{[derived]},$$

and the exponential rates are meaningful only above it. Second, a warning that the product of Lipschitz constants $\prod K_j$ is a worst case that can be far from the truth. For the multiplication tree of Section 4 the ledger's $K=4$ per level gives $4^L=d^2$; the actual first-order sensitivity of a product of factors bounded by one is $1$ per level, so the expected floor is $u\,d$, not $u\,d^2$. Which one is right in fp64 is Experiment A.

### 3.4 The structure classes [theorem / measured, per row]

The content of a theory of depth is this table.

| structure of $f$ | cost with the block | shallow cost | how it is found from data | status |
|---|---|---|---|---|
| active dimension $m$: $f=g(U^\top x)$, $U\in\mathbb R^{d\times m}$ | $(c\,k\mathfrak r)^{m-1}N$ inside the subspace | $(c\,k\mathfrak r)^{d-1}N$ | eigenvectors of the gradient covariance of a first even fit, iterated | measured: composition in $d=3$ from $4\times10^{-7}$ to $2\times10^{-11}$; sheet in $d=5$ from $10^{-6}$ to $5\times10^{-11}$ (expH04) |
| function of a quadratic form: $g(x^\top Ax+b^\top x+c)$ | $(\mathrm{rank}A+1)N_1+N_2$, linear in $d$ | $(c\,k\mathfrak r)^{d-1}N$ | none needed: any orthonormal frame; the anchor from the data | theorem (View B's radial case, generalized); E1 oracle arm of expI01 |
| short ridge sum inside a channel: $g\big(\sum_{r\le S}p_r(v_r^\top x)\big)$ | $SN_1+N_2$ | atoms exact once found; $(c\,k\mathfrak r)^{d-1}N$ otherwise | projection pursuit on the residual, then variable-projection Gauss-Newton with the readout eliminated | measured for $g=\mathrm{id}$: sums of 1 to 8 ridges recovered to $10^{-13}$ with directions exact to fp64 in $d=3,4$ (expH06); open for $g\ne\mathrm{id}$ |
| product, arithmetic circuit with $S$ multiplications at depth $L$ | $2S$ quadratic profiles at depth $L$; for $x_1\cdots x_d$: $O(d\log d)$ units, depth $\lceil\log_2d\rceil$ | $2^{d-1}$ directions with degree-$d$ profiles, exact | no known procedure | theorem (View A); learnability open |

Two remarks on the rows. The second row is where View B's theorem belongs, one line wider: radial symmetry is the case $A=I$, and the general statement costs nothing more. The fourth row is View A's product theorem; Appendix B states it with the facts it rests on and the one caveat that matters.

### 3.5 Learnability, stated plainly [measured / open]

In the shallow model the certificate transfers to the fit because the readout is linear: the empirical least-squares solution is the orthogonal projection onto the span, so it does at least as well as the certificate on the training data (measured: the fit often lands far below the certificate). In the block only the last readout $\Psi$ is such a projection. Everything upstream (directions $V$, channel tensors $C$) is a nonlinear search. What is known:

- Directions on a ridge component are recoverable to fp64 exactness from a coarse projection-pursuit start (expH06, three seeds each, 24 cases). A single multiplication gate is a two-ridge sum, $xy=\tfrac12(v_+^\top z)^2-\tfrac12(v_-^\top z)^2$ with $v_\pm=(1,\pm1)/\sqrt2$; the record's product-sines target is the same object with sinusoidal profiles and the hierarchy found it at $2\times10^{-14}$. So one gate is a solved problem.
- A single channel $g(P)$ is identifiable up to a monotone reparametrization ($\nabla f=g'(P)\nabla P$ fixes the level sets); two or more channels are not unique. This is the record's own statement and it is where expI01's E3 arm lives.
- A computation graph (the fourth row) has no known data-driven recovery procedure at any precision. The first expI01 rows (rank 1 at $1.1\times10^{-6}$, untied at $5.9\times10^{-7}$ on fast waves in $d=3$, one seed, far above the oracle floor) say the optimization gap is real even for one quadratic channel.

The honest summary: constructions exist for every row when the structure is known (all four are explicit). What does not exist is a construction from data for unknown structure beyond the first three rows, and for the fourth row not even a stable procedure. That is the research question, and the paper should end on it rather than on "the central question is whether these factorizations can be learned".

---

## 4. What "practical like 1-D" means, and whether depth can have it

In 1-D the recipe is: geometry by rule ($\lambda$, uniform mesh, halo), one solve, no per-target tuning, $n\ge8$ to $16$ times the unit count. In 2-D and 3-D the record kept that shape: even directions by rule, offsets by the two-floor crossing, atoms by projection pursuit plus Gauss-Newton, one solve. The cost grew as $(k\mathfrak r)^{d-1}$, which Section 2 says is unavoidable for generic targets.

For the block the shape is: per-layer geometry by rule (mesh, $\lambda$, collar, range tracking of every channel), a solved last readout, and a nonlinear search over a small parameter set. The size of that set is the measure of practicality:

$$\#\text{nonlinear parameters}\;=\;Md+MN_1K\;(+\,K\text{ biases}),$$

for example $d=8$, $M=16$, $N_1=32$, $K=2$: about $1.2\times10^3$. Variable-projection Gauss-Newton at that size is cheap (the Jacobian is $n\times10^3$ on $n\sim10^4$ to $10^5$ rows) and it is the method that reached fp64 on directions. So "solvable" is plausible in the sense that matters: a second-order method on a parameter set that does not grow with the data, everything else fixed by rule or solved. The record's practicality gate is satisfied by construction (no per-sample state, $O(\#\text{parameters})$ optimizer state, the design matrix is the memory). What is not known is the basin: whether the search converges from a random start, and for $K\ge2$ what it converges to, since the factorization is not unique.

The decision procedure this gives for a target presented as data is short, and every step of it is either measured or is one of the experiments below:

1. Fit the even shallow model once; read the active dimension $m$ from the gradient covariance. If $m\le3$ or $4$, finish shallow inside the subspace (measured to $10^{-11}$ and better).
2. If $m$ is larger, the shallow model is out by Section 2. Try a block with small $K$ by VarPro and Gauss-Newton; read the refit error against the oracle floor.
3. If that fails, fp64 is not reachable for that target at that budget by any method we know, and the honest output is the shallow error at the affordable budget with the two-floor split.

---

## 5. Experiments

Each uses the expI01 notebook machinery (fixed meshes, range tracking, QR-then-SVD readout at rcond $10^{-14}$, the confound checklist) and each answers a question the theory cannot.

**A. The composed floor against depth.** Build the multiplication tree for $x_1\cdots x_d$, $d\in\{2,4,8,16\}$, as an oracle arm: leaves in $[-1,1]$, every gate two fixed directions $(1,\pm1)/\sqrt2$ and one profile $\tfrac12t^2$ solved once on its range with $N_1=24$, the last readout solved. Units $2(d'-1)N_1$ with $d'$ the padded leaf count: 48, 144, 336, 720. Measure the test error against $d$. The ledger predicts a floor of order $u\,d^2$; the first-order sensitivity predicts $u\,d$. Either way this is the first measured number for a composed fp64 floor, and it tells us whether the ledger's Lipschitz products are usable or hopelessly pessimistic.

**B. The Waring cliff in fp64.** Shallow ridge model on $x_1\cdots x_d$ for $d=4,5,6$ ($2^{d-1}=8,16,32$), error against $M$, with (i) the Waring directions $(1,\varepsilon_2,\dots,\varepsilon_d)/\sqrt d$ and (ii) the record's even nested directions; profiles resolved with $N\ge30$ so only $M$ binds; rcond $10^{-14}$; readout norm logged. Prediction: with (i) the floor arrives exactly at $M=2^{d-1}$ (the polarization identity), with (ii) later. The question of interest is whether fewer than $2^{d-1}$ directions ever reach the floor through cancellation. The border-rank route requires coefficients of size $1/\varepsilon$, which the truncation and the readout-norm rule remove, so the expectation is no, and then View A's exact lower bound is the operative one in fp64. That is a claim the theory cannot make and one run can.

**C. One gate, then one tree, from a random start.** VarPro plus Gauss-Newton on $x_1x_2$ in $d=2$ with a single-channel block ($M=2$ learned directions, $K=1$, $N_1=N_2=24$): must recover $v_\pm$ to fp64 and the floor (the product-sines result says it will). Then $x_1x_2x_3x_4$ with a depth-2 block ($K=2$ channels feeding one gate): success is the floor with exact directions from three seeds. This is the depth analogue of the expH06 ridge recovery and the first data point for the fourth row of the table.

If A and B come out as predicted and C succeeds, the paper has a measured depth result. If C fails, that is the finding, and the theory's fourth row stays labeled open.

---

## 6. What to tell the two agents

Merge, then cut.

Keep from View A: the three-resource framing, the ledger (Theorem 1), the product theorem and the circuit corollary (Theorem 3, Corollary 4), the direction-error term. Keep from View B: the four-claims discipline with a status on every claim, the radial example as one row of the table, the demotion of KAT to a remark.

Add what neither has: the entropy lower bound (Section 2) as the first statement; the quadrature certificate as the shallow cost, so the comparison is honest; the composed floor in the ledger; the learnability section (3.5) with the record's numbers; the experiments.

Drop: the near-radial bound; the Pareto class $\mathcal C_{L,B}(\Gamma)$ as a definition (keep the frontier as a figure of speech until it is characterized); the KAT modulus theorem as a numbered result (one paragraph); the Sprecher architecture (a footnote, with the shared-outer-function count correction from the review); the confidence percentages.

Then stop writing theory until Experiments A, B and C are run.

---

## Appendix A. Numbers used above

- Degree $p$ at $\varepsilon=10^{-13}$ from $(e\,k\mathfrak r/2p)^p\le\varepsilon$: $k\mathfrak r=1,2,4,5.3,7.5\Rightarrow p=14,17,22,25,29$.
- Fast waves: $k=6\pi/\sqrt2=13.3$; $k\mathfrak r=4.0$ on the $\mathfrak r=0.3$ ball (expH06), $5.3$ on the $\mathfrak r=0.4$ ball (expH05).
- Monomial counts $\binom{22+d}{d}$: 276, 2300, 14950, 80730 for $d=2,3,4,5$; $5.9\times10^6$ at $d=8$; $6.5\times10^7$ at $d=10$. At $p=30$ (a strip of width one, $\rho=e$): 496, 5456, 46376, 324632; $8.5\times10^8$ at $d=10$.
- Record: 3-D floor at 9216 units, $4.0\times$ the $p=22$ count; 4-D crossing 25k to 50k, $1.7$ to $3.3\times$.
- Product tree, $N_1=24$: units 48, 144, 144, 336, 336, 720 for $d=2,3,4,5,8,10$. Shallow Waring representation: $2^{d-1}$ directions with a degree-$d$ profile each, about $30\cdot2^{d-1}$ units: $2.4\times10^2$ at $d=4$, $1.5\times10^4$ at $d=10$.

## Appendix B. The product theorem, stated with its sources

**Upper bound (View A, correct).** $xy=\tfrac12(v_+^\top z)^2-\tfrac12(v_-^\top z)^2$ with $v_\pm=(1,\pm1)/\sqrt2$; a balanced binary tree over $d'=2^{\lceil\log_2d\rceil}$ leaves (padded with ones) has $d'-1<2d$ gates, $2(d'-1)<4d$ quadratic profiles, depth $\lceil\log_2d\rceil$.

**Lower bound (View A, correct as an exact statement).** If $x_1\cdots x_d=\sum_{r=1}^Mp_r(v_r^\top x+b_r)$ on an open set with $p_r\in C^d$, then $D^d(x_1\cdots x_d)=\sum_rp_r^{(d)}(\cdot)\,v_r^{\otimes d}$, so $M$ is at least the symmetric (Waring) rank of the monomial. The complex Waring rank of $x_1^{a_1}\cdots x_n^{a_n}$ is $\prod_i(a_i+1)/(\min_ia_i+1)$ (Carlini, Catalisano and Geramita 2012), which is $2^{d-1}$ for the square-free monomial; the real rank is the same for this monomial (Carlini, Kummer, Oneto and Ventura 2016). The polarization identity $x_1\cdots x_d=\frac{1}{2^{d-1}d!}\sum_{\varepsilon\in\{\pm1\}^{d-1}}\big(\prod\varepsilon_i\big)\big(x_1+\sum_{i\ge2}\varepsilon_ix_i\big)^d$ attains it.

**The caveat that matters.** The lower bound is for exact representation. For $\varepsilon$-approximation the relevant quantity is the border rank, which can be smaller than the rank for some monomials ($x^{d-1}y$ has rank $d$ and border rank 2), and I cannot cite the border rank of $x_1\cdots x_d$ with confidence. Border-rank constructions realize the limit $\lim_{t\to0}[(x+ty)^d-x^d]/(dt)$ with coefficients of size $1/t\sim1/\varepsilon$. In fp64 with truncation at rcond $10^{-14}$ and the $O(1)$ readout-norm rule those solutions do not exist, so in the working regime the exact rank should be the operative bound. Experiment B measures exactly this.

**Where this sits in the literature.** The shallow-versus-deep separation for compositional functions is Mhaskar and Poggio (2016) and the deep-approximation literature View A cites; the Waring-rank view of shallow ridge representations of polynomials is classical. What is specific to this program is the QI accounting: each edge is a block with $\gamma h=0.25$, a fixed mesh on a tracked range, a rate $e^{-aN}$ above the floor, and a solvable last readout. That accounting is what turns "depth $\log d$, width $O(d)$" into a unit count with a floor, and it is what Experiment A tests.
