# P001: Training-free, fit-free Radon-QUILL construction for known functions

Status: **proposed** (agent-drafted Oct 2, 2026; awaiting Sam). Serves: paper 2 core result. INBOX item 2.

## Proposed claim
For a target $f$ on a ball in $\mathbb R^d$ whose filtered Radon profiles $q_v$, defined by $\widehat q_v(\omega)=C_d|\omega|^{d-1}\widehat f(\omega v)$ with $C_d=|S^{d-1}|/(2(2\pi)^{d-1})$, can be evaluated at complex arguments, a single-hidden-layer tanh network
$$\widehat f(x)=b+\sum_{m=1}^{M}\sum_{j}a_{mj}\tanh\big(\gamma(v_m\cdot x-c_j)\big)$$
is written down in closed form, with no least squares and no training: directions and weights $(v_m,w_m)$ from a positive sphere quadrature rule; per direction the paper-1 geometry ($N$ interior centers, $R=\lceil\sqrt N\rceil$ halo, $\gamma=\lambda/h$); readouts $a_{mj}=\tfrac{hw_m}{2}\operatorname{Im}q_{v_m}(c_j+iA)/A$, $A=\pi/(2\gamma)$, plus the paper-1 finite-contour halo correction. The error splits exactly into an angular quadrature part and a per-profile encoding part (no factor $M$ for positive rules), and construction costs about one inference pass.

## Evidence (agent-side; preliminary)
- 3-D rotated anisotropic Gaussian, validated on 549 fresh points (`agent/imported/precisionMLPs/results/checkpoint_F_applications/expF19_radon_direct_pde/quill_review/allocation_optimized_selected.json`, produced by `experiments/expF19_radon_direct_pde/quill_sweep.py`; re-read Oct 2 by the coordinator):

| Neurons | $M$ | $N$ | $\lambda$ | value rel. err | gradient | Laplacian | readout $\ell_1$ |
|---|---|---|---|---|---|---|---|
| 6,480 | 144 | 33 | 0.40 | 3.8e-5 | 2.5e-4 | 2.9e-4 | 2.32 |
| 18,000 | 400 | 33 | 0.40 | 2.3e-8 | 1.8e-7 | 2.9e-7 | 2.32 |
| 46,080 | 1024 | 33 | 0.40 | 2.9e-10 | 1.8e-9 | 2.8e-8 | 2.32 |
| 84,992 | 1024 | 65 | 0.30 | 2.1e-13 | 1.6e-12 | 7.7e-12 | 2.03 |
| 244,800 | 1600 | 129 | 0.25 | 6.4e-16 | 1.5e-15 | 7.9e-15 | 1.90 |

- Origin: Oct 1 subagent `radon_interpretation_test`; 205,824 neurons, 2.26e-15 on the older long-halo grid (`results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/radon/report.md`).
- Representation identity and general-$d$ Gaussian profile ${}_1F_1$ re-derived by the theory audit (`agent/workspace/2026-10-02_import_audit/audit_theory.md` §1.2).
- Without the complex-shift correction (leading derivative samples) the error is 18-36%; with a plain $\sqrt N$ halo and no rational correction, ~4e-5.

## Caveats Sam should weigh
1. $(M,N,\lambda)$ were selected by sweeping against the known target, then validated on fresh points; validation set is 549 points.
2. All floor results with $d\ge3$ are Gaussians (entire profiles). The method needs $q_v$ at complex arguments; "arbitrary $f$" is not covered (Gaussian mixtures, trig polynomials, polynomials, known ridge sums, closed-form radial functions are).
3. 4-D ($8.15\times10^{-15}$, 6.59M neurons) and 5-D ($6.4\times10^{-11}$ anisotropic, angular-limited) used the superseded long plain halo, so neuron counts are ~4x inflated; they should be rerun with the corrected encoder before being cited.
4. The angular error theorem for $d\ge3$ (spherical-design certificate) is only sketched, outside the original repo (`papers/related_notes/review_compositional_ridge_QI.md` §2.2).
5. Derivative accuracy is empirical only.

## Suggested human experiment (if Sam wants a clean claim)
h01: corrected-encoder construction in $d=2..5$ on a frozen target family (Gaussian mixtures + one non-Gaussian closed-form class), fixed allocation rule chosen before running, fresh-point $L_2$ and $L_\infty$ for value/gradient/Laplacian, neurons-vs-error curves.
