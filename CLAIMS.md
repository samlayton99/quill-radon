# Claims

The only lasting conclusions in this repo. **Only Sam adds, edits, or retires entries.** Agents may cite these as settled; agents propose new claims in `agent/proposals/` and flag them in `human/INBOX.md`.

Entry format: `ID` | statement | scope and caveats | evidence | date accepted.

## Paper 1: QUILL (submitted to ICLR 2027; `papers/paper1_quill/v3_iclr2027_submission.pdf`)

Seeded from the submitted paper, which is Sam's reviewed work. Statements paraphrase the paper; the paper's own wording governs.

- **P1-1 Construction.** For $f$ with a bounded analytic extension to a complex neighborhood of $[-1,1]$, the single-hidden-layer tanh network $q(x)=b+\sum_j w_j\tanh(\gamma(x-x_j))$ on a uniform grid ($h=2/N$, $\gamma=\lambda/h$, $0<\lambda\le1$, halo $R=\lceil\sqrt N\rceil$ per side) achieves geometric error decay in width $W$ plus an aliasing term $\propto e^{-\pi^2/\lambda}/W$ plus sampling/computation allowances, with a width-independent readout $\ell_1$ bound. Scope: Theorem 3.1 / A.1 conditions (admissible geometry, numerical and sampling conditions, reference-scaled truncated SVD recovery). Evidence: paper Section 3.2, Appendix A. Accepted: submission (Sep 26, 2026).
- **P1-2 Logarithmic width.** For target error above the finite-precision floor, $W=O(\log(1/\varepsilon))$ with $O(\log(1/\varepsilon))$ working precision (Corollary 3.1.1; Table 1). Accepted: submission.
- **P1-3 Finite-halo boundary correction.** The boundary contribution is corrected inside the existing halo slots by a prescribed-denominator Pade-type construction with explicit, width-independently bounded partial-fraction coefficients (Appendix A, Props. A.1.1-A.1.2; source `papers/paper1_quill/appendix_source_v7/sections/7_appendix/01_Construction/09_26_sl.tex`). Accepted: submission.
- **P1-4 Bandwidth rule.** The high-precision relative bandwidth is predicted by $2\exp(-\pi^2/\lambda)\sinh(2\pi\bar\omega/(N\lambda))=\epsilon_{\rm eff}$ (Eq. 7; Appendix B). Accepted: submission.
- **P1-5 Slow scale acquisition.** Small slopes delay readout gradient descent (Theorem 3.2), and under dispersed gradient signal joint GD does not reach the construction's slope scale within the analyzed budget (Theorem 3.3; Appendices E-F). Accepted: submission.
- **P1-6 Circuit compilation.** Bounded arithmetic circuits compile into QUILL networks with $O(S_\times(1+\log_+(\kappa/\varepsilon)))$ hidden units and $O(D_\times)$ nonlinear depth (Theorem 4.1); tested polynomial programs reach sampled relative $L_2$ below $6.5\times10^{-14}$ in FP64. Accepted: submission.
- **P1-7 LLM installation (proof of concept).** A width-80 QUILL $x^2$ module installed in three frozen LMs with engineered routing answers all 2,250 evaluated 3/5/7-digit squaring problems. Accepted: submission.

## Paper 2: Radon / higher-dimensional QUILL

*No entries yet.* Candidates awaiting Sam's review are in `agent/proposals/` and listed in `human/INBOX.md`.
