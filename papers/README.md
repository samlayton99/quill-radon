# Papers

## Paper 1: QUILL (the gold standard)

**Local only:** the PDFs and LaTeX below are gitignored (paper 1 is under double-blind review) and exist only in local clones on Sam's machines.

Three versions, oldest to newest. Each is authoritative for what it covers; where they disagree, the newest governs.

| File | What it is |
|---|---|
| `paper1_quill/v1_workshop_QIs.pdf` | First version: "Constructing Machine-Precision Neural Networks with Quasi-Interpolants". Expressivity vs optimization framing; $\lambda\to0$ under training. (Was `precisionMLPs/papers/QIs_workshop.pdf`.) |
| `paper1_quill/v2_section3_rewrite.pdf` | The rewrite of Section 3 only (the construction: cardinal function, tanh realization, error-width tradeoff). Read together with v1, not alone. (Was `precisionMLPs/papers/Section_3_Rewrite.pdf`.) |
| `paper1_quill/v3_iclr2027_submission.pdf` | Final submission, ICLR 2027: "QUILL: In-weight computation with MLPs". Theorem 3.1 (construction), bandwidth rule (Eq. 7), Theorems 3.2-3.3 (slow scale acquisition), Theorem 4.1 (circuit compilation), LLM installation. 50 pages with appendices. (Was `~/Downloads/QI_MLPs___ICLR_2027_Submission (18).pdf`, Sep 26, 2026.) |
| `paper1_quill/appendix_source_v7/` | LaTeX source of the submitted appendices. `sections/7_appendix/01_Construction/09_26_sl.tex` is the 1-D construction theorem with the finite-contour halo correction that every Radon profile in paper 2 uses. |
| `paper1_quill/practical_implementation.tex` | Older fp64/mpmath implementation note; superseded by `09_26_sl.tex` for the encoder. |

The paper 1 results that this repo treats as settled are in `../CLAIMS.md`.

## Related notes (`related_notes/`)

Earlier theory notes that were outside the precisionMLPs repo (from `~/Downloads`, Sep 3, 2026), copied because paper 2's missing angular theorem is sketched in them:

- `review_compositional_ridge_QI.md`: §2.2 spherical-design angular certificate (T1), direction-count estimate $M\asymp(L+\tfrac e2kr+O(\log1/\varepsilon))^{d-1}$.
- `depth_theory_response.md`: §3 shallow-cost proposition; table of structure classes (active subspaces, sparse ridges, low-degree) vs $(k\rho)^{d-1}$.

These are agent-written notes of uncertain status, not claims.

## Paper 2

Drafts go in `paper2/` when writing starts.
