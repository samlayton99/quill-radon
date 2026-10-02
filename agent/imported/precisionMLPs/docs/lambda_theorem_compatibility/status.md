# Lambda section compatibility review

Coordinator: current Codex task. Started 2026-09-14.

Purpose: revise `choosing_optimal_lambda_bundle (2).zip`, preserving Sam's one-page argument and companion appendix, against the updated proof. Neither reference manuscript is an editing target.

Sources: the pasted conversation, both attached lambda bundles, the older `QIs_workshop.pdf`, and the September 13, 2026 `theorem_for_sam.pdf`. Existing manuscript and experiment files remain untouched.

Current status: completed. The new local density exactly matches the retained Fourier amplitude; the actual finite-contour pole term is bounded by twice the old unanchored replica sum. The revised compact bound includes that factor two. The appendix proves the connection in three lemmas and proves the monotone budget-selection argument. The representative-frequency step remains explicitly an approximation.

Latest deliverable: `choosing_optimal_lambda_bundle_revised_v7.zip` and the source/output folder `choosing_optimal_lambda/`. Earlier ZIPs are preserved. Main: one page, 498 extracted words, 11 pt, 7 pt paragraph spacing. Combined review: eight pages (main, landscape figure, six-page appendix).

Iteration 7 replaces Gaussian with exact GELU in the 2-by-4 comparison, and replaces the original targets with sin(2*pi*x)+0.5*sin(6*pi*x)+0.25*sin(10*pi*x) and exp(sin(3*pi*x)). It retains the expC08 viridis layout and original measured curves. Precision panels use p=24,32,40,48,53 because the p=16 refined GELU roots exceed the measured lambda interval. The appendix and standalone selector now support the GELU second-derivative kernel and match the displayed targets. The main text remains byte-identical to iteration 6.

Validation: 96 standalone-selector checks, 43 first-pair evaluations, 18 logarithmic-sensitivity checks, and the existing high-precision tail and finite-pole checks pass. The displayed later-pair ratio is below 2.29e-7. PDFs build with no overfull boxes or unresolved references; the figure and revised appendix layout were visually inspected. Original source observations are byte-identical to iteration 6. ZIP integrity passed. No reference-paper edits, new network fits, commits or publication.

Independent work:
- `qi_proof_audit.md`: actual QI alias term, assumptions, and output integration.
- `compact_bound_audit.md`: compact first-pair bound and its sufficient conditions.
- Experiment context: read-only report through task messaging.

Shared context: local revision `1a626ad59f8931c5592c625afdfd0da80cb0bb5a`; sync failed and local installation check passed.
