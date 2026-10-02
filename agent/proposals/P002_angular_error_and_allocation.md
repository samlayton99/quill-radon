# P002: Where the curse of dimensionality lives in the direct construction

Status: **proposed** (agent-drafted Oct 2, 2026; awaiting Sam). Serves: paper 2 scaling theory. INBOX item 3.

## Proposed claim (two parts, different strength)
(a) **Error split (proven, elementary).** For the direct construction (P001), $f-\widehat f=E_{\rm ang}+E_{\rm prof}+E_{\rm fp}$ exactly, with $\|E_{\rm prof}\|_\infty\le\max_v\|q_v-\mathcal Q_Nq_v\|_\infty$ for positive rules. So the $M$-vs-$N$ tradeoff is an exact sum of an angular floor and a 1-D profile floor, not the least-squares "max law" of checkpoint H.

(b) **Angular error depends on the rule's exactness degree $t$, nearly independently of $d$ (empirical).** At matched $t$ the 4-D and 5-D anisotropic-Gaussian angular errors agree within 1.6x ($t=15$: 7.3e-4 vs 4.6e-4; $t=47$: 6.8e-11 vs 6.4e-11); the isotropic Gaussian reaches the floor at $t=31$ in both. Hence the dimension cost is the direction count $M(t,d)$ of the sphere rule. Product rules use $M=n^{d-1}$ ($t=2n-1$), about a $(d-1)!$-order factor above the Delsarte-Goethals-Seidel lower bound (19x at $d=5$, $t=47$).

## Evidence (agent-side)
- Theory audit regrouping of `results/.../expF19_radon_direct_pde/scaling/structured*.json` by $t$ (`agent/workspace/2026-10-02_import_audit/audit_theory.md` §1.7, §1.11).
- Direct-construction held-out checks: max-of-floors ratio 1.000-1.026 (value) over 27 cells (`quill_review/allocation_prediction.json`).
- Spherical-design certificate sketch: $M\asymp(L+\tfrac e2kr+O(\log1/\varepsilon))^{d-1}$ (`papers/related_notes/review_compositional_ridge_QI.md` §2.2). The isotropic floor degree is predicted ($t\approx28$ vs 31 measured); the anisotropic case needs ~25 more degrees (angular smoothness term).

## Caveats
- (b) is two Gaussian families on one rule family; 3-D used a different rule and evaluation set.
- The 4-D/5-D data used the long plain halo; angular errors are unaffected by that, profile errors are not.
- Profiles sharpen with $d$ through the $|\omega|^{d-1}$ filter (effective bandwidth ~$\sqrt d$ for the isotropic Gaussian), so $N$ is not fully dimension-free.

## What would make it a claim
A human experiment fixing a target family and testing (i) $E$ vs $t$ across $d=3..6$ with one rule family, and (ii) whether a better positive sphere rule (designs, Lebedev in 3-D) reaches the same $t$-error with far fewer directions.
