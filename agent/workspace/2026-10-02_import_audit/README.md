# Import audit of the precisionMLPs Radon work (Oct 2, 2026)

- **Question:** What did the Sep 29 - Oct 2 Codex agent loop (and the earlier ridge/physics work) actually establish about the Radon construction, what is noise or overclaim, and what should paper 2 carry forward?
- **Who asked:** Sam, Oct 2: "look at everything that has been done with radon ... read, understand, parse, and organize that work so that we have a clean repo to work from for the second paper."
- **Serves:** repo setup; INBOX items 1-7; proposals P001-P004.
- **Status:** done. Six read-only audit subagents plus the coordinator. No experiments were run. Numbers were spot-checked against saved JSON (the coordinator re-read `quill_review/allocation_optimized_selected.json` and ran 88 core tests in the snapshot).

## Files

| File | Scope |
|---|---|
| `audit_theory.md` | The Radon-QUILL theory for known functions: statement, proven/derived/empirical ledger, open theorems |
| `audit_known_function.md` | All known-function experiments (checkpoints E, H, I; expF19 encoding scripts); best-results table; M-vs-N evidence; 22 overclaims |
| `audit_prior_physics.md` | Checkpoint F (expF01-F18) + cdeng: the global-lstsq memory wall, recommended benchmark/baseline set |
| `audit_expF19_early.md` | expF19 phases A-C: encoded external solves, native spectral dynamics, product gates, depth compiler, ridge PINN |
| `audit_route2_current.md` | Route 2 / streamed native residual solver: algorithm as math, requirement check, what results show, overfitting, dependencies |
| `audit_codex_sessions.md` | Chat-history timeline, rejected approaches, retractions, insights that live only in chat, final state |
| `sessions/` (local only; gitignored) | Sam's messages from the Radon sessions; full dialog extracts (main loop, spoke session, theory session); session index; three Codex HTML visualizations; `RAW_SESSIONS.md` (paths to raw logs) |

## Synthesis (observed; not conclusions)

1. **The strongest result is the known-function construction** (P001): explicit readouts, no fit, one hidden layer, construction about one inference pass; 3-D anisotropic Gaussian at 6.4e-16 with 244,800 neurons. Its scope is targets whose Radon profiles are available in closed form at complex arguments. Its error is exactly angular + profile, and the dimension cost lives in the sphere rule (P002).
2. **The PDE program did not reach a method that meets Sam's requirements.** The sequence was: encode external solves (rejected), native spectral dynamics, product gates (rejected), depth compiler (rejected), then route 2. Route 2 is a polynomial spectral least-squares solve compiled exactly into a flat tanh MLP (P003). It works on smooth, polynomial-friendly problems, is brittle in $d$ through $M=(p+1)^{d-1}$, and its headline NS numbers come from tuned chained runs. Under a frozen policy it failed Helmholtz $n=4$, Allen-Cahn and Burgers inverse.
3. **The motivating wall is real** (P004): global least squares stopped at ~12.3k unknowns / 7 GB for $d=3$ NS.
4. **The open problem for the PDE half** (agent's reading): getting profile coordinates for an *unknown* solution without a global solve and without a polynomial intermediate whose size grows like $\binom{p+d}{d}$. The known-function theory says the representation is cheap; nothing yet makes the PDE solve cheap.
5. **No like-for-like physics benchmark exists** for the Radon work (INBOX 7).

## Overclaims and errors in the imported docs (left unedited in the snapshot)

From the six audits; details and sources in each file.

- `results/.../expF19_radon_direct_pde/status.md` lists Cole-Hopf Burgers and the Fourier-Galerkin 3-D NS (then encoded) as "validated outcomes"; `nonlinear/report.md` frames it as a route to neural NS. Sam rejected this as cheating.
- `docs/quill_pde_checkpoint.md`: "dramatically more accurate and faster than ... PINNs" credits a product-gate tensor solver on a matched sine eigenbasis; the classical solver was faster still.
- "Native Burgers 2.52e-14 at dt .0005" appears in no saved JSON (best saved 2.50e-13, equal to the FFT baseline).
- NS animation and "867 tanh units" labeled "Native QUILL PDE dynamics": product-gate spectral solver, 59,049 coefficients.
- `docs/radon_method_catalogue.md` E01 rates the 2-layer depth compiler "Req 7, successful pure-MLP repair"; `docs/quill_pure_mlp_status.md` says "both architecture gates passed". Its only PDE test has a quadratic solution inside the basis.
- The wave dense-autograd check (5.9e-16) is quoted beside "~1e-14" but was run on the 8.7e-4 model.
- Inverse headlines cherry-pick: "2.8e-9 tensor" only at 400 directions; "nu 1.14e-11" is the K64 refinement (K32: 2.1e-6).
- Phase-A "corrected" encoder is the long-halo complex-shift density; its neuron counts are mixed with corrected ones.
- 4-D/5-D construction results use the superseded long plain halo (~4x neuron inflation) and are reported beside the corrected 3-D result without saying so (`methods/analytical_construction.md`).
- Least-squares laws (max-of-floors with an unproven $\sqrt2$ bracket, $e_N\sim N^{-10}$, $\alpha<1/2$, polynomial-floor direction count) are blurred into the direct construction; `ridge_quadrature_theory.md` labels the $\sqrt2$ bracket a theorem.
- `ridge_quadrature_theory.md` "no null space" holds only on all of $\mathbb R^d$; "the 1-D leg is dimension-free" ignores the $|\omega|^{d-1}$ sharpening.
- "d=256 with 804 neurons" is a 4-ridge target; QMC results (0.06-2.3%) are not precision results.
- The F19 allocation selector picks the largest M and N in all 9 budgets (no-op); `quill_review` allocation and lambda choices are made against the known function.
- Commit b8ffdf0 changed the direction rule, so saved E01/H01/H05 data will not reproduce exactly.
- Route 2 docs ("analytical readout coordinates", "polynomials are not in the deployed model") never state that the solve is polynomial spectral collocation; the "p32" result had done zero p32 corrections; soft time budgets were not enforced (120 s became 3,749 s).
- Pre-Radon: `expF_results.md` "geometry wins 19-685x" (revised by expF03 to ties / 2-100x vs tuned ELM; spectral wins 7/9 and 9/9); expF06 "nonlinear floor 1e-7" was an artifact; expF03 PINN Adam ran on MPS fp32 though the text says CPU fp64; expF13 Poisson-CG "below reference noise" is false; expF17 "BWLer (explicit)" is a Chebyshev basis, not BWLer; expF12 "the moonshot works" is 1.5e-6 on a tensor architecture with gradient-only advection.

## What lives only in chat (now preserved here)

Degree-of-freedom count of ridge decompositions $D_k(M,d)=\max(0,M-\binom{d+k-1}{k})$; filter constant $C_d$ and worked 3-D Gaussian; uniqueness vs stable recovery; fixed-cutoff blowup argument; NS Taylor recurrence; all-at-once inverse and closed-form $\nu^*$; ball-frame contraction bound; memory arithmetic. See `audit_codex_sessions.md` §4 and `sessions/`.
