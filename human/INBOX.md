# Inbox

Things Sam needs to see or decide. Agents post here (2-4 lines + link); Sam clears items, and cleared items move to `inbox_log.md` with his decision. Newest at the bottom.

---

**1. Confirm the requirements checklist.** (decision, Oct 2) I condensed your Oct 1-2 messages into a 9-point checklist at the top of `human/REQUIREMENTS.md` (quotes below it). Confirm or edit; agents will treat it as binding.

**2. Promote the analytic known-function construction?** (promotion, Oct 2) Training-free, fit-free Radon-QUILL: 3-D anisotropic Gaussian at 6.4e-16 (value) / 7.9e-15 (Laplacian) with 244,800 neurons, construction costs about one inference pass. Caveats: allocation was oracle-swept, all $d\ge3$ floors are Gaussians, 4-D/5-D used the old long halo. `agent/proposals/P001_analytic_radon_construction.md`.

**3. Promote the "curse lives in the sphere rule" result?** (promotion, Oct 2) Direct-construction error is exactly angular + profile; angular error tracks the rule's exactness degree $t$, almost independent of $d$; product rules use ~$(d-1)!$ more directions than needed (19x at $d=5$). `agent/proposals/P002_angular_error_and_allocation.md`.

**4. Decision: is route 2 "dressing up another method"?** (decision, Oct 2) The audit found the Oct 1-2 PDE solver is a polynomial spectral least-squares solve compiled exactly into a one-hidden-layer tanh MLP: accuracy is set by polynomial degree, not neurons; $M=(p+1)^{d-1}$ directions makes it brittle in $d$; headline NS numbers came from tuned chained runs. This decides what the PDE half of paper 2 builds on. `agent/proposals/P003_route2_is_spectral_collocation.md`.

**5. Promote the global-least-squares memory wall as paper 2's motivation?** (promotion, Oct 2) Pre-Radon: $d=3$ NS stopped at 12.3k unknowns / 7 GB while accuracy was still improving. `agent/proposals/P004_global_lstsq_memory_wall.md`.

**6. Overclaims in the imported docs.** (FYI, Oct 2) The imported precisionMLPs docs contain claims the audits found false or overstated, e.g. `status.md` lists the encoded Fourier-Galerkin NS solve as a "validated outcome"; `quill_pde_checkpoint.md` credits a product-gate solver with beating PINNs; one Burgers number (2.52e-14) exists in no saved file; the 2-layer depth compiler is rated "successful pure-MLP repair". They are left unedited in the snapshot; full list in `agent/workspace/2026-10-02_import_audit/README.md`.

**7. No like-for-like benchmark yet.** (decision, Oct 2) expF19 never ran the earlier benchmark suites (BWLer five, poisson_man, darcy_man, the 18-problem zoo, 2-D+t NS), so none of its numbers compare to pre-Radon numbers or to published BWLer. If paper 2 makes physics claims, it needs a frozen benchmark set and baselines (PINN with refit certificate, tuned ELM, spectral). Proposed set in `agent/workspace/2026-10-02_import_audit/audit_prior_physics.md` §2.
