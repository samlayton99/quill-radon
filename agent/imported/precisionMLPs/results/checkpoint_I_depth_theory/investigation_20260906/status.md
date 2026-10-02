# Checkpoint I investigation — 2026-09-06

Coordinator: Codex root task. Status: complete; final report and artifact review passed.

Scope: read both papers and checkpoints A–C; audit checkpoint F initialization and depth results; reproduce checkpoint I's positive and negative evidence; run controlled tests of QI block initialization and training; preserve all pre-existing work. No commits or publication requested.

Context: local reviewed guidance revision `d7eb04d1bf15a82975546d5302bae8caf6be7fa8`; installation check passed, context synchronization failed at transport. Project evidence is read directly from the local repository.

Completed: read the workshop paper, replacement Section 3, implementation notes, and A–C writeups. The combined verification suite passes 33 tests: 24 existing I02 checks, four analytic checks, two fixed-bank invariant tests and three corrected-GN checks. All new Python files compile; main-report and README links resolve.

Findings: the saved I01 notebook has only E0/E6 outputs, whereas I02 contains useful A2 and Lorenz positives. Deep failures often exhibit mesh escape and constant channels. F04's tenfold gains are on 1D synthetic tasks; real-data initializer gains are smaller, but depth gains survive matched parameter counts. A compact fixed-bank model with layerwise initialization beats approximately parameter-matched dense QI on airfoil and Parkinsons and is competitive on bike. Root screening rejects small coefficient steps and smooth random profiles as sufficient fixes. A positive quadratic initialization helps radial waves even after releasing its coefficient restriction (same 539 parameters, median relative L2 .0749 to .000977 over three paired runs). It helps other tested targets relative to the original initialization but does not beat the dense control there. Extra depth from a trained good model buys only 5–6% in the controlled one-seed continuation.

Completed studies: analytic screens; 48-run confirmation; 18 paired quadratic initialization/resolution/release follow-ups; four refinement diagnostics; canonical real-data and resolution comparisons; 53 depth fits; isolated corrected-GN diagnostic. Existing source/result files remain untouched. Python environment: repository `.venv/bin/python`, PyTorch 2.4.1, CPU/fp64.

Final independent review verified the nine released-quadratic runs and all main-report medians against raw records. Its scope and parameter-range clarifications were incorporated. Summary plots and regenerated depth legends were visually inspected. No material unresolved implementation or reporting fault remains in these bounded scalar/canonical experiments; the main report explicitly records the limits on generalization and precision claims.

Scoped reports: `evidence_audit/evidence_audit.md` (saved evidence/protocols), `f_init/` (F initialization and real-data probes), `depth/` (depth and occupancy). Root experiment code: `experiments/expI03_investigation/probe.py`; root screening outputs: `screen*.json`, `ridge_screen.json`; held-out seeds: `confirm_d3.json`.

Main report: [expI03_results.md](../expI03_investigation/expI03_results.md). Reproduction and model usage: [README](../../../experiments/expI03_investigation/README.md). All new experiment code and outputs have separate paths. Existing checkpoint I and F edits are preserved. No commits or publication.
