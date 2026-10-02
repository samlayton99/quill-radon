# quill-radon: house contract

Paper 2 of the QUILL program. Paper 1 (`papers/paper1_quill/`, submitted to ICLR 2027) constructs single-hidden-layer tanh MLPs for univariate analytic functions to the fp64 floor by quasi-interpolation. Paper 2 extends the construction to higher-dimensional inputs and outputs through the Radon transform (ridge functions $\sum_m q_m(v_m\cdot x)$, each profile a 1-D QUILL), and makes it practical, with physics (PINN-style forward and inverse PDE problems) as the main application.

Read `README.md` for the map, `human/REQUIREMENTS.md` for Sam's binding rules, and `human/INBOX.md` for what is open. `AGENTS.md` is a copy of this file for Codex; keep them identical.

## The two sides (the core rule)

This repo is split by **authority**, not by who typed the code.

- **Human side** (`CLAIMS.md`, `human/`, `papers/`): what Sam directs, reviews, and keeps clean. Experiments whose question and design Sam set. **The only conclusions that last are in `CLAIMS.md`, and only Sam adds to it.** Anything Sam needs to see must land on the human side; he cannot review everything on the agent side.
- **Agent side** (`agent/`): a structured scratchpad and staging ground. Agents explore, prototype, audit, and run side studies here with more freedom. Everything here is **preliminary evidence**, however convincing it looks. It feeds the human side; it never concludes.

Agents work on **both** sides. When Sam directs an experiment, agents implement it on the human side (code, runs, figures, a draft writeup) under Sam's frozen `SPEC.md`. The fence is authority, not location:

1. Agents never write or edit a **Conclusions** section, never add to or edit `CLAIMS.md`, and never change a human experiment's `SPEC.md` without Sam's explicit sign-off in the conversation. Agents may draft a proposed claim in `agent/proposals/`.
2. Agent prose uses "observed", "suggests", "not yet checked". Never "proves", "establishes", "solves".
3. Agents may cite `CLAIMS.md` entries as settled. They may **not** cite another agent document as settled; cite it as evidence ("agent/workspace/... reports X"). This stops one agent's guess from becoming the next agent's premise.
4. The imported precisionMLPs snapshot (`agent/imported/`) is agent-side evidence, including its 60-method catalogue, rankings, and "validated" labels. Its docs were written by agents and contain known overclaims (see `agent/workspace/2026-10-02_import_audit/`).

## The two sides talk: no blindsides

`human/INBOX.md` is the channel. Agents **must** post a short item (2-4 lines + link to the agent-side detail) when:

- a decision is Sam's to make (research direction, experiment design, scope, cost, requirement interpretation);
- agent evidence **contradicts** a `CLAIMS.md` entry or a human experiment -- post immediately, never leave a contradiction sitting in a scratch file;
- a human experiment failed, broke, or deviated from its SPEC;
- a requirement in `human/REQUIREMENTS.md` looks wrong, too strict, or is being quietly violated;
- a finding is ready for promotion (link its proposal).

Keep the inbox short. Sam clears items (promote / reject / direct a human experiment / dismiss); cleared items move to `human/inbox_log.md` with his decision.

Links run both ways: every agent workspace folder states what it serves (`serves: h03`, a CLAIMS id, or `exploratory`); every human experiment has an "Agent evidence" section linking agent work that bears on it, marked preliminary.

## Promotion path

agent exploration -> `agent/proposals/NNN_*.md` + INBOX item -> Sam decides -> either Sam accepts it into `CLAIMS.md`, or Sam directs a clean human experiment (`human/experiments/hNN_*`) whose result he then claims.

## Human experiments (`human/experiments/hNN_name/`) -- tight

- `SPEC.md` first: Sam's question, the hypothesis, the exact design, metrics, and what would count as success/failure. Frozen before running; changes need Sam's sign-off and are logged at the bottom of SPEC.md.
- Code (`run.py`, modules), `results/` (data gitignored unless small), `figures/`.
- `WRITEUP.md` in this order: Title + Status (draft / awaiting Sam / signed off) -> TL;DR (2-4 bullets) -> Question -> Design (the exact math, parameters, metric definitions; one "Code & data" block with paths) -> Results (plain language; one bullet per figure: layout and what to look for) -> Additional details (only if load-bearing) -> Agent evidence (links, preliminary) -> **Conclusions (empty until Sam signs off)** -> Open questions.
- Every claimed number traces to a data file and the command that produced it.

## Agent workspace (`agent/workspace/YYYY-MM-DD_topic/`) -- structured, freer

Required: `README.md` with: question, who asked (Sam's request, quoted, or "agent-initiated"), `serves:`, status (running / done / abandoned / superseded-by), and a findings section where **every number cites its data file and the command**. Beyond that, organize as the work needs. Abandoned work stays, labeled. Do not create top-level folders elsewhere.

## Before proposing any PDE / physics mechanism

Check it against `human/REQUIREMENTS.md` on paper first: one hidden layer on the original inputs; no external solver whose answer is encoded; memory bounded, not brittle in $d$; general across equations (no per-equation tuning); honest accuracy labels. The previous program's biggest failures were mechanisms that quietly broke these (Fourier solve then encode; product gates; polynomial spectral solve described as "native"; tuned chained resumes reported as one policy). The import audit lists them.

## Numerical conventions

- fp64 everywhere (`torch.set_default_dtype(torch.float64)`); construction may use mpmath offline to produce fp64 coefficients.
- QUILL geometry: uniform centers, $\gamma=\lambda/h$, $\lambda\approx0.25$ (rule in paper 1 Appendix B), halo $R=\lceil\sqrt N\rceil$ per side **with** the finite-contour rational correction (paper 1 Appendix A; `09_26_sl.tex`). $N$ = interior centers. A plain $\sqrt N$ halo without the correction fails (~1e-4).
- Report errors with their exact meaning: relative field error vs a known reference on held-out points; raw PDE residual (units); scaled residual; encoding error; refinement difference; parameter error. "Near floor" always names the quantity.
- Inverse problems: never generate observations with the same method at a finer setting without saying so (inverse crime).

## Figures and writing

- Every experimental claim ships with a plot made before reporting: fixed meaningful axes, trajectories not lone points. Follow Sam's plot directions exactly.
- Legends go outside the axes, above (`ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=..., borderaxespad=0)` or one shared `fig.legend` across the top).
- Math in LaTeX (`$...$`, `$$...$$`, KaTeX-safe). Do not hard-wrap prose. No emojis.
- Number discipline: raw numbers only where they carry the point.

## Environment

- Python venv: `~/venv/quill-radon` (Python 3.12, `requirements.txt`; set `SCIKIT_LEARN_DATA=~/.cache/scikit-learn`). New repo-level tests go in `tests/` when paper-2 code exists.
- Imported code runs in place from `agent/imported/precisionMLPs/` (scripts write into that snapshot's `results/` via `parents[2]`). Run with `OMP_NUM_THREADS=1 MPLCONFIGDIR=$TMPDIR/mpl`.
- Large binaries (npz, pt, npy, gif, mp4) are gitignored; keep data inside this repo.

## Commits

Use the default git identity; never add an AI co-author. Commit when Sam asks.
