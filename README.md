# quill-radon

Paper 2 of the QUILL program: high-precision single-hidden-layer MLPs for **multivariate inputs and outputs** via the Radon transform, made practical, with physics (PINN-style forward and inverse PDEs) as the main application.

Paper 1 (`papers/paper1_quill/`, ICLR 2027 submission) builds one-hidden-layer tanh networks for 1-D analytic functions that converge geometrically in width to the fp64 floor. Paper 2 writes a $d$-dimensional target as a superposition of ridge profiles, $f(x)\approx\sum_m w_m\,q_m(v_m\cdot x)$, with each profile a paper-1 QUILL, so the whole thing is still one hidden layer.

## Start here

| If you want | Read |
|---|---|
| What is settled | `CLAIMS.md` (paper 1 only, so far) |
| What is open and needs Sam | `human/INBOX.md` |
| Sam's binding rules for the program | `human/REQUIREMENTS.md` |
| How this repo works (human vs agent side) | `CLAUDE.md` (= `AGENTS.md`) |
| The three versions of paper 1 | `papers/README.md` |
| What the earlier Radon work actually established, and what was overclaimed | `agent/workspace/2026-10-02_import_audit/README.md` |
| Where every imported file is, tagged by usefulness | `agent/imported/README.md` |
| Candidate claims awaiting review | `agent/proposals/` |

## Layout

```
CLAIMS.md            the only lasting conclusions; Sam-owned
CLAUDE.md, AGENTS.md house contract (identical)
papers/              paper 1 (v1 workshop, v2 Section-3 rewrite, v3 ICLR submission, appendix LaTeX), related notes, paper 2 drafts
human/               Sam's side: INBOX.md, REQUIREMENTS.md, experiments/hNN_*, notes/
agent/               agent side: workspace/, proposals/, imported/precisionMLPs/ (verbatim snapshot)
```

**Two sides.** The human side holds what Sam directs and claims; it stays tight and clean. The agent side is a structured scratchpad where agents explore and stage evidence; nothing there is a conclusion. Agents work on both sides, and they surface anything Sam needs to see in `human/INBOX.md`. Promotion: agent evidence -> proposal + INBOX item -> Sam accepts into `CLAIMS.md` or directs a human experiment.

## Environment

`~/venv/quill-radon` (Python 3.12; `requirements.txt`). All 574 imported tests pass in place (Oct 2, 2026). Imported code runs in place:

```bash
cd agent/imported/precisionMLPs
OMP_NUM_THREADS=1 MPLCONFIGDIR=$TMPDIR/mpl ~/venv/quill-radon/bin/python -m pytest -q tests/test_route2_driver.py
```

Origin: material imported from `samlayton99/precision-mlps` at commit `ca6dc9d` (Oct 2, 2026); that repo is unchanged.
