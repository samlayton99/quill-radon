# Agent side

A structured scratchpad and staging ground. Everything here is **preliminary evidence**, not conclusions; conclusions live only in `../CLAIMS.md`, and only Sam writes them. Rules are in `../CLAUDE.md`.

| Folder | What goes there |
|---|---|
| `workspace/YYYY-MM-DD_topic/` | Agent work: explorations, prototypes, audits, side studies. Each has a `README.md` (question, who asked, `serves:`, status, findings with data path + command per number). |
| `proposals/PNNN_*.md` | Candidate claims for Sam: proposed statement, evidence links, caveats, what would make it a claim. Each has a matching `human/INBOX.md` item. Sam's decision is recorded at the top (accepted -> CLAIMS id / rejected / converted to human experiment). |
| `imported/precisionMLPs/` | Verbatim snapshot of all Radon-relevant material from the precisionMLPs repo at commit `ca6dc9d` (Oct 2, 2026). Do not edit; see its README for the map. |

## Index

**Proposals**
- `P001` training-free, fit-free Radon-QUILL construction for known functions.
- `P002` direct-construction error split; angular error tracks sphere-rule exactness degree; where the curse of dimensionality lives.
- `P003` route 2 is polynomial spectral collocation compiled into a one-hidden-layer MLP (framing decision).
- `P004` the global least-squares memory wall (motivating negative).

**Workspace**
- `2026-10-02_import_audit/`: six read-only audits of all prior Radon work (theory, known-function experiments, pre-Radon physics, early expF19, route 2, Codex session history), plus Sam's messages and session extracts.
