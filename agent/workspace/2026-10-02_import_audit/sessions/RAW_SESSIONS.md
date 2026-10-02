# Raw Codex session logs (not copied: size)

They remain on the Mac mini under `~/.codex/sessions/2026/`. JSONL; stream with Python, never cat whole.

**Essential**
- `09/30/rollout-2026-09-30T20-19-36-01a0f571-40a6-7e13-a1a8-7b582519ec45_01a0f579-84f0-7742-b4b8-37b53ad20b21.jsonl` (123 MB): the Radon -> PINN main loop, Oct 1 03:19 - Oct 2 20:20 UTC. Extract: `main_dialog.md`.
- `09/29/rollout-2026-09-29T12-53-07-01a0eeba-6172-7f21-bda0-34c1ced4ce87.jsonl` (16 MB): Radon spoke interpretability; filter derivation; degree-of-freedom counts. Extract: `spoke_dialog.md`.
- `09/30/rollout-2026-09-30T14-24-23-01a0ded8-c952-7301-87c2-2541403061d0_01a0f434-4d1f-7893-8989-4ba72f2adde6.jsonl` (31 MB): theory session where the training-free construction originated; depth/composition. Extract: `theory_dialog.md`.
- `09/30/rollout-2026-09-30T19-29-07-01a0f54b-4a74-77d0-89d4-74a2e2ef9ae9.jsonl`: subagent `radon_interpretation_test`, which built the analytic 3-D construction.

**Secondary (large worker transcripts):** `09/30/*01a0f590-79cc*`, `09/30/*01a0f5a1-6d06*`, `10/01/*01a0fa0f-17a7*`, `*01a0fa0f-52cc*`, `*01a0fa0f-7b28*`, `10/01/*01a0fa4f*`, `*01a0fa50*`, `10/02/*01a0fb6f*`. Full list with tasks: `../audit_codex_sessions.md` §6 and `sessions_index.tsv`.
