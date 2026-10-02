# Organization status and provenance

Coordinator: root. Evidence cut and organization date: **October 1, 2026**.

## October 2 continuation

The user explicitly resumed numerical work on the representation-to-solve gap and requested a per-run scratchpad. The original organization record below remains historical. New work is tracked in [current findings/scratchpad.md](current_findings/scratchpad.md) and consolidated in [coefficient-access evidence](evidence/coefficient_access_2026_10_02.md). New outputs are in `route2_gap_diagnosis/`; they are outside the October 1 file-inventory snapshot. Existing results and production solver modules are preserved.

The first batch is complete: 25 root runs and seven focused controls. Both adaptive native diffusion runs reach ordinary relative field error below 1e-15; contrast 1000 passes sampled driver checks, whereas contrast 10 ends at its time budget with normalized held-out PDE RMS 1.66e-13 above its 1e-13 threshold. This is a paired linear verification result, not a new general-solver claim. All numerical runs in this batch have finished.

## Scope and disposition

The user requested organizing existing Radon results before testing anything new, then clarified that the top three must be **per problem class**, including known-function approximation, noisy observations and clean/noisy inverse problems. The current requirements and PINN aims are consolidated here. There is no universal top-three ranking.

- Original 60 method IDs are preserved, with present-utility classifications and architecture qualifications added.
- Twelve later components/extensions are indexed separately; they are not twelve independent solver inventions.
- All 17 expF19 result subfolders are mapped by utility; the file inventory records 1,491 existing files at this cut.
- Relevant preceding construction, placement and noise studies remain linked through the catalogue and rankings. The inventory does not claim to cover every unrelated result in the repository.
- Historical numerical scores are retained as historical, not reinterpreted under the newer one-hidden-layer requirement.
- No raw model/data/figure files were moved, deleted or regenerated. Original experiment paths remain valid.
- No new fits, sweeps, scientific evaluations or numerical tests were run for this organization. Documentation checks read files and validate links/schema only.
- No commits or external publication were made.

Documentation verification passed: **329 local links resolve**, all 60 original CSV rows/columns are preserved exactly, and all 1,491 inventory paths exist. [Check record](catalogue/documentation_check.json). This is a file/link check, not a numerical validation of the experiments.

## Paused work

| Campaign | Preserved state | Correct classification |
|---|---|---|
| Wall-NS reduced inverse with sigma=.001 | Twelve completed forwards; best saved feasible candidate; current forward stopped | User-interrupted, not converged and not a numerical failure |
| Physical recurrence campaign | Two completed counterrotation budget exits; stirring viscosity .01 checkpoint | Third case user-interrupted; viscosity .003 case unstarted |

Exact administrative records: [noisy inverse interruption](../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/inverse/wall_ns_reduced_inverse/noise_0.001/interruption_status.json), [physical campaign interruption](../../results/checkpoint_F_applications/expF19_radon_direct_pde/route2_battletest/flows_recurrence/interruption_status.json). Previously completed failures retain their actual status.

## Evidence handling

The library is based on this workspace's saved results and source records. Local agent-context checks passed, but cross-machine context synchronization failed transport; this is not claimed to be a newly synchronized laptop inventory. Historical timings shown here came from the existing M4 Mac mini workspace, not laptop benchmarks.

Collaborative archival work was scoped to three pages: forward/control evidence, NS/dimensional evidence, and inverse evidence. Root owns the classification, rankings, requirements, PINN aims, indexing and link checks. The latest user constraints override older documents' architecture allowances. Other agents' and unrelated user work were preserved.

Raw arrays and model archives may be gitignored; linked local availability does not imply those artifacts have been committed or transferred to another machine. The inventory is a filesystem catalogue, not a backup.

## Entry points

[Library](README.md) · [Problem-class rankings](rankings.md) · [PINN aims](pinn_aims.md) · [Historical methods](catalogue/README.md) · [Result inventory](catalogue/result_inventory.csv).
