# P004: The global least-squares memory wall (motivating negative)

Status: **proposed** (agent-drafted Oct 2, 2026; awaiting Sam). Serves: paper 2 motivation. INBOX item 5.

## Proposed claim
Before the Radon construction, every precision PDE result in the program used a frozen tanh ridge dictionary plus one dense global least-squares (Gauss-Newton) solve over all neurons. That solve, not accuracy, stopped the program at about three inputs on a 16 GB machine:
- $d=2$, one field: $W=9216$ columns, ~27.6k x 9.2k, ~2 GB, ~3 min (expF13).
- $d=3$ (2-D+t drifting Taylor-Green NS, expF18): 4096 units/field x 3 fields = 12.3k unknowns, ~46k rows, 7 GB peak after an augmented-QR rewrite (naive route swapped at 12 GB); velocity 1.07e-11 while the oracle ceiling was still falling (1.1e-12).
- $d=4$ (expF12): reachable only by abandoning one hidden layer for a Kronecker tensor basis, capped near $\sqrt{\varepsilon}$ (1.5e-6).

## Evidence
`agent/workspace/2026-10-02_import_audit/audit_prior_physics.md` §1; source writeups under `agent/imported/precisionMLPs/results/checkpoint_F_applications/{expF13_bwler_suite,expF18_ns_spacetime,expF12_tensor_ns3d}/` and `agent/imported/precisionMLPs/experiments/expF01_linear_de_zoo/PINN_FEASIBILITY.md`.

## Caveats
- expF18 numbers are checked against `status.md`/`cells*.json`; expF12 json files are absent on this machine (md and png only).
- The global solve also has an accuracy pathology independent of memory: the collocation minimizer is displaced from the best fit (reaction, NS) by up to two orders.
