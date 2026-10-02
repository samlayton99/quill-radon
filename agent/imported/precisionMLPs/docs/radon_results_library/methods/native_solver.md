# Streamed native Radon residual solver

**Use when the solution is unknown and the available information is the equation, domain, boundary/initial conditions and optional measurements.** This is the current main forward-PDE platform, not a proved universal solver. Historical E09–E11 are its streamed predecessors; E02/E03 are cached versions. [Rankings](../rankings.md).

## Actual procedure

1. Declare the unknown fields, PDE residuals, domain, boundary/initial constraints and any physical parameters. A manufactured reference may supply prescribed forcing and boundary values; interior target values do not enter the blind solve.
2. Choose Radon directions and corrected scalar tanh features. Represent readouts using independent analytical coordinates instead of treating every redundant neuron coefficient as an unrelated unknown. Disk and box coordinate maps differ.
3. Start from zero readouts or an explicitly recorded earlier native PDE iterate. Evaluate the residual of the **actual neural function** at collocation points.
4. Find numerical coordinate corrections using streamed Gauss–Newton/Krylov products and bounded preconditioner blocks. Ideal analytical coordinate functions can approximate the correction Jacobian. Actual neural residuals and acceptance checks still decide whether a candidate is accepted.
5. Increase coordinate degree, angular resolution or scalar resolution when the driver permits; record stalls, budgets and ordinary-arithmetic discrepancies separately.
6. Export and check an ordinary `Linear/Tanh/Linear` model at held-out points. Compare richer resolution where available. A small training residual alone is insufficient evidence.

Polynomial functions appear in coordinate algebra and inexpensive correction operators. They are **not an extra evaluator inside the exported model**. There is no supplied external PDE solution that is subsequently encoded as the answer.

## Strongest saved evidence, with its scope

| Case | Main result | Interpretation |
|---|---|---|
| Smooth manufactured steady 3D NS | Velocity relative L2 7.93e-15; pressure 1.56e-14; 84,099 neurons | Near-floor smooth forced benchmark |
| Smooth manufactured 3D NS plus time | Velocity 1.84e-15; pressure 6.37e-14; 751,689 neurons | Four-input success, but corner errors exceed 1e-13 and solve ended on budget |
| Zero-forcing wall-driven disk NS, viscosity .1/.03 | Momentum RMS 3.93e-15 / 6.71e-15 with stable richer-network checks | Real boundary-driven setup; true field unknown, sampled residuals are not true-error certificates |
| Parameterized diffusion family | Relative field 1.63e-15; raw PDE RMS 3.31e-14 | One network takes position and parameter inputs |
| Full-rank five-input semilinear elliptic control | Relative field 2.73e-9 with 1,003,833 neurons | Feasibility/scaling evidence, not a floor or hard-5D victory |

Failures are part of this method's evidence: stiff Allen–Cahn, low-viscosity inverse Burgers, cornered cavities and harder counterrotating flows remain unresolved. Operator scaling improves one contrast-1000 diffusion case from 0.00432 to 8.03e-11 field error, but its raw PDE RMS remains about 1.15e-7. This is not a universal precision repair.

## Useful variants, not separate inventions

- **All-actual-Jacobian correction:** useful fidelity baseline and safeguard; can be expensive or stall. It is a variant of this solver, not a separate top-three family.
- **Ideal correction / ideal preconditioning:** cheaper numerical updates, with the actual neural model retained as the acceptance criterion.
- **Cached disk coordinates:** faster for tractable small cases, but explicitly retain point-by-coordinate data. Useful baseline when memory permits.
- **Factored box coordinates:** avoid storing the full direction-by-degree-by-coordinate tensor. Their arithmetic cost remains substantial.
- **Residual scaling, initial-plane pairing, recurrences, tensor contractions and stopping-policy controls:** see [component register](../components.md). Some help; some fail.

## Resources and current limit

Streaming removes the dense global feature/Jacobian allocation. It does not guarantee construction costs equal inference costs. Coordinate conversion, collocation passes, preconditioner setup and repeated Krylov/outer iterations all count. Tensor angular rules retain their dimension dependence. See [costs](../requirements_and_costs.md) for explicit symbols and terms.

The strongest 3D and space-time timings are final-stage continuations, not cold total solve times. Historical experiments changed implementations and budgets; this library does not infer a matched speedup from unrelated rows.

## Documentation and implementation

- [Forward evidence](../evidence/forward_and_controls.md), [NS and dimensional evidence](../evidence/navier_stokes_and_dimensions.md), [failures](../evidence/failures_and_limits.md).
- [Native driver](../../../experiments/expF19_radon_direct_pde/solver/route2.py), [general residual solve](../../../experiments/expF19_radon_direct_pde/solver/general_residual.py), [streamed operator](../../../experiments/expF19_radon_direct_pde/solver/streamed_residual.py).
- [Disk profiles](../../../experiments/expF19_radon_direct_pde/route2_disk_profiles.py), [box profiles](../../../experiments/expF19_radon_direct_pde/route2_box_profiles.py).
- [Original hardening record](../../quill_streamed_hardening_status.md), [experiment narrative](../../../results/checkpoint_F_applications/expF19_radon_direct_pde/expF19_results.md).
