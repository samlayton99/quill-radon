# Failures, limits and incomplete results

This page makes the boundaries of the successes easy to find. It is based entirely on saved evidence; no reruns were performed for organization.

| Finding | Evidence / quantitative example | What it rules out |
|---|---|---|
| Corrected halo is essential | Naive sqrt(N) halo: 3D value error 3.96e-5; corrected: 6.40e-16 | Halo count alone is not the construction algorithm |
| Angular resolution remains independent of center resolution | H05 direction cliffs and A09 failed direction snapping | Adding centers cannot repair missing angular content |
| Small nonlinear correction residual does not imply successful full solve | Six-case forward stress suite includes Helmholtz and Allen–Cahn failures | A general declaration API is not a generally reliable solver |
| Residual normalization is useful but incomplete | Contrast-1000 field .00432→8.03e-11; raw PDE about 1.15e-7 | Scaled stopping cannot be reported as a raw machine floor |
| Tighter solves can spend the budget at inadequate resolution | Two normalized precision controls stopped at p24 with no field improvement | Lower tolerance alone does not establish a scaling route to arbitrary accuracy |
| Exact-IC algebra can be numerically unstable | Allen–Cahn field errors .506/.637; readout L1 about 4e11 | A formal constraint does not guarantee accurate ordinary tanh evaluation |
| Ordinary exported arithmetic can lose digits | Known-target p64 encoding about 2.2e-15 under stable evaluation, 9.91e-12 ordinary tanh | Ideal coordinates/stable evaluator cannot substitute for deployed-model validation |
| Difficult physical flows remain hard | Counterrotation viscosity .03: momentum RMS .00778, max .506; square cavity corner residuals remain large | Smooth manufactured NS and disk stirring do not settle general difficult fluids |
| Low-viscosity inverse Burgers remains unresolved | True viscosity .02 fitted about .0534; field error 16.8%; physics-only gates also fail | Sensors do not rescue a failing native forward; no valid reduced-inverse answer follows |
| 5D progress is below requested floor | Million-neuron control: field 2.73e-9, corner max 5.37e-7 | No generic high-dimensional machine-precision demonstration |
| Native inverse reference independence is limited | Wall-NS measurements from a finer same-method model | This is not yet independent-discretization or experimental inverse validation |
| Noisy NS and later physical flows are incomplete | Administrative interruption at user's organization request | Do not label unfinished runs either converged successes or numerical failures |
| General weak/shock/blowup machinery is absent | C09 supplied front family; C10 prescribed shrinking core; C04 finite-volume reference | These are not a discovered entropy solution or NS blowup solution |
| Three noisy-regression winners are absent | B01 validates one 1D QI/SVD family; G01 gives qualitative regularization evidence | No established noisy high-dimensional approximation ranking |

Sources: [constructor](../methods/analytical_construction.md), [forward controls](forward_and_controls.md), [NS/dimensions](navier_stokes_and_dimensions.md), [inverse problems](inverse_problems.md), [problem-class rankings](../rankings.md). The historical [catalogue](../catalogue/README.md) retains all failed method IDs, rather than silently replacing them with their successful descendants.

For equations with discontinuities, a smooth finite tanh network cannot equal a jump uniformly. For a true singularity, a finite bounded smooth representation cannot have finite uniform error on a domain including the singular point. A meaningful goal needs a weak/integral norm, a time interval before singularity, or an explicit excluded singular set. This is a question of what accuracy means, separate from optimization failure.
