# expF15 — matched-parameter tensor audit on BWLer and dysts

## Result

Low-rank tensor structure is a major parameter-efficiency win only when the
solution is actually low separation-rank.  It decisively wins on BWLer's two
convection cases and wave, helps on reaction and Burgers at looser accuracy, and
does not beat the Radon result on the perforated Poisson geometry.  It does not
compress the original five dysts trajectories: dropping one state-space rank
saves roughly a third of the readout but causes 10%–55% error (and 35% on
rank-3 Lorenz96).  MacArthur, added below, is the important higher-dimensional
exception.

These are deliberately labelled ceilings/diagnostics:

- BWLer uses an oracle rank-r tensor-Chebyshev fit to the reference solution,
  with exactly `2*r*(degree+1)` stored coefficients.  It is not a PDE solve.
- dysts first performs the existing shared-QI ODE solve and then takes the
  optimal function-space rank truncation.  Its temporal factors remain inside
  the QI span.  It is not a rank-constrained ODE solve.

## BWLer suite

The tensor column is the best tested cell at or below the Radon parameter budget.
The Radon values are the existing expF13 results.

| problem | Radon rel-L2 | Radon params | tensor rel-L2 | tensor params | rank, degree |
|---|---:|---:|---:|---:|---:|
| convection c40 | 6.2e-12 | 9,216 | 2.55e-15 | 260 | 2, 64 |
| convection c80 | 1.0e-9 | 9,216 | 5.03e-15 | 388 | 2, 96 |
| reaction | 4.6e-7 | 6,400 | 8.44e-16 | 6,208 | 32, 96 |
| wave | 1.6e-13 | 9,216 | 1.05e-15 | 260 | 2, 64 |
| Burgers | 2.9e-1 | 4,096 | 4.37e-2 | 2,064 | 8, 128 |
| Poisson-CG | 9.4e-5 | 4,096 | 7.10e-3 | 272 | 8, 16 |
| Poisson manufactured | 1.1e-6 | 4,096 | 8.66e-6 | 528 | 8, 32 |

The headline numbers need different interpretations.  Convection and wave are
true rank-2 identities, so their 24–47x coefficient reduction is structural and
matches the constructive Toeplitz-QI result in `mlp-interpolants-C`.  Reaction
needs rank 32 to reach the floor, so its best-accuracy result is not a strong
low-rank win; it does, however, beat the Radon error with 1,568 coefficients at
9.28e-9.  Burgers beats the failed Radon solve at half the coefficients, but this
is only an oracle representation result and therefore diagnoses the solve rather
than replacing it.  The perforated domain breaks ordinary rectangular tensor
structure; masked/scattered ALS does not recover the Radon accuracy.

## dysts suite

At N=384 the shared QI dictionary has `p=697` features.  Full readout storage is
`d*p`; rank-r factor storage is `r*(p+d)`.

| system | full QI params | certified full rel-L2 | reduced rank | reduced params | reduced rel-L2 |
|---|---:|---:|---:|---:|---:|
| Lorenz | 2,091 | 1.1e-13 | 2/3 | 1,400 | 1.02e-1 |
| Rössler | 2,091 | 4.3e-13 | 2/3 | 1,400 | 3.16e-1 |
| Thomas | 2,091 | 1.4e-12 | 2/3 | 1,400 | 2.01e-1 |
| Halvorsen | 2,091 | 4.9e-13 | 2/3 | 1,400 | 5.52e-1 |
| Lorenz96 | 2,788 | 1.3e-13 | 3/4 | 2,103 | 3.54e-1 |

The conclusion is negative but clean: sharing the time dictionary is already the
right parameterization, while low-rank coupling across the few state channels is
not.  Each chaotic coordinate carries an independent direction over a three-
Lyapunov-time window.  Full-rank factorization even stores slightly more numbers
than the original dense readout, so there is no hidden tensor saving on these
low-dimensional ODEs.

### Additional systems

These systems use the same N=384 shared-QI dictionary, fitted directly to a
tight DOP853 trajectory, followed by the optimal function-space rank
truncation.  This is an oracle interpolation/compression diagnostic because the
current dense Gauss-Newton implementation is especially impractical at
MacArthur's dimension 10.

| system | d | full QI floor / params | best lower rank / params | reduced rel-L2 | intrinsic rank fact |
|---|---:|---:|---:|---:|---:|
| InteriorSquirmer | 3 | 3.89e-6 / 2,091 | 2 / 1,400 | 1.30e-1 | full rank required |
| DoublePendulum | 4 | 1.04e-14 / 2,788 | 3 / 2,103 | 3.53e-4 | rank 4 required for precision |
| MacArthur | 10 | 4.67e-8 / 6,970 | 6 / 4,242 | 4.67e-8 | rank 6: 2.03e-12; rank 7: fp64 floor |

MacArthur is the first meaningful output-tensor win in this suite.  Rank 6
preserves the entire currently resolved QI approximation with 39% fewer stored
coefficients; rank 3 uses 2,121 coefficients (70% fewer) at 1.44e-3 relative
error.  Increasing the time resolution confirms that the N=384 basis, not
output rank, limits the full fit: at N=512 the MacArthur floor/rank-6 error is
2.14e-8 with 5,562 rather than 9,270 coefficients; at N=768 it is 7.74e-9 with
8,394 rather than 13,890.  Rank 6 therefore preserves the floor and the 39%
storage saving as resolution grows.
DoublePendulum has a useful low-accuracy compression curve (rank 1 is 1.09e-2),
but precision requires all four output directions.  InteriorSquirmer is both
full-rank and under-resolved because of its sharp switching protocol: its full
floor improves from 3.89e-6 (N=384) to 7.45e-7 (N=512) and 2.90e-8 (N=768),
while its rank-2 error remains fixed at 0.130.

## Artifacts

- `data.json`: every degree/rank cell and the accounting metadata.
- `tensor_suite.png`: Pareto curves and the dysts rank-compression cliff.
- `extra_dysts_data.json` and `extra_dysts_tensor_compression.png`: the three
  additional systems and their full rank curves.
- `run.py`: reproducible experiment.  The BWLer reference files are local; the
  dysts run uses the existing `jaxpi` environment because the repository's
  lightweight `.venv` does not contain the `dysts` metadata package.

## Direct tensor-QI versus Radon comparison

The invariant comparison changes the conclusion from “tensor is generally
better” to a sharper one: **tensor-QI wins on true low-separation-rank fields;
dense Radon wins on the tested chaotic trajectories.**

### Constructive BWLer wins

These three solutions have exact rank 2.  Their factors are derived from the
PDE characteristics/eigenmodes and built by the original analytic
Toeplitz-cardinal derivative convolution—no least-squares fit and no sampled
2-D solution.

| problem | constructive tensor-QI rel-L2 / coefficients | expF13 Radon rel-L2 / width | error gain |
|---|---:|---:|---:|
| convection c40 | 2.80e-13 / 1,216 | 6.2e-12 / 9,216 | 22x |
| convection c80 | 2.83e-12 / 1,344 | 1.0e-9 / 9,216 | 354x |
| wave | 3.60e-14 / 1,216 | 1.6e-13 / 9,216 | 4.5x |

Thus QI is simultaneously more accurate and roughly 7x smaller by the stated
coefficient/width accounting.  Reaction and Burgers have encouraging oracle
tensor ceilings but are not yet constructive-QI PDE solves.  The two perforated
Poisson cases remain Radon wins because the rectangular tensor factorization
does not respect the domain geometry.

### Analytic rank-2 control

For

`sin(x+y)/sqrt(1+x^2)`,

the exact rank-2 factorization plus original Toeplitz-QI reaches **2.41e-15**
with 1,156 factor-readout coefficients.  An oracle-tuned 2-D Radon value-lstsq
fit at exactly 1,156 readout coefficients reaches **7.55e-14**.  This is a
**31x matched-parameter QI win** at the fp64 floor.

### Dysts: Radon wins

For the original five systems, compare the largest compressed tensor rank
(`d-1`) against dense 1-D Radon at the same coefficient budget:

| system | budget | tensor rel-L2 | Radon rel-L2 |
|---|---:|---:|---:|
| Lorenz | 1,400 | 1.02e-1 | 3.30e-14 |
| Rössler | 1,400 | 3.16e-1 | 7.41e-12 |
| Thomas | 1,400 | 2.01e-1 | 1.31e-13 |
| Halvorsen | 1,400 | 5.52e-1 | 5.13e-14 |
| Lorenz96 | 2,103 | 3.54e-1 | 3.77e-14 |

The additional systems show the same ordering at every tested budget.
InteriorSquirmer rank 2 gives 1.30e-1 versus Radon's 1.23e-6 at 1,400
coefficients; DoublePendulum rank 3 gives 3.53e-4 versus 2.92e-15 at 2,103;
MacArthur rank 6 gives 4.67e-8 versus 3.49e-8 at 4,242.  Even MacArthur's real
low-rank structure does not overcome the temporal resolution that dense Radon
buys at matched storage.

The direct explicit-QI and Hermite-lstsq probes also failed to reverse these
Dysts comparisons.  The credible next QI attempt is adaptive/windowed or
multiband time geometry; further output-rank compression is the wrong axis.
