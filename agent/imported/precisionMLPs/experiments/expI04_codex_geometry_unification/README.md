# Codex geometry investigation

This additive experiment answers five questions: learned second-layer geometry, controlled uniform placement, unsupervised low-dimensional compression followed by a ridge head, Radon coefficient meaning, and target-derived readout interpretation.

All paths are relative to the repository root. Existing experiment implementations and results were left intact. CPU float64, one BLAS/PyTorch thread per run. These are small diagnostic dense solves, not a proposed scalable optimizer.

## Results

`results/checkpoint_I_depth_theory/expI04_codex_geometry_unification/` contains:

- `report.md`: integrated outcomes and measured boundaries.
- `theory.md`: objects, exact decompositions, coordinate derivatives, proofs, and controlled predictions.
- `deep/`: ordinary trained tanh models, placement interventions and controls.
- `manifold/`: two unsupervised encoders, three downstream targets, compression controls, and PCA collision audit.
- `radon/`: independent 3D analytic target-to-readout construction and directional ambiguity.
- `theory/`: controlled ReLU chart, folding obstruction and adaptive knot-density checks.

## Reproduce

Use the existing `.venv/bin/python`. Before each command, set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`.

```sh
.venv/bin/python experiments/expI04_codex_geometry_unification/deep.py --reproduce-final
.venv/bin/python experiments/expI04_codex_geometry_unification/deep_figures.py
.venv/bin/python experiments/expI04_codex_geometry_unification/manifold.py train --steps 6000 --seed 0
.venv/bin/python experiments/expI04_codex_geometry_unification/manifold.py train --steps 6000 --seed 1
.venv/bin/python experiments/expI04_codex_geometry_unification/manifold.py evaluate --large --seed 0
.venv/bin/python experiments/expI04_codex_geometry_unification/manifold.py evaluate --large --seed 1
.venv/bin/python experiments/expI04_codex_geometry_unification/manifold_summary.py
.venv/bin/python experiments/expI04_codex_geometry_unification/radon.py
.venv/bin/python experiments/expI04_codex_geometry_unification/theory_relu_checks.py
.venv/bin/python experiments/expI04_codex_geometry_unification/theory_learned_relu.py --target quadratic --width 256 --steps 12000
```

The deep run records its exploratory polishing stages; see its report for the exact schedule and helper commands. Outputs preserve full validation sweeps, final independent test errors, input grids, learned checkpoints, and standalone figures. Reports distinguish designed geometries from actually learned ones and readout interpolation from least squares. No derivative reconstructed by summing a trained network is presented as an independent coefficient prediction.
