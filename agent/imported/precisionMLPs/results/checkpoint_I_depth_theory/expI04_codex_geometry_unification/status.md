# Codex geometry investigation — final coordinator status

Date: 2026-09-30. Coordinator: root. All scoped experiments and audits completed.

## Outcomes

1. Actual trained two-hidden-layer tanh networks: learned test relative L2 6.72e−6/8.53e−6; second-layer transitions have a precise hyperplane-on-hidden-curve interpretation. A curve from scalar input is not evidence by itself of manifold discovery.
2. Crossing-only arclength uniformization improves refitted errors by 6.0x/2.2x; all-neuron uniformization is mixed. Random and local-width-preserving controls completed. Improvement is in noiseless approximation, not uniformly better conditioning.
3. Unlabeled learned 24D→3D chart plus 2048-feature ridge head works for three targets and two seeds; independent test errors 1.51e−4–1.48e−3. True-coordinate control remains much better. PCA folding confirmed by orientation reversal and a numerical input collision with different target values.
4. Analytic 3D Radon readout construction predicts entire coefficient arrays without any LS; independently evaluated Gaussian-family errors near 1e−13 at 205824neurons. Cost and analytic-target assumptions reported.
5. Controlled deep ReLU chart predicts independent solved readouts to 0.0584% coefficient error. Simple pointwise predictions fail on actual trained tanh and ReLU models. Exact directional gauge demonstrates nonunique raw readout allocation. No universal arbitrary-Adam decoder is claimed.

## Owned outputs

- Integrated report: report.md
- Theory and proofs: theory.md, theory/
- Ordinary learned-model interventions: deep/report.md and figures/checkpoints/metrics
- Manifold transfer and controls: manifold/report.md, comparison.png, metrics and checkpoints
- Radon construction and coefficient ambiguity: radon/report.md and figures/metrics
- Source/reproduction: experiments/expI04_codex_geometry_unification/README.md

## Validation

Workers inspected plots; coordinator inspected main figures and corrected a PCA-dimension annotation. Independent audit found no label leakage in the manifold experiment and identified required favorable-manifold / matched-head-budget qualifications. ReLU final model checked on 4096 genuinely held-out Sobol points after detecting shared points in an earlier dense evaluation grid; results unchanged. Analytic identities, model synthesis, finite result tables and Python compilation checked. No data or existing user edits were reverted; all work is additive. No deployment, commit, or external publication.

## Scope

The mathematical decomposition is established projection/conditional-expectation theory. The new work supplies controlled empirical predictions, interventions and counterexamples. The broad hypothesis is narrowed by evidence; a general theory proving deep-learning optimization and generalization across architectures, losses and stop-gradient systems has not been established. This scope is explicit in the report and user-facing findings.
