# Learned 3D chart and Radon-style ridge head on a noiseless 24D manifold

Codex experiment, 2026-09-30. Question 3. Two independent encoder initializations; one shared fixed train/validation/test split.

**Yes, the hybrid works on this controlled manifold, but compression quality controls its precision.** The learned encoder supports three separate target functions, reaching relative L2 errors from 1.51e−4 to 1.48e−3. Using the true coordinates on the identical head budget reaches 3.12e−7 to 1.70e−5. PCA compression folds the manifold and fails badly. None of these results establishes a universal manifold learner.

![Held-out comparisons](comparison.png)

## Protocol and scope

The intrinsic variable u is uniform on [−1,1]^3. The 24D embedding concatenates u, quadratic and product terms, sines/cosines, and mixed cubic/trigonometric terms, then applies a fixed orthogonal mixing. It is an injective graph embedding, and a favorable global linear projection back to u exists. Thus this tests curved-manifold chart recovery, not an arbitrary topology or an embedding with no linear inverse chart. The encoder never receives u or any target labels.

There are 4,096 input-training points, 1,024 validation points, and 4,096 test points, generated with independent scrambled Sobol sequences. The training-input PCA projection initializes a residual nonlinear autoencoder: 24→64→64→3 encoder and 3→64→64→24 decoder, tanh hidden layers, float64 CPU. Each seed uses 6,000 Adam steps followed by 40 blocks of up to 100 L-BFGS iterations. Validation reconstruction selects the checkpoint. See config_s0.json and config_s1.json for settings.

Each representation receives 128 directions and 16 evenly spaced tanh centers per direction, with a 25% collar around the training projection range. Latent 3D directions use the existing spherical-Fibonacci rule; ambient 24D directions use the existing seeded Gaussian-direction rule. The bandwidth gamma*h is selected from {.15,.25,.4}, and the SVD cutoff from {1e−8,1e−10,1e−12,1e−14}, using validation error separately for each downstream target. The head is a frozen finite Radon-style ridge dictionary plus a fitted output bias. It is a least-squares head, not the analytic no-solve Radon construction tested separately.

The controls match 2,048 ridge features plus a bias. They do not match total model parameters or end-to-end training cost: learned compression has an additional encoder and its training. The ambient baseline is a fixed ridge bank, not an optimized deep 24D network. The encoder is shared across all three targets, but the selected head bandwidth may differ by target. All reported function errors are unsquared test relative L2.

## Results

| Coordinates | Product wave | Composition | Gaussian bump |
|---|---:|---:|---:|
| True 3D coordinates | 6.31423e-07 | 1.69521e-05 | 3.11881e-07 |
| PCA 3D | 0.415443 | 0.106976 | 0.253889 |
| Learned 3D, seed 0 | 0.00140468 | 0.000189151 | 0.000511202 |
| Learned 3D, seed 1 | 0.0014758 | 0.00015086 | 0.000431324 |
| Ambient 24D | 0.00407614 | 0.000239553 | 0.00105643 |

The target functions are sin(pi*u1)*cos(pi*u2), exp(.5*sin(pi*u1)+.3*u2*u3), and exp(−2||u−(.2,−.15,.1)||²). All targets vary in, or are evaluated across, the same 3D domain. The product wave happens not to depend on u3; the other targets do. The encoder sees none of these labels during representation learning.

The two learned encoders reconstruct held-out ambient inputs to relative errors 0.00450 and 0.00537. These are reconstruction errors, not proven irreducible downstream error floors: a different decoder or richer downstream basis can improve despite the same encoder. The learned latent coordinate is not equal to u. Its sampled intrinsic Jacobian has minimum singular values 0.885 and 0.880 and median condition numbers 1.304 and 1.276, across 256 unseen points. This shows sampled local regularity; it does not prove global injectivity.

## A specific explanation for PCA failure

PCA retains 64.99% of input variance, but its 3D coordinate map has Jacobian determinants from −1.655 to 6.540. A smooth injective map between connected open subsets of equal dimension cannot reverse orientation between regular points. Thus this projection cannot be a globally injective chart. Low dimension or high retained variance does not establish preservation of the information needed by the target.

The independent audit also located two numerical preimages with almost identical PCA codes. Re-evaluation gives:

```json
{
  "intrinsic_points": [
    [
      -0.8656066786497831,
      0.003151705488562584,
      0.7727957330644131
    ],
    [
      -0.6304931540941848,
      0.0031728427694085576,
      0.6962785444136483
    ]
  ],
  "pca_codes": [
    [
      -0.9143289169877036,
      -0.8355447054417768,
      -0.1462466489777025
    ],
    [
      -0.9143289169876271,
      -0.8355447054417663,
      -0.1462466489776386
    ]
  ],
  "pca_distance": 1.0024238441388211e-13,
  "intrinsic_distance": 0.24725139035960403,
  "target_values": [
    [
      -0.4097564483121703,
      0.8153338807130408,
      0.03982524489951311
    ],
    [
      -0.9170926660150939,
      0.6326067113235273,
      0.11795432537110297
    ]
  ],
  "method": "Independent input-only root search; supplied pair independently re-evaluated here. Numerical collision, not an interval-certified root."
}
```

The product-wave values differ by about 0.507 despite a PCA-code difference near 1e−13. This is direct numerical evidence of task-relevant folding, in addition to the orientation argument. An isolated near-collision by itself is not a numerical lower bound on population error.

## What this says about the theory

A learned chart plus a classical approximation family is a workable hybrid. The three stages must be separated: preserve target-relevant distinctions; express the target simply in the retained coordinates; then resolve it with enough directions and centers. A small latent dimension does not establish any of the three. The oracle/learned gap is a measured precision gap, not evidence that Radon approximation fails or that the encoder has necessarily discarded all of that information.

An early 24-directions×24-centers pilot was angular-resolution limited even on the true coordinates. The final 128×16 allocation was chosen after this pilot, and all four representations receive the same final allocation. Results are exploratory, not a preregistered confirmation. Per-target bandwidth/cutoff selection is validation-only; test values never select those settings.

Reproduction: run manifold.py train --steps 6000 --seed 0 (and seed 1), then manifold.py evaluate --large --seed 0 (and seed 1), then manifold_summary.py. Explicit shell commands are in the root experiment README. Saved checkpoints, coordinates, full sweeps, histories, and metrics are beside this report. Independent read-only audit found no target leakage or error-metric bug.
