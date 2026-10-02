# F initialization and the practical value of a compact QI block

Investigation date: September 6, 2026. Source context revision: `d7eb04d1bf15a82975546d5302bae8caf6be7fa8`; root's sync transport failed, local context check passed. Existing experiment code/results were preserved. New code is confined to `experiments/expI03_investigation/f_init/`.

**There is a real depth benefit, and a compact fixed-bank block retains it. The roughly tenfold F04 initialization gain is a low-dimensional synthetic result, not the size of the general real-data gain.** The strongest new result is a small ordinary-Adam model with fixed midpoint banks, enough latent channels, and simple layerwise data calibration. It improves over a dense MLP at approximately the same parameter count on airfoil and parkinsons and is competitive on bike sharing. No head solve, whitening, geometry updates, or band penalty is needed for this result.

## 1. What F04 actually initialized

`experiments/expF04_qi_init_real_data/model.py:84`, `qi_ridge_init_layer_`, partitions total hidden width W into bundles of P=floor(sqrt(W)) neurons. Each bundle shares a random unit direction u. From training-input projections t=x·u it estimates A=quantile(|t|,.999), then sets gamma=.25 P/(2A), row weights gamma u, and biases -gamma c. The winning `scaled_psqrt` version samples c from observed projections. The two-layer `qi2` arm repeats this over the actual first-layer activations. The final readout keeps PyTorch's default initialization.

Two distinctions matter:

- The scale depends on **centers per direction**, P, rather than total width W. A flat W-direction initializer using gamma proportional to W makes every isolated feature sharp without providing a dense collection along any particular direction. The successful bundle initializer is much softer.
- F04's sharing is only an initial condition. All individual weight rows become independent trainable parameters. It therefore does not directly establish the value of permanently tied QI blocks.

Three audit issues in the original comparisons:

1. `all20/run.py:123` reports final evaluation loss; `all20_2layers/run.py:119` selects the best evaluation loss over 50 epochs. Those are different estimands, and the latter uses the test/evaluation split to select an epoch. I compare original final losses where available and use separate validation selection in all new probes.
2. Equal width is not equal parameter count. Moving from one to two hidden layers at width256 adds 65,792 parameters. The new depth controls explicitly match parameter counts.
3. The documented “centers-only” uniform-versus-sampled ablation does not preserve later directions: sampling centers consumes RNG draws between successive random directions. Also, uniform endpoints have actual spacing 2A/(P-1), while gamma uses h=2A/P. Thus gamma times actual spacing is .25 P/(P-1), not exactly .25. These issues are removed in the canonical model below.

## 2. Where the tenfold gain is present

The preserved `ladder/ladder_rows.json` gives these three-seed mean final relative L2 errors at d=1, using the GELU model selected by `ladder/run.py:91`:

| Target | Default initialization | `scaled_psqrt` | Error reduction |
|---|---:|---:|---:|
| Ridge mixture | .048686 | .004955 | 9.83× |
| Gaussian bump | .135250 | .013527 | 10.00× |
| Sine (`tensor4` at d=1) | .041394 | .004217 | 9.82× |

These gains largely disappear by dimensions8–16. The fully sharp P=W initialization is worse than the moderate bundle resolution on these trained GELU examples. This is evidence for a useful scale/optimization regime, not a general high-dimensional precision guarantee.

Across the original sixteen real-data tasks, one-layer tanh initialization gains in final loss range from about .98× to 1.28×. For two-layer width 256, the geometric-mean final-loss gain of `qi2` versus the same architecture's default initialization is 1.19× on eight regression tasks and 1.16× on eight classification tasks. See [historical ratios](historical_ratios.png) and [aggregated raw values](historical_summary.json).

## 3. Depth helps beyond parameter count

New controlled runs use fp64, one CPU thread, Adam lr=.001, batch128, exactly5000 shared update batches, and three seeds. Existing test splits are retained. Each seed holds out20% of the original training split for validation, trains on at most4096 remaining examples, and standardizes targets using those training examples. Validation selects among checkpoints every100 updates; test data never select a model. No schedules, solve-based training, or per-task hyperparameters are used.

All table values below are test MSE divided by training target variance, averaged across the three seeds. They are **MSE**, so a factor in this table is the square of the corresponding RMSE factor.

| Task | 1-layer QI, width128 | 2-layer QI, matched parameters | 2-layer QI, width128 |
|---|---:|---:|---:|
| Airfoil | .33823 /897 params | .16173 /885 params | .06790 /17,409 params |
| Parkinsons | .61414 /2,689 | .37654 /2,689 | .23513 /19,201 |
| Bike sharing | .29896 /1,793 | .08793 /1,751 | .06685 /18,305 |

The equal-parameter depth advantage survives at5000 updates after being weaker at1000. It is therefore not solely the original parameter-count or evaluation-selection artifact. Composing learned ridge features appears useful here. This experiment does not isolate interaction order from every other optimization effect of depth.

Initialization itself is a smaller effect: at the matched two-layer widths, default-versus-QI2 losses are .22123/.16173 (airfoil), .38622/.37654 (parkinsons), and .08961/.08793 (bike). [Depth plot](controlled_depth.png); raw trajectories: `probe_rows_1000.json` and `probe_rows.json`.

## 4. The initial frozen dictionary is not the source of the real-data win

I froze initial features and selected a ridge-regularized readout on the independent validation split. At width128, the initial standard/QI test errors are .23763/.33120 on airfoil, .66591/.76374 on parkinsons, and .50219/.59319 on bike. QI starts with a **worse** frozen dictionary on all three tasks. This remains a statement about this width and regularization grid, not a universal statement about initial representability.

Training improves the QI geometry. Initially identical direction rows separate: after5000 updates, mean sine of the within-bundle angle is .39, .39, and .56 for the one-layer QI models. F's high-dimensional benefit therefore cannot be explained as simply installing an already-superior frozen approximation space. [Dictionary diagnostic](initial_dictionary.png).

Nevertheless, enforcing the sharing throughout training only moderately hurts the wide two-layer network and cuts parameter count by approximately9×. Free versus tied two-layer test MSE is .0679/.0909 (airfoil), .2351/.3470 (parkinsons), and .0669/.0830 (bike). This motivates an actual compact block, rather than abandoning the shared structure.

## 5. A reusable fixed-bank model that works

The primary reusable implementation is `fixed_qi_mlp.py:FixedQIMLP`. The default model has two stages, twelve projected channels per stage, and eleven centers per channel. At each stage,

`p = Linear(z); H[r,j] = tanh(gamma * (p[r] - c[j])); z = flatten(H)`.

The fixed bank has h=2/N, midpoint centers c_j=-1+(j+.5)h, and gamma=.25/h. The affine projections and final linear head are trained. Bank centers and gamma are immutable buffers. A stage's affine projection is computed once per channel, avoiding the repeated matrix products of the expanded dense MLP.

The default initializer is simple:

1. Initialize each projection/mixer row as an independent unit Gaussian direction.
2. On at most4096 training examples, divide the row by quantile(|projection|,.999). This places that quantile at1 in the fixed bank coordinate. Bias starts at0. Repeat using the previous stage's actual bank outputs.
3. Initialize the final head with ordinary Linear uniform bounds. All subsequent training is Adam on the trainable parameters. There is no forward normalization or range tracking.

The calibration does not fit labels, and its temporary work is bounded by a fixed sample cap. Training state is ordinary Adam state. The prototype supports different channel/center counts and additional stages, although the experiments here only validate two stages.

The canonical model removes F04's endpoint-spacing discrepancy. Dense two-layer controls have approximately the same parameter count, using standard initialization or F04 QI initialization in both layers. These are the **primary** practical results:

| Task | Dense default | Dense QI both | Fixed bank, random mixer | Fixed bank, I02 polynomial mixer |
|---|---:|---:|---:|---:|
| Airfoil | .16296 ±.00290 | .12558 ±.00315 | **.09414 ±.00412** | .08966 ±.01250 |
| Parkinsons | .40174 ±.01760 | .36830 ±.01814 | **.27692 ±.01500** | .31870 ±.01813 |
| Bike sharing | .08676 ±.00535 | **.07818 ±.00391** | .08118 ±.00263 | .07952 ±.00180 |

Values after ± are seed standard deviations, not confidence intervals. Parameter counts are1801/1834 (block/dense airfoil),1969/1996 (parkinsons), and1885/1925 (bike). The block has slightly fewer parameters in every comparison. The random-mixer block wins clearly on two tasks and is competitive, without winning, on bike. [Canonical comparison](canonical_comparison.png), [summary](canonical_summary.json), [all trajectories](canonical_rows.json).

The I02 control uses independent degree3 Chebyshev profiles, represented by truncated SVD coefficients on [-.8,.8] at rcond1e-10, then the **same** layerwise calibration and head initialization. Thus the mixer prior is isolated within the same architecture. Polynomial initialization is not intrinsically broken: it also works at this modest bank resolution. It is unnecessary here and worse on parkinsons. Its initialized second-projection norm is approximately2000–2300 versus3.8–4.7 for the simple random mixer, despite similar initial occupied ranges. The raw random mixer avoids this cancellation burden without losing the practical benefit.

An important limit: after training, roughly60–73% of second-stage coordinates lie outside [-1,1]. These successful tabular models use the tanh tails. Their performance does **not** establish uniform precision on the bank interval or prove that maintaining full bank occupancy is needed during ordinary regression training. The result supports a practical trainable architecture, not a precision theorem. It also argues against assuming that a band penalty is automatically beneficial.

## 6. Checks and reproduction

`test_fixed_qi_mlp.py` verifies dense-network equivalence before and after updates; exact fixed gamma·h=.25; midpoint spacing; calibration; nonzero projection gradients; and bitwise bank invariance after training. The two tests pass. Canonical training additionally asserts immutable bank buffers in every run. Preliminary tied-bank probes assert bitwise equality with their parent dense models at initialization.

Commands, from repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/precision-mpl-cache .venv/bin/python experiments/expI03_investigation/f_init/canonical_run.py
.venv/bin/python -m pytest -q experiments/expI03_investigation/f_init/test_fixed_qi_mlp.py
```

Historical/probe reproduction is in `probe.py`, `tied_probe.py`, and `summarize.py`. Earlier fixed-bank conversions deliberately preserved the original F04 initializer; their results are `fixed_*_rows.json` and `fixed_bank_comparison.png`. They are discovery evidence, superseded for precise geometry claims by `canonical_*`.

The task scope is bounded: three real-data regression tasks, three seeds, fixed5000-update budget, small parameter counts, fp64 CPU. There is no broad benchmark, noise-free precision claim, validated large-depth result, or wall-time speedup claim. The canonical block took roughly2.0–2.7s per run including evaluations, versus1.1–1.3s for the small dense controls on this machine. Kernel overhead matters at this scale; parameter efficiency is the established benefit.

## 7. Resolution versus channels

The final bounded sweep fixes an approximately 1800-parameter budget and varies N in {4,8,16,32}, choosing equal first/second channel count M=K to minimize the parameter-count difference. It uses the canonical random initializer and the same three tasks, three seeds, and 5000-update protocol.

| Task | N=4 | N=8 | N=16 | N=32 |
|---|---:|---:|---:|---:|
| Airfoil | .21110 (M20) | .11298 (M14) | .09655 (M10) | .08669 (M7) |
| Parkinsons | .56000 (M18) | .34024 (M13) | .32761 (M9) | .34282 (M7) |
| Bike sharing | .10331 (M19) | .07792 (M14) | .08044 (M10) | .10127 (M7) |

The parenthesized value is the number of channels at each stage. Integer channel counts imply parameter budgets from 1630 to 1971 (all exact counts and seed deviations are in [resolution summary](resolution_summary.json)). In particular, parkinsons N16 has 1630 parameters and N32 has 1940, so that comparison slightly favors N32 in capacity.

The result rejects both simple extremes. Four very soft features per channel are inadequate at this update budget, despite providing more channels. Increasing resolution to32 is useful on airfoil, neutral on parkinsons, and harmful on bike; it is not a universal improvement. At fixed parameter count, bank resolution and channel count must be treated as a tradeoff. The successful default N11/M12 sits in an intermediate region, but this small sweep does not establish it as a universal optimum. [Resolution plot](resolution_comparison.png), [all trajectories](resolution_rows.json).

Reproduce with:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/precision-mpl-cache .venv/bin/python experiments/expI03_investigation/f_init/canonical_run.py --resolution-sweep
```

The strongest transferable recipe supported here is therefore **fixed moderate-resolution banks, enough learned channels, layerwise input calibration once, and modest random mixing coefficients**. It is a measured practical baseline for checkpoint I, not a proof of uniformly precise composition. Further changes should beat this simple baseline rather than assuming a more elaborate smooth-profile initialization, range constraint, or precision readout solve is beneficial.
