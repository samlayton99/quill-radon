# Checkpoint I investigation

Read the [main results](../../results/checkpoint_I_depth_theory/expI03_investigation/expI03_results.md). All original I01/I02 and F experiments remain unchanged. The complete new protocol is recorded in [config.yaml](config.yaml); saved per-run JSON also includes configurations and trajectories.

The practical model is [f_init/fixed_qi_mlp.py](f_init/fixed_qi_mlp.py). It supports ordinary PyTorch training with fixed midpoint banks and learned affine projections. The measured default uses two stages, twelve channels and eleven centers at each stage. Train-input calibration runs once:

```python
import torch
from experiments.expI03_investigation.f_init.fixed_qi_mlp import FixedQIMLP

torch.set_default_dtype(torch.float64)  # The dtype used in the recorded experiments.
model = FixedQIMLP(X_train.shape[1]).to(X_train)
model.initialize_(X_train, seed=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# X_batch and y_batch come only from the training split.
optimizer.zero_grad()
prediction = model(X_batch).squeeze(-1)
loss = (prediction - y_batch).square().mean()
loss.backward()
optimizer.step()
```

Use separate validation for checkpoint selection. The recorded experiments standardized regression targets from the actual training subset. The results validate fp64 CPU regression on three cached random-row splits; other dtypes, devices and deployment distributions have not been benchmarked. Extra stages are supported by the interface but do not yet have a broad accuracy result.

From the repository root, reproduce the primary practical study and resolution control:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/precision-mpl-cache .venv/bin/python experiments/expI03_investigation/f_init/canonical_run.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/precision-mpl-cache .venv/bin/python experiments/expI03_investigation/f_init/canonical_run.py --resolution-sweep
```

The analytic probes use dense per-step SVD head fits; they are numerical diagnostics, with a different compute profile from the practical model. Reproduce the main comparison, then its three follow-up controls:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/expI03_investigation/probe.py --name confirm_d3 --targets fast_waves,composition,product_peak,random_ridges --seeds 3,4,5 --steps 2000 --lr .005 --beta .01 --ridge 1e-6 --ntrain 2560 --ntest 4096 --arms raw,poly3,quadratic,mlp --save-models
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/expI03_investigation/probe.py --name quadratic_init_ablation --targets fast_waves --seeds 3,4,5 --steps 2000 --lr .005 --beta .01 --ridge 1e-6 --ntrain 2560 --ntest 4096 --arms poly2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/expI03_investigation/probe.py --name quadratic_resolution --targets fast_waves --seeds 3,4,5 --steps 2000 --lr .005 --beta .01 --ridge 1e-6 --ntrain 2560 --ntest 4096 --arms poly2,quadratic --N1 64
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/expI03_investigation/probe.py --name quadratic_release --targets fast_waves,composition,random_ridges --seeds 3,4,5 --steps 2000 --lr .005 --beta .01 --ridge 1e-6 --ntrain 2560 --ntest 4096 --arms quadratic_free --save-models
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/expI03_investigation/refine.py --source confirm_d3 --target fast_waves --seeds 3
MPLCONFIGDIR=/tmp/precision-mpl-cache .venv/bin/python experiments/expI03_investigation/summarize.py
```

Runners resume existing JSON by row key. Use a new output name when changing a protocol; do not reuse a populated name with different arguments. Original screening runs are exploratory and are retained under the dated results directory. Their settings differ from the primary confirmation.

The [depth report](../../results/checkpoint_I_depth_theory/investigation_20260906/depth/report.md) gives its reproduction commands. The [solver report](../../results/checkpoint_I_depth_theory/investigation_20260906/evidence_audit/solver_audit.md) explains the corrected dense GN reference and its limitations.

Verification:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_expI02_block_prior.py tests/test_expI03_investigation.py experiments/expI03_investigation/f_init/test_fixed_qi_mlp.py results/checkpoint_I_depth_theory/investigation_20260906/evidence_audit/test_solver_audit.py
```
