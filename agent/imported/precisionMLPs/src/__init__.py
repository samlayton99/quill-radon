"""precisionMLPs: Machine-precision MLP training via quasi-interpolant theory.

All computation runs in float64 (machine-eps precision is the whole point).

Device note: Apple's MPS (Metal) backend does NOT support float64 -- it raises
on any float64 tensor -- so MPS is intentionally never selected. CUDA does
support float64; everything else falls back to CPU. In practice these are tiny
1-D problems (width <= a few hundred), so CPU is the expected device.
"""

import torch

torch.set_default_dtype(torch.float64)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
