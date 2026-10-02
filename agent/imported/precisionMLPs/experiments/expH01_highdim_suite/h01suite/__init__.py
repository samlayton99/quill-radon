"""The checkpoint-H high-dimensional benchmark suite.

Eighty designed approximation problems on the cube ``[-1,1]^d`` for ``d = 1..5``
(16 per dimension), specified in
``results/checkpoint_H_highdim/expH01_highdim_suite/SUITE_SPEC.md`` (Version 3).

The suite is only a measuring device: it carries the target functions, the data
geometries, the test sets, the error breakdowns, and one frozen reference model that
uses evenly spread directions and evenly spaced centers. No adaptive model lives here.

Module map
----------
``basis``      the fixed DCT-II directions ``u_k`` and the normalized coordinates ``z_k``.
``targets``    the twelve function families, each with an analytic gradient.
``normalize``  the fixed uniform reference set used to center and scale every target.
``densities``  the eight data geometries and the three test sets.
``tasks``      the 80-task list.
``metrics``    errors, region breakdowns, and the predicted center density.
``baseline``   the reference model (even directions, even centers) and a random-features control.
"""

from h01suite import basis, baseline, densities, metrics, normalize, tasks, targets  # noqa: F401
from h01suite.tasks import TASKS, get_task, task_ids  # noqa: F401
