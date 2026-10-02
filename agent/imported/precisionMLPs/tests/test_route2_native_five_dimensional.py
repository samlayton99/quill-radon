"""Manufactured forcing and native declaration, independently differentiated."""
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_native_five_dimensional import (B, VALUE, SECONDS, ORDERS, PROTOCOL,
    exact_numpy, boundary_torch, forcing_torch, make_declaration)


def test_full_rank_dense_manufactured_forcing_matches_independent_autograd():
    assert np.linalg.matrix_rank(B) == 5
    assert np.count_nonzero(B-np.diag(np.diag(B))) == 20
    points = np.random.default_rng(994).uniform(-1, 1, (11, 5))
    x = torch.tensor(points, dtype=torch.float64, requires_grad=True)
    u = boundary_torch(x)
    gradient = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    second = torch.stack([torch.autograd.grad(gradient[:, i].sum(), x, create_graph=True)[0][:, i]
                          for i in range(5)], dim=1)
    value_np, second_np = exact_numpy(points)
    np.testing.assert_allclose(u.detach().numpy().ravel(), value_np, atol=5e-16, rtol=5e-16)
    np.testing.assert_allclose(second.detach().numpy(), second_np, atol=5e-16, rtol=5e-16)
    residual = -second.sum(dim=1, keepdim=True)+u**3-forcing_torch(x)
    assert float(residual.detach().abs().max()) < 1e-15


def test_declaration_has_only_pde_and_boundary_no_interior_observations():
    declaration = make_declaration()
    blocks = declaration.make_blocks(97, 17)
    assert [b.name for b in blocks] == ['semilinear_elliptic', 'Dirichlet']
    assert blocks[0].derivatives == ORDERS
    assert np.all(np.max(abs(blocks[1].points), axis=1) == 1)
    x = torch.tensor(blocks[0].points, dtype=torch.float64)
    value, second = exact_numpy(blocks[0].points)
    jets = {VALUE: torch.tensor(value[:, None])}
    jets.update({d: torch.tensor(second[:, i:i+1]) for i, d in enumerate(SECONDS)})
    r = blocks[0].function(x, jets, torch.empty((len(x), 0)))
    assert float(r.abs().max()) < 1e-15
    assert PROTOCOL['angular_rule'] == 'tensor'
