"""Angular folding must preserve moments and the declared model size."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from ridge_frame_dimension_study import sphere_rule


@pytest.mark.parametrize('d,p',[(2,0),(2,12),(3,7),(3,12),(4,8),(4,12)])
def test_fold_has_exact_cardinality_and_matches_even_full_rule(d,p):
    directions,weights=sphere_rule(d,p)
    full,full_weights=sphere_rule(d,p,fold=False)
    assert len(directions)==(p+1)**(d-1)
    np.testing.assert_allclose(np.linalg.norm(directions,axis=1),1,atol=3e-16)
    np.testing.assert_allclose(weights.sum(),1,atol=3e-16)
    rng=np.random.default_rng(31)
    for _ in range(8):
        a=rng.normal(size=d);a/=np.linalg.norm(a)
        for n in (0,2*p):
            np.testing.assert_allclose(weights@((directions@a)**n),
                                       full_weights@((full@a)**n),atol=2e-15,rtol=2e-14)


def test_odd_integrands_are_not_claimed_preserved():
    directions,weights=sphere_rule(3,3)
    assert weights@directions[:,0]>.1
