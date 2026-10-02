from pathlib import Path
import json
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
from route2_battletest_flows import compatible_native_start


def save(path,degree,viscosity=.1):
    np.savez(path,metadata=json.dumps(dict(kind='native_boundary_driven_ns',case='disk_counterrotating',
        viscosity=viscosity,lid_speed=1.,degree=degree,coordinate_basis='disk')))


def test_continuation_uses_saved_matching_stage_without_discarding_modes(tmp_path):
    high=tmp_path/'flow_p64_n513.npz';low=tmp_path/'flow_p48_n513.npz'
    save(high,64);save(low,48)
    assert compatible_native_start(high,48)==str(low)
    assert compatible_native_start(high,64)==str(high)
    assert compatible_native_start(low,64)==str(low)
    save(low,48,viscosity=.03)
    with pytest.raises(ValueError,match='changes the problem'):
        compatible_native_start(high,48)


def test_missing_matching_stage_never_silently_truncates(tmp_path):
    high=tmp_path/'flow_p64_n513.npz';save(high,64)
    with pytest.raises(ValueError,match='will not discard'):
        compatible_native_start(high,48)
