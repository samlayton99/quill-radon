"""Equation and oracle-quarantine checks for the fixed forward campaign."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'experiments/expF19_radon_direct_pde'))
import route2_battletest_forward as campaign


class Manufactured(torch.nn.Module):
    def forward(self, x):
        return campaign.manufactured_jet(x)[0]


def test_manufactured_forcing_matches_independent_autograd_derivatives():
    points=np.random.default_rng(717).uniform(0.,1.,(19,2))
    orders=((0,0),(1,0),(0,1),(2,0),(0,2))
    x,jets=campaign.ordinary_jets(Manufactured(),points,orders)
    for order,exact in zip(orders,campaign.manufactured_jet(x)):
        torch.testing.assert_close(jets[order],exact,rtol=3e-14,atol=3e-14)
    for contrast in [10.,1000.]:
        residual,_=campaign.equation(dict(family='diffusion',contrast=contrast))
        assert float(residual(x,jets,torch.empty(0)).abs().max()) < 1e-11


@pytest.mark.parametrize('n',[2,4])
def test_published_helmholtz_sign_and_frequency(n):
    class Target(torch.nn.Module):
        def forward(self,x):
            return torch.sin(2*np.pi*n*x[:,0:1])*torch.sin(2*np.pi*n*x[:,1:2])
    callback,orders=campaign.equation(dict(family='helmholtz',n=n))
    points=np.random.default_rng(813).uniform(0.,1.,(21,2))
    x,jets=campaign.ordinary_jets(Target(),points,orders)
    assert float(callback(x,jets,torch.empty(0)).abs().max()) < 3e-12


def test_allen_cahn_zero_state_is_not_an_initial_condition_solution():
    declaration=campaign.make_declaration(campaign.CASES['allen_cahn_d00001'])
    blocks=declaration.make_blocks(29,7)
    assert [b.name for b in blocks]==['PDE','initial','periodic_value','periodic_dx']
    for block in blocks:
        x=torch.tensor(block.points,dtype=torch.float64)
        jets={a:torch.zeros((len(x),1),dtype=torch.float64) for a in block.derivatives}
        result=block.function(x,jets,torch.empty(0)).detach().numpy()
        if block.aggregation is not None:result=block.aggregation@result
        if block.name=='initial':assert np.max(abs(result))>.5
        else:np.testing.assert_array_equal(result,0.)


def test_factories_do_not_query_reference_oracles(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Evaluation reference queried by fitting declaration')
    monkeypatch.setattr(campaign,'analytical_reference',forbidden)
    monkeypatch.setattr(campaign,'allen_reference',forbidden)
    for case in campaign.CASES.values():
        declaration=campaign.make_declaration(case,1)
        for block in declaration.make_blocks(23,17):
            x=torch.tensor(block.points,dtype=torch.float64)
            jets={a:torch.zeros((len(x),1),dtype=torch.float64) for a in block.derivatives}
            result=block.function(x,jets,torch.empty(0))
            assert torch.all(torch.isfinite(result))
            assert block.name in {'PDE','initial','Dirichlet','periodic_value','periodic_dx'}


def test_frozen_protocol_rejects_silent_revision(tmp_path):
    fingerprint=campaign.freeze_protocol(tmp_path)
    assert campaign.freeze_protocol(tmp_path)==fingerprint
    path=tmp_path/'protocol.json'
    data=json.loads(path.read_text());data['seeds']=[17]
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='Frozen protocol differs'):
        campaign.freeze_protocol(tmp_path)


def test_accuracy_extension_changes_budgets_uniformly_and_starts_cold():
    try:
        campaign.configure('accuracy_v2')
        changed={key:value for key,value in campaign.OPTIONS.items()
                 if campaign.BASE_OPTIONS.get(key)!=value}
        assert changed=={'max_seconds':600.,'max_iterations':60}
        assert campaign.PROTOCOL['seeds']==[0]
        assert campaign.PROTOCOL['cases']==campaign.BASE_PROTOCOL['cases']
        assert 'initial_solution' not in campaign.OPTIONS
        assert campaign.OUT.name=='accuracy_v2'
        with pytest.raises(ValueError,match='outside this frozen campaign'):
            campaign.fit('helmholtz_n2',1)
    finally:
        campaign.configure('screening_v1')


def test_atomic_case_claim_prevents_duplicate_worker_fit(monkeypatch,tmp_path):
    monkeypatch.setattr(campaign,'OUT',tmp_path)
    claim=tmp_path/'helmholtz_n2_s0.running.json'
    claim.write_text('{"pid":123}')
    def forbidden(*args):raise AssertionError('Duplicate worker entered fit')
    monkeypatch.setattr(campaign,'_fit_case',forbidden)
    assert campaign.fit('helmholtz_n2',0)['status']=='already_running_elsewhere'
    assert claim.exists()


def test_paired_precision_protocol_is_uniform_and_restricts_case_selection():
    try:
        campaign.configure('normalized_diffusion_precision_v3')
        assert set(campaign.PROTOCOL['cases'])=={'diffusion_c10','diffusion_c1000'}
        assert campaign.OPTIONS['tolerance']==1e-13
        assert campaign.OPTIONS['centers']==257 and campaign.OPTIONS['lam']==.2
        assert campaign.OPTIONS['max_seconds']==600. and campaign.OPTIONS['max_iterations']==60
        assert campaign.PROTOCOL['normalization']['blocks']==['PDE']
        assert campaign.PROTOCOL['normalization']['fit_truth_or_reference_used'] is False
        assert 'initial_solution' not in campaign.OPTIONS
        with pytest.raises(ValueError,match='Case is outside'):
            campaign.fit('helmholtz_n4',0)
    finally:
        campaign.configure('screening_v1')


def test_loaded_code_snapshot_distinguishes_runtime_patch_from_disk(monkeypatch):
    before=campaign.source_snapshot()
    monkeypatch.setattr(campaign,'equation',lambda case:('patched',()))
    after=campaign.source_snapshot()
    assert before['disk_files_sha256']==after['disk_files_sha256']
    assert before['loaded_function_code_sha256']['equation']!=after['loaded_function_code_sha256']['equation']
