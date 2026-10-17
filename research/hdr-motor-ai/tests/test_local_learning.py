import pytest
import torch
from hdr_motor_ai.mechanics import GearNetwork
from hdr_motor_ai.local_learning import (local_phase_gradients, local_update, relaxed_phase,
                                          predict_local, residual)


def test_nudge_is_teaching_only_and_v0_moves():
    m = GearNetwork(channels=1)
    x=torch.tensor([[.7,-.3,.1,-.6,.5,-.2]])
    free=relaxed_phase(m,x,steps=380)
    teacher=torch.tensor([[1.0]])
    taught=relaxed_phase(m,x,steps=380,state=free,beta=.15,teacher=teacher)
    assert float(taught.angle[:,m.v0_idx,0]) > float(free.angle[:,m.v0_idx,0])
    assert abs(float(free.angle[:,m.v0_idx,0])) > 1e-4
    assert not hasattr(m,'teacher_motor')


def test_local_gradient_formula_matches_explicit_energy_parameter_derivative():
    torch.manual_seed(21)
    m=GearNetwork(channels=1)
    x=torch.randn(3,6)*.3
    a=relaxed_phase(m,x,steps=160)
    b=relaxed_phase(m,x,steps=160,state=a,beta=.15,teacher=torch.ones(3,1)*.5)
    grad_k, grad_r, grad_q=local_phase_gradients(m,a,b,.15)
    # Explicit autograd only at FIXED phase coordinates; no BPTT in production.
    def static_energy(state):
        k,q,r=m.coefficients()
        e=state.angle[:,m.a,:].detach()+r[None]*state.angle[:,m.b,:].detach()
        return (.5*k[None]*e.square()+.25*q[None]*e.pow(4)).sum()/3
    assert torch.isfinite(grad_k).all() and torch.isfinite(grad_r).all()
    g1=torch.autograd.grad(static_energy(b)-static_energy(a),[m.log_k,m.ratio_logits,m.log_cubic])
    for actual,expected in zip((grad_k,grad_r,grad_q),g1):
        torch.testing.assert_close(actual,expected.detach()/.15,atol=1e-5,rtol=1e-4)


def test_local_learning_does_not_run_backprop_and_respects_edge_masks():
    torch.manual_seed(4)
    m=GearNetwork(channels=1)
    x=torch.randn(8,6)*.4
    y=(x[:,0]>0).float()
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    local_update(m,x,y,free_steps=160,nudged_steps=160,train_kinds=('hub',),lr=.1)
    for n,p in m.named_parameters():
        assert p.grad is None
        if n in ('motor_gain','motor_bias','readout.weight','readout.bias'):
            torch.testing.assert_close(p,before[n])
    for i,kind in enumerate(m.edge_kinds):
        if kind!='hub':
            torch.testing.assert_close(m.log_k[i],before['log_k'][i])
            torch.testing.assert_close(m.ratio_logits[i],before['ratio_logits'][i])
    assert (m.log_k!=before['log_k']).any()


def test_no_learning_preserves_every_parameter():
    m=GearNetwork(channels=1)
    x=torch.randn(2,6)*.2
    prior={k:v.clone() for k,v in m.state_dict().items()}
    result=local_update(m,x,torch.tensor([1.,0.]),free_steps=90,nudged_steps=90,fit=False)
    for name,v in prior.items():torch.testing.assert_close(v,m.state_dict()[name])
    assert result.free_output.shape==(2,)


def test_reject_unsafe_teacher_and_multichannel():
    m=GearNetwork(channels=1);x=torch.zeros(2,6)
    with pytest.raises(ValueError):relaxed_phase(m,x,steps=0)
    with pytest.raises(ValueError):relaxed_phase(m,x,beta=.1)
    with pytest.raises(ValueError):local_update(m,x,torch.ones(2),train_kinds=('bogus',))
    with pytest.raises(ValueError):local_update(GearNetwork(channels=3),x,torch.ones(2))
