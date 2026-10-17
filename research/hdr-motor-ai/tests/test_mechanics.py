import json
import math
import subprocess
from pathlib import Path

import pytest
import torch

from hdr_motor_ai.mechanics import GearNetwork, MotorState, topology

ROOT = Path(__file__).resolve().parents[1]


def test_four_layers_and_passive_reference():
    e = topology(6, 2)
    assert {k: sum(z[2] == k for z in e) for k in ('pair','neighbor','group','hub')} == {
        'pair': 15, 'neighbor': 5, 'group': 6, 'hub': 2}
    assert all(a != 8 for a,b,kind,orientation in e)
    assert [t[-1] for t in e if t[2] == 'hub'] == [1., -1.]


def test_v0_undriven_and_reacts_through_couplings():
    m = GearNetwork(channels=2)
    u = torch.tensor([[1., -.8, .2, -.4, .9, -.5]])
    s = m.make_state(u)
    _, initial_acc = m.derivatives(s,u)
    assert torch.allclose(initial_acc[:,m.v0_idx,:],torch.zeros(1,2))
    s2 = m.evolve(u,steps=150,state=s)
    assert s2.angle[:,m.v0_idx,:].abs().max() > 0.001
    assert s2.velocity.shape == s2.angle.shape


def test_continuity_impulse_not_recompute():
    m=GearNetwork(channels=1)
    u=torch.tensor([[.8, -.7, .4, -.5, .6, -.4]])
    old=m.evolve(u,60)
    new=u.clone();new[0,0]=-0.8
    after=m.step(old,new)
    assert torch.allclose(after.angle,old.angle+m.dt*after.velocity)
    assert torch.allclose(after.velocity[:,m.v0_idx],old.velocity[:,m.v0_idx]+m.dt*m.derivatives(old,new)[1][:,m.v0_idx])


def test_energy_torque_gradient_physics():
    torch.manual_seed(8)
    m=GearNetwork(channels=2)
    u=torch.tensor([[.5,-.2,.3,-.1,.7,-.3]])
    s=m.make_state(u)
    angle=torch.randn_like(s.angle,requires_grad=True)*0.1
    st=MotorState(angle,torch.zeros_like(angle))
    e=m.energy(st,u).sum()
    analytical=-torch.autograd.grad(e,angle)[0]
    _,accel=m.derivatives(st,u)
    torch.testing.assert_close(analytical,accel*m.inertia,atol=3e-6,rtol=1e-5)


def test_dissipation_under_constant_input():
    m=GearNetwork(channels=3)
    u=torch.tensor([[.5,-.3,.2,-.1,.6,-.4]])
    st=m.make_state(u)
    energies=[]
    for t in range(220):
        energies.append(m.energy(st,u).item())
        st=m.step(st,u)
    assert energies[-1] < energies[0]*0.5
    assert all(math.isfinite(v) for v in energies)


def test_unique_linear_stationary_configuration():
    m=GearNetwork(channels=2)
    k,_,r=m.coefficients()
    for c in range(2):
        B=torch.zeros(len(m.edges),m.total)
        B[torch.arange(len(m.edges)),m.a]=1
        B[torch.arange(len(m.edges)),m.b]+=r[:,c]
        H=B.T@torch.diag(k[:,c])@B
        H[:m.n,:m.n]+=torch.eye(m.n)*m.motor_stiffness
        assert torch.linalg.eigvalsh(H).min()>1e-5


def test_positive_stiffness_bounded_ratios_and_trainability():
    m=GearNetwork(channels=3)
    k,cubic,r=m.coefficients()
    assert torch.all(k>0) and torch.all(cubic>0)
    assert torch.all(r.abs()>=1-m.ratio_span)
    x=torch.randn(5,6)*.3
    y,_=m(x,steps=40)
    y.square().mean().backward()
    for p in [m.log_k,m.ratio_logits,m.motor_gain,m.readout.weight]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0


def test_dynamic_state_shape_and_bad_parameters():
    with pytest.raises(ValueError):topology(2,3)
    m=GearNetwork()
    with pytest.raises(ValueError):m.evolve(torch.randn(2,4))
    with pytest.raises(ValueError):m.evolve(torch.randn(2,6),steps=-1)
    assert m.evolve(torch.zeros(2,6),steps=0).angle.abs().sum()==0


def test_export_round_trip_and_node_consistency():
    torch.manual_seed(3)
    m=GearNetwork(channels=2)
    u=torch.tensor([[.8,-.75,.35,-.2,.6,-.45]],dtype=torch.float32)
    st=m.evolve(u,steps=80)
    out=m.browser_export()
    payload={'model':out,'input':u[0].tolist(),'steps':80}
    code="""let E=require(process.argv[1]);let p=JSON.parse(require('fs').readFileSync(0,'utf8'));
let s=E.makeState(p.model);E.evolve(p.model,s,p.input,p.steps);
console.log(JSON.stringify({angles:s.angle,velocities:s.velocity,output:E.read(p.model,s)}));"""
    result=subprocess.run(['node','-e',code,str(ROOT/'web'/'engine.js')],input=json.dumps(payload),
                          text=True,capture_output=True,check=True)
    j=json.loads(result.stdout)
    torch.testing.assert_close(torch.tensor(j['angles']),st.angle[0],rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(torch.tensor(j['velocities']),st.velocity[0],rtol=2e-5,atol=2e-5)
    y=m.readout(st.angle[:,m.v0_idx,:])[0,0]
    assert j['output']==pytest.approx(float(y.detach()),abs=1e-5)
