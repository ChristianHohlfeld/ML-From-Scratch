import json
import math
import unittest
import numpy as np
from gear_ai import Gear,Design,Mechanism,fit_structure,mse,search

class TestMachine(unittest.TestCase):
    def setUp(self):
        self.design=Design(.3,-.1,[Gear(1,.8,.2),Gear(-2,.12,-.18),Gear(5,.2,.09)])
    def test_ratio_constraint(self):
        machine=Mechanism(self.design)
        for _ in range(90):machine.step(.01,drive_speed=2.)
        for g in self.design.gears:
            self.assertAlmostEqual(machine.sensed_phases()[g.turns],g.turns*machine.angle+g.phase,places=12)
    def test_unique_resultant(self):
        p=self.design.tip(1.7)
        self.assertTrue(np.allclose(p,self.design.sample(np.array([1.7]))[0]))
        self.assertTrue(np.isfinite(p).all())
    def test_phase_shift_gauge(self):
        # Rotation of every mounting vector and base rotates the whole output.
        angle=.44
        z=complex(math.cos(angle),math.sin(angle))
        rotated=Design(*(v for v in ((complex(.3,-.1)*z).real,(complex(.3,-.1)*z).imag)),
                       [Gear(g.turns,(complex(g.x,g.y)*z).real,(complex(g.x,g.y)*z).imag) for g in self.design.gears])
        original=complex(*self.design.tip(.9)); other=complex(*rotated.tip(.9))
        self.assertAlmostEqual(abs(other-original*z),0,places=10)
    def test_change_preserves_shaft(self):
        m=Mechanism(self.design)
        for _ in range(10):m.step(.01,drive_speed=3.)
        a,v=m.angle,m.velocity
        m.replace_structure(Design(.3,-.1,self.design.gears[:2]))
        self.assertEqual((a,v),(m.angle,m.velocity))
    def test_torque_bound(self):
        m=Mechanism(self.design,max_torque=.5,drag=0.)
        initial=m.velocity
        m.step(.01,command=100.)
        self.assertLessEqual(abs(m.velocity-initial),.01*.5/m.inertia+1e-12)
    def test_validation(self):
        with self.assertRaises(ValueError):Design(0,0,[Gear(2,1,0),Gear(2,1,0)])
        with self.assertRaises(ValueError):Gear(0,1,0)
        with self.assertRaises(ValueError):m=Mechanism(self.design);m.step(.1,drive_speed=1)
        with self.assertRaises(ValueError):Design.from_json({'format':'not-hdr','base':[0,0],'gears':[]})
    def test_roundtrip(self):
        self.assertEqual(self.design.to_json(),Design.from_json(json.loads(json.dumps(self.design.to_json()))).to_json())
    def test_closed_curve(self):
        self.assertTrue(np.allclose(self.design.tip(0),self.design.tip(2*np.pi),atol=1e-12))
    def test_offline_identification(self):
        x=np.linspace(0,2*np.pi,100,endpoint=False)
        d=fit_structure(x,self.design.sample(x),[1,-2,5])
        self.assertLess(mse(d,x,self.design.sample(x)),1e-15)
    def test_selects_sparse_structure(self):
        x=np.linspace(0,2*np.pi,128,endpoint=False)
        z=self.design.sample(x)
        fit=search(x[::2],z[::2],x[1::2],z[1::2],max_ratio=6,max_gears=6,penalty=1e-7)
        self.assertEqual(set(g.turns for g in fit.design.gears),{1,-2,5})
    def test_prune_noninformative(self):
        x=np.linspace(0,2*np.pi,96,endpoint=False)
        z=self.design.sample(x)
        out=search(x[::2],z[::2],x[1::2],z[1::2],max_ratio=8,max_gears=8,penalty=1e-5)
        self.assertEqual(len(out.design.gears),3)
    def test_no_inference_optimizer(self):
        # Inference can advance after fitting without touching search.
        m=Mechanism(self.design)
        for _ in range(15):m.step(.01,drive_speed=2)
        self.assertTrue(np.isfinite(m.read()).all())

if __name__=='__main__':unittest.main()
