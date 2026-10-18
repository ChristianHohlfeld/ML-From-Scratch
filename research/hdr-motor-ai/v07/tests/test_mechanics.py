import unittest,math
import numpy as np
from gear_ai.two_link import TwoLink,PhysicalTwin
from gear_ai.protocol import parse_telemetry, integrate_joules

class Mechanical(unittest.TestCase):
 def setUp(self):self.m=TwoLink()
 def test_home_position(self):
  np.testing.assert_allclose(self.m.tip(0,0),[.105,0],atol=1e-12)
 def test_second_joint_mixed(self):
  # Rectangular mixed difference is nonzero; two independent additive arms cannot produce it.
  self.assertGreater(np.linalg.norm(self.m.rectangular_interaction(-.5,.5,-.5,.5)),.005)
 def test_jacobian_finite_difference(self):
  a,b=.2,.3;d=1e-6
  for j in [0,1]:
   va=(self.m.tip(a+d,b)-self.m.tip(a-d,b))/(2*d) if j==0 else (self.m.tip(a,b+d)-self.m.tip(a,b-d))/(2*d)
   np.testing.assert_allclose(va,self.m.jacobian(a,b)[:,j],atol=1e-9)
 def test_mass_symmetric_positive_over_range(self):
  for b in np.linspace(-1,1,31):
   M=self.m.mass_matrix(b)
   np.testing.assert_allclose(M,M.T,atol=1e-16)
   self.assertTrue(np.min(np.linalg.eigvalsh(M))>0)
 def test_energy_coriolis_identity(self):
  # qdot . (Mdot - 2C) . qdot = 0, for analytical mechanical Coriolis vector.
  for a,b,da,db in [(0,.2,.3,.1),(.3,-.2,.4,-.3),(-.2,.6,.2,.2)]:
   eps=1e-6;qdot=np.array([da,db]);
   mdot=(self.m.mass_matrix(b+eps)-self.m.mass_matrix(b-eps))/(2*eps)*db
   self.assertAlmostEqual(float(qdot@mdot@qdot),2*float(qdot@self.m.coriolis(b,da,db)),places=8)
 def test_twin_tracks_position(self):
  twin=PhysicalTwin(self.m);target=(.25,-.2)
  p=twin.settle(target,seconds=1.0)
  self.assertLess(np.linalg.norm(p-self.m.tip(*target)),.01)
 def test_target_guard(self):
  twin=PhysicalTwin(self.m)
  for target in [(math.pi,0),(0,float('nan'))]:
   with self.assertRaises(ValueError):twin.step(.001,target)
 def test_frame_mass_parameters(self):
  with self.assertRaises(ValueError):TwoLink(l1=-.06)
 def test_protocol_complete(self):
  row=parse_telemetry('T 100 10.000 -20.000 5.0000 0.5000 2.5000')
  self.assertEqual(row.millisecond,100)
  self.assertAlmostEqual(integrate_joules([row,parse_telemetry('T 110 10 -20 5 0.5 2.5')]),.025)
 def test_protocol_sensor_absence(self):
  p=parse_telemetry('T 100 10 -20 nan nan nan')
  self.assertIsNone(integrate_joules([p,p]))
 def test_protocol_reject_partial(self):
  with self.assertRaises(ValueError):parse_telemetry('T 100 0 0 5 nan nan')
if __name__=='__main__':unittest.main()
