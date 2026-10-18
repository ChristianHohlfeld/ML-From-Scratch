import math,json,unittest
import numpy as np
from gear_ai import Stage,Machine,Drive,rectangular_interaction,fit,learn,loss

class Tests(unittest.TestCase):
    def setUp(self):
        self.m=Machine(0.1-.3j,[Stage(1,0,.5,0),Stage(0,2,.2,.15),Stage(2,-1,.3,.25)])
    def test_ratio_exact(self):
        a,b=.7,-1.4
        self.assertAlmostEqual(self.m.ports(a,b)[(2,-1)],2*a-b+math.atan2(.25,.3))
    def test_joint_nonseparable(self):
        self.assertGreater(abs(rectangular_interaction(self.m,.2,.8,-1.,.8)),.05)
    def test_additive_has_no_interaction(self):
        m=Machine(0j,[Stage(2,0,.4,.3),Stage(0,3,.4,.2)])
        self.assertLess(abs(rectangular_interaction(m,.2,.8,-1.,.8)),1e-12)
    def test_two_inputs_change_output_independently(self):
        self.assertGreater(abs(complex(*self.m.read(.1,.2))-complex(*self.m.read(.4,.2))),.01)
        self.assertGreater(abs(complex(*self.m.read(.1,.2))-complex(*self.m.read(.1,.4))),.01)
    def test_angle_periodicity(self):
        np.testing.assert_allclose(self.m.read(.2,.3),self.m.read(.2+2*math.pi,.3),atol=1e-12)
        np.testing.assert_allclose(self.m.read(.2,.3),self.m.read(.2,.3+2*math.pi),atol=1e-12)
    def test_equal_run_inertia(self):
        d=Drive(self.m)
        for i in range(180):d.step(.01,(.8,-.9))
        self.assertTrue(np.isfinite(d.read()).all())
        self.assertGreater(abs(d.angle[0]),.2)
        self.assertGreater(abs(d.angle[1]),.2)
    def test_mechanical_backdrive_and_positive_inertia(self):
        coupled=Machine(0j,[Stage(2,-1,.4,.2)])
        d=Drive(coupled)
        self.assertLess(d.mass_matrix[0,1],0)
        self.assertGreater(np.linalg.eigvalsh(d.mass_matrix).min(),0)
        d.kp=0;d.kd=0
        d.max_torque=0.0
        # External torque on A transmits through the mixed inertia to B.
        force=np.array([1.,0.])
        acceleration=np.linalg.solve(d.mass_matrix,force)
        self.assertGreater(abs(acceleration[1]),0)
    def test_torque_bounded(self):
        d=Drive(self.m,max_torque=.2,drag=0)
        d.step(.01,[100,-100])
        self.assertLessEqual(max(abs(d.speed)),.002+1e-10)
    def test_structure_change_preserves_both_shafts(self):
        d=Drive(self.m)
        for i in range(12):d.step(.01,(1.,-1.))
        a=d.angle.copy();v=d.speed.copy()
        d.replace(Machine(0j,[Stage(1,1,.4,.1)]))
        np.testing.assert_array_equal(a,d.angle)
        np.testing.assert_array_equal(v,d.speed)
    def test_json_roundtrip_and_bad_ratios(self):
        m=Machine.from_json(json.loads(json.dumps(self.m.to_json())))
        np.testing.assert_allclose(m.read(1,2),self.m.read(1,2))
        for k,l in [(0,0),(1.2,1)]:
            with self.assertRaises(ValueError):Stage(k,l,1,0)
        with self.assertRaises(ValueError): Machine(0j,[Stage(1,1,1,0),Stage(1,1,1,0)])
        with self.assertRaises(ValueError):Machine.from_json({'format':'hdr-gear-ai/1'})
    def test_fits_cross_mechanism(self):
        rng=np.random.default_rng(11)
        a=rng.uniform(-math.pi,math.pi,320);b=rng.uniform(-math.pi,math.pi,320)
        m=fit((a,b),self.m.complex_sample(a,b),[(1,0),(0,2),(2,-1)])
        self.assertLess(loss(m,(a,b),self.m.complex_sample(a,b)),1e-14)
    def test_structure_selects_interaction(self):
        rng=np.random.default_rng(41)
        a=rng.uniform(-math.pi,math.pi,350);b=rng.uniform(-math.pi,math.pi,350)
        out=learn((a[:180],b[:180]),self.m.complex_sample(a[:180],b[:180]),
                  (a[180:270],b[180:270]),self.m.complex_sample(a[180:270],b[180:270]),
                  max_ratio=3,max_stages=5,penalty=1e-6)
        self.assertEqual({(g.k,g.l) for g in out.machine.stages},{(1,0),(0,2),(2,-1)})
        self.assertLess(loss(out.machine,(a[270:],b[270:]),self.m.complex_sample(a[270:],b[270:])),1e-12)
    def test_without_mixed_terms_cannot_fit(self):
        rng=np.random.default_rng(43)
        a=rng.uniform(-math.pi,math.pi,350);b=rng.uniform(-math.pi,math.pi,350)
        mixed=learn((a[:200],b[:200]),self.m.complex_sample(a[:200],b[:200]),(a[200:260],b[200:260]),self.m.complex_sample(a[200:260],b[200:260]),max_ratio=3,max_stages=5,penalty=1e-6,mixed=True)
        separate=learn((a[:200],b[:200]),self.m.complex_sample(a[:200],b[:200]),(a[200:260],b[200:260]),self.m.complex_sample(a[200:260],b[200:260]),max_ratio=3,max_stages=5,penalty=1e-6,mixed=False)
        target=self.m.complex_sample(a[260:],b[260:])
        self.assertLess(loss(mixed.machine,(a[260:],b[260:]),target),loss(separate.machine,(a[260:],b[260:]),target)/100)
if __name__=='__main__':unittest.main()
