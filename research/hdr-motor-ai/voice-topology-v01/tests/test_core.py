import unittest
import numpy as np
from hdr_voice import synth_vowel,prepare,pitch_estimate,embed,betti_curve,signature,mfcc

class TestVoiceShapes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.audio=synth_vowel('a',132,seed=17);cls.feat=signature(cls.audio)
    def test_source_is_finite(self):self.assertTrue(np.isfinite(self.audio).all())
    def test_pitch(self):self.assertAlmostEqual(pitch_estimate(prepare(self.audio)),132,delta=6)
    def test_high_pitch(self):self.assertAlmostEqual(pitch_estimate(prepare(synth_vowel('i',225))),225,delta=10)
    def test_three_dimensional_phase_space(self):self.assertEqual(embed(self.audio,14).shape,(52,3))
    def test_loudness_invariance(self):
        other=signature(self.audio*.17)
        np.testing.assert_allclose(self.feat['combined'],other['combined'],atol=2e-4,rtol=5e-4)
    def test_translation_invariance(self):
        a=embed(self.audio,14);b=embed(self.audio+2,14)
        np.testing.assert_allclose(a,b,atol=1e-5)
    def test_rotation_scale_topology(self):
        p=embed(self.audio,14);t=.45;R=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
        np.testing.assert_allclose(betti_curve(p),betti_curve(p@R*2.3+11))
    def test_triangle_has_no_hole(self):
        t=betti_curve(np.array([[0,0,0],[1,0,0],[0,1,0]]),[.1,2])
        self.assertEqual(t[0],1);self.assertAlmostEqual(t[2],1/3);self.assertEqual(t[3],0)
    def test_circle_produces_betti1(self):
        angles=np.linspace(0,2*np.pi,24,endpoint=False)
        cloud=np.c_[np.cos(angles),np.sin(angles),np.zeros_like(angles)]
        b=betti_curve(cloud,[.25,.5,.8,1])
        self.assertGreater(np.max(b[1::2]),0)
    def test_all_feature_lengths(self):
        f=self.feat;self.assertEqual(len(f['combined']),175)
        self.assertEqual(len(f['topology']),84)
        self.assertEqual(len(f['coupling']),15)
        self.assertAlmostEqual(np.sum(f['energy']),1,places=4)
    def test_mfcc(self):
        m=mfcc(self.audio);self.assertEqual(len(m),26)
        self.assertTrue(np.isfinite(m).all())
    def test_silence_rejected(self):
        with self.assertRaises(ValueError):signature(np.zeros(4000))
    def test_too_short_rejected(self):
        with self.assertRaises(ValueError):signature(np.ones(50))
    def test_bands_different_vowels(self):
        a=signature(synth_vowel('a',130))['energy'];i=signature(synth_vowel('i',130))['energy']
        self.assertGreater(np.linalg.norm(a-i),.2)
    def test_pure_sine_wave(self):
        t=np.arange(4000)/8000
        f=signature(np.sin(2*np.pi*160*t).astype('float32'))
        self.assertTrue(np.isfinite(f['combined']).all())
    def test_ensure_h1_nonnegative(self):self.assertTrue(np.all(self.feat['topology']>=0))

if __name__=='__main__':unittest.main()
