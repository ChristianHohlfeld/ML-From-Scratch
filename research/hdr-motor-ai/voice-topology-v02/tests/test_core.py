import numpy as np
import pytest
from waveform import Shape,geometry,propagate,signature,topology_2d,fit_shape

def test_geometry_3d_spatial():
    mask,dx,info=geometry(Shape())
    assert mask.ndim==3 and mask.shape==(38,19,19)
    assert mask.sum()>500 and mask.sum()<mask.size
    assert np.all(info['area']>0)
    assert dx>0 and info['dy']>0

def test_acoustic_cfl():
    r=propagate(Shape(),steps=300)
    assert 0 < r['cfl'] < 1

def test_all_outputs_finite():
    for variant in ('chirp','multitone','impulse'):
        r=propagate(Shape(),steps=420,excitation=variant)
        assert np.isfinite(r['signal']).all()
        assert np.isfinite(r['energy']).all()
        assert (r['energy']>=0).all()

def test_deterministic_solver():
    a=propagate(Shape(),steps=600)
    b=propagate(Shape(),steps=600)
    assert np.array_equal(a['signal'],b['signal'])

def test_geometry_alters_wave_response():
    a=propagate(Shape((.65,1.3,.68,1.3)),steps=1100)
    b=propagate(Shape((1.25,.6,1.25,.63)),steps=1100)
    assert np.linalg.norm(a['signal']-b['signal'])>.01

def test_topology_distinct_boundary():
    assert topology_2d(Shape(rod=False))==(1,0)
    assert topology_2d(Shape(rod=True))==(1,1)
    full,*_=geometry(Shape(rod=False))
    hole,*_=geometry(Shape(rod=True))
    assert hole.sum()<full.sum()
    assert np.any(full & ~hole)

def test_scale_invariant_profile_but_not_acoustics():
    a=Shape(length=.14);b=Shape(length=.21)
    aa,dx,fa=geometry(a);bb,dx2,fb=geometry(b)
    assert np.array_equal(aa,bb)
    assert np.allclose(fa['radial'],fb['radial'])
    assert not np.array_equal(propagate(a,steps=700)['signal'],propagate(b,steps=700)['signal'])

def test_signature_finite_normalized():
    sig=signature(propagate(Shape(),steps=1200))
    assert len(sig)==32 and np.isfinite(sig).all()
    assert np.linalg.norm(sig)==pytest.approx(1,abs=1e-8)

def test_input_rejects_invalid_shapes():
    for vals in [(0,1,1,1),(1,1,1),(2,1,1,1)]:
        with pytest.raises(ValueError):Shape(tuple(vals))
    with pytest.raises(ValueError):Shape(length=.5)
    with pytest.raises(ValueError):geometry(Shape(),nx=6)

def test_physics_learning_reduces_training_and_holdout():
    truth=Shape((.69,1.25,.72,1.3)); flat=Shape((1,1,1,1))
    target=signature(propagate(truth,steps=2100,excitation='chirp'))
    fit=fit_shape(target,initial=flat,max_evals=125,steps=2100)
    assert fit['after']<fit['before']*.15
    hold=signature(propagate(truth,steps=2100,excitation='multitone'))
    flat_err=np.linalg.norm(signature(propagate(flat,steps=2100,excitation='multitone'))-hold)
    fit_err=np.linalg.norm(signature(propagate(fit['shape'],steps=2100,excitation='multitone'))-hold)
    assert fit_err < flat_err*.8

def test_true_topology_not_inferred_by_classifier():
    # The topology flag is a structural ground-truth property, not a
    # claim that a single pressure sensor can recover all 3D holes.
    assert topology_2d(Shape(rod=True))[1]==1
