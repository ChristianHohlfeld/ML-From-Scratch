import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from benchmark import CLASSES, base_mesh, perturb_mesh, equalize_vertices, topology, wave_features, geometry_features, run

@pytest.mark.parametrize('label',CLASSES)
def test_all_manifolds_closed(label):
    m=base_mesh(label)
    assert m.is_watertight
    assert topology(m)['first_betti'] >= 0

@pytest.mark.parametrize('label',CLASSES)
def test_edge_splits_preserve_topology(label):
    m=base_mesh(label)
    altered=equalize_vertices(m,96,123)
    assert len(altered.vertices)==96
    assert topology(m)==topology(altered)

@pytest.mark.parametrize('label',CLASSES)
def test_rotation_scale_warp_preserves_topology(label):
    m=equalize_vertices(base_mesh(label),96,12)
    for seed in (41,42,43):
        x=perturb_mesh(m,seed)
        assert topology(m)==topology(x)

@pytest.mark.parametrize('label',CLASSES)
def test_geometry_baseline_is_rotation_scale_invariant(label):
    m=equalize_vertices(base_mesh(label),96,12)
    x=m.copy()
    x.apply_scale(2.75)
    x.apply_translation([100.,-44.,20.])
    assert np.allclose(geometry_features(m),geometry_features(x),atol=1e-8)

@pytest.mark.parametrize('label',CLASSES)
def test_wave_signal_is_translation_scale_invariant(label):
    m=equalize_vertices(base_mesh(label),96,12)
    x=m.copy()
    x.apply_scale(.75)
    x.apply_translation([300.,-24.,100.])
    assert np.allclose(wave_features(m,steps=60),wave_features(x,steps=60),atol=1e-8)

def test_topology_distinguishes_hole_and_disconnection():
    assert topology(base_mesh('sphere'))['first_betti']==0
    assert topology(base_mesh('torus'))['first_betti']==2
    assert topology(base_mesh('two_spheres'))['components']==2
    assert topology(base_mesh('dumbbell'))==topology(base_mesh('sphere'))

def test_coupling_network_matters_and_deterministic():
    m=equalize_vertices(base_mesh('torus'),96,18)
    normal=wave_features(m,steps=80)
    wrong=wave_features(m,steps=80,shuffle_edges=True,rng_seed=18)
    assert np.max(np.abs(normal-wrong))>.00001
    assert np.allclose(normal,wave_features(m,steps=80))

def test_full_benchmark_not_just_topology():
    results=run(n_per_class=10,seed=20261018,steps=80)
    assert results['train']==20 and results['test']==20
    assert results['metrics']['exact_topology_baseline']==.75
    assert set(results['metrics'])=={'coupled_wave','shuffled_wave_edges','geometry_baseline','exact_topology_baseline'}
    assert all(0<=acc<=1 for acc in results['metrics'].values())
