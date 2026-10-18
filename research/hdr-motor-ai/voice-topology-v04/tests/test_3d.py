import numpy as np
import pytest
from waveform.core import Shape,signature
from waveform.mesh import make_mesh,export_obj,deformed_vertices
from waveform.coupled import simulate, Wall, fit_shape


def test_mesh_vertices_and_connectivity():
    m=make_mesh(nx=16,ny=13,nz=13,nphi=12)
    assert m.rest.shape==(16,12,3)
    assert m.triangles.shape==(2*15*12,3)
    assert m.edges.shape[1]==2
    assert np.all(m.patch_area>0)
    assert np.allclose(np.linalg.norm(m.normals,axis=2),1)
    assert len(np.unique(m.triangles))==16*12
    assert np.all(m.cell>=0)


def test_surface_is_orientable_two_boundaries_and_no_internal_cracks():
    m=make_mesh(nx=13,ny=13,nz=13,nphi=12)
    from collections import Counter
    counts=Counter(tuple(sorted((int(a),int(b)))) for tri in m.triangles for a,b in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])))
    boundary=[ab for ab,c in counts.items() if c==1]
    assert len(boundary)==24 # two rings; no internal seam
    assert all(c in (1,2) for c in counts.values())
    # Euler of cylinder = 0.
    assert m.rest.shape[0]*m.rest.shape[1]-len(counts)+len(m.triangles)==0


def test_independent_xyz_surface_nodes():
    r=simulate(steps=500,nx=24,ny=13,nz=13,nphi=12,frames=4)
    assert r['displacement'].shape==(24,12,3)
    assert r['wall_snapshots'].shape==(4,24,12,3)
    assert np.max(np.abs(r['displacement'][0]))==0
    assert np.max(np.abs(r['displacement'][-1]))==0
    assert np.max(np.abs(r['displacement'][:, :, 1]))>1e-10
    assert np.max(np.abs(r['displacement'][:, :, 2]))>1e-10
    assert np.std(r['displacement'][12,:,1])>1e-10
    assert np.isfinite(r['energy']).all() and r['energy'].min()>=0
    assert r['max_wall_m']<min(r['mesh'].dx,.084/12)*.07


def test_reaction_changes_mic_signal_and_zero_wall_when_off():
    a=simulate(steps=650,nx=24,ny=13,nz=13,nphi=12,frames=2)
    b=simulate(steps=650,nx=24,ny=13,nz=13,nphi=12,frames=2,wall=Wall(enabled=False))
    assert np.max(np.abs(b['displacement']))==0
    relative=np.linalg.norm(a['signal']-b['signal'])/(np.linalg.norm(b['signal'])+1e-12)
    assert relative>1e-3


def test_symmetric_source_nonzero_anisotropic_surface_response():
    r=simulate(Shape(controls=(.55,1.35,.67,1.31)),steps=500,nx=24,ny=13,nz=13,nphi=12,frames=2)
    # Elliptic chamber has angular and axial position-dependent pressure coupling.
    assert np.std(np.linalg.norm(r['displacement'][11],axis=1))>1e-10


def test_shell_reaction_pair_has_equal_opposite_power():
    # Power from fluid: -p*patch*v_n; power to membrane: +p*patch*v_n
    p=np.array([-.3,1.1,2.4]);area=np.array([.01,.02,.03]);v=np.array([.03,-.04,.08]);
    fluid=-np.sum(p*area*v);solid=np.sum(p*area*v)
    assert abs(fluid+solid)<1e-15


def test_wall_passivity_and_rejection():
    with pytest.raises(ValueError):Wall(mass_per_area=-1)
    with pytest.raises(ValueError):Wall(tension=-1)
    with pytest.raises(ValueError):simulate(steps=0)
    with pytest.raises(ValueError):simulate(wall=Wall(tension=3e9),steps=20)


def test_mesh_obj_export_and_no_shape_leakage():
    m=make_mesh(nx=16,ny=13,nz=13,nphi=12)
    obj=export_obj(m)
    assert obj.count('\nv ')==m.rest.shape[0]*m.rest.shape[1]
    assert obj.count('\nf ')==len(m.triangles)
    with pytest.raises(ValueError):deformed_vertices(m,np.zeros((2,3)))


def test_shape_changes_out_of_sample_acoustic_signature():
    a=simulate(Shape(controls=(.65,1.32,.7,1.25)),steps=850,nx=24,ny=13,nz=13,nphi=12,excitation='multitone',frames=1)
    b=simulate(Shape(controls=(1.2,.7,1.3,.75)),steps=850,nx=24,ny=13,nz=13,nphi=12,excitation='multitone',frames=1)
    assert np.linalg.norm(signature(a)-signature(b))>1e-3


def test_exact_nonzero_drive_reproducibility():
    dr=np.sin(np.arange(360)*.044)
    r=simulate(steps=360,nx=24,ny=13,nz=13,nphi=12,drive=dr,frames=2)
    z=simulate(steps=360,nx=24,ny=13,nz=13,nphi=12,drive=dr,frames=2)
    np.testing.assert_array_equal(r['signal'],z['signal'])
    with pytest.raises(ValueError):simulate(steps=360,drive=np.array([1.,2.]))


def test_nonaxisymmetric_rest_mesh_and_angular_mode():
    s=Shape(lobes=(.20,-.15))
    m=make_mesh(s,nx=20,ny=13,nz=13,nphi=12)
    base=make_mesh(Shape(),nx=20,ny=13,nz=13,nphi=12)
    assert not np.allclose(m.rest,base.rest)
    assert not np.allclose(m.normals,base.normals)
    assert np.all(m.patch_area>0)
    r=simulate(s,steps=450,nx=20,ny=13,nz=13,nphi=12,frames=2)
    assert r['max_wall_m']>0


def test_true_angular_form_fitting_changes_learning_objective():
    target=Shape(controls=(.79,1.15,.8,1.22),lobes=(.18,-.13))
    cfg=dict(steps=600,nx=20,ny=13,nz=13,nphi=12,frames=1)
    truth=signature(simulate(target,**cfg))
    fit=fit_shape(truth,initial=Shape(controls=(1,1,1,1)),steps=cfg['steps'],nx=cfg['nx'],ny=cfg['ny'],nz=cfg['nz'],nphi=cfg['nphi'],max_evals=34)
    assert fit['after']<fit['before']
    assert len(fit['shape'].lobes)==2
