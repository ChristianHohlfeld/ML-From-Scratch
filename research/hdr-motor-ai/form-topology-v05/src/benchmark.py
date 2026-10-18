"""HDR form/topology falsification benchmark.

Closed triangle-mesh tests. True topology labels are computed from combinatorics,
not fitted from a picture or introduced as machine features.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.spatial.transform import Rotation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

CLASSES = ('sphere', 'torus', 'two_spheres', 'dumbbell')

def base_mesh(label: str, detail: int = 0) -> trimesh.Trimesh:
    if label == 'sphere':
        return trimesh.creation.icosphere(subdivisions=1 + detail)
    if label == 'dumbbell':
        m = trimesh.creation.icosphere(subdivisions=1 + detail)
        x = m.vertices.copy()
        factor = .33 + .9 * np.abs(x[:,0]) ** 1.5
        x[:,1:] *= factor[:,None]
        x[:,0] *= 1.6
        m.vertices = x
        return m
    if label == 'torus':
        return trimesh.creation.torus(major_radius=1.2, minor_radius=.40,
                                     major_sections=12 + 4 * detail, minor_sections=8 + 4 * detail)
    if label == 'two_spheres':
        a = trimesh.creation.icosphere(subdivisions=1 + detail, radius=.64)
        b = a.copy()
        a.apply_translation([-1.03, 0, 0])
        b.apply_translation([1.03, 0, 0])
        return trimesh.util.concatenate((a, b))
    raise ValueError(label)

def equalize_vertices(mesh: trimesh.Trimesh, target: int = 96, seed: int = 0) -> trimesh.Trimesh:
    """Split mesh edges without changing connected components, genus or boundaries.

    Each split adds one vertex and two triangles in a closed 2-manifold.
    Use controlled random edge choice for a distinct triangulation per specimen.
    """
    rng = np.random.default_rng(seed)
    vertices = np.asarray(mesh.vertices, dtype=float).tolist()
    faces = np.asarray(mesh.faces, dtype=int).tolist()
    while len(vertices) < target:
        ef = np.asarray(faces, dtype=int)
        ea = np.vstack((ef[:,[0,1]],ef[:,[1,2]],ef[:,[2,0]]))
        eu = np.unique(np.sort(ea,axis=1),axis=0)
        vv = np.asarray(vertices)
        lengths = np.linalg.norm(vv[eu[:,0]]-vv[eu[:,1]],axis=1)
        # Prefer long springs while permitting a different triangulation each run.
        weights = (lengths/np.max(lengths))**6
        weights /= weights.sum()
        k = int(rng.choice(len(eu),p=weights))
        a,b=map(int,eu[k]); mid=len(vertices)
        vertices.append(((vv[a]+vv[b])*.5).tolist())
        new_faces=[]
        touched=0
        for face in faces:
            if a in face and b in face:
                touched += 1
                ia=face.index(a); ib=face.index(b)
                other=next(v for v in face if v!=a and v!=b)
                if (ia+1)%3 == ib:
                    new_faces.extend([[a,mid,other],[mid,b,other]])
                else:
                    new_faces.extend([[b,mid,other],[mid,a,other]])
            else:
                new_faces.append(face)
        if touched != 2:
            raise ValueError('Expected closed manifold, got edge incidence '+str(touched))
        faces=new_faces
    m=trimesh.Trimesh(vertices=np.asarray(vertices),faces=np.asarray(faces),process=False)
    assert m.is_watertight and m.euler_number==mesh.euler_number
    return m

def perturb_mesh(mesh: trimesh.Trimesh, seed: int, level: float = 1.) -> trimesh.Trimesh:
    """Continuous invertible-ish small warp; faces are NEVER modified."""
    rng = np.random.default_rng(seed)
    m = mesh.copy()
    x = m.vertices.copy().astype(float)
    # Affine warp with positive determinant, plus smooth nonrigid field.
    rotation = Rotation.random(random_state=rng).as_matrix()
    anis = rng.uniform(.78, 1.22, 3)
    k = rng.uniform(-.12, .12, 3) * level
    y = x * anis
    y[:, 0] += k[0] * np.sin(1.3 * x[:, 1] + .4 * x[:, 2])
    y[:, 1] += k[1] * np.sin(1.1 * x[:, 2] - .6 * x[:, 0])
    y[:, 2] += k[2] * np.sin(1.4 * x[:, 0] + .3 * x[:, 1])
    y = y @ rotation.T
    y *= rng.uniform(.45, 2.5)
    y += rng.normal(0, 2.0, 3)
    m.vertices = y
    return m

def topology(mesh: trimesh.Trimesh) -> dict:
    # For triangulated closed oriented surfaces beta1 = 2*c - chi (beta2=c).
    parts = mesh.split(only_watertight=False)
    c = len(parts)
    chi = int(mesh.euler_number)
    return dict(components=c, euler=chi, first_betti=2*c-chi,
                watertight=bool(mesh.is_watertight))

def edges(mesh):
    return np.asarray(mesh.edges_unique, dtype=np.int32)

def wave_features(mesh: trimesh.Trimesh, steps=160, seeds=4, shuffle_edges=False, rng_seed=0) -> np.ndarray:
    """Damped elastic waves on actual triangulated 3D surface springs.

    m u'' + gamma u' = - K u + f(t), where K is a weighted graph
    spring Laplacian. Changes in 3D edge lengths alter spring constants.
    Only time traces of wave energy/response are read out; no label is used.
    """
    x = np.asarray(mesh.vertices, dtype=float)
    n = len(x)
    x = x - x.mean(axis=0)
    x /= max(np.linalg.norm(x, axis=1).max(), 1e-9)
    e = edges(mesh).copy()
    if shuffle_edges:
        # Same vertex count and geometry, but destroys original neighborhood topology.
        rng = np.random.default_rng(rng_seed)
        p = rng.permutation(n)
        e = p[e]
    lengths = np.linalg.norm(x[e[:,0]]-x[e[:,1]],axis=1)
    # Spring connections with normalized geometric rest lengths.
    w = np.clip(np.median(lengths) / np.maximum(lengths, 1.e-6), .2, 4.)
    a = sp.coo_matrix((np.r_[w, w], (np.r_[e[:,0], e[:,1]],
                                      np.r_[e[:,1], e[:,0]])), shape=(n,n)).tocsr()
    degrees = np.asarray(a.sum(axis=1)).reshape(-1)
    lap = sp.diags(degrees) - a
    # Stable under vertex-degree changes and mesh refinement.
    lap = sp.diags(1.0 / np.sqrt(np.maximum(degrees, 1.e-8))) @ lap @ sp.diags(1.0 / np.sqrt(np.maximum(degrees, 1.e-8)))
    # Probes: deterministic invariant-ish landmark selection, does not use label.
    # Four excitations, chosen by maximal geometric distance from previous probes.
    probes = [0]  # deterministic landmark under translation, rotation, and scale
    for _ in range(seeds - 1):
        mind = np.min([np.linalg.norm(x-x[q],axis=1) for q in probes], axis=0)
        probes.append(int(np.argmax(np.round(mind, 10))))
    obs = []
    dt = .22
    for source in probes:
        u = np.zeros(n); v = np.zeros(n)
        u[source] = 1.0
        energies = []
        for t in range(steps):
            # Symplectic Euler with viscous damping; medium propagation by real mesh edges.
            v += dt * (-3.2*(lap @ u) - .12*v)
            u += dt * v
            if t%4 == 0:
                energies.append(float(np.mean(u*u)))
        energies=np.asarray(energies)
        # Bands of response power and time-resolved coherent field.
        p = np.abs(np.fft.rfft(energies-energies.mean()))**2
        bands = [p[a:b].sum() for a,b in [(1,3),(3,6),(6,11),(11,21)]]
        obs.append([*np.log1p(np.array(bands)*100),
                    *np.log1p(energies[np.rint(np.linspace(0,len(energies)-1,8)).astype(int)]*100)])
    obs = np.asarray(obs)
    return np.r_[obs.mean(axis=0),obs.std(axis=0)]

def geometry_features(mesh):
    """Coordinate shape baseline: rotation/scale-invariant radius and pairwise-distance summaries."""
    x = np.asarray(mesh.vertices, dtype=float)
    x = x-x.mean(axis=0)
    norm = np.sqrt(np.mean(np.sum(x*x,axis=1)))
    x /= max(norm,1.e-10)
    radii = np.linalg.norm(x,axis=1)
    rng = np.random.default_rng(1122)
    a = rng.integers(0,len(x),size=4000); b=rng.integers(0,len(x),size=4000)
    d = np.linalg.norm(x[a]-x[b],axis=1)
    cov = np.linalg.eigvalsh(np.cov(x.T))
    return np.r_[np.quantile(radii,np.linspace(0,1,17)),
                 np.quantile(d,np.linspace(0,1,17)),cov]

def run(n_per_class=24, seed=20261018, out=None, steps=160):
    labels = []; splits=[]; waves=[]; shuffled=[]; geometry=[]; tops=[]; meshes=[]
    for i,label in enumerate(CLASSES):
        for j in range(n_per_class):
            # Held-out test uses unseen randomly generated transforms and perturbations.
            is_test=j>=n_per_class//2
            m=perturb_mesh(equalize_vertices(base_mesh(label),target=96,seed=seed+10000*i+j+1),seed+10000*i+j+1,level=1.0 if not is_test else 1.3)
            assert m.is_watertight
            t=topology(m)
            labels.append(label); splits.append('test' if is_test else 'train')
            geometry.append(geometry_features(m))
            waves.append(wave_features(m,steps=steps))
            shuffled.append(wave_features(m,steps=steps,shuffle_edges=True,rng_seed=seed+123*i+5*j))
            tops.append(t); meshes.append(m)
    labels=np.asarray(labels)
    train=np.asarray(splits)=='train'; test=~train
    outobj={'experiment':'HDR 3D surface spring-wave vs invariants',
            'synthetic':True,'seed':seed,'n_per_class':n_per_class,
            'train':int(train.sum()),'test':int(test.sum()), 'classes':list(CLASSES),
            'mesh_invariants':{label:topology(base_mesh(label)) for label in CLASSES},'matched_vertices':96,
            'metrics':{},'confusions':{},'note':'Accuracy here only means classifying four generated closed mesh families under geometric warps; not full geometric reasoning, language comprehension, or arbitrary topology.'}
    for name,x in [('coupled_wave',np.asarray(waves)),('shuffled_wave_edges',np.asarray(shuffled)),('geometry_baseline',np.asarray(geometry))]:
        model=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=3))
        model.fit(x[train],labels[train]); pred=model.predict(x[test])
        outobj['metrics'][name]=float(np.mean(pred==labels[test]))
        outobj['confusions'][name] = {lab:{predlab:int(np.sum((labels[test]==lab)&(pred==predlab))) for predlab in CLASSES} for lab in CLASSES}
    topo_labels=[(z['components'],z['first_betti']) for z in tops]
    ref={}
    for k in np.where(train)[0]:
        ref.setdefault(tuple(topo_labels[k]),labels[k])
    pred=np.array([ref.get(tuple(topo_labels[k]),'unknown') for k in np.where(test)[0]])
    outobj['metrics']['exact_topology_baseline']=float(np.mean(pred==labels[test]))
    # A precise but honest note: labels are defined by topology, so this baseline is intentionally trivial.
    outobj['limitations']=['Sphere and dumbbell have exactly the same topological invariants, so an exact-topology-only baseline necessarily confuses them.',
        'A spring-wave sensor on the mesh is not a volumetric soft-body acoustic fluid simulation.',
        'A classifier downstream of the wave network is still ordinary machine learning.',
        'All shapes are procedurally generated; no scanned real-world objects.',
        'Edges are randomly split to an equal 96 vertices per shape; continuous warps preserve topology. No occlusion or noisy real measurements.']
    if out:
        Path(out).parent.mkdir(parents=True,exist_ok=True)
        Path(out).write_text(json.dumps(outobj,indent=2,ensure_ascii=False)+'\n')
    return outobj

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--samples',type=int,default=24)
    parser.add_argument('--steps',type=int,default=160)
    parser.add_argument('--output',default='results/benchmark.json')
    args=parser.parse_args()
    print(json.dumps(run(n_per_class=args.samples,out=args.output,steps=args.steps),indent=2))
