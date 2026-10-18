# HDR WaveForm v0.4 — 3D soft-surface acoustic learning

**Status: experimental, digitally tested, not validated on hardware or human speech.**

This is a step toward *form learns form*. A 3D triangulated membrane is coupled bidirectionally to a 3D acoustic field. Each surface vertex has independent **x, y, z displacement and velocity**; neighbouring membrane nodes are coupled by axial and circumferential springs. Sound pressure pushes each vertex along its local surface normal, and the very same local vertex velocity displaces air, altering the wave pressure in the nearby fluid voxel. Form learning changes the **rest geometry** of that physical computational system, not the dynamics equations.

## Open the project

- Offline browser simulation: [`dist/index.html`](dist/index.html), no CDN or external libraries. Drag on the 3D surface to rotate; run, pause, change elasticity, compare against a rigid wall, learn a shape, or export the six learned parameters.
- Python acoustics: [`waveform/coupled.py`](waveform/coupled.py), Numba-accelerated voxel FDTD for 3D air velocity / pressure, with a vector-valued membrane.
- Triangular membrane and OBJ export: [`waveform/mesh.py`](waveform/mesh.py).
- Reference forward acoustics and acoustic signatures: [`waveform/core.py`](waveform/core.py). The legacy rigid-domain `propagate` baseline is kept for comparisons, but **not** the v0.4 training solver.
- Synthetic inverse problem and out-of-distribution sound tests: [`scripts/benchmark.py`](scripts/benchmark.py).
- Build script: [`scripts/build.py`](scripts/build.py).

## Equations and dimensions

The air field solves linear 3D acoustics on a staggered Cartesian grid:

\[
\rho\, \partial_t\mathbf v=-\nabla p,\qquad \partial_t p=-K(\nabla\cdot\mathbf v + \mathcal B\dot{\mathbf u}) -\gamma p + s,
\]

where \(K=\rho c^2\), \(\mathbf v=(v_x,v_y,v_z)\), and \(\mathbf u_{ij}\in\mathbb R^3\) is a **true three-component 3D membrane displacement** at the \(i\)-th axial and \(j\)-th circumferential vertex. For local patch \(a_{ij}\), outward rest-surface normal \(\mathbf n_{ij}\), and matched pressure voxel \(q\), the feedback to that voxel is

\[
(\mathcal B\dot{\mathbf u})_q=\frac1{V_q}\sum_{ij\mapsto q} a_{ij}(\dot{\mathbf u}_{ij}\cdot\mathbf n_{ij}).
\]

The corresponding pressure traction is \(p_q\mathbf n_{ij}\). The pair exchanges equal and opposite instantaneous work, \(p_q a_{ij}\,\mathbf n_{ij}\cdot\dot{\mathbf u}_{ij}\), before numerical losses and discretisation errors. Each elastic vertex is coupled to its two axial and two circular neighbours; its vector dynamics obey

\[
m_a\ddot{\mathbf u}_{ij}=p_q\mathbf n_{ij}-k_a\mathbf u_{ij}-b_a\dot{\mathbf u}_{ij}+T(\Delta_x\mathbf u_{ij}+\Delta_\varphi\mathbf u_{ij}).
\]

The end rings are clamped. The tube surface has 2D material topology and is **embedded in 3D** with a properly triangulated surface; there is no assumption that all angular locations share one radial displacement. Fluid is fully 3D. This is not merely a 1D waveguide.

## Geometry learning

The rest surface has **six learnable controls**: four nonuniform axial radius controls and two angular deformation modes (cosine \(2\varphi\), sine \(3\varphi\), tapered at the ends). The angular modes create truly non-axisymmetric rest geometries. A Powell search changes these controls, reruns the coupled simulator, and measures spectral discrepancy. A separate `multitone` excitation, never used for fitting, evaluates generalisation. The algorithm does not read or directly optimize hidden target geometry.

### Measured results

On three synthetically generated, non-axisymmetric target chambers (24×13×13 fluid cells, 12 angular wall sectors, 950 acoustic steps, 48 nominal optimisation evaluations), held-out acoustic spectral MSE:

| Target | Initial MSE | Final MSE | Reduction |
|---|---:|---:|---:|
| A | 0.000558 | 0.00000490 | 99.1% |
| B | 0.001054 | 0.00007456 | 92.9% |
| C | 0.001301 | 0.00003376 | 97.4% |
| **Mean** | **0.000971** | **0.00003774** | **96.5%** |

The learned *geometric parameters do not reliably match the hidden originals*: mean absolute parameter discrepancy = **0.160** in their normalized units. An acoustic match is **not** proof of unique geometric reconstruction or topology recovery. See `results/benchmark.json` for unrounded numbers and each learned configuration. This experiment uses the **same numerical physical laws** to generate targets and to learn; success may not transfer to a different instrument, a real vocal tract or arbitrary world geometry.

## Validity limits and what remains

1. The surface is a 3D XYZ-displacement membrane, but the **air voxel occupancy remains fixed**. Moving-wall flux is linearised; a guard rejects displacements above 7% of the minimum voxel spacing. This is **not** a full arbitrary-deformation moving-mesh FSI solver, a solid tetrahedral FEM, or a strain-hardening/contact model.
2. The spring discretisation and local pressure mapping are idealisations; no measured material parameters, turbulence, glottal opening/closing or tissue-contact mechanics.
3. The angular rest geometry is constrained to two harmonics and four longitudinal controls. Mesh topology does not change during learning. The removable central rod changes fluid cross-section topology by configuration, but this **is not inferred** automatically.
4. The physical source is a synthesized chirp or multitone signal, not actual human speech, and the extracted signatures do not represent semantics or tokens.
5. Python and JavaScript solvers use the same finite-difference update and match numerically; independent physical validation, mesh convergence, identifiability, measured computational/energy costs and comparison with acoustic reconstruction baselines are future work.
6. Despite earlier analogies, this project does **not** use Box3D: Box3D solves rigid constraints, not the 3D acoustic PDE. Using a dedicated wave solver is necessary for the air pressure field. One could couple it to Box3D for rigid components later.

## Reproduce

```bash
python -m pip install -e '.[test]'
python -m pytest -q
node --test tests/test_engine.cjs
python scripts/build.py
PYTHONPATH=. python scripts/benchmark.py
python scripts/browser_smoke.py   # requires installed Chromium and Playwright
```

All runtime files are local. Python and Node required for offline reproduction; Web browser lab itself runs without servers or build tools.
