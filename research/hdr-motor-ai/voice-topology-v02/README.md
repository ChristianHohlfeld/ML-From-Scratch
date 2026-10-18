# HDR WaveForm v0.2 · Learning a 3D form from its sound

**Research status: working 3D numerical proof of concept; not measured human speech, not a validated vocal-tract reconstruction, not a soft-body fluid-structure solver or a new LLM.**

Builds on [HDR Voice Topology v0.1](../voice-topology-v01) and is informed by the mechanically generated 3D forms in [`nostack-dev/kurzlernen-de/topology.html`](https://github.com/nostack-dev/kurzlernen-de/blob/main/topology.html). The earlier project only embedded one-dimensional microphone samples into an *abstract* 3D phase space; here the **wave itself propagates through a genuine discretized three-dimensional volume**.

## Principle: form shapes wave; sound can constrain form

1. Model an adjustable three-dimensional acoustic tube by **four dimensionless, coupled axial radius controls**. Two radii per cross-section form an ellipse; the 3D fluid volume is voxelized. A coaxial solid rod is an **explicit topology change** (simply connected fluid cross-section versus an annulus).
2. Excite one end with a controlled chirp (stylized vocal-fold drive). **Solve the 3D scalar acoustic wave equation** using an explicit damped finite-difference time-domain (FDTD) update, not a 1D vocal filter, a sound-to-3D projection, Box3D or a neural approximation of physics. Walls use a simple zero-normal-neighbour-pressure-flux condition. Monitor the pressure at the other end.
3. The forward solver predicts a frequency-response signature. **Fit the form itself** by proposing modified radius controls, physically simulating again, and retaining better-fitting forms (SciPy bounded Powell search; a simple coordinate search in the browser). The solver and governing physics remain unchanged during learning.
4. **Test the learned form on a different excitation**, a multitone signal never used for fitting. The target and learner share the same simulated PDE/discretization and family of shapes, so this tests controlled inverse recovery only, not out-of-family or real-world generalization.

For fluid cells, schematically:

\[
p^{t+1}_{ijk}=(2-\gamma)p^t_{ijk}-(1-\gamma)p^{t-1}_{ijk}+c^2\Delta t^2\nabla_h^2p^t_{ijk}+s^t_{ijk}.
\]

Boundary cells outside the masked tube do not participate in the stencil. The 3D Courant condition `c² dt² (1/dx² + 1/dy² + 1/dz²) < 1` is enforced (effective CFL **0.78**). Varying wall geometry changes the acoustic response directly; there is no learned substitute for the wave equation.

### What "soft form" means here (important)

The tube is deformable **between simulations**: learning updates its three-dimensional boundary and the solver follows the new solid/fluid geometry. While a pulse propagates, the tube walls are **stationary**. Thus **this is a quasi-static shape learner, *not* a pressure-driven elastic soft-body or fully coupled fluid–structure interaction.** We must introduce shell stiffness, boundary velocity/pressure continuity and two-way energy transfer before claiming dynamically deforming tissue. Likewise, Box3D is a rigid-body constraint solver; it does not natively model compressible sound-wave propagation. Calling Box3D in place of an acoustic PDE would not make the acoustics real.

### World topology versus acoustics

Our tube-with-rod example has different cross-sectional Betti numbers: **(H₀=1, H₁=0)** for an unobstructed tube; **(H₀=1, H₁=1)** when a continuous central rod introduces an annulus. These topological properties are **known from the constructed geometry**, not reliably inferred from the recorded pressure. One microphone and a restricted frequency range do **not** uniquely specify 3D topology or vocal-tract anatomy. Inverse reconstruction can agree acoustically while differing geometrically.

**Scale**: the four radius controls are proportions and their *normalised geometry* is size-invariant, but physical acoustic resonances vary with scale. We deliberately do not conceal this distinction by normalizing away meaningful pitch/formant differences.

## Reproduce locally

Python 3.10+ and Node 20+:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
node --test tests/test_engine.cjs
python scripts/benchmark.py
python scripts/export_mesh.py
python scripts/build.py
```

Open [`dist/index.html`](dist/index.html) in your browser for the self-contained, offline interactive experiment (no CDN): 3D projected tube and a physical 2D slice of the **actual 3D pressure field**, independent form sliders, topology toggle, unknown reference forms, physically evaluated parameter fitting, signatures, waveform, playback, and exported geometry. No server required. Optional browser smoke test: `python scripts/browser_smoke.py` (requires Playwright and Chromium; **not** part of regular CI).

Running `scripts/export_mesh.py` generates OBJ files in `results/`: watertight triangular surface exports of one synthetic target and fitted tube. They are not anatomy or hardware-ready CAD. The `rod` is not exported by the mesh exporter. Large OBJ meshes are included in the downloadable ZIP but can be regenerated from the benchmark results instead of being versioned in GitHub.

## Quantitative results (synthetic only)

Full experiment and parameters: [`results/benchmark.json`](results/benchmark.json). Three *fixed* examples with distinct four-control wall profiles. Train on broadband chirp; test on a separate multitone. Baseline: **same 3D FDTD solver with an untrained uniform-radius tube**, so improvements isolate *fitting*, not a model-family advantage.

| Synthetic shape | Untrained held-out L2 | Fitted held-out L2 | Relative improvement |
|---|---:|---:|---:|
| A: wide/narrow | 0.10199 | 0.02936 | **71.2%** |
| B: subtle shape | 0.01370 | 0.01022 | **25.4%** |
| C: alternating | 0.13849 | 0.03921 | **71.7%** |
| **Mean** | — | — | **56.1%** |

The mean absolute error of the **underlying four normalized geometry controls is ~0.210**: the measured sound can match substantially better even when the recovered physical geometry is wrong. In particular, the C case illustrates ambiguity and local minima. The optimizer is finite-budget (at most 140 full FDTD evaluations per target), and those values are not claims of exact reconstruction.

### Verification

- `python -m pytest -q`: **12 tests**, including a genuine **Python-versus-JavaScript equality test of the 3D wave solver**, finite pressure response, 3D CFL safety, deterministic source, explicit topology alteration, independent unseen source and reduced acoustic error.
- `node --test tests/test_engine.cjs`: **7 tests**.
- Chromium desktop/mobile smoke: preset switches, non-inert learner control, measurable improvement, no horizontal overflow, no page errors.
- Every new shape is actually discretized and solved; there is no use of the target form itself in the inverse procedure.

## What would constitute a genuine breakthrough?

1. Reconstruct deformable **real** vocal-tract shapes from synchronized 3D imaging and audio, with **speaker-disjoint** held-out tests; compare established inverse-acoustic tract models and spectral baselines.
2. Upgrade to **two-way moving elastic walls** driven by pressure with physically justified tissue stiffness and damping, plus independent validation of mass/energy stability.
3. Learn topology (number of connected components, holes and surfaces) and changing boundaries from **multiple spatial observations** instead of specifying them by hand.
4. Couple many wave-defined shapes to learn stable composition/containment/relations in the world. Then examine whether those structures support geometric reasoning and eventually language.

The present prototype provides an experimentally falsifiable bridge from "sound as 3D trajectory" toward **"sound is generated by form, and form is learned from how sound propagates."**

MIT license. Third-party dependencies retain their own licenses. The earlier `nostack-dev/kurzlernen-de/topology.html` remains untouched.
