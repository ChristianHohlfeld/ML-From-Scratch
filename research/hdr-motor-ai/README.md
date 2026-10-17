# HDR Motor AI v0.4 · Local Physical Learning Lab

**Status:** Experimental numerical implementation of a *physically inspired* analog-computing architecture, with trainable mechanical couplings and a passive reference rotor `V₀`. This is an independent research prototype; it is **not** a hardware build, a language model, or an established improvement over Transformers.


## v0.4: mechanism learns locally, not through backpropagation

**Core contribution**: supervised two-phase equilibrium propagation (EP) applied to the SAME elastic gear network as v0.3. At inference, the motor inputs drive the rotating shafts, and V₀ is entirely undriven. During **training only**, a weak physical teaching spring is temporarily coupled to V₀. Every elastic connection changes its own stiffness, cubic coefficient and gear ratio based exclusively on **the change in strain at its two endpoints between free and nudged phases**. There are no global gradients or BPTT used for this local learning; the phase dynamics are still numerically simulated.

This approach is related to established equilibrium propagation / contrastive physical-learning literature; **it is not claimed as a new discovery of EP**. This project evaluates its suitability for the HDR-inspired all-to-all/neighbour/group/V₀ gear topology. No evidence of hardware energy or speed advantage is claimed.

**Exact two-phase rule**, link `a–b`, channel `c`:

```text
e = θ_a + r θ_b                       (local link extension)
E_link = ½ k e² + ¼ q e⁴             (local passive energy)
free:       inertia dynamics → θ⁰    (V₀ no drive)
nudged:     same dynamics + β(t−θ_V₀) torque at V₀ (training only)
Δ log k = −η (k/2β) (e_β² − e_0²)
Δ log q = −η (q/4β) (e_β⁴ − e_0⁴)
Δ z     = −η/β [((k e_β + q e_β³) θ_b,β − (k e_0 + q e_0³) θ_b,0) ∂r/∂z]
r = sign ⋅ [1 + span ⋅ tanh(z)]
```

This is an approximation of the equilibrium gradient for *sufficiently relaxed* phases and small teaching strength `β`. Finite numbers of dynamic steps and finite `β` introduce bias. The controller also needs to provide the teacher value `t`; there is no claim that the apparatus learns without supervised feedback. Tests compare the local formulas to static partial derivatives calculated by autograd *only in tests*, and verify no `p.grad` is generated during learning.

### Experiments / reproducibility

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests
node tests/browser_engine.test.cjs
python scripts/build_web.py --check
python -m hdr_motor_ai.local_train --task group_contrast --epochs 12 --output results/v04/group_seed7
python -m hdr_motor_ai.local_train --task group_contrast --epochs 12 --no-learning --output results/v04/group_control
python -m hdr_motor_ai.local_train --task nonlinear_relation --epochs 28 --train-samples 512 --baseline-epochs 28 --free-steps 300 --nudged-steps 300 --output results/v04/nonlinear_seed7
PYTHONPATH=. python scripts/calibration_study.py --seed 2 --epochs 50 --lr 1 --steps 260 --output results/v04/calibration
PYTHONPATH=. python scripts/calibration_study.py --seed 3 --epochs 32 --lr 1 --steps 260 --output results/v04/calibration_seed3
PYTHONPATH=. python scripts/calibration_study.py --seed 4 --epochs 32 --lr 1 --steps 260 --output results/v04/calibration_seed4
```

**Calibration**: fixed randomly distorted teacher mechanism generates target V₀ angles; student sees motor commands and supervised teacher readout, but never teacher's gear settings. Inputs are split train/validation/test, with checkpoints chosen by *validation* MSE.

| Calibration seed | Untuned held-out MSE | After local EP | Error reduction |
|---|---:|---:|---:|
| 2 | 0.0083785 | 0.0019927 | **76.2%** |
| 3 | 0.0037739 | 0.0022851 | **39.4%** |
| 4 | 0.0037875 | 0.0024136 | **36.3%** |

**Hard controls, same simple classification task** (group contrast, seed 7): local EP: **99.2%**, *never train any link* control: **99.2%**, baseline MLP: **99.6%**. Thus this task provides **no demonstrated learning gain**. Nonlinear relation task: local EP: **43.8%**, MLP: **92.2%** — a substantial failure for current mechanical representation.

The local calibration experiment establishes learnability on a controlled synthetic physical system, **not** language understanding, reasoning, compression, low energy consumption or superiority to Transformers. Numerical simulation is much slower than the MLP here; local learning is measured in tens of CPU seconds against milliseconds for an MLP. Cross-seed variability is material.

**Browser:** `dist/motor_lab.html` includes a new *Lokale Lernphase* panel. Moving a slider changes a motor's target; pressing *Lokal lernen* teaches a chosen desired V₀ angle using the JS implementation of two-phase local EP. It does **not** reset the ongoing rotor positions and velocities. Teacher force is applied only to an isolated teaching-phase copy. The browser demo trains its first physical channel; Python EP study uses one channel for an unambiguous V₀ sensor. Previous 3-channel v0.3 presets remain usable.

**Known limitations / falsifiability**: supervised signal must arrive at V₀, two relaxation phases are costly, no real on-device adjustable compliant gears exist, initialization can already solve easy problems, stable time-step selection matters, and the current topology cannot yet solve nonlinear XOR-like tasks. Next meaningful frontier: build and instrument a physical 2–3-rotor elastically coupled prototype and compare *measured* total power/latency against CPU simulation with equal accuracy targets.

---

## Idea

**Do not re-solve the whole system when input data changes.** Encode input variables in the target angles of actuated rotor shafts. Continuous torque propagation through *elastic, proportional, signed gearing* changes neighbouring rotors, local groups and the unique central reference rotor `V₀`. **Read V₀ directly** as a physical sensor. It is never externally actuated and is not recalculated by `argmin` at runtime.

```text
                       learnable pair connections (every↔every)
   motors M1,M3,M5  <===========================> motors M2,M4,M6
         |                                                       |
         | learnable group / neighbour links                      |
         v                                                       v
    passive group A                                       passive group B
         \                                                       /
          \                   elastic gears                      /
           +-------------------- V₀ ----------------------------+
                                  |
                             sensor readout
```

The current topology contains four simultaneously active relation classes:

1. All-to-all pair links, `n(n−1)/2` connections.
2. Ordered neighbour links, `n−1` connections, independent of all-to-all links.
3. One motor→group link per motor (alternating group membership in the prototype).
4. Group→V₀ links, with an opposite orientation for the second group.

This is a multi-channel generalized torsion network; the simulated gears are compliant, **not** rigid meshing pinions. Elasticity avoids overconstraining closed cycles.

## Dynamics (no algebraic V₀ solver)

For each link `i–j` and channel `c`:

```text
e_ij,c    = θ_i,c + r_ij,c θ_j,c
U_ij,c    = ½ k_ij,c e_ij,c² + ¼ q_ij,c e_ij,c⁴
torque_i  = -∂U/∂θ_i
I θ̈_i,c  = torque_i,c - b θ̇_i,c + motor_torque_i,c
motor_torque_i,c = km (gain_i,c * input_i + bias_i,c - θ_i,c) [only motors]
V₀ output = readout_bias + Σ_c readout_weight_c * θ_V₀,c
```

`V₀` has **no motor torque**. The dynamics are integrated by a fixed-step semi-implicit Euler method; previous angle and velocity are retained while targets change. Positive `k`, `q`, `b` and `I` give a nonnegative energy function and dissipative continuous-time system. The selected graph plus motor grounding makes the **linear static stiffness matrix positive definite**, so a unique static equilibrium exists for constant inputs; nevertheless the **actual behavior is transient**, is numerically stepped rather than directly solved, and may take time to settle.

**Important nuance:** The ratio and stiffness can be learned. The v0.3 path trains them digitally with backpropagation; the **v0.4 local EP** alternative uses two mechanical phases with a teaching signal and no gradients through the whole network. Both are simulated digitally; no physical self-adjusting hardware is built. These learned constants are *mechanically interpretable* designs for a future prototype.

## Run it

### 1. Offline browser, no install / CDN

Open [`dist/motor_lab.html`](dist/motor_lab.html) in Chrome, Safari, Firefox or mobile. It is a **single self-contained HTML file** with the identical JS physics engine used in parity tests. Its **Erlerntes Preset** button loads a real trained `group_contrast` model embedded in the HTML. Drag motor target sliders; click connection lines to adjust `k` and `|r|` live; add/remove motors while preserving existing shaft angles and angular velocities. JSON import/export supported.

Manual slider changes are physical adjustments. The new local teaching panel can also update the individual couplers in the browser from two simulated dynamical phases, with no BPTT. When a new motor is added, only new couplings receive default constants; all retained couplings keep their existing learned values.

### 2. Train / compare

```bash
python -m pip install -e '.[test]'
python -m hdr_motor_ai.train --task group_contrast --seed 7 --epochs 80 --steps 55 --output runs/group
python -m hdr_motor_ai.train --task nonlinear_relation --seed 7 --epochs 80 --steps 55 --output runs/nonlinear
python scripts/diagnose.py --run runs/group --output runs/group_ablation.json
cp runs/group/motor_gears.json results/example_gears.json
python scripts/build_web.py
python scripts/build_web.py --check
```

Outputs: best PyTorch checkpoints (`.pt`) for each model, training CSV curves, held-out test metrics JSON, and a browser-loadable `motor_gears.json`. The ZIP release includes the trained checkpoint; source repository versions are reproducible with the above command.

Three models are compared on **identical synthetic datasets**:

- **motor** — 3-channel, all-to-all plus group+neighbour elastic machinery, read only at V₀.
- **mlp** — conventional small fully-connected MLP (no relational mechanical constraints).
- **attention** — pedagogical small single-head attention model, **not** a modern optimized Transformer.

The train, validation and test datasets use separate pseudorandom seeds. Checkpoints are chosen using *validation loss only*, never test accuracy.

### 3. Test it

```bash
python -m pytest -q tests
node tests/browser_engine.test.cjs
python scripts/build_web.py --check
```

Python tests cover force as the negative energy derivative, mechanical dissipation, the absence of drive torque on V₀, its passive response, state continuity, uniqueness of the linear equilibrium, positive physical parameters, gradients and **Python↔JS simulation parity**. JS tests cover state conservation across add/remove, energy, link tuning and malicious JSON rejection. The optional Chromium/Playwright test verifies desktop and mobile UI (and saves screenshots). All tests run without servers or external URLs; browser test skips when Playwright or Chromium is missing.

## Actual measured results

> **NOT a general-AI evaluation.** Results are short synthetic classification tasks, intentionally tiny and easy enough for controlled comparisons; single-core CPU timing depends on host and should not be treated as hardware-independent. Do not infer that HDR outperforms Transformers or that mechanical computing is intrinsically faster.

| Task / seed | Motor test accuracy | MLP | 1-head attention | Motor training time |
|---|---:|---:|---:|---:|
| group contrast / 7 (80 epochs) | **99.80%** | 99.41% | 81.25% | 22.42 s |
| group contrast / 13 (65 epochs) | 99.22% | 99.22% | 96.48% | 19.46 s |
| group contrast / 23 (65 epochs) | 98.63% | 99.41% | 95.31% | 19.37 s |
| nonlinear relation / 7 (80 epochs) | **66.02%** | **92.58%** | 68.55% | 24.22 s |

Ablation for the first run, using the fixed trained model and the *same held-out test set*, with removed edge classes replaced by near-zero stiffness:

| Test | Accuracy |
|---|---:|
| intact V₀ mechanical network | **99.80%** |
| without V₀ hub connections | 53.32% |
| without motor→group connections | 65.04% |
| without all-to-all pair connections | 98.63% |
| without separate neighbour connections | 99.80% |
| fixed 1:1 ratio (learned ratios reset only) | 99.61% |

**What these numbers mean:** the group contrast task is primarily solved through **group↔hub physical coupling**. This particular task barely benefits from the pair/neighbor cross-couplings; that is an *unfavourable* finding for any claim that all-to-all proportionality automatically makes it superior. The nonlinear task exposes current limitations (mechanical capacity, readout, settling/training). Further benchmarks and redesigned topologies are necessary.

All raw metrics are under `results/`, including `ablation.json`, `group_seed*/summary.json`, `nonlinear_seed7/summary.json`. A reference **pretrained mechanical parameter JSON** is at `results/example_gears.json` and is embedded in `dist/motor_lab.html`.

## What would make this a genuinely new AI architecture?

1. Hardware or efficient analog emulation whose physical relaxation computes a useful task at lower **end-to-end** energy/latency than ordinary numerical linear algebra.
2. A mechanism for *local plasticity* of stiffness/ratios while preserving passive/stable dynamics (current backprop is offline and nonlocal).
3. Successful, held-out **nonlinear and long-horizon** benchmarks, and parameter-/energy-/latency-matched comparisons against conventional models.
4. A mathematical definition and falsifiable benefit for the special self-consistent `V₀` beyond being a passive shared reference rotor.

This is a real, falsifiable first engineering step, not a validation of HDR's original compression claims; **compression is deliberately out of scope**.