# HDR Motor AI v0.3 · Physical Learning Lab

**Status:** Experimental numerical implementation of a *physically inspired* analog-computing architecture, with trainable mechanical couplings and a passive reference rotor `V₀`. This is an independent research prototype; it is **not** a hardware build, a language model, or an established improvement over Transformers.

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

**Important nuance:** The ratio and stiffness can be learned, but the current **training procedure is digital** (backpropagation through simulated mechanical time). There is no claimed purely mechanical on-device learning mechanism or physically fabricated tunable gear yet. These learned constants are *mechanically interpretable* designs for a future prototype.

## Run it

### 1. Offline browser, no install / CDN

Open [`dist/motor_lab.html`](dist/motor_lab.html) in Chrome, Safari, Firefox or mobile. It is a **single self-contained HTML file** with the identical JS physics engine used in parity tests. Its **Erlerntes Preset** button loads a real trained `group_contrast` model embedded in the HTML. Drag motor target sliders; click connection lines to adjust `k` and `|r|` live; add/remove motors while preserving existing shaft angles and angular velocities. JSON import/export supported.

Manual changes made in the browser are **physical adjustments**, not online gradient training. When a new motor is added, only new couplings receive default constants; all retained couplings keep their existing learned values.

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