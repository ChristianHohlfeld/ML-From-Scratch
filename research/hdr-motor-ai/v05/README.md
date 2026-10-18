# HDR Gear AI v0.5 – Structural Gear Computer

**Status:** Experimental analog-computing architecture inspired by the [Umriss-Motor](https://github.com/nostack-dev/kurzlernen-de/blob/main/topology.html) and Hohlfeld's HDR proportional-coupling idea. **Not** an established new general-purpose AI, a hardware build, or an alternative to a pretrained transformer.

## What changed from v0.4

v0.4 trained virtual spring constants around a passive rotor. v0.5 instead treats the **topology of a mechanical machine as the model**. A single common shaft drives integer-ratio gears with learned radii and mounting phases. A serial chain of arms sums their displacements. The unique measured pen/resultant position is the experimental output `V₀`:

\[
\phi(t) = \text{shaft angle},\quad \theta_i=k_i\phi+\alpha_i,\quad
V_0(\phi)=c+\sum_{i=1}^m r_i\exp(i\theta_i),\qquad k_i\in\mathbb Z\setminus\{0\}.
\]

This formula is used to **fit and verify** the simulator. In the intended physical mechanism, a single shaft and gearing **embody** the relation; the resultant pen location is observed, not found by a runtime `argmin`/global optimizer. The web simulator still uses numerical integration and trigonometry, and hardware has not been built.

- **Learning phase:** Residual-driven search selects and removes integer-ratio gear stages. Coefficients/phase are refitted *offline* whenever the structure changes; a separate validation set and complexity penalty select the final structure.
- **Execution phase:** Fixed gear structure, finite-torque motor dynamics, no structure search or re-optimization at each input.
- **Dynamic V₀:** Unique **resultant/output** for a given shaft state; shifts automatically when the motor turns. This is a **candidate mechanization**, *not* proof that it is identical to the uniquely determined, fully coupled `V₀` proposed in the HDR paper.
- **Inputs:** One continuous scalar/phase input; this demonstrator **does not** yet compute general multidimensional input-to-output functions. More independent inputs require mechanical input ports and a demonstrably adequate architecture.
- **Gears:** Ideal integer ratio constraints and linkages; physical build must handle backlash, compliance, collision, energy, and manufacturing tolerances.

## Use

- Open `dist/gear_lab.html` directly in a browser: no CDN, no server, no installation. The goal can be selected or drawn. Try **Eine Stufe lernen**, **Bis zum Stopp**, **Stufe entfernen**, play/pause, and **JSON sichern**.
- Python package requires NumPy: `python -m pip install -e .`.
- Run tests: `PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py' -v` and `node --test tests/engine.test.cjs`.
- Rebuild standalone web: `python scripts/build_web.py`.
- Benchmarks (optional PyTorch baseline): `PYTHONPATH=. python scripts/benchmark.py --out results/v05 --seeds 3,7,11 --epochs 600`.
- Browser regression via installed Playwright + Chromium: `python tests/browser_smoke.py` (uses inline page injection, no network).

## Evaluation, held-out clean test MSE (lower is better)

| Task | Seed | Gear AI (selected stages) | Dense signed Fourier (18 stages) | Small MLP |
|---|---|---:|---:|---:|
| known mechanical function | 3 | 0.000048 (4) | 0.000359 | 0.001086 |
| known mechanical function | 7 | 0.000090 (4) | 0.000312 | 0.001211 |
| known mechanical function | 11 | 0.000053 (4) | 0.000344 | 0.001577 |
| sharp/nonharmonic function | 3 | 0.003504 (6) | 0.004089 | 0.002411 |
| sharp/nonharmonic function | 7 | 0.003527 (5) | 0.004629 | 0.001840 |
| sharp/nonharmonic function | 11 | 0.003587 (5) | 0.005581 | 0.003368 |

Dataset: 80 noisy training phases, 80 noisy validation phases, 200 clean held-out test phases, Gaussian noise σ=0.035. A three-seed synthetic study; task mechanically generated from a sparse gear series naturally favors this architecture. Validation is part of model selection; test is separate. No physical performance, energy consumption, or wall-clock superiority has been established. Results and full configs in `results/v05/`.

**Crucial limitation:** The structure learner is a regularized sparse harmonic regression (Fourier-like pursuit). This known technique is not claimed to be a new mathematical discovery. Novelty, if any, would need to come from the *physical embodiment*, scalable couplings and experimentally demonstrated efficiency/generalization. In particular, the current chain is **not** the user's fully coupled element↔neighbor↔group↔global HDR relation; it is an executable, falsifiable first mechanical specialization.

## Research next gate

1. Generalize to multiple independent mechanical input ports and demonstrate tasks with genuinely nonseparable interactions; compare to compact conventional networks using matched parameter and energy budgets.
2. Measure backlash, torque, stiffness, damping, inertia, errors and energy on an actual 3D-printed mechanism rather than simulated perfect gears.
3. Define a **mechanically constrained learning actuator** (shifting gears/phase/geometry) and measure learning energy. The current structure update runs externally in Python/JS and is not physically learned.
4. Define the stronger HDR `V₀` unambiguously and prove existence/uniqueness under topology changes; the present pen/resultant has a unique output for a given motor angle but is not such a fixed point.

## Integrity

- No changes to `nostack-dev/kurzlernen-de/topology.html`. Its code provides the conceptual, independently verified mechanical precedent.
- Offline synthesis and runtime are intentionally separated; no `argmin` or `fit_structure()` inside `Mechanism.step()`.
- Python, JS and browser tests cover geometry, signed ratios, dynamic shaft state, structural learning and reproducible packaging.

MIT license for new code in this repository. Box3D and Three.js are **not bundled**; attribution to the original Umriss-Motor is for inspiration only, not a copy of its 1.6MB self-contained file.