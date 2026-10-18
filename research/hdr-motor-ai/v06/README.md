# HDR Gear AI v0.6 — multi-input compound gearbox computer

**Status: research simulation.** Two independently actuated motor inputs can jointly drive a compound gear stage, forming an output that cannot be expressed as the sum of two independently calculated motor effects. Offline learning selects whole-number ratios and removes unnecessary stages. Runtime advances two motor states and reads the position of a linkage tip called `V₀` here.

**Not** a trained language model, a general-purpose AI, a physical prototype, an implementation of all-to-all HDR neighbor/group proportionality, or proof of superiority over Transformers/MLPs. Inspired by [Umriss-Motor](https://github.com/nostack-dev/kurzlernen-de/blob/main/topology.html); its actual code is **not copied or changed**.

## New architecture

Two independent inputs: unwrapped angles `α(t)` and `β(t)`. Stage `j` has signed *integer* ratios `k_j`, `l_j`, radius `r_j`, mounting angle `φ_j`, and rigid arm. The stage is constrained to

```text
θ_j = k_j α + l_j β + φ_j
```

The serial linkage's measured tip, our **candidate** `V₀`, is

\[
 V_0(\alpha,\beta)=c+\sum_j r_j e^{i(k_j\alpha+l_j\beta+\phi_j)}.
\]

Mixed stages with `k_j * l_j ≠ 0` create joint interactions. An exclusively additive collection, in which each gear is driven by only one motor, cannot create a nonzero rectangular mixed difference

\[
 f(a_1,b_1)-f(a_1,b_0)-f(a_0,b_1)+f(a_0,b_0).
\]

A practical mixer **requires a summing differential** (e.g. a planetary differential combined with signed gear reduction), not two ordinary rigid gears simultaneously connected to an incompatible common axle. The angular relationship is an idealized kinematic design target. Physical feasibility, backlash, load sharing and torque capacity must be verified on a real mechanism.

### Physical backreaction in the software simulator

The two servo input shafts each have inertia and bounded motor torque. A gear with effective rotational inertia `J_j = 0.02 r_j²` contributes

\[
M = \begin{bmatrix}J_A&0\\0&J_B\end{bmatrix} +
    \sum_j J_j\begin{bmatrix}k_j^2 & k_j l_j\\k_j l_j&l_j^2\end{bmatrix},
\qquad M\ddot{q}=\tau-\gamma\dot{q}.
\]

The off-diagonal inertia **couples motor accelerations** and represents a necessary part of physical back-driving. This is still a simplified reduced-coordinate model, not Box3D rigid-body collision or strain/contact mechanics. Numerical time integration uses a small fixed two-by-two solve, **not** an optimization of `V₀` at each input.

A fixed machine determines exactly one ideal linkage tip for any `(α,β)`. This is **not proof of existence/uniqueness of the self-consistent HDR `V₀`** described in the broader hypothesis; that stronger identification remains open.

### Learning is offline; inference is mechanical

1. Observe training pairs of input phases `(α,β)` and desired output positions.
2. Scan possible signed compound ratios `(-3..3, -3..3)` and add a useful stage using residual Fourier correlations. Refit selected coefficients by complex ridge regression. Validate and prune under a complexity penalty. **No test data is used in model selection.**
3. Freeze ratios, lengths and mount angles. During inference the two motors turn, the fixed gear constraints set stage phases, and the linkage yields `V₀`. Adding/removing stages is a deliberate *hardware reconfiguration*, **not** a per-token or per-sample computation.

This is a sparse 2D Fourier feature/structure search in a mechanical representation. The fitting algorithm and basis are known mathematics. New scientific merit would require a working embodiment, scaled coupling, demonstrated resource advantages, or an objectively superior task class.

## Run

```bash
python -m pip install -e .
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/engine.test.cjs
python scripts/build_web.py
python tests/browser_smoke.py  # local Chromium + Playwright required
PYTHONPATH=. python scripts/benchmark.py --seeds 3,7,11 --epochs 260 --out results/benchmarks.json
```

Open `dist/gear_lab.html` on phone or desktop; one self-contained HTML file, **no CDN**. Move either input independently. Choose a target (`gekoppelt`, `additiv`, `nicht harmonisch`), click **Alles lernen**, disable cross-couplings for the ablation, remove weak stages, export/import JSON. Only the training phase inspects target data.

## Benchmarks (three seeded synthetic tasks)

Each task: 520 noisy training observations, 170 noisy validation, 420 clean random held-out input pairs, 420 clean **unseen combinations** from the previously unobserved `(α > 0, β > 0)` quadrant. The model was selected only using training and validation; the hidden quadrant was not accessed until evaluation. Gaussian observation noise standard deviation 0.035 **per output channel**. Mean **complex squared error** `|pred-target|²`, *lower is better*. All values are 3-seed means; full per-run records in `results/benchmarks.json`.

| Task | Architecture | Held-out IID MSE | Unseen quadrant MSE |
|---|---|---:|---:|
| Known compound gearbox | **Compound gear (5 stages)** | **0.000039** | **0.000054** |
| | Additive gear, no cross terms | 0.220366 | 0.843844 |
| | 2-layer MLP (1282 parameters) | 0.000807 | 0.283313 |
| Additive function | Compound gear | **0.000032** | **0.000037** |
| | Additive-only gear | **0.000032** | **0.000037** |
| | MLP | 0.000888 | 0.005376 |
| Sharp nonlinear interaction | Compound gear | 0.010978 | **0.030698** |
| | Additive-only gear | 0.062779 | 0.986730 |
| | **MLP** | **0.004660** | 0.109441 |

**Interpretation:** The compound model reconstructs a sparse target deliberately generated by **the same structural family**; its extrapolation is impressive but expected from the correct inductive bias. It has a useful compactness advantage for those targets. The MLP is more accurate on IID examples from the difficult sharp task, while the periodic compound basis generalizes better into this particular missing quadrant; this is task dependent, not proof of general superiority. Compared to true analog hardware, all runtime and energy figures would be speculative.

## Quality gates

- 13 deterministic Python tests: signed ratios, two independent drives, true nonadditivity, finite torque, backreaction and positive inertia matrix, held-out structure selection and ablation.
- 10 JavaScript tests: matching coefficient identities, dynamic motors, positive inertia, learned composition and ablation, JSON validation, cross-language numeric fixture.
- Browser smoke test: actual Chromium on desktop and Android-sized viewport; no runtime errors, offline rendered graphics, slider interaction, learning, ablation, JSON export and no horizontal overflow.
- Python benchmark: fixed seeds, independent train/validation/test and locked fourth quadrant.
- If a GitHub PR has CI, **inspect actual check results** before merging, not merely local tests.

## What must be proven next

1. Actually assemble a compound planetary differential with two motor ports, angle encoders and a physical linkage. Measure gear ratio residual, backlash, motor backreaction, load-dependent tip error and steady power.
2. Demonstrate a native mechanically adjustable parameter/mixer and **local physical** learning (instead of this offline numeric fit).
3. Scale to `N` inputs without unreasonable mechanical complexity, measure physical latency, watts/inference, precision, maintenance and reproducibility against an equal-power digital baseline.
4. Formalize the uniquely self-determined HDR `V₀`, with global, group, neighbour and element relationships, and prove any claimed global fixed-point uniqueness. `V₀` here is only the directly read resultant.

MIT license for new code; no external JS libraries, WASM assets, or Box3D code copied from the Umriss-Motor.
