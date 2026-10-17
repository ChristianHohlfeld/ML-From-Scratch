# v0.4 independent benchmark results
All tasks use disjoint PRNG train / validation / test sets. Calibration models use a hidden synthetic teacher mechanism, not real measured hardware. Selection is based on held-out validation. Main research README gives exact reproduction commands.

| Calibration seed | Untrained test MSE | Local EP test MSE | Relative reduction |
| --- | ---: | ---: | ---: |
| 2 | 0.0083785 | 0.0019927 | 76.2% |
| 3 | 0.0037739 | 0.0022851 | 39.4% |
| 4 | 0.0037875 | 0.0024136 | 36.3% |

- Easy group task local EP = 99.22%, UNTRAINED control = 99.22%, MLP = 99.61% — no established learning gain.
- Hard nonlinear task local EP = 43.75%, MLP = 92.19% — architecture fails current benchmark.
- CPU simulation runs substantially slower than MLP. Physical energy and latency not measured.
