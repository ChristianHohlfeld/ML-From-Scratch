# Reproduced benchmark outputs

These are deterministic CPU runs of v0.3; exact timing depends on host.
- `group_contrast_run` (seed 7, 80 epochs, steps 55)
- `group_seed13` (seed 13, 65 epochs, steps 55)
- `group_seed23` (seed 23, 65 epochs, steps 55)
- `nonlinear_seed7` (seed 7, 80 epochs, steps 55)
- `ablation.json` from the trained seed-7 group model (edge stiffness reduced)
- `example_gears.json` = trained seed-7 gear coefficients exported as JSON

`.pt` checkpoints and CSV curves are in the downloadable ZIP; regenerate by running the documented command. These data are for synthetic classification only, not LLM performance.
