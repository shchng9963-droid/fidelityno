# MLST revision results

All files in this directory were generated on 2026-08-06 in the isolated server
worktree `fidelityno_mlst_revision_20260806` using Python 3.11.15, NumPy 2.4.6,
SciPy 1.17.1, and PyTorch 2.12.0.

Key outputs:

- `exact_composition_optimized.csv`: deterministic composition error and CPU latency.
- `collision_family_ood_label_budget.csv`: identical-label-budget baselines.
- `dfe_*_stratified.csv`: stratified DFE at 2,000--200,000 shots.
- `dfe_family_ood_low_shot.csv`: 4--1,024 total-shot crossover.
- `noisy_ood_calibration.csv`: five measurement repeats for finite-shot labels.
- `noisy_ood_calibration_summary.csv`: per-checkpoint summary.
- `figures/`: plots generated only from the CSV files above.

The collision filename `family_ood` is retained for compatibility and means a
memory-strength shift in eta, not a different functional channel family.
