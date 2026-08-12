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
- `eta_sweep_aggregate.csv`: bath-retention sweep used in the decision-rule figure.
- `calibration_ece_summary.csv`: model-wise ID calibration-error summary.
- `reliability_curves.csv`: seed-level reliability curves exported from the checkpoints.
- `figures/`: earlier diagnostic plots retained with the machine-readable results.


## Information-limited collision audit and measurement-conditioned estimator

The MLST novelty revision adds two paired evaluations on the high-memory
collision family.

- `collision_ood_identifiability_{inputs,per_base,models,summary}.*` replays
  2048 fixed observable marginal sequences at 15 values of the hidden
  retention parameter in `[0.85, 0.99]`.  It reports the representation
  ambiguity diameter, the pointwise minimax absolute-error lower bound, and
  the uniform-grid Bayes MAE.
- `measurement_conditioned_hybrid_independent_rng{,_summary}.csv` compares
  stratified DFE with a constrained fusion of the marginal prior and a
  same-query DFE pilot.  Finite-label rows include independent 64-shot target
  labels and independent pilot shots used to fit the fusion weight.
- `independent_rng_label_budget.csv` verifies the marginal-only label-budget
  ordering on a split whose physical parameters and retention values use
  independent random streams.

Reproduce the independent split, audits, and figures with:

```bash
python scripts/gen_collision_independent_eta_split.py \
  --out data_mlst/collision_family_ood_independent_rng.npz
python scripts/eval_collision_ood_identifiability.py \
  --n-base 2048 --n-eta 15 --eta-min 0.85 --eta-max 0.99 \
  --n-calib 64 --calib-repeats 20 --ckpts <five-checkpoints>
python scripts/eval_measurement_conditioned_hybrid.py \
  --data data_mlst/collision_family_ood_independent_rng.npz \
  --ckpts <five-checkpoints> --budgets 4,8,16,32,48,64,96,128 \
  --n-calib 64 --n-test 4032 --label-shots 64 --repeats 5 \
  --out results_mlst/measurement_conditioned_hybrid_independent_rng.csv
python scripts/export_reliability_curves.py
python scripts/build_fresh_submission_figures.py
```

The checkpoint arguments must point to the five seed-controlled bidirectional
collision models used by the original benchmark.
