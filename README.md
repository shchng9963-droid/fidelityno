# FidelityNO: cost--accuracy audits for composed-channel fidelity

## Revision evaluation (2026-10-02)

The latest technical follow-up is documented in
`revision/REPRODUCE_INDEPENDENT_WEIGHTS.txt` and fixed by `revision/stage5_design.json`.
It replaces outcome-dependent fusion variances with a calibration-derived proxy,
at unchanged 4096-shot calibration cost. All validation budgets are recomputed
before new independent tests. Ridge and neural still select 20/36 dephasing shots
versus DFE 32/48; constant selects 28/44. Neural has lower mean errors but no extra
shot saving over ridge at either headline target. Its six primary comparisons
per physical family include DFE, constant and ridge at both targets. In exchange,
60-shot DFE fails the 0.065 mean-risk target, while zero-query ridge meets both.
The two-bath 8-shot excursion is substantially reduced by the independent rule.
Results and old-versus-new diagnostics are under `results_revision/stage5`.

The additive asset `mlst_independent_weights_20261002.zip`, release
`mlst-independent-weights-20261002`, contains the new code, data and per-query
errors. It is used alongside the base asset `mlst_revision_data_code_20261002.zip`
for the original calibration/validation data and frozen checkpoints. Prior releases
remain unchanged. The independent-weight readout stress test has not been run;
stage-three readout results describe the older estimators only.

### Earlier outcome-dependent evaluation (preserved)

The corrected evaluation is documented in `revision/REPRODUCE_STAGE2_STAGE3.txt`
and `revision/REPRODUCE_FOLLOWUP.txt`. The follow-up separately selects constant,
ridge and neural shrinkage before fresh independent tests. Both ridge and neural
use 20/36 shots at the two dephasing targets versus DFE's 32/48: there is no
additional neural query-shot saving over ridge at either target. All priors pay
4096 calibration shots. Constant shrinkage uses 32/44 shots. Neural versus
constant at the looser target fails the 0.002 noninferiority check despite both
passing the mean-risk target. Single-fit risk distributions are separate from
mean-risk intervals. Exchange ridge alone meets both targets without query shots.

The two-bath-qubit extension and numerical continuous-retention extrema are
included. The extension is non-neural and does not establish neural scaling.
These are deployment measurement-shot costs, not full lifecycle costs.
The earlier measurement comparisons in the archival commands below are
superseded by these corrected evaluations. Original and follow-up tests remain
separate in `results_revision/stage2` and `results_revision/stage4`.

The complete generated data, saved per-query errors, five source checkpoints,
and file hashes are distributed as the data/code-only release asset
`mlst_revision_data_code_20261002.zip` under release `mlst-revision-20261002`.
The archive includes code for reproducing figures, but no manuscript, response
letter, editorial correspondence or reviewer reports.

The manuscript and reviewer response are deliberately excluded from the
data/code release archive.

This repository contains the generators, baselines, sequence models, and tests
used for the MLST manuscript. The scientific question is estimator selection,
not whether one neural architecture wins universally.

This public repository contains data-generation and experimental code only.
The manuscript source and submission files are not distributed here.

## Important information boundary

For Markovian datasets the input contains the full Choi matrix of every step.
Exact superoperator composition is therefore available and is the required
deterministic baseline. In the collision dataset, the input contains reset-bath
marginals while the label comes from joint system--bath propagation. The target
distribution uses bath retention `eta` 0.85--0.99 versus 0--0.7 in training.
For fixed collision
parameters the input marginals are independent of `eta`; models predict a
distribution-conditional correction and do not infer the realised `eta`.

## Environment

Python 3.11 is used in the reported runs. Install the declared dependencies in
an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run the validation suite:

```bash
make test
```

## Data and checkpoints

Generated arrays and checkpoints are intentionally not committed to Git because
of their size. Generate the collision dataset with:

```bash
make collision-data
```

The default output is `data/collision`. The generator writes a manifest with
split sizes, seeds, lengths, eta intervals, fidelity convention, and feature
representation. Neural checkpoints are produced by `scripts/train_collision.sh`.

## Reproduce the corrected MLST audits

After data and collision checkpoints are available:

```bash
make mlst-audit
make mlst-figures
```

The audit includes:

- deterministic exact composition from full Choi inputs;
- constant, affine-product, affine-exact, and summary-ridge controls using the
  same labelled OOD indices as the neural model;
- DFE with enumeration of the target Pauli support, fixed total-shot allocation,
  and exact binomial measurement outcomes;
- finite-shot DFE labels for the offline OOD calibration set.
- a second exchange-coupled memory family with noncommuting Hamiltonian terms;
- symmetric readout-error stress tests and two-stage shot allocation;
- zero-shot rank and candidate-selection metrics;
- exact composition on all three two-qubit order-sensitive splits.

All audit CSV files record sample counts, seeds, and cost variables. The
manuscript-specific public snapshot is tagged `mlst-submission-v4`. An archival
DOI will be added when the versioned data and result release is deposited.

The headline checkpoints belong under `checkpoints/collision/`. Files with the
same basename directly under `checkpoints/` are older runs and should not be
used for the MLST tables. The SHA-256 values for the five reported checkpoints
are stored in `results_mlst/checkpoint_manifest.sha256`. Run the extended audit
with:

```bash
make mlst-independent-data
make mlst-enhanced-audit
```
