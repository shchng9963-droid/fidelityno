# FidelityNO: cost--accuracy audits for composed-channel fidelity

This repository contains the generators, baselines, sequence models, and tests
used for the MLST manuscript. The scientific question is estimator selection,
not whether one neural architecture wins universally.

## Important information boundary

For Markovian datasets the input contains the full Choi matrix of every step.
Exact superoperator composition is therefore available and is the required
deterministic baseline. In the collision dataset, the input contains reset-bath
marginals while the label comes from joint system--bath propagation. The legacy
file name `family_ood.npz` denotes a bath-retention shift (`eta` 0.85--0.99
versus 0--0.7 in training), not a held-out collision family. For fixed collision
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
manuscript-specific public snapshot is tagged `mlst-submission-v2`. An archival
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
