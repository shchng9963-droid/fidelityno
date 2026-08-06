# Portable entry points for FidelityNO and the MLST evidence audit.
SHELL := /bin/bash
PYTHON ?= python
MARKOVIAN_DATA ?= data/benchmarks
COLLISION_DATA ?= data/collision
COLLISION_CKPTS ?= checkpoints/collision
RESULTS_MLST ?= results_mlst

.PHONY: test smoke data collision-data mlst-audit mlst-figures

test:
	$(PYTHON) -m pytest -q

smoke:
	EPOCHS=3 N_TRAIN=1000 N_TEST=300 BATCH=128 MC_BUDGETS=10,100 MC_MAX_EVAL=100 bash scripts/run_all_baselines.sh

data:
	$(PYTHON) scripts/gen_data.py --n-train 200000 --n-test 10000 --seed 0 --holdout-family pauli

collision-data:
	$(PYTHON) scripts/gen_collision_data.py --outdir $(COLLISION_DATA) --seed 0

mlst-audit:
	$(PYTHON) scripts/eval_exact_composition.py --data \
		$(MARKOVIAN_DATA)/single_qubit_mixed/id_test.npz \
		$(MARKOVIAN_DATA)/single_qubit_mixed/length_ood.npz \
		$(MARKOVIAN_DATA)/two_qubit_mixed/id_test.npz \
		$(MARKOVIAN_DATA)/two_qubit_mixed/length_ood.npz \
		$(COLLISION_DATA)/id_test.npz \
		$(COLLISION_DATA)/length_ood.npz \
		$(COLLISION_DATA)/family_ood.npz \
		--out $(RESULTS_MLST)/exact_composition_optimized.csv
	$(PYTHON) scripts/eval_label_budget_baselines.py --data $(COLLISION_DATA)/family_ood.npz --ckpts $(wildcard $(COLLISION_CKPTS)/bidir_seed*.pt) --out $(RESULTS_MLST)/collision_family_ood_label_budget.csv
	$(PYTHON) scripts/eval_dfe.py --data $(COLLISION_DATA)/family_ood.npz --strategy stratified --M 1 --pauli-budgets 4,8,16,32,64,128,256,512,1024 --n-eval 1024 --out $(RESULTS_MLST)/dfe_family_ood_low_shot.csv
	$(PYTHON) scripts/eval_noisy_ood_calibration.py --data $(COLLISION_DATA)/family_ood.npz --ckpts $(wildcard $(COLLISION_CKPTS)/bidir_seed*.pt) --calib-sizes 16,32,64,128 --shots-per-label 16,32,64,128,512 --repeats 5 --out $(RESULTS_MLST)/noisy_ood_calibration.csv

mlst-figures:
	$(PYTHON) scripts/build_mlst_revision_figures.py --results $(RESULTS_MLST) --outdir $(RESULTS_MLST)/figures
