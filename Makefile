# Portable entry points for FidelityNO and the MLST evidence audit.
SHELL := /bin/bash
PYTHON ?= python
MARKOVIAN_DATA ?= data/benchmarks
COLLISION_DATA ?= data/collision
COLLISION_CKPTS ?= checkpoints/collision
RESULTS_MLST ?= results_mlst
INDEPENDENT_DATA ?= data_mlst/collision_family_ood_independent_rng.npz
ORDER_SENSITIVE_DATA ?= data/benchmarks/two_qubit_order_sensitive

.PHONY: test smoke data collision-data mlst-audit mlst-independent-data mlst-enhanced-audit mlst-figures

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

mlst-independent-data:
	$(PYTHON) scripts/gen_collision_independent_eta_split.py --out $(INDEPENDENT_DATA) --n 4096 --parameter-seed 20260807 --eta-seed 920260807 --lengths 8,16,24 --eta-range 0.85,0.99

mlst-enhanced-audit:
	$(PYTHON) scripts/eval_exchange_memory_identifiability.py --n-base 1024 --n-eta 15 --eta-min 0.85 --eta-max 0.99 --lengths 8,16,24 --seed 20260810 --out-prefix $(RESULTS_MLST)/exchange_memory_identifiability
	$(PYTHON) scripts/eval_adaptive_measurement_allocation.py --data $(INDEPENDENT_DATA) --ckpts $(wildcard $(COLLISION_CKPTS)/bidir_seed*.pt) --average-budgets 32,64 --pilot-shots 8 --n-calib 64 --n-test 4032 --label-shots 64 --repeats 5 --bootstrap 10000 --out $(RESULTS_MLST)/adaptive_measurement_allocation.csv
	$(PYTHON) scripts/eval_readout_noise_robustness.py --data $(INDEPENDENT_DATA) --ckpts $(wildcard $(COLLISION_CKPTS)/bidir_seed*.pt) --budgets 32,64 --readout-errors 0,0.01,0.03,0.05 --n-calib 64 --n-test 4032 --label-shots 64 --repeats 5 --out $(RESULTS_MLST)/readout_noise_robustness.csv
	$(PYTHON) scripts/eval_ranking_utility.py --data $(INDEPENDENT_DATA) --ckpts $(wildcard $(COLLISION_CKPTS)/bidir_seed*.pt) --fractions 0.05,0.10,0.20 --out $(RESULTS_MLST)/ranking_utility.csv
	$(PYTHON) scripts/eval_exact_composition.py --data $(ORDER_SENSITIVE_DATA)/id_test.npz $(ORDER_SENSITIVE_DATA)/length_ood.npz $(ORDER_SENSITIVE_DATA)/family_ood.npz --repeats 5 --warmup 1 --out $(RESULTS_MLST)/exact_composition_order_sensitive.csv

mlst-figures:
	$(PYTHON) scripts/build_mlst_revision_figures.py --results $(RESULTS_MLST) --outdir $(RESULTS_MLST)/figures
