# FidelityNO Makefile
# Usage:
#   make all          # Full experiment (200k data, 100 epochs, 5 seeds)
#   make smoke        # Quick smoke test (1k data, 3 epochs)
#   make test         # Run pytest
#   make figures      # Regenerate figures from results/summary.csv

SHELL := /bin/bash
ENV := /home/wangshuchang/miniforge3/envs/fidelityno/bin
PYTHON := $(ENV)/python

.PHONY: all smoke test figures clean data

all:
	EPOCHS=100 N_TRAIN=200000 N_TEST=10000 BATCH=128 bash scripts/run_all_baselines.sh

smoke:
	EPOCHS=3 N_TRAIN=1000 N_TEST=300 BATCH=128 MC_BUDGETS=10,100 MC_MAX_EVAL=100 bash scripts/run_all_baselines.sh

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

figures:
	$(PYTHON) scripts/make_figures.py

data:
	$(PYTHON) scripts/gen_data.py --n-train 200000 --n-test 10000 --seed 0 --holdout-family pauli

demo:
	$(PYTHON) scripts/demo_cutoff_bo.py --ckpt checkpoints/fidelityno_seed0.pt

clean:
	rm -rf checkpoints/*.pt results/*.csv results/figs/*.png results/figs/*.pdf wandb/offline-run-* logs/*.log
