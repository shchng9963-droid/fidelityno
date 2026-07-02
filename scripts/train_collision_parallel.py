#!/usr/bin/env python3
"""Parallel training driver for the non-Markovian collision dataset
(PRXQ P1.1).  Mirrors train_device_regime_parallel.py but with a
parametric data-dir / ckpt-dir / log-dir.
"""
from __future__ import annotations
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("COLL_DATA_DIR", REPO / "data" / "collision"))
CKPT_DIR = Path(os.environ.get("COLL_CKPT_DIR", REPO / "checkpoints" / "collision"))
LOG_DIR = Path(os.environ.get("COLL_LOG_DIR", REPO / "results_prxq" / "collision" / "training_logs"))
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = int(os.environ.get("DEVICE_EPOCHS", 80))
BATCH = int(os.environ.get("DEVICE_BATCH", 256))
PATIENCE = int(os.environ.get("DEVICE_PATIENCE", 20))

MODELS = [
    ("fidelityno", True),
    ("mlp", False),
    ("deepsets", False),
    ("bidir", False),
]
SEEDS = [0, 1, 2, 3, 4]
GPUS = [g.strip() for g in os.environ.get("DEVICE_GPUS", "0,1").split(",") if g.strip()]


def cmd_for(model: str, is_group: bool, seed: int) -> list[str]:
    ckpt = f"{model}_seed{seed}.pt"
    overrides = [
        f"seed={seed}",
        f"data.train={DATA_DIR}/train.npz",
        f"data.val={DATA_DIR}/calib.npz",
        f"train.epochs={EPOCHS}",
        f"train.batch_size={BATCH}",
        f"train.patience={PATIENCE}",
        f"train.ckpt_dir={CKPT_DIR}",
        f"train.ckpt_name={ckpt}",
    ]
    overrides.insert(0, f"model={model}" if is_group else f"model.name={model}")
    return ["python", "train.py", *overrides]


def main() -> None:
    queue = []
    for model, is_group in MODELS:
        for seed in SEEDS:
            ckpt = CKPT_DIR / f"{model}_seed{seed}.pt"
            if ckpt.exists():
                print(f"[skip] {ckpt} exists")
                continue
            queue.append((model, is_group, seed))
    print(f"[queue] {len(queue)} runs on GPUs={GPUS}")

    running: dict[str, dict] = {g: None for g in GPUS}
    finished, failed = [], []
    while queue or any(v is not None for v in running.values()):
        for g in GPUS:
            slot = running[g]
            if slot is None:
                continue
            ret = slot["proc"].poll()
            if ret is None:
                continue
            slot["log"].close()
            elapsed = time.time() - slot["start"]
            tag = slot["name"]
            (finished if ret == 0 else failed).append(tag)
            print(f"[{'done' if ret==0 else 'FAIL'}] gpu{g} {tag} ({elapsed:.0f}s) ret={ret}")
            running[g] = None

        for g in GPUS:
            if running[g] is not None or not queue:
                continue
            model, is_group, seed = queue.pop(0)
            tag = f"{model}_seed{seed}"
            log_path = LOG_DIR / f"{tag}.log"
            log_f = log_path.open("w")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = g
            env["WANDB_MODE"] = env.get("WANDB_MODE", "offline")
            env["TQDM_DISABLE"] = "1"
            cmd = cmd_for(model, is_group, seed)
            print(f"[launch] gpu{g} {tag} -> {log_path}")
            print(f"         $ CUDA_VISIBLE_DEVICES={g} {' '.join(shlex.quote(c) for c in cmd)}")
            proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running[g] = {"proc": proc, "name": tag, "log": log_f, "log_path": log_path, "start": time.time()}
        time.sleep(15)

    print(f"[summary] finished={len(finished)} failed={len(failed)}")
    if failed:
        for t in failed:
            print(" ", t)
        sys.exit(1)


if __name__ == "__main__":
    main()
