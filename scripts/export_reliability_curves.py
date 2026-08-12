#!/usr/bin/env python3
"""Export reliability curves as a compact, plotting-independent CSV.

This script uses the same checkpoints, data splits, conformal offsets, and
seed aggregation as ``make_reliability_figs.py``.  The exported table lets the
submission figures be rebuilt without bundling model checkpoints.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.make_reliability_figs as reliability


def configure_prediction_device(device: torch.device) -> None:
    """Route the unchanged checkpoint forward passes through one device."""
    def predict(model, dataset, batch_size: int = 1024):
        model = model.to(device)
        predictions, targets = [], []
        with torch.inference_mode():
            for x, mask, y, _stats in DataLoader(dataset, batch_size=batch_size):
                quantiles, _ = model(x.to(device), mask.to(device))
                predictions.append(quantiles.detach().cpu().numpy())
                targets.append(y.numpy())
        return np.concatenate(predictions), np.concatenate(targets), 0.0

    reliability.predict = predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results_mlst/reliability_curves.csv",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, or a specific CUDA device.",
    )
    args = parser.parse_args()

    requested = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if requested == "auto":
        requested = "cpu"
    device = torch.device(requested)
    configure_prediction_device(device)
    print(f"[device] {device}", flush=True)

    rows: list[dict[str, float | int | str]] = []
    split_names = {"id": "ID", "len_ood": "Length OOD"}
    for label, model_name in reliability.MODELS.items():
        bundle, levels = reliability.gather_predictions(model_name, target_length=48)
        if levels is None:
            continue
        for split_key, split_label in split_names.items():
            for seed_index, (quantiles, targets) in enumerate(bundle[split_key]):
                coverage = reliability.empirical_coverage(quantiles, targets, levels)
                for level, value in zip(levels, coverage):
                    rows.append(
                        {
                            "split": split_label,
                            "model": label,
                            "seed_index": seed_index,
                            "nominal_level": float(level),
                            "empirical_coverage": float(value),
                        }
                    )

    if not rows:
        raise RuntimeError("No checkpoint predictions were available.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"[saved] {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
