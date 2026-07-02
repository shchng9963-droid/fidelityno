from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from physics.channels.collision_nonmarkov import collision_sequence
from scripts.eval_calibrated import conformal_offsets, apply_offsets
from scripts.eval_mc import (
    features_to_choi,
    infer_dim_from_feature_dim,
    kraus_from_choi,
    mc_process_fidelity,
)
from train import make_model, prediction_to_quantiles


def build_dataset(n_samples: int, *, eta: float, lengths: list[int], seed: int, out_path: Path, max_len: int = 48) -> Path:
    rng = np.random.default_rng(seed)
    X = None
    M = None
    y = np.zeros(n_samples, dtype=np.float32)
    stats = np.zeros((n_samples, 2), dtype=np.float32)
    length_arr = np.zeros(n_samples, dtype=np.int32)
    per_fid = np.ones((n_samples, max_len), dtype=np.float32)
    eta_arr = np.full(n_samples, eta, dtype=np.float32)
    true_choi_real = np.zeros((n_samples, 4, 4), dtype=np.float32)
    true_choi_imag = np.zeros((n_samples, 4, 4), dtype=np.float32)
    fam = np.empty(n_samples, dtype=object)

    for i in range(n_samples):
        L = int(rng.choice(lengths))
        sample = collision_sequence(num_collisions=L, eta=eta, rng=rng)
        if X is None:
            feat_dim = sample.marginals[0].choi.size * 2
            X = np.zeros((n_samples, max_len, feat_dim), dtype=np.float32)
            M = np.zeros((n_samples, max_len), dtype=np.float32)
        from physics.composition import sequence_features, composed_stats
        x, m = sequence_features(sample.marginals, max_len, 2, "choi_hermitian")
        X[i] = x
        M[i] = m
        y[i] = sample.true_F_e
        length_arr[i] = L
        true_choi_real[i] = sample.true_choi.real.astype(np.float32)
        true_choi_imag[i] = sample.true_choi.imag.astype(np.float32)
        s = composed_stats(sample.marginals)
        stats[i] = (s["trace"], s["purity"])
        fam[i] = "collision"
        for t, ch in enumerate(sample.marginals[:max_len]):
            from physics.fidelity import entanglement_fidelity
            per_fid[i, t] = entanglement_fidelity(ch)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=X,
        mask=M,
        y=y,
        stats=stats,
        length=length_arr,
        per_fid=per_fid,
        eta=eta_arr,
        true_choi_real=true_choi_real,
        true_choi_imag=true_choi_imag,
        family_prefix=fam,
        family_counts=np.zeros((n_samples, 1), dtype=np.int32),
        family_idx_seq=np.full((n_samples, max_len), 0, dtype=np.int16),
        family_names=np.array(["collision"], dtype=object),
        perm_gap_random=np.zeros(n_samples, dtype=np.float32),
        perm_gap_reverse=np.zeros(n_samples, dtype=np.float32),
        fidelity_random_perm=y.copy(),
        fidelity_reverse=y.copy(),
    )
    return out_path


def load_eval_split(raw: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    n = len(raw["y"])
    n_cal = max(2, n // 2)
    rng = np.random.default_rng(20260614 + int(float(raw["eta"][0]) * 1000))
    perm = rng.permutation(n)
    return perm[:n_cal], perm[n_cal:]


def _mc_one(args: tuple[np.ndarray, np.ndarray, int, int, int]) -> float:
    x_i, mask_i, dim, n_samples, seed = args
    rng = np.random.default_rng(seed)
    seq = [kraus_from_choi(features_to_choi(x_i[j]), dim) for j in range(int(mask_i.sum()))]
    return float(mc_process_fidelity(seq, dim, n_samples, rng))


def baseline_rows(data_path: str, *, mc_samples: int = 1000, mc_workers: int = 1) -> pd.DataFrame:
    raw = np.load(data_path, allow_pickle=True)
    eta = float(raw["eta"][0])
    y = raw["y"].astype(np.float64)
    cal_idx, eval_idx = load_eval_split(raw)

    per_fid = raw["per_fid"].astype(np.float64)
    mask = raw["mask"] > 0
    prod = np.prod(np.where(mask, per_fid, 1.0), axis=1)

    dim = infer_dim_from_feature_dim(raw["x"].shape[-1])
    seeds = [20270614 + int(eta * 1000) * 1_000_000 + i for i in range(len(y))]
    work = [(raw["x"][i], raw["mask"][i], dim, mc_samples, seeds[i]) for i in range(len(y))]
    if mc_workers and mc_workers > 1:
        with ProcessPoolExecutor(max_workers=mc_workers) as ex:
            mc_pred = np.asarray(list(ex.map(_mc_one, work, chunksize=max(1, len(work) // (mc_workers * 8)))), dtype=np.float64)
    else:
        mc_pred = np.asarray([_mc_one(w) for w in work], dtype=np.float64)

    return pd.DataFrame([
        {"eta": eta, "model": "product_bound", "mae": float(np.abs(prod[eval_idx] - y[eval_idx]).mean()), "ece": np.nan, "ckpt": "baseline"},
        {"eta": eta, "model": f"mc_{mc_samples}", "mae": float(np.abs(mc_pred[eval_idx] - y[eval_idx]).mean()), "ece": np.nan, "ckpt": "baseline"},
    ])


def eval_split(ckpt_path: str, data_path: str) -> pd.DataFrame:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["cfg"])
    levels = np.asarray(cfg.model.quantiles, dtype=float)
    raw = np.load(data_path, allow_pickle=True)
    model = make_model(cfg.model.name, raw["x"].shape[-1], raw["x"].shape[1], cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = TensorDataset(
        torch.tensor(raw["x"]).float(),
        torch.tensor(raw["mask"]).float(),
        torch.tensor(raw["y"]).float(),
        torch.tensor(raw["stats"]).float(),
    )
    preds, ys = [], []
    with torch.no_grad():
        for x, m, y, _ in DataLoader(ds, batch_size=256):
            pred, _ = model(x, m)
            q = prediction_to_quantiles(pred, torch.tensor(levels, dtype=torch.float32))
            preds.append(q.numpy())
            ys.append(y.numpy())
    q = np.concatenate(preds)
    y = np.concatenate(ys)

    cal_idx, eval_idx = load_eval_split(raw)
    offsets = conformal_offsets(q[cal_idx], y[cal_idx], levels)
    q_corr = apply_offsets(q, offsets)

    mae_raw = float(np.abs(q.mean(1)[eval_idx] - y[eval_idx]).mean())
    mae_corr = float(np.abs(q_corr.mean(1)[eval_idx] - y[eval_idx]).mean())
    ece_raw = float(np.abs((y[:, None] <= q).mean(0) - levels).mean())
    ece_corr = float(np.abs((y[:, None] <= q_corr).mean(0) - levels).mean())
    return pd.DataFrame([
        {"eta": float(raw["eta"][0]), "model": "fidelityno_raw", "mae": mae_raw, "ece": ece_raw},
        {"eta": float(raw["eta"][0]), "model": "fidelityno_cal", "mae": mae_corr, "ece": ece_corr},
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", default=[str(p) for p in sorted(Path("checkpoints/collision").glob("fidelityno_seed*.pt"))])
    ap.add_argument("--etas", default="0.0,0.3,0.6,0.85,0.99")
    ap.add_argument("--n-samples", type=int, default=2048)
    ap.add_argument("--lengths", default="8,16,24,32,48")
    ap.add_argument("--out-dir", default="results_prxq/collision/eta_sweep")
    ap.add_argument("--mc-samples", type=int, default=1000)
    ap.add_argument("--mc-workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--reuse-data", action="store_true", help="Reuse existing eta_*.npz shards when present.")
    args = ap.parse_args()

    etas = [float(x) for x in args.etas.split(",") if x.strip()]
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for eta in etas:
        data_path = out_dir / f"eta_{eta:.2f}.npz"
        if args.reuse_data and data_path.exists():
            print(f"reuse eta={eta:.2f} data={data_path}", flush=True)
        else:
            data_path = build_dataset(
                args.n_samples,
                eta=eta,
                lengths=lengths,
                seed=12345 + int(round(eta * 1000)),
                out_path=data_path,
            )
        all_rows.append(baseline_rows(str(data_path), mc_samples=args.mc_samples, mc_workers=args.mc_workers))
        print(f"done eta={eta:.2f} baselines", flush=True)
        for ckpt in args.ckpts:
            df = eval_split(ckpt, str(data_path))
            df["ckpt"] = Path(ckpt).name
            all_rows.append(df)
            print(f"done eta={eta:.2f} ckpt={Path(ckpt).name}")

    res = pd.concat(all_rows, ignore_index=True)
    out_csv = out_dir / "eta_sweep.csv"
    res.to_csv(out_csv, index=False)
    agg = (
        res.groupby(["eta", "model"])
        .agg(mae_mean=("mae", "mean"), mae_std=("mae", "std"), ece_mean=("ece", "mean"), ece_std=("ece", "std"))
        .reset_index()
        .sort_values(["eta", "model"])
    )
    agg.to_csv(out_dir / "eta_sweep_aggregate.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colors = {"product_bound": "#2ca02c", "fidelityno_raw": "#d62728", "fidelityno_cal": "#1f77b4", "mc_1000": "#9467bd"}
    for model in ["product_bound", "mc_1000", "fidelityno_raw", "fidelityno_cal"]:
        sub = agg[agg["model"] == model].sort_values("eta")
        ax.plot(sub["eta"], sub["mae_mean"], "-o", label=model, color=colors[model], lw=2.0, ms=5)
    ax.set_xlabel(r"Bath retention $\eta$")
    ax.set_ylabel("MAE")
    ax.set_title("Collision-regime bath-retention sweep")
    ax.grid(True, alpha=0.3, lw=0.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "eta_sweep.pdf")
    fig.savefig(out_dir / "eta_sweep.png", dpi=170)
    print(f"[saved] {out_csv}")
    print(f"[saved] {out_dir / 'eta_sweep_aggregate.csv'}")
    print(f"[saved] {out_dir / 'eta_sweep.pdf'}")


if __name__ == "__main__":
    main()
