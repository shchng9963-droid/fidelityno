"""Demo B: Surrogate-based protocol selection (nested vs parallel purification).

Scenario:
  Given a fixed budget of noisy CPTP channels, choose between two protocol arms:
    - Arm P (parallel): apply all N channels in series, no intermediate
      purification.  Long chain, native per-link noise.
    - Arm N (nested):   apply N/2 channels in series, but each channel has
      *halved* per-link noise (modelling one round of distillation that
      reduces depth at the cost of a constant prefactor in resources).

  Both arms come from the SAME training noise family distribution (mixed,
  excluding the held-out family at training time), so this is a valid
  in-distribution decision task.

  We rank candidates by predicted fidelity F̂(arm) and compare to ground
  truth F*(arm) computed by exact Choi composition.

Methods compared:
    exact_grid       Reference. Picks F*.  (Oracle.)
    mc_K_bo          Monte-Carlo with K samples per arm.
    fidelityno_*     Trained neural surrogate(s).

Metrics per method:
    agreement_rate   P[ argmax F̂ == argmax F* ]
    regret_mean      E[ F*(oracle) - F*(method choice) ]   (≥0)
    wall_clock_ms    mean ms per scenario  (decision cost)

Outputs (default args):
    results/demo_protocol/raw.csv         per (scenario, method, arm) row
    results/demo_protocol/summary.csv     one row per method
    results/figs/demo_protocol_*.{pdf,png}
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from physics.channels.single_qubit import sample_single_qubit
from physics.channels.lindblad import sample_lindblad
from physics.channels.base import Channel
from physics.composition import exact_sequence_fidelity, sequence_features
from scripts.eval_mc import features_to_choi, kraus_from_choi, mc_process_fidelity
from train import make_model, mean_from_prediction


FAMILIES_NO_PAULI = ["amplitude_damping", "phase_damping", "depolarizing", "lindblad"]


def _scale_channel_noise(family: str, scale: float, rng: np.random.Generator) -> Channel:
    """Sample a channel from the given family but with the noise parameter
    multiplied by `scale` (≤1 reduces noise, modeling distillation gain)."""
    if family == "lindblad":
        # The lindblad sampler integrates over a random time t; just shrink t.
        # Implement by sampling and rescaling its raw parameters.
        ch = sample_lindblad(rng)
        # ch.params layout depends on internal; cleanest is to *resample* with
        # a smaller t-range. We approximate by composing the channel with
        # itself a fractional number of times via choi-eigvalue interpolation
        # toward identity:  C_scaled = (1-scale)·I + scale·C  (CPTP for scale∈[0,1]).
        I = Channel("identity", ch.dim, kraus=[np.eye(ch.dim, dtype=np.complex128)])
        new_choi = (1.0 - scale) * I.choi + scale * ch.choi
        return Channel("lindblad_scaled", ch.dim, choi=new_choi, params=ch.params)
    # For the parametric single-qubit families, draw and then convex-combine
    # with identity by `scale`. This rescales the *fidelity loss* by `scale`,
    # which is exactly what one round of distillation gives in the leading
    # order of small noise rates.
    ch = sample_single_qubit(rng, family)
    I = Channel("identity", ch.dim, kraus=[np.eye(ch.dim, dtype=np.complex128)])
    new_choi = (1.0 - scale) * I.choi + scale * ch.choi
    return Channel(f"{family}_scaled", ch.dim, choi=new_choi, params=ch.params)


def make_scenario(rng: np.random.Generator, n_par: int, families: list[str],
                  scale_range: tuple[float, float] = (0.3, 1.5)):
    """Draw one (parallel arm, nested arm) pair sharing the same family
    distribution but with the nested arm having half the depth and a
    *random* per-link noise rescaling drawn from `scale_range`.

    Why a random scale: if the scale is always <1 the nested arm trivially
    wins. Real distillation gain depends on operating regime — sometimes the
    overhead exceeds the depth saving. Drawing the scale per-scenario from a
    range that straddles 1 ensures both arms can win and the surrogate has to
    actually predict.

    Returns:
        arm_par   : list[Channel]  length n_par
        arm_nest  : list[Channel]  length n_par // 2
        scale     : float
    """
    scale = float(rng.uniform(*scale_range))
    arm_par = []
    for _ in range(n_par):
        fam = rng.choice(families)
        arm_par.append(_scale_channel_noise(str(fam), 1.0, rng))
    arm_nest = []
    for _ in range(n_par // 2):
        fam = rng.choice(families)
        arm_nest.append(_scale_channel_noise(str(fam), scale, rng))
    return arm_par, arm_nest, scale


# ---------------------- Surrogates ----------------------------------------

def exact_arm(arm: list[Channel]) -> float:
    return exact_sequence_fidelity(arm)


def mc_arm(arm: list[Channel], n_mc: int, dim: int, max_len: int, seed: int) -> float:
    feats, _ = sequence_features(arm, max_len, dim)
    kraus_seq = [kraus_from_choi(features_to_choi(feats[j]), dim) for j in range(len(arm))]
    rng = np.random.default_rng(seed)
    return mc_process_fidelity(kraus_seq, dim, n_mc, rng)


def neural_arm(model, arm: list[Channel], dim: int, max_len: int, device: str) -> float:
    feats, mask = sequence_features(arm, max_len, dim)
    x = torch.tensor(feats).float().unsqueeze(0).to(device)
    m = torch.tensor(mask).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred, _ = model(x, m)
        return float(mean_from_prediction(pred).cpu().item())


# ---------------------- Driver --------------------------------------------

def load_model(ckpt_path: Path, device: str, max_len: int, input_dim: int):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["cfg"])
    model = make_model(cfg.model.name, input_dim, max_len, cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg.model.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenarios", type=int, default=200)
    ap.add_argument("--n-par",       type=int, default=12,
                    help="Parallel-arm depth; nested arm uses n_par//2.")
    ap.add_argument("--mc-budgets",  type=str, default="100,1000")
    ap.add_argument("--ckpts",       nargs="+", default=[
        "checkpoints/fidelityno_seed0.pt",
        "checkpoints/gnn_seed0.pt",
    ], help="Neural surrogate checkpoints to evaluate.")
    ap.add_argument("--seed",        type=int, default=2026)
    ap.add_argument("--out-dir",     default="results/demo_protocol")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("results/figs"); fig_dir.mkdir(parents=True, exist_ok=True)

    dim = 2
    max_len = 48  # match training data
    input_dim = 2 * (dim * dim) ** 2  # choi_hermitian feature dim

    # -- Build scenarios up front.
    rng = np.random.default_rng(args.seed)
    scenarios = []
    scales = []
    for i in range(args.n_scenarios):
        arm_par, arm_nest, scale = make_scenario(rng, args.n_par, FAMILIES_NO_PAULI)
        scenarios.append((arm_par, arm_nest))
        scales.append(scale)

    # -- Compute exact ground truth (oracle).
    print(f"[1/{2 + len(args.mc_budgets.split(','))}] Computing exact ground truth for "
          f"{args.n_scenarios} scenarios x 2 arms ...")
    rows = []
    t0 = time.perf_counter()
    exact = []
    for i, (par, nest) in enumerate(scenarios):
        F_par  = exact_arm(par)
        F_nest = exact_arm(nest)
        exact.append((F_par, F_nest))
        rows.append({"scenario": i, "method": "exact", "arm": "par",  "fidelity": F_par,
                     "decision_t_ms": 0.0})
        rows.append({"scenario": i, "method": "exact", "arm": "nest", "fidelity": F_nest,
                     "decision_t_ms": 0.0})
    exact_total = time.perf_counter() - t0
    exact_per_scenario = exact_total / args.n_scenarios * 1000.0
    print(f"  exact: {exact_total:.2f}s total, {exact_per_scenario:.3f} ms/scenario")

    # -- MC baselines
    for budget_str in args.mc_budgets.split(","):
        K = int(budget_str)
        print(f"[mc] MC-{K} budget ...")
        t0 = time.perf_counter()
        for i, (par, nest) in enumerate(scenarios):
            t_arm0 = time.perf_counter()
            F_par  = mc_arm(par,  K, dim, max_len, args.seed * 31 + 2 * i)
            F_nest = mc_arm(nest, K, dim, max_len, args.seed * 31 + 2 * i + 1)
            dt_ms = 1000.0 * (time.perf_counter() - t_arm0)
            rows.append({"scenario": i, "method": f"mc_{K}", "arm": "par",  "fidelity": F_par,
                         "decision_t_ms": dt_ms})
            rows.append({"scenario": i, "method": f"mc_{K}", "arm": "nest", "fidelity": F_nest,
                         "decision_t_ms": dt_ms})
        print(f"  mc_{K}: {time.perf_counter()-t0:.2f}s total")

    # -- Neural surrogates
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ckpt_path in args.ckpts:
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            print(f"  WARNING: {ckpt} not found, skipping")
            continue
        model, name = load_model(ckpt, device, max_len, input_dim)
        # Distinguish ckpt seeds by the method label.
        method_label = f"surrogate_{name}_{ckpt.stem}"
        print(f"[surrogate] {method_label} on {device} ...")
        t0 = time.perf_counter()
        # Warm up CUDA kernels.
        with torch.no_grad():
            _ = neural_arm(model, scenarios[0][0], dim, max_len, device)
        for i, (par, nest) in enumerate(scenarios):
            t_arm0 = time.perf_counter()
            F_par  = neural_arm(model, par,  dim, max_len, device)
            F_nest = neural_arm(model, nest, dim, max_len, device)
            dt_ms = 1000.0 * (time.perf_counter() - t_arm0)
            rows.append({"scenario": i, "method": method_label, "arm": "par",  "fidelity": F_par,
                         "decision_t_ms": dt_ms})
            rows.append({"scenario": i, "method": method_label, "arm": "nest", "fidelity": F_nest,
                         "decision_t_ms": dt_ms})
        print(f"  {method_label}: {time.perf_counter()-t0:.2f}s total")

    raw = pd.DataFrame(rows)
    raw.to_csv(out_dir / "raw.csv", index=False)
    print(f"\nWrote {out_dir/'raw.csv'} ({len(raw)} rows)")

    # ----- Aggregate -----
    # For each method, derive per-scenario decision: which arm has higher F̂.
    methods = [m for m in raw["method"].unique() if m != "exact"]
    summary_rows = []
    exact_pivot = raw[raw["method"] == "exact"].pivot(index="scenario", columns="arm", values="fidelity")
    exact_decision = (exact_pivot["nest"] > exact_pivot["par"]).astype(int)  # 1 if nest wins
    exact_max = np.maximum(exact_pivot["par"].values, exact_pivot["nest"].values)

    for m in methods:
        sub = raw[raw["method"] == m]
        pivot = sub.pivot(index="scenario", columns="arm", values="fidelity")
        decision = (pivot["nest"] > pivot["par"]).astype(int)
        agreement = float((decision == exact_decision).mean())

        # regret = F*(oracle) - F*(picked)
        picked_F_true = np.where(decision == 0,  exact_pivot["par"].values, exact_pivot["nest"].values)
        regret = exact_max - picked_F_true
        # decision cost (single value per scenario; we logged the arm-level dt
        # but each scenario evaluates two arms → sum dt across arms).
        per_scenario_dt = sub.groupby("scenario")["decision_t_ms"].sum().values
        summary_rows.append({
            "method": m,
            "n_scenarios": len(decision),
            "agreement_rate":  agreement,
            "regret_mean":     float(regret.mean()),
            "regret_max":      float(regret.max()),
            "decision_t_ms_mean": float(per_scenario_dt.mean()),
            "decision_t_ms_p95":  float(np.percentile(per_scenario_dt, 95)),
        })

    # Add exact as reference (zero regret, oracle agreement, "infinite" cost ref)
    summary_rows.insert(0, {
        "method": "exact",
        "n_scenarios": len(exact_decision),
        "agreement_rate": 1.0,
        "regret_mean": 0.0,
        "regret_max":  0.0,
        "decision_t_ms_mean": exact_per_scenario,
        "decision_t_ms_p95":  exact_per_scenario,
    })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nWrote {out_dir/'summary.csv'}\n")
    print(summary.to_string(index=False))

    # ----- Plots -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figures")
        return

    # Bar chart: agreement vs decision cost.
    PRETTY_METHOD = {
        "exact":   "Exact (oracle)",
        "mc_100":  "MC-100",
        "mc_1000": "MC-1000",
        "surrogate_fidelityno_fidelityno_seed0":   "FidelityNO-T",
        "surrogate_gnn_gnn_seed0":                 "FidelityNO-G",
        "surrogate_generic_gnn_generic_gnn_seed0": "Generic-GNN (no enc.)",
        "surrogate_mlp_mlp_seed0":                 "Flat MLP",
        "surrogate_deepsets_deepsets_seed0":       "DeepSets",
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    methods_plot = [m for m in summary["method"] if m != "exact"]
    sub = summary[summary["method"].isin(methods_plot)].sort_values("agreement_rate")
    xs = np.arange(len(sub))
    bars = ax.barh(xs, sub["agreement_rate"], color=plt.cm.viridis(sub["agreement_rate"]/1.0))
    ax.set_yticks(xs, [PRETTY_METHOD.get(m, m) for m in sub["method"]])
    ax.set_xlim(0.5, 1.05)
    ax.axvline(1.0, color="gray", lw=1, ls=":", label="oracle")
    for i_row, (_, row) in enumerate(sub.iterrows()):
        x = row["agreement_rate"]
        t = row["decision_t_ms_mean"]
        ax.text(x + 0.005, i_row, f"{x:.3f}  ({t:.1f} ms)", va="center", fontsize=8)
    ax.set_xlabel("Agreement with exact decision")
    ax.set_title("Demo B: nested vs parallel — surrogate decision quality")
    fig.tight_layout()
    fig.savefig(fig_dir / "demo_protocol_agreement.pdf")
    fig.savefig(fig_dir / "demo_protocol_agreement.png", dpi=180)

    # Regret distribution (boxplot).
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    method_order = methods_plot
    regret_per_method = []
    for m in method_order:
        sub = raw[raw["method"] == m]
        pivot = sub.pivot(index="scenario", columns="arm", values="fidelity")
        decision = (pivot["nest"] > pivot["par"]).astype(int)
        picked_F_true = np.where(decision == 0,  exact_pivot["par"].values, exact_pivot["nest"].values)
        regret = exact_max - picked_F_true
        regret_per_method.append(regret)
    ax.boxplot(regret_per_method,
               tick_labels=[PRETTY_METHOD.get(m, m) for m in method_order],
               showfliers=False, vert=False)
    ax.set_xlabel("Regret  F*(oracle) − F*(picked)")
    ax.set_title("Demo B: regret distribution")
    fig.tight_layout()
    fig.savefig(fig_dir / "demo_protocol_regret.pdf")
    fig.savefig(fig_dir / "demo_protocol_regret.png", dpi=180)

    print(f"\nFigures: {fig_dir/'demo_protocol_agreement.pdf'} and {fig_dir/'demo_protocol_regret.pdf'}")


if __name__ == "__main__":
    main()
