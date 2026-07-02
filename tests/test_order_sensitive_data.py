from pathlib import Path
import json
import subprocess
import sys
import numpy as np

from scripts.gen_data import build_dataset
from physics.channels.two_qubit import sample_order_sensitive_two_qubit_sequence
from physics.composition import exact_sequence_fidelity


def test_order_sensitive_two_qubit_dataset_enforces_permutation_gap(tmp_path):
    out = tmp_path / "order_sensitive_train.npz"
    meta = build_dataset(
        out,
        n=24,
        seed=11,
        lengths=[8, 16],
        family="order_sensitive",
        dim=4,
        max_len=16,
    )
    d = np.load(out, allow_pickle=True)
    assert "perm_gap_random" in d.files
    assert "perm_gap_reverse" in d.files
    assert np.all(d["perm_gap_random"] >= 0.01)
    assert np.mean(d["perm_gap_random"]) > 0.05
    assert np.quantile(d["perm_gap_reverse"], 0.5) >= 0.0
    assert meta["family"] == "order_sensitive"
    assert meta["benchmark_tag"] == "order_sensitive"


def test_order_sensitive_train_excludes_holdout_family(tmp_path):
    out = tmp_path / "order_sensitive_train_exclude_cdep.npz"
    meta = build_dataset(
        out,
        n=32,
        seed=17,
        lengths=[8, 16],
        family="order_sensitive",
        dim=4,
        max_len=16,
        exclude_family="correlated_dephasing",
    )
    d = np.load(out, allow_pickle=True)
    names = d["family_names"].tolist()
    holdout_idx = names.index("correlated_dephasing")
    assert d["family_counts"][:, holdout_idx].sum() == 0
    assert "correlated_dephasing" not in meta["active_families"]
    assert np.mean(d["perm_gap_random"]) > 0.01


def test_order_sensitive_manifest_from_generator(tmp_path):
    outdir = tmp_path / "bench"
    outdir.mkdir(parents=True, exist_ok=True)
    meta = build_dataset(
        outdir / "train.npz",
        n=12,
        seed=3,
        lengths=[8],
        family="order_sensitive",
        dim=4,
        max_len=8,
    )
    manifest = {
        "splits": [
            json.loads(json.dumps({"family": meta["family"], "dim": meta["dim"], "benchmark_tag": meta["benchmark_tag"]}))
        ]
    }
    assert manifest["splits"][0]["benchmark_tag"] == "order_sensitive"


def test_order_sensitive_cli_family_ood_keeps_permutation_gap_and_holdout(tmp_path):
    outdir = tmp_path / "bench_cli"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/gen_data.py",
            "--outdir",
            str(outdir),
            "--n-train",
            "24",
            "--n-calib",
            "8",
            "--n-test",
            "12",
            "--seed",
            "5",
            "--dim",
            "4",
            "--family",
            "order_sensitive",
            "--holdout-family",
            "imperfect_swap",
            "--train-lengths",
            "8",
            "--id-lengths",
            "8",
            "--length-ood-lengths",
            "24",
            "--family-ood-lengths",
            "8",
        ],
        cwd=Path(__file__).resolve().parents[1],
    )
    family_ood = np.load(outdir / "family_ood.npz", allow_pickle=True)
    names = family_ood["family_names"].tolist()
    swap_idx = names.index("imperfect_swap")
    assert np.all(family_ood["family_counts"][:, swap_idx] > 0)
    assert np.mean(family_ood["perm_gap_random"]) > 0.01
    manifest = json.loads((outdir / "manifest.json").read_text())
    split_meta = next(item for item in manifest["splits"] if item["file"].endswith("family_ood.npz"))
    assert split_meta["benchmark_tag"] == "order_sensitive"
    assert split_meta["family"] == "order_sensitive"
    assert split_meta["required_family"] == "imperfect_swap"


def _family_signature(seq):
    labels = []
    for ch in seq:
        name = ch.name
        if "imperfect_cnot" in name:
            labels.append("cnot")
        elif "imperfect_swap" in name:
            labels.append("swap")
        elif "correlated_dephasing" in name:
            labels.append("cdep")
        elif "two_qubit_depolarizing" in name:
            labels.append("depol")
        else:
            labels.append(name)
    return tuple(labels)


def test_order_sensitive_sequence_fidelity_should_not_collapse_to_identity_floor_when_ideal_sequence_is_nonidentity():
    rng = np.random.default_rng(3)
    seq, _ = sample_order_sensitive_two_qubit_sequence(
        rng,
        length=8,
        required_family="correlated_dephasing",
    )
    fid = exact_sequence_fidelity(seq)
    assert fid > 0.1



def test_order_sensitive_sampler_should_not_use_one_fixed_family_order_pattern():
    rng = np.random.default_rng(7)
    signatures = set()
    for _ in range(12):
        seq, _ = sample_order_sensitive_two_qubit_sequence(
            rng,
            length=8,
            exclude_family="correlated_dephasing",
        )
        signatures.add(_family_signature(seq))
    assert len(signatures) > 1
