from pathlib import Path
import json
import numpy as np
from scripts.gen_data import build_dataset


def test_mixed_train_excludes_holdout_family(tmp_path):
    out = tmp_path / "train.npz"
    meta = build_dataset(out, n=30, seed=7, lengths=[2,4], family="mixed", dim=2, max_len=8, exclude_family="pauli")
    d = np.load(out, allow_pickle=True)
    names = d["family_names"].tolist()
    pauli_idx = names.index("pauli")
    assert d["family_counts"][:, pauli_idx].sum() == 0
    assert "pauli" not in meta["active_families"]


def test_family_ood_split_contains_requested_holdout(tmp_path):
    out = tmp_path / "family_ood.npz"
    build_dataset(out, n=20, seed=8, lengths=[2], family="pauli", dim=2, max_len=4)
    d = np.load(out, allow_pickle=True)
    names = d["family_names"].tolist()
    pauli_idx = names.index("pauli")
    assert np.all(d["family_counts"][:, pauli_idx] == d["length"])
