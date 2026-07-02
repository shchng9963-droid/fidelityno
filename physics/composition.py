"""Channel composition and Choi-based fidelity utilities.

For PRX Quantum P0.4 (fidelity definition cleanup): all fidelity primitives
now live in ``physics.fidelity``. This module re-exports them under their
historical names so v1 call sites keep working unchanged.

If you need a fidelity formula, import from ``physics.fidelity`` directly.
"""

from __future__ import annotations
import numpy as np
from physics.channels.base import Channel
from physics.representations import choi_to_features, feature_dim_for_representation
from physics.fidelity import (
    entanglement_fidelity,
    process_fidelity,
    average_gate_fidelity,
    state_fidelity,
    ef_to_avg,
    avg_to_ef,
    FIDELITY_KIND,
)

__all__ = [
    "compose_channels",
    "process_fidelity",
    "entanglement_fidelity",
    "average_gate_fidelity",
    "state_fidelity",
    "ef_to_avg",
    "avg_to_ef",
    "FIDELITY_KIND",
    "identity_channel",
    "reference_target_for_channel",
    "reference_target_for_sequence",
    "channel_reference_fidelity",
    "exact_sequence_fidelity",
    "composed_stats",
    "sequence_features",
]


def compose_channels(channels: list[Channel]) -> Channel:
    """Right-to-left composition: returns Lambda_n o ... o Lambda_1.

    The resulting channel acts on state rho as
    ``(Lambda_n o ... o Lambda_1)(rho) = Lambda_n(... Lambda_1(rho))``.
    """
    if not channels:
        raise ValueError("empty channel sequence")
    dim = channels[0].dim
    s = np.eye(dim * dim, dtype=np.complex128)
    names = []
    params = []
    for ch in channels:
        if ch.dim != dim:
            raise ValueError("mixed dimensions in one sequence are unsupported")
        s = ch.superop @ s
        names.append(ch.name)
        params.append(ch.params)
    return Channel(
        "composed",
        dim,
        superop=s,
        params=np.concatenate(params) if params else np.zeros(0),
        metadata={"names": names},
    )


def identity_channel(dim: int) -> Channel:
    return Channel("identity", dim, kraus=[np.eye(dim, dtype=np.complex128)])


def reference_target_for_channel(ch: Channel) -> Channel:
    """Pick the natural ideal target for a single channel.

    - imperfect_cnot, imperfect_swap -> the ideal two-qubit unitary.
    - everything else -> identity (modeling pure noise / memory).
    """
    name = ch.name
    if "imperfect_cnot" in name:
        from physics.channels.two_qubit import cnot_unitary
        return Channel("ideal_cnot", ch.dim, kraus=[cnot_unitary()])
    if "imperfect_swap" in name:
        from physics.channels.two_qubit import swap_unitary
        return Channel("ideal_swap", ch.dim, kraus=[swap_unitary()])
    return identity_channel(ch.dim)


def reference_target_for_sequence(channels: list[Channel]) -> Channel:
    if not channels:
        raise ValueError("empty channel sequence")
    return compose_channels([reference_target_for_channel(ch) for ch in channels])


def channel_reference_fidelity(ch: Channel) -> float:
    """Per-step entanglement fidelity vs the channel's natural ideal target."""
    return entanglement_fidelity(ch, reference_target_for_channel(ch))


def exact_sequence_fidelity(channels: list[Channel], target: Channel | None = None) -> float:
    """Entanglement fidelity F_e of the composed sequence vs target.

    If target is None, uses the sequence's natural reference (identity for
    noise channels, ideal unitaries for imperfect gates).

    This is the canonical training signal for FidelityNO.
    """
    if target is None:
        target = reference_target_for_sequence(channels)
    return entanglement_fidelity(compose_channels(channels), target)


def composed_stats(channels: list[Channel]) -> dict[str, float]:
    """Auxiliary physics-consistency statistics of the composed Choi matrix.

    Returns trace, purity, and F_e to identity. Used as auxiliary loss in
    FidelityNO and as a sanity gauge.
    """
    comp = compose_channels(channels)
    c = comp.choi
    return {
        "trace": float(np.trace(c).real),
        "purity": float(np.trace(c.conj().T @ c).real / (comp.dim ** 2)),
        "process_fidelity_identity": entanglement_fidelity(comp),
    }


def sequence_features(channels: list[Channel], max_len: int, dim: int, representation: str = "choi_hermitian") -> tuple[np.ndarray, np.ndarray]:
    feat_dim = feature_dim_for_representation(dim, representation)
    x = np.zeros((max_len, feat_dim), dtype=np.float32)
    mask = np.zeros((max_len,), dtype=np.float32)
    for i, ch in enumerate(channels[:max_len]):
        x[i] = choi_to_features(ch.choi, dim, representation)
        mask[i] = 1.0
    return x, mask
