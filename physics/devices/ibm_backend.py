"""Pull T1, T2, gate-time, gate-error data from archived IBM backend
snapshots (qiskit_ibm_runtime.fake_provider.Fake*V2) and convert each
qubit/edge into FidelityNO's amplitude-damping + phase-damping +
depolarizing channel triple.

The Fake*V2 classes ship with calibration data captured from real
hardware (Falcon r5/r10 and Eagle r1 generations). They are ideal for a
PRX Quantum honest "real-noise" validation: zero auth, deterministic,
fully-cited.

Conversions (see physics/channels/units.py for the formulas):
  - Amplitude damping:  γ_AD  = 1 - exp(-t_gate / T1)
  - Phase damping:      λ_PD  = 1 - exp(-2 t_gate / T_φ),  T_φ from T1, T2.
  - Depolarizing:       p_dep = (d+1)/d * RB_error_per_gate, d=2 single-qubit.

For two-qubit gates we use the gate's reported error directly as the
total Pauli weight in a `two_qubit_depolarizing` channel (a standard
approximation in QPU benchmarking literature). Any leakage / coherent
component beyond depolarizing-plus-T1/T2 is not modelled here; that is a
known limitation flagged in PRXQ_PLAN.md and the paper Limitations.

Public API
----------
- ``device_qubit_channel(backend, qubit) -> Channel``
  Single-qubit channel = amplitude_damping ∘ phase_damping ∘ depolarizing
  for one qubit's calibration.
- ``device_two_qubit_channel(backend, edge) -> Channel``
  Two-qubit imperfect-cnot channel using the edge's reported CNOT/ECR
  error as a depolarizing-equivalent.
- ``available_fake_backends() -> list[str]``
- ``load_fake_backend(name)``
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from physics.channels.base import Channel, kraus_to_superop
from physics.channels.single_qubit import (
    amplitude_damping,
    depolarizing,
    phase_damping,
)
from physics.channels.two_qubit import (
    cnot_unitary,
    two_qubit_depolarizing,
)
from physics.channels.units import (
    Tphi_from_T1_T2,
    depolarizing_p_from_rb,
    gamma_from_T1,
)

# Modern IBM hardware that has Fake*V2 snapshots in qiskit_ibm_runtime.
# Eagle (127 qubits) and Falcon r10 (27 qubits) cover the full 2022-2024
# generation. Honeywell / Quantinuum / IonQ are not in qiskit fake_provider.
_PRIORITY_BACKENDS: tuple[str, ...] = (
    "FakeWashingtonV2",   # Eagle r1, 127 qubits
    "FakeKolkataV2",      # Falcon r10, 27 qubits
    "FakeMumbaiV2",       # Falcon r5, 27 qubits
    "FakeHanoiV2",        # Falcon r5, 27 qubits
    "FakeMontrealV2",     # Falcon r4, 27 qubits
    "FakeCairoV2",        # Falcon r5, 27 qubits
    "FakeKolkataV2",      # duplicate-safe
    "FakeBrooklynV2",     # Hummingbird r2, 65 qubits
    "FakeManhattanV2",    # Hummingbird r2, 65 qubits
)


def available_fake_backends() -> list[str]:
    """List Fake*V2 backend class names that ship with qiskit_ibm_runtime."""
    import qiskit_ibm_runtime.fake_provider as fp
    return sorted(n for n in dir(fp) if n.startswith("Fake") and n.endswith("V2"))


def load_fake_backend(name: str):
    """Instantiate ``qiskit_ibm_runtime.fake_provider.<name>``."""
    import qiskit_ibm_runtime.fake_provider as fp
    if not hasattr(fp, name):
        raise KeyError(
            f"Unknown fake backend {name!r}. Available: {available_fake_backends()}"
        )
    return getattr(fp, name)()


def _two_qubit_gate_name(backend) -> str:
    for g in ("ecr", "cx", "cz"):
        if g in backend.operation_names:
            return g
    raise RuntimeError(f"Backend {backend.name} exposes no 2q entangling gate.")


def _single_qubit_gate_name(backend) -> str:
    for g in ("sx", "x"):
        if g in backend.operation_names:
            return g
    raise RuntimeError(f"Backend {backend.name} exposes no 1q gate.")


def _qubit_calibration(backend, qubit: int) -> dict:
    """Return T1, T2, gate-time, gate-error for one qubit (all SI seconds)."""
    qp = backend.qubit_properties(qubit)
    g1 = _single_qubit_gate_name(backend)
    target_entry = backend.target[g1].get((qubit,))
    if target_entry is None:
        raise RuntimeError(f"Qubit {qubit} on {backend.name} has no calibrated {g1}.")
    return {
        "qubit": qubit,
        "T1_s": float(qp.t1) if qp.t1 is not None else math.inf,
        "T2_s": float(qp.t2) if qp.t2 is not None else math.inf,
        "gate_time_s": float(target_entry.duration) if target_entry.duration is not None else 0.0,
        "gate_error": float(target_entry.error) if target_entry.error is not None else 0.0,
        "gate_name": g1,
    }


def _edge_calibration(backend, edge: tuple[int, int]) -> dict:
    g2 = _two_qubit_gate_name(backend)
    entry = backend.target[g2].get(tuple(edge))
    if entry is None:
        raise RuntimeError(f"Edge {edge} on {backend.name} has no calibrated {g2}.")
    return {
        "edge": tuple(edge),
        "gate_time_s": float(entry.duration) if entry.duration is not None else 0.0,
        "gate_error": float(entry.error) if entry.error is not None else 0.0,
        "gate_name": g2,
    }


def device_qubit_channel(backend, qubit: int) -> Channel:
    """Build a single-qubit Choi for a real-device qubit's noise.

    The returned channel is the cascade
        AD(γ) ∘ PD(λ_p) ∘ Depolarizing(p_dep)
    where γ, λ_p, p_dep come from the device's T1, T2, single-qubit RB
    error and gate time (seconds → microseconds via 1e6).
    """
    cal = _qubit_calibration(backend, qubit)
    T1_us = cal["T1_s"] * 1e6
    T2_us = cal["T2_s"] * 1e6
    gate_us = cal["gate_time_s"] * 1e6

    gamma = gamma_from_T1(T1_us, gate_us) if T1_us > 0 else 0.0

    if T2_us > 0 and 2.0 * T1_us > 0:
        Tphi_us = Tphi_from_T1_T2(T1_us, T2_us)
        if math.isinf(Tphi_us):
            lambda_p = 0.0
        else:
            lambda_p = float(1.0 - math.exp(-2.0 * gate_us / Tphi_us))
    else:
        lambda_p = 0.0

    p_dep = depolarizing_p_from_rb(cal["gate_error"], dim=2)
    p_dep = float(np.clip(p_dep, 0.0, 0.99))

    ch_ad = amplitude_damping(gamma)
    ch_pd = phase_damping(lambda_p)
    ch_dp = depolarizing(p_dep)
    superop = ch_dp.superop @ ch_pd.superop @ ch_ad.superop
    return Channel(
        f"ibm_qubit_{qubit}",
        2,
        superop=superop,
        params=np.array([gamma, lambda_p, p_dep]),
        metadata={
            "source": "qiskit_ibm_runtime.fake_provider",
            "backend": backend.name,
            "T1_us": T1_us,
            "T2_us": T2_us,
            "gate_time_us": gate_us,
            "rb_error": cal["gate_error"],
            "components": ["amplitude_damping", "phase_damping", "depolarizing"],
        },
    )


def device_two_qubit_channel(backend, edge: tuple[int, int]) -> Channel:
    """Build a 2-qubit imperfect-CNOT channel for a real-device edge.

    Channel = ideal CNOT (or ECR cast as CNOT-equivalent) followed by
    a 2-qubit depolarizing channel with weight equal to the device's
    reported gate error. Two-qubit dim=4.
    """
    cal = _edge_calibration(backend, edge)
    p = float(np.clip(cal["gate_error"], 0.0, 0.5))

    # Ideal CNOT Kraus
    U = cnot_unitary()
    cnot_channel = Channel("ideal_cnot", 4, kraus=[U])
    # Mix with depolarizing
    dep = two_qubit_depolarizing(p)
    superop = dep.superop @ cnot_channel.superop
    return Channel(
        f"ibm_edge_{edge[0]}_{edge[1]}",
        4,
        superop=superop,
        params=np.array([p]),
        metadata={
            "source": "qiskit_ibm_runtime.fake_provider",
            "backend": backend.name,
            "edge": list(edge),
            "gate_name": cal["gate_name"],
            "gate_error": cal["gate_error"],
            "gate_time_us": cal["gate_time_s"] * 1e6,
            "components": ["ideal_cnot", "two_qubit_depolarizing"],
        },
    )


def list_calibrated_qubits(backend) -> list[int]:
    """All qubits that have a single-qubit gate calibration."""
    g1 = _single_qubit_gate_name(backend)
    keys = list(backend.target[g1].keys())
    return sorted({k[0] for k in keys if isinstance(k, tuple) and len(k) == 1})


def list_calibrated_edges(backend) -> list[tuple[int, int]]:
    """All directed edges with a two-qubit gate calibration."""
    g2 = _two_qubit_gate_name(backend)
    return sorted(tuple(k) for k in backend.target[g2].keys()
                  if isinstance(k, tuple) and len(k) == 2)
