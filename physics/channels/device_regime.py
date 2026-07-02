"""Device-regime channel samplers for FidelityNO PRXQ track P0.1b.

These samplers use the *same* constructors as v1 but with parameter
ranges narrowed to physically realistic device numbers. Used to retrain
FidelityNO on the noise distribution it will actually be deployed on.

Range source: ``physics/channels/units.py`` Appendix A — modern IBM
Falcon/Eagle, Google Sycamore, Quantinuum H1/H2, IonQ Aria, neutral-atom
QA. We bracket their union with a small safety margin to allow a bit of
extrapolation toward harder noise.

Single-qubit ranges (after extrapolation):
    γ (amp damping)       ∼ U(0,  0.01)
    λ_p (phase damping)   ∼ U(0,  0.02)
    p_dep (depolarizing)  ∼ U(0,  0.005)
    Pauli total weight    ∼ U(0,  0.005)
    Lindblad t            ∼ U(0,  0.05)  with weak Hamiltonian / jumps

These cover IBM Cairo's 2q error (~7e-3 on Pauli weight per gate when
chained as 1q-equivalents at the noisy end) while excluding the
"break it" regime γ > 0.1 that v1 trained on.

The two-qubit device regime is added below; we keep the v1 ranges as
the "broad" default and just provide a "device" alias.

Convention: every sampler returns a Channel of dim 2 (single-qubit) or
dim 4 (two-qubit) compatible with the rest of the codebase, and stores
``regime`` in metadata so dataset analysis can group by regime later.
"""

from __future__ import annotations

import numpy as np

from physics.channels.base import Channel
from physics.channels.lindblad import lindblad_channel
from physics.channels.single_qubit import (
    I2,
    X,
    Y,
    Z,
    amplitude_damping,
    depolarizing,
    pauli_channel,
    phase_damping,
)
from physics.channels.two_qubit import (
    correlated_dephasing,
    imperfect_gate,
    two_qubit_depolarizing,
)


# ---- single-qubit device-regime sampler ----

DEVICE_RANGES_1Q = {
    "amplitude_damping": (0.0, 0.01),
    "phase_damping":     (0.0, 0.02),
    "depolarizing":      (0.0, 0.005),
    "pauli_total":       (0.0, 0.005),
    "lindblad_omega":    (-0.05, 0.05),
    "lindblad_gd":       (0.0, 0.005),
    "lindblad_gp":       (0.0, 0.004),
    "lindblad_t":        (0.0, 0.05),
}


def sample_single_qubit_device(rng: np.random.Generator, family: str | None = None) -> Channel:
    """Single-qubit channel sampled from the *device-regime* ranges.

    Identical Kraus / Choi structure as v1's ``sample_single_qubit``;
    only the parameter range is narrowed. Lindblad uses the same
    superoperator as v1 with smaller jump rates.
    """
    fams = ["amplitude_damping", "phase_damping", "depolarizing", "pauli", "lindblad"]
    family = family or rng.choice(fams)

    if family == "amplitude_damping":
        ch = amplitude_damping(rng.uniform(*DEVICE_RANGES_1Q["amplitude_damping"]))
    elif family == "phase_damping":
        ch = phase_damping(rng.uniform(*DEVICE_RANGES_1Q["phase_damping"]))
    elif family == "depolarizing":
        ch = depolarizing(rng.uniform(*DEVICE_RANGES_1Q["depolarizing"]))
    elif family == "pauli":
        probs = rng.dirichlet([1, 1, 1]) * rng.uniform(*DEVICE_RANGES_1Q["pauli_total"])
        ch = pauli_channel(*probs)
    elif family == "lindblad":
        wx = rng.uniform(*DEVICE_RANGES_1Q["lindblad_omega"])
        wz = rng.uniform(*DEVICE_RANGES_1Q["lindblad_omega"])
        gd = rng.uniform(*DEVICE_RANGES_1Q["lindblad_gd"])
        gp = rng.uniform(*DEVICE_RANGES_1Q["lindblad_gp"])
        t  = rng.uniform(*DEVICE_RANGES_1Q["lindblad_t"])
        sm = np.array([[0, 1], [0, 0]], dtype=np.complex128)
        H = 0.5 * (wx * X + wz * Z)
        jumps = []
        if gd > 0: jumps.append(np.sqrt(gd) * sm)
        if gp > 0: jumps.append(np.sqrt(gp) * Z)
        ch = lindblad_channel(H, jumps, t, params=np.array([wx, wz, gd, gp, t]))
    else:
        raise ValueError(f"unknown family {family}")

    ch.metadata = {**(ch.metadata or {}), "regime": "device", "family_label": family}
    return ch


# ---- two-qubit device-regime sampler ----

DEVICE_RANGES_2Q = {
    "correlated_dephasing": (0.0, 0.01),
    "two_qubit_depolarizing": (0.0, 0.015),
    "imperfect_cnot_theta": (-0.02, 0.02),  # tighter than v1 (-0.08, 0.08)
    "imperfect_cnot_p":     (0.0, 0.012),
    "imperfect_swap_theta": (-0.02, 0.02),
    "imperfect_swap_p":     (0.0, 0.012),
}


def sample_two_qubit_device(rng: np.random.Generator, family: str | None = None) -> Channel:
    fams = ["correlated_dephasing", "two_qubit_depolarizing", "imperfect_cnot", "imperfect_swap"]
    family = family or rng.choice(fams)
    if family == "correlated_dephasing":
        ch = correlated_dephasing(rng.uniform(*DEVICE_RANGES_2Q["correlated_dephasing"]))
    elif family == "two_qubit_depolarizing":
        ch = two_qubit_depolarizing(rng.uniform(*DEVICE_RANGES_2Q["two_qubit_depolarizing"]))
    elif family == "imperfect_cnot":
        ch = imperfect_gate("cnot",
                            theta=rng.uniform(*DEVICE_RANGES_2Q["imperfect_cnot_theta"]),
                            p=rng.uniform(*DEVICE_RANGES_2Q["imperfect_cnot_p"]))
    elif family == "imperfect_swap":
        ch = imperfect_gate("swap",
                            theta=rng.uniform(*DEVICE_RANGES_2Q["imperfect_swap_theta"]),
                            p=rng.uniform(*DEVICE_RANGES_2Q["imperfect_swap_p"]))
    else:
        raise ValueError(f"unknown 2q family {family}")
    ch.metadata = {**(ch.metadata or {}), "regime": "device", "family_label": family}
    return ch


def regime_summary() -> dict:
    """Return a dict suitable for embedding in a manifest."""
    return {
        "regime": "device",
        "single_qubit_ranges": DEVICE_RANGES_1Q,
        "two_qubit_ranges": DEVICE_RANGES_2Q,
        "rationale": (
            "Narrow sampling around modern QPU calibration (IBM Falcon/Eagle, "
            "Google Sycamore, Quantinuum H1/H2, IonQ Aria). All ranges allow "
            "a small safety margin above the device's typical operating point "
            "to support transfer to noisier days. See physics/channels/units.py."
        ),
    }
