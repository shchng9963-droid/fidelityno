"""Physical-units bridge for FidelityNO channel parameters (PRX Quantum P0.2).

PRX Quantum reviewers will reject anything that says "γ ∼ U(0, 0.25)"
without an explanation of what physical regime that covers. This module
turns the abstract parameters used by ``physics/channels`` into and out
of the standard device-side numbers (T₁, T₂, gate time, RB error per
Clifford, etc.) that experimentalists report.

All physical times are in **microseconds** unless otherwise stated.
This is the convention of IBM Quantum, Quantinuum, and IonQ datasheets.

Channel parameter ↔ device-physics dictionary
---------------------------------------------

Amplitude damping
~~~~~~~~~~~~~~~~~
The amplitude-damping Kraus rep with parameter ``γ ∈ [0, 1]`` corresponds
to an excited-state population that has decayed by

    γ = 1 − exp(−t_gate / T₁)

so that ``T₁ = -t_gate / ln(1 − γ)``. v1 used ``γ ∼ U(0, 0.25)``; with a
50 ns gate that is ``T₁ ∈ [174 ns, ∞)`` if you read it loosely. Realistic
quantum-memory regimes (γ small) correspond to ``T₁ >> t_gate``.

Phase damping
~~~~~~~~~~~~~
With parameter ``λ_p ∈ [0, 1]`` and pure-dephasing time T_φ:

    λ_p = 1 − exp(−2 t_gate / T_φ).

If only T₂ is reported (the usual case), the relation
``1/T_φ = 1/T₂ − 1/(2 T₁)`` recovers T_φ. We provide both helpers.

Depolarizing
~~~~~~~~~~~~
The depolarizing parameter ``p`` we use is "total error probability"
``P(any non-I Pauli applied) = p``, distributed equally among X, Y, Z.
The conversion to RB-style Clifford error rate per gate is:

    r_RB = (d / (d + 1)) · p          (eq. 6 in Magesan et al. PRA 85, 042311)

For d=2: ``r_RB = (2/3) p``. So an IBM single-qubit RB error of 1e-3
corresponds to ``p = 1.5e-3`` in our convention.

Pauli channel
~~~~~~~~~~~~~
Triple ``(p_X, p_Y, p_Z)`` with sum ≤ 1; identity probability
``p_0 = 1 − p_X − p_Y − p_Z``. In our regimes summary we report
``p_total = p_X + p_Y + p_Z``.

Lindblad
~~~~~~~~
The Lindblad family integrates ``L = −i[H, ·] + Σ_k D[c_k]`` for time t.
Sampled ranges:

    ω_x, ω_z ∈ U(−0.5, 0.5)   [angular frequency in 1 / unit-time]
    γ_damp   ∈ U(0, 0.15)     [collapse rate of σ_minus]
    γ_phase  ∈ U(0, 0.12)     [collapse rate of σ_z]
    t        ∈ U(0.05, 1.0)   [unit-time]

Adopting ``unit-time = μs`` makes ω_{x,z} ∈ [−0.5, 0.5] MHz Rabi
amplitudes and (γ_damp, γ_phase) in [0, 0.15] MHz which corresponds to
T₁ ≥ 6.7 μs and T_φ ≥ 8.3 μs — i.e., a noisy NISQ regime. Adopting
``unit-time = ms`` (trapped-ion) maps them to kHz / ms.

Device regimes summary table
----------------------------
Read from ``DEVICE_REGIMES`` below or call ``device_regime_table()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceRegime:
    """Reference operating point of a representative quantum device.

    All times are in microseconds.
    """
    name: str                 # short label, e.g. "ibm_superconducting"
    description: str          # one-line free text for paper/table
    gate_time_us: float       # representative single-qubit gate time
    T1_us: float              # representative T1
    T2_us: float              # representative T2 (>= 0, <= 2 T1)
    rb_1q_error: float        # representative single-qubit RB error per Clifford
    rb_2q_error: float        # representative two-qubit RB error
    citation: str             # one-line attribution


# Numbers below are *representative*, not actual single-day calibration
# data. They are used only to compute a "what regime does γ ∼ U(0,X) cover"
# annotation table in the paper appendix. Real-data validation lives in
# physics/devices/ (PRXQ track P0.1).
DEVICE_REGIMES: dict[str, DeviceRegime] = {
    "ibm_superconducting": DeviceRegime(
        name="ibm_superconducting",
        description="IBM Eagle-class superconducting transmon (e.g. ibm_kyiv, ibm_brisbane, 2024).",
        gate_time_us=0.05,        # 50 ns single-qubit
        T1_us=120.0,
        T2_us=90.0,
        rb_1q_error=2.5e-4,
        rb_2q_error=8e-3,
        citation="IBM Quantum Documentation, accessed 2025; arXiv:2106.00675 reports similar values for Falcon/Eagle.",
    ),
    "google_superconducting": DeviceRegime(
        name="google_superconducting",
        description="Google Sycamore / Willow-class transmon.",
        gate_time_us=0.025,       # 25 ns
        T1_us=20.0,
        T2_us=15.0,
        rb_1q_error=8e-4,
        rb_2q_error=6e-3,
        citation="Arute et al., Nature 574 (2019); Acharya et al., Nature 614 (2023).",
    ),
    "quantinuum_trapped_ion": DeviceRegime(
        name="quantinuum_trapped_ion",
        description="Quantinuum H1 / H2 trapped-ion (Yb-171).",
        gate_time_us=5.0,         # ~5 us single-qubit
        T1_us=50_000_000.0,       # >50 s ≈ effectively infinite
        T2_us=3_000_000.0,        # ~3 s coherence
        rb_1q_error=2e-5,
        rb_2q_error=2e-3,
        citation="Quantinuum H1 datasheet 2024; Pino et al., Nature 592 (2021).",
    ),
    "ionq_aria": DeviceRegime(
        name="ionq_aria",
        description="IonQ Aria trapped-ion (Yb-171).",
        gate_time_us=130.0,
        T1_us=10_000_000.0,
        T2_us=1_000_000.0,
        rb_1q_error=5e-4,
        rb_2q_error=4e-3,
        citation="IonQ Aria specifications, 2024.",
    ),
    "neutral_atom_qa": DeviceRegime(
        name="neutral_atom_qa",
        description="Neutral-atom array (QuEra, Atom Computing) Rydberg gates.",
        gate_time_us=0.5,
        T1_us=4_000_000.0,
        T2_us=1_500.0,
        rb_1q_error=2e-3,
        rb_2q_error=5e-3,
        citation="Bluvstein et al., Nature 626 (2024); Graham et al., Nature 622 (2023).",
    ),
}


# ---- Conversions: amplitude damping ----

def gamma_from_T1(T1_us: float, gate_time_us: float) -> float:
    """γ = 1 − exp(−t_gate / T₁). Valid for T₁ > 0, t > 0."""
    if T1_us <= 0 or gate_time_us < 0:
        raise ValueError("T1 and gate_time must be positive.")
    return float(1.0 - math.exp(-gate_time_us / T1_us))


def T1_from_gamma(gamma: float, gate_time_us: float) -> float:
    """T₁ = −t_gate / ln(1 − γ). Returns +inf if γ = 0."""
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1).")
    if gamma == 0.0:
        return math.inf
    return float(-gate_time_us / math.log(1.0 - gamma))


# ---- Conversions: phase damping ----

def Tphi_from_T1_T2(T1_us: float, T2_us: float) -> float:
    """1/T_φ = 1/T₂ − 1/(2 T₁). Returns +inf if RHS ≤ 0."""
    inv_phi = 1.0 / T2_us - 1.0 / (2.0 * T1_us)
    if inv_phi <= 0:
        return math.inf
    return float(1.0 / inv_phi)


def lambda_p_from_T2(T1_us: float, T2_us: float, gate_time_us: float) -> float:
    """λ_p = 1 − exp(−2 t_gate / T_φ)."""
    Tphi = Tphi_from_T1_T2(T1_us, T2_us)
    if math.isinf(Tphi):
        return 0.0
    return float(1.0 - math.exp(-2.0 * gate_time_us / Tphi))


def T2_from_T1_lambda_p(T1_us: float, lambda_p: float, gate_time_us: float) -> float:
    """Solve for T₂ given T₁, λ_p, gate time."""
    if not 0.0 <= lambda_p < 1.0:
        raise ValueError("lambda_p must be in [0, 1).")
    if lambda_p == 0.0:
        return 2.0 * T1_us
    Tphi = -2.0 * gate_time_us / math.log(1.0 - lambda_p)
    inv_T2 = 1.0 / Tphi + 1.0 / (2.0 * T1_us)
    return float(1.0 / inv_T2)


# ---- Conversions: depolarizing / RB ----

def depolarizing_p_from_rb(rb_error: float, dim: int = 2) -> float:
    """Convert RB error rate (Clifford-average) to our depolarizing-p convention.

    Magesan, Gambetta, Emerson PRA 85, 042311 (2012):
        r_RB = (d / (d+1)) p,   so   p = (d+1)/d · r_RB.
    """
    return float((dim + 1) / dim * rb_error)


def rb_from_depolarizing_p(p: float, dim: int = 2) -> float:
    """Inverse of ``depolarizing_p_from_rb``."""
    return float(dim / (dim + 1) * p)


# ---- "What does the v1 sampling range cover?" annotation ----

@dataclass(frozen=True)
class FamilyRangeAnnotation:
    family: str
    code_range: tuple[float, float]
    physical_meaning: str
    regimes_covered: dict[str, str]   # device_name -> short regime tag


def annotate_amplitude_damping_range(
    code_range: tuple[float, float] = (0.0, 0.25),
) -> FamilyRangeAnnotation:
    regimes = {}
    for name, dev in DEVICE_REGIMES.items():
        gamma_dev = gamma_from_T1(dev.T1_us, dev.gate_time_us)
        if gamma_dev <= code_range[1]:
            regimes[name] = (
                f"covered (γ_dev≈{gamma_dev:.2e} at t_gate={dev.gate_time_us}μs, "
                f"T1={dev.T1_us}μs)"
            )
        else:
            regimes[name] = f"OUTSIDE (γ_dev≈{gamma_dev:.2e}, would need γ_max > {gamma_dev:.2e})"
    return FamilyRangeAnnotation(
        family="amplitude_damping",
        code_range=code_range,
        physical_meaning=(
            f"γ ∈ [{code_range[0]}, {code_range[1]}] = "
            f"T1 ∈ [{T1_from_gamma(code_range[1], 0.05):.2f} μs at 50 ns gate, ∞)"
        ),
        regimes_covered=regimes,
    )


def annotate_phase_damping_range(
    code_range: tuple[float, float] = (0.0, 0.30),
) -> FamilyRangeAnnotation:
    regimes = {}
    for name, dev in DEVICE_REGIMES.items():
        lp_dev = lambda_p_from_T2(dev.T1_us, dev.T2_us, dev.gate_time_us)
        regimes[name] = (
            f"λ_p_dev≈{lp_dev:.2e} (T1={dev.T1_us}, T2={dev.T2_us}, t_gate={dev.gate_time_us}μs) — "
            + ("covered" if lp_dev <= code_range[1] else "OUTSIDE")
        )
    return FamilyRangeAnnotation(
        family="phase_damping",
        code_range=code_range,
        physical_meaning=(
            f"λ_p ∈ [{code_range[0]}, {code_range[1]}] for representative t_gate=50 ns and T1=120 μs "
            f"=> T2 ∈ [{T2_from_T1_lambda_p(120, code_range[1], 0.05):.2f} μs, 240 μs)"
        ),
        regimes_covered=regimes,
    )


def annotate_depolarizing_range(
    code_range: tuple[float, float] = (0.0, 0.20),
) -> FamilyRangeAnnotation:
    regimes = {}
    for name, dev in DEVICE_REGIMES.items():
        # interpret as 1q regime
        p_dev = depolarizing_p_from_rb(dev.rb_1q_error, dim=2)
        regimes[name] = (
            f"p_dev_1q≈{p_dev:.2e} (1q RB={dev.rb_1q_error:.1e}) — "
            + ("covered" if p_dev <= code_range[1] else "OUTSIDE")
        )
    return FamilyRangeAnnotation(
        family="depolarizing",
        code_range=code_range,
        physical_meaning=(
            f"p ∈ [{code_range[0]}, {code_range[1]}] = total error probability per channel "
            f"= RB error per Clifford in [{rb_from_depolarizing_p(code_range[0]):.2e}, "
            f"{rb_from_depolarizing_p(code_range[1]):.2e}] (d=2)"
        ),
        regimes_covered=regimes,
    )


def device_regime_table() -> str:
    """Return a Markdown table of device regimes for the paper appendix."""
    lines = [
        "| Device | Gate (μs) | T₁ (μs) | T₂ (μs) | RB-1q | RB-2q | γ_AD | λ_PD | p_dep |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, dev in DEVICE_REGIMES.items():
        gamma = gamma_from_T1(dev.T1_us, dev.gate_time_us)
        lp = lambda_p_from_T2(dev.T1_us, dev.T2_us, dev.gate_time_us)
        p_dep = depolarizing_p_from_rb(dev.rb_1q_error, dim=2)
        lines.append(
            f"| {name} | {dev.gate_time_us:g} | {dev.T1_us:g} | {dev.T2_us:g} "
            f"| {dev.rb_1q_error:.1e} | {dev.rb_2q_error:.1e} "
            f"| {gamma:.2e} | {lp:.2e} | {p_dep:.2e} |"
        )
    return "\n".join(lines)


def all_family_annotations() -> list[FamilyRangeAnnotation]:
    return [
        annotate_amplitude_damping_range(),
        annotate_phase_damping_range(),
        annotate_depolarizing_range(),
    ]
