"""Canonical fidelity primitives for FidelityNO.

Single source of truth for *every* "fidelity" computed in the project.
PRX Quantum reviewers will demand unambiguous definitions and explicit
relationships among the variants; this module enforces both.

Conventions (used everywhere in this codebase)
----------------------------------------------
- Hilbert-space dimension: ``d``.
- Channel ``Lambda``: CPTP map ``Lambda: M_d -> M_d``.
- Choi matrix (column-major / "vec" convention used by ``physics.channels.base``)::

      C_Lambda = sum_{i,j} |i><j| (X) Lambda(|i><j|)         in C^{d^2 x d^2}.

  Trace-preserving means ``Tr_out[ C_Lambda ] = I_d``.
- For a unitary ``U``, the unitary channel has Kraus ``{U}`` and Choi
  ``C_U = d |Phi_U><Phi_U|`` where ``|Phi_U> = (I (X) U) |Phi+>`` and
  ``|Phi+> = sum_i |ii> / sqrt(d)``.

Fidelity variants and the formulas this module implements
---------------------------------------------------------
1) Entanglement fidelity (``entanglement_fidelity``)
       F_e(Lambda, Lambda_target) = Tr[ C_Lambda^dagger C_Lambda_target ] / d^2.
   For Kraus reps ``{K_i}``, ``{M_j}``::
       F_e = (1/d^2) sum_{i,j} | Tr(M_j^dagger K_i) |^2.
   When ``Lambda_target = U`` (a unitary)::
       F_e(Lambda, U) = (1/d^2) sum_i | Tr(U^dagger K_i) |^2.
   This is what ``physics.composition.process_fidelity`` historically
   computed and what the v1 model is trained on.

2) Average gate fidelity (``average_gate_fidelity``)
       F_avg(Lambda, U) = ( d * F_e(Lambda, U) + 1 ) / ( d + 1 ).
   Standard Horodecki / Nielsen result (Nielsen, Phys. Lett. A 303
   (2002) 249). Reported by experimentalists from RB. Cheaply derived
   from F_e at no extra cost.

3) Process fidelity (``process_fidelity``)
   The literature is inconsistent. We adopt the most common QI
   convention::
       F_pro(Lambda, Lambda_target) := F_e(Lambda, Lambda_target).
   ``process_fidelity`` is therefore an alias of ``entanglement_fidelity``.
   We keep it because (a) v1 code uses the name and (b) Nielsen-Chuang
   §9.3 uses it as a synonym.

4) Uhlmann state fidelity (``state_fidelity``)
       F_state(rho, sigma) = ( Tr sqrt( sqrt(rho) sigma sqrt(rho) ) )^2.
   For pure ``rho = |psi><psi|``: F_state = <psi| sigma |psi>.
   Note: some papers define it without the square. We use the
   "squared" convention so that for pure states F_state in [0, 1].

What we do *not* compute
------------------------
- Diamond distance / 1->1 norm. See ``models/baselines/diamond_sdp.py``
  (added in PRXQ track P1.6).
- Bures angle / Hellinger fidelity.

References
----------
- M. A. Nielsen, "A simple formula for the average gate fidelity of a
  quantum dynamical operation," Phys. Lett. A 303, 249 (2002).
- Horodecki et al., "General teleportation channel...", PRA 60, 1888.
- Nielsen & Chuang, "Quantum Computation and Quantum Information",
  Cambridge (2010), §9.2.2.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm

from physics.channels.base import Channel

__all__ = [
    "FIDELITY_KIND",
    "entanglement_fidelity",
    "process_fidelity",
    "average_gate_fidelity",
    "state_fidelity",
    "ef_to_avg",
    "avg_to_ef",
    "fidelity_formula",
]


# Canonical kind tag written into dataset manifests.
# v1 trained F_e against per-step reference targets; we keep that as the
# canonical training signal but report F_avg as a derived quantity.
FIDELITY_KIND = "entanglement_fidelity"


def fidelity_formula() -> str:
    """LaTeX string of the canonical training-signal formula. Used in tables."""
    return r"F_e(\Lambda,\Lambda^{\mathrm{tgt}}) = \mathrm{Tr}[C_\Lambda^\dagger C_{\Lambda^{\mathrm{tgt}}}]/d^2"


def _identity_channel(dim: int) -> Channel:
    return Channel("identity", dim, kraus=[np.eye(dim, dtype=np.complex128)])


def entanglement_fidelity(ch: Channel, target: Channel | None = None) -> float:
    """Entanglement fidelity F_e(Lambda, target) = Tr[C_Lambda^dagger C_target] / d^2.

    Parameters
    ----------
    ch : Channel
        The actual (noisy) channel Lambda.
    target : Channel or None
        Target channel; if None, taken to be the identity channel on dim(ch).

    Returns
    -------
    float in [0, 1]
        Numerically clipped.
    """
    target = target if target is not None else _identity_channel(ch.dim)
    if target.dim != ch.dim:
        raise ValueError(f"dimension mismatch: ch.dim={ch.dim}, target.dim={target.dim}")
    val = np.trace(ch.choi.conj().T @ target.choi).real / (ch.dim ** 2)
    return float(np.clip(val, 0.0, 1.0))


# Process fidelity := entanglement fidelity in our convention.
process_fidelity = entanglement_fidelity


def ef_to_avg(F_e: float, dim: int) -> float:
    """Convert entanglement fidelity to average gate fidelity (Nielsen 2002).

    F_avg = (d * F_e + 1) / (d + 1).
    """
    return float((dim * F_e + 1.0) / (dim + 1.0))


def avg_to_ef(F_avg: float, dim: int) -> float:
    """Inverse of ``ef_to_avg``."""
    return float(((dim + 1.0) * F_avg - 1.0) / dim)


def average_gate_fidelity(ch: Channel, target: Channel | None = None) -> float:
    """Average gate fidelity. Derived as ef_to_avg(F_e(ch, target), dim)."""
    F_e = entanglement_fidelity(ch, target)
    return ef_to_avg(F_e, ch.dim)


def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann state fidelity ( Tr sqrt(sqrt(rho) sigma sqrt(rho)) )^2.

    For pure rho = |psi><psi|: equals <psi| sigma |psi>. Clipped to [0, 1].
    """
    sr = sqrtm(rho)
    inner = sqrtm(sr @ sigma @ sr)
    val = (np.trace(inner).real) ** 2
    return float(np.clip(val, 0.0, 1.0))
