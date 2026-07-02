"""True diamond-norm SDP baseline (wrapper around qutip.dnorm).

We delegate the actual SDP to QuTiP's implementation of the Watrous SDP
(see qutip.dnorm; reference: Watrous, "Simpler semidefinite programs for
completely bounded norms", arXiv:1207.5726). We use SCS as the solver
because the more accurate CVXOPT solver is not in our environment.

The fidelity LOWER bound from the diamond distance to identity is
   F_e(Lambda, id) >= 1 - 0.5 * ||Lambda - id||_diamond
(this is the standard 1-norm/diamond -> fidelity inequality applied to
the Choi state of Lambda). For unitary targets one substitutes the
unitary's super-operator for the identity.

Note: ||Lambda - id||_diamond is provably *subadditive* under
composition, so we can also report a "telescoped" upper bound on the
composed diamond norm by summing per-channel diamond norms; this is the
loose telescope baseline already in eval_analytic.py.
"""
from __future__ import annotations

import numpy as np
import qutip as qt


def diamond_norm_of_difference(
    Lambda_choi: np.ndarray,
    d: int,
    solver: str = "SCS",
) -> float:
    """Return ||Lambda - id||_diamond for a CPTP map Lambda on a d-dim
    system given as the Choi matrix Lambda_choi (d*d, d*d).
    """
    # qutip wants a Qobj super-operator, not a Choi.  Convert.
    from physics.channels.base import choi_to_superop
    S = choi_to_superop(Lambda_choi, d)
    # qutip's super has dims=[[[d],[d]],[[d],[d]]]; our superop has the
    # right column-major vec convention which qutip uses too.
    S_qt = qt.Qobj(S, dims=[[[d], [d]], [[d], [d]]], superrep="super")
    S_id = qt.to_super(qt.qeye(d))
    return float(qt.dnorm(S_qt - S_id, solver=solver))


def fidelity_lower_bound_from_diamond(d_norm: float) -> float:
    """Convert ||Lambda - id||_diamond -> F_e lower bound (FvG)."""
    return float(np.clip(1.0 - 0.5 * d_norm, 0.0, 1.0))
