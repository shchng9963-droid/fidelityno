
"""Simulator backend selection.

Primary differentiable backend choice for v1 is dynamiqs (JAX/autodiff friendly).  The
lightweight closed-form channel generators and scipy matrix exponentials are used in
unit tests and smoke runs for speed/reproducibility; dynamiqs is imported here and is
available for differentiable Lindblad extensions without changing call sites.
"""
from __future__ import annotations
import importlib.metadata as md

def backend_versions() -> dict[str,str]:
    out={}
    for pkg in ["dynamiqs", "qiskit-dynamics", "qutip", "qiskit", "neuraloperator", "torch"]:
        try: out[pkg]=md.version(pkg)
        except md.PackageNotFoundError: out[pkg]="not-installed"
    return out

PREFERRED_BACKEND = "dynamiqs"
