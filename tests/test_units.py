"""Tests for physical-units conversions in physics/channels/units.py
(PRX Quantum P0.2)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from physics.channels.units import (
    DEVICE_REGIMES,
    T1_from_gamma,
    T2_from_T1_lambda_p,
    Tphi_from_T1_T2,
    annotate_amplitude_damping_range,
    depolarizing_p_from_rb,
    device_regime_table,
    gamma_from_T1,
    lambda_p_from_T2,
    rb_from_depolarizing_p,
)


@pytest.mark.parametrize("T1_us,gate_us", [(50.0, 0.05), (120.0, 0.025), (10_000.0, 5.0)])
def test_amplitude_damping_round_trip(T1_us, gate_us):
    gamma = gamma_from_T1(T1_us, gate_us)
    T1_recovered = T1_from_gamma(gamma, gate_us)
    assert math.isclose(T1_recovered, T1_us, rel_tol=1e-9)


def test_T1_from_gamma_zero_is_infinity():
    assert math.isinf(T1_from_gamma(0.0, 0.05))


@pytest.mark.parametrize("T1_us,T2_us,gate_us", [(120.0, 90.0, 0.05), (20.0, 15.0, 0.025)])
def test_phase_damping_round_trip(T1_us, T2_us, gate_us):
    lp = lambda_p_from_T2(T1_us, T2_us, gate_us)
    T2_recovered = T2_from_T1_lambda_p(T1_us, lp, gate_us)
    assert math.isclose(T2_recovered, T2_us, rel_tol=1e-6)


def test_phase_damping_T2_eq_2T1_means_no_phase_damping():
    # When T2 = 2 T1, the pure-dephasing time T_phi is +inf; lambda_p = 0.
    assert math.isinf(Tphi_from_T1_T2(120.0, 240.0))
    assert lambda_p_from_T2(120.0, 240.0, 0.05) == 0.0


@pytest.mark.parametrize("rb,dim", [(2.5e-4, 2), (8e-3, 4), (5e-4, 8)])
def test_depolarizing_round_trip(rb, dim):
    p = depolarizing_p_from_rb(rb, dim=dim)
    assert math.isclose(rb_from_depolarizing_p(p, dim=dim), rb, rel_tol=1e-12)


def test_depolarizing_d2_factor_two_thirds():
    # r_RB = (d/(d+1)) p; for d=2 this is 2/3.
    assert math.isclose(rb_from_depolarizing_p(0.003, dim=2), 0.002, rel_tol=1e-12)
    assert math.isclose(depolarizing_p_from_rb(0.001, dim=2), 0.0015, rel_tol=1e-12)


def test_device_regime_table_renders():
    md = device_regime_table()
    # five known devices in the table.
    for name in DEVICE_REGIMES.keys():
        assert name in md
    # markdown table header present.
    assert md.count("|") >= 4 * (len(DEVICE_REGIMES) + 2)


def test_amplitude_damping_annotation_marks_known_regimes():
    ann = annotate_amplitude_damping_range((0.0, 0.25))
    assert ann.family == "amplitude_damping"
    # IBM and Google numbers comfortably inside U(0, 0.25).
    assert "covered" in ann.regimes_covered["ibm_superconducting"]
    assert "covered" in ann.regimes_covered["google_superconducting"]


def test_lambda_p_clamped_to_zero_when_T2_geq_2T1():
    # Use T2 slightly > 2 T1 (allowed only as numerical artefact); lambda_p = 0.
    assert lambda_p_from_T2(100.0, 199.99, 0.05) > 0.0
    assert lambda_p_from_T2(100.0, 250.0, 0.05) == 0.0
