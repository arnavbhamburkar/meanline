import math

import pytest
from CoolProp.CoolProp import PropsSI

from meanline.thermo import (
    ThermoState,
    h0_from_h,
    h_from_h0,
    static_from_total,
    total_from_static,
)

FLUID = "Air"
P_ATM = 101_325.0
T_REF = 300.0


def _state_with_enthalpy(h: float) -> ThermoState:
    # Only h matters for the enthalpy relations; other fields are placeholders.
    return ThermoState(fluid=FLUID, p=P_ATM, T=T_REF, h=h, s=0.0, rho=1.0, a=340.0)


# --- enthalpy relations (hand calculations) --------------------------------


def test_h0_from_h_hand_calculation():
    # Static state h = 300 kJ/kg, c = 100 m/s:
    # h0 = 300000 + 0.5*100**2 = 305000 J/kg.
    static_state = _state_with_enthalpy(300_000.0)
    assert h0_from_h(static_state, 100.0) == pytest.approx(305_000.0)


def test_h_from_h0_hand_calculation():
    # Total state h0 = 305 kJ/kg (its .h), c = 100 m/s:
    # h = 305000 - 0.5*100**2 = 300000 J/kg.
    total_state = _state_with_enthalpy(305_000.0)
    assert h_from_h0(total_state, 100.0) == pytest.approx(300_000.0)


def test_h_from_h0_inverts_h0_from_h():
    static_state = _state_with_enthalpy(426_297.8)
    total_state = _state_with_enthalpy(h0_from_h(static_state, 180.0))
    assert h_from_h0(total_state, 180.0) == pytest.approx(static_state.h)


# --- constructors ----------------------------------------------------------


def test_state_from_pT_matches_coolprop():
    state = ThermoState.state_from_pT(FLUID, P_ATM, T_REF)
    assert state.p == P_ATM
    assert state.T == T_REF
    assert state.h == pytest.approx(PropsSI("H", "P", P_ATM, "T", T_REF, FLUID))
    assert state.s == pytest.approx(PropsSI("S", "P", P_ATM, "T", T_REF, FLUID))
    assert state.rho == pytest.approx(PropsSI("D", "P", P_ATM, "T", T_REF, FLUID))
    assert state.a == pytest.approx(PropsSI("A", "P", P_ATM, "T", T_REF, FLUID))


def test_state_from_hs_round_trips_pT():
    ref = ThermoState.state_from_pT(FLUID, P_ATM, T_REF)
    state = ThermoState.state_from_hs(FLUID, ref.h, ref.s)
    assert state.p == pytest.approx(ref.p)
    assert state.T == pytest.approx(ref.T)
    assert state.rho == pytest.approx(ref.rho)
    assert state.a == pytest.approx(ref.a)


def test_state_from_ps_round_trips_pT():
    ref = ThermoState.state_from_pT(FLUID, P_ATM, T_REF)
    state = ThermoState.state_from_ps(FLUID, ref.p, ref.s)
    assert state.T == pytest.approx(ref.T)
    assert state.h == pytest.approx(ref.h)
    assert state.rho == pytest.approx(ref.rho)
    assert state.a == pytest.approx(ref.a)


def test_speed_of_sound_matches_ideal_gas_estimate():
    # Air at 1 atm behaves almost ideally: a ~ sqrt(gamma * R * T), with
    # gamma = cp/cv and R = cp - cv taken from CoolProp (no hard-coded constants).
    state = ThermoState.state_from_pT(FLUID, P_ATM, T_REF)
    cp = PropsSI("CPMASS", "P", P_ATM, "T", T_REF, FLUID)
    cv = PropsSI("CVMASS", "P", P_ATM, "T", T_REF, FLUID)
    a_ideal = math.sqrt((cp / cv) * (cp - cv) * T_REF)
    assert state.a == pytest.approx(a_ideal, rel=0.01)


def test_state_is_immutable():
    state = ThermoState.state_from_pT(FLUID, P_ATM, T_REF)
    with pytest.raises(AttributeError):
        state.p = 2.0 * P_ATM  # type: ignore[misc]


# --- static <-> total conversions -----------------------------------------


def test_total_from_static_enthalpy_and_entropy():
    static = ThermoState.state_from_pT(FLUID, P_ATM, T_REF)
    total = total_from_static(static, 100.0)
    assert total.h - static.h == pytest.approx(0.5 * 100.0**2)
    assert total.s == pytest.approx(static.s)
    assert total.p > static.p
    assert total.T > static.T


def test_total_from_static_temperature_rise_hand_estimate():
    # T0 - T ~ c^2 / (2 cp) for small c; cp comes from CoolProp, not memory.
    c = 100.0
    static = ThermoState.state_from_pT(FLUID, P_ATM, T_REF)
    cp = PropsSI("CPMASS", "P", P_ATM, "T", T_REF, FLUID)
    total = total_from_static(static, c)
    assert total.T - static.T == pytest.approx(c**2 / (2.0 * cp), rel=0.01)


def test_total_from_static_matches_ideal_gas_isentropic_relations():
    # Ideal-gas cross-check: p0/p = (1 + (g-1)/2 M^2)^(g/(g-1)), T0/T = 1 + (g-1)/2 M^2.
    # Air at 1 atm is close to ideal, so the real-gas result should agree closely.
    c = 150.0
    static = ThermoState.state_from_pT(FLUID, P_ATM, T_REF)
    cp = PropsSI("CPMASS", "P", P_ATM, "T", T_REF, FLUID)
    cv = PropsSI("CVMASS", "P", P_ATM, "T", T_REF, FLUID)
    gamma = cp / cv
    mach = c / static.a
    t_ratio = 1.0 + 0.5 * (gamma - 1.0) * mach**2
    p_ratio = t_ratio ** (gamma / (gamma - 1.0))
    total = total_from_static(static, c)
    assert total.T / static.T == pytest.approx(t_ratio, rel=2e-3)
    assert total.p / static.p == pytest.approx(p_ratio, rel=2e-3)


def test_total_equals_static_at_zero_velocity():
    static = ThermoState.state_from_pT(FLUID, P_ATM, T_REF)
    total = total_from_static(static, 0.0)
    assert total.p == pytest.approx(static.p)
    assert total.T == pytest.approx(static.T)
    assert total.h == pytest.approx(static.h)
    assert total.rho == pytest.approx(static.rho)


def test_static_total_static_round_trip():
    static = ThermoState.state_from_pT(FLUID, 250_000.0, 350.0)
    back = static_from_total(total_from_static(static, 220.0), 220.0)
    assert back.p == pytest.approx(static.p)
    assert back.T == pytest.approx(static.T)
    assert back.h == pytest.approx(static.h)
    assert back.s == pytest.approx(static.s)


# --- input validation (ValueError) -----------------------------------------


@pytest.mark.parametrize("bad_p", [0.0, -1.0])
def test_state_from_pT_rejects_non_positive_pressure(bad_p):
    with pytest.raises(ValueError):
        ThermoState.state_from_pT(FLUID, bad_p, T_REF)


@pytest.mark.parametrize("bad_temperature", [0.0, -10.0])
def test_state_from_pT_rejects_non_positive_temperature(bad_temperature):
    with pytest.raises(ValueError):
        ThermoState.state_from_pT(FLUID, P_ATM, bad_temperature)


def test_direct_construction_rejects_non_positive_pressure_and_temperature():
    with pytest.raises(ValueError):
        ThermoState(FLUID, p=-1.0, T=T_REF, h=0.0, s=0.0, rho=1.0, a=340.0)
    with pytest.raises(ValueError):
        ThermoState(FLUID, p=P_ATM, T=0.0, h=0.0, s=0.0, rho=1.0, a=340.0)


def test_negative_velocity_rejected():
    state = ThermoState.state_from_pT(FLUID, P_ATM, T_REF)
    with pytest.raises(ValueError):
        h0_from_h(state, -1.0)
    with pytest.raises(ValueError):
        h_from_h0(state, -1.0)
    with pytest.raises(ValueError):
        total_from_static(state, -1.0)
    with pytest.raises(ValueError):
        static_from_total(state, -1.0)
