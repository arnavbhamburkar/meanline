"""Thermodynamic state handling via CoolProp.

Planned scope:
    - Static and total (stagnation) state representation.
    - Static <-> total conversions for a given velocity.
    - Real-gas properties from CoolProp; no hard-coded ideal-gas constants
      unless a function is explicitly ideal-gas.

Model level: mean-line, one radius. SI units.


References
----------
TODO(source): Dixon & Hall, total/static relations (chapter and equation).
"""

from dataclasses import dataclass

from CoolProp.CoolProp import PropsSI


@dataclass(frozen=True)
class ThermoState:
    """Thermodynamic state representation.

    Attributes
    ----------
    fluid: str
        Fluid name, e.g. 'Air', 'Nitrogen', 'Hydrogen'. See CoolProp for the
        supported fluids.
    p : float
        Static pressure [Pa].
    T : float
        Static temperature [K].
    h : float
        Static enthalpy [J/kg].
    s : float
        Static entropy [J/kg/K].
    rho : float
        Density [kg/m^3].
    a : float
        Speed of sound [m/s].
    """

    fluid: str
    p: float  # Pa
    T: float  # K
    h: float  # J/kg
    s: float  # J/kg/K
    rho: float  # kg/m^3
    a: float  # m/s

    def __post_init__(self):
        """Validate that pressure and temperature are positive."""
        if self.p <= 0:
            raise ValueError(f"Pressure must be positive, got p={self.p} Pa")
        if self.T <= 0:
            raise ValueError(f"Temperature must be positive, got T={self.T} K")

    @classmethod
    def state_from_pT(cls, fluid: str, p: float, T: float) -> "ThermoState":
        """Build a state from pressure [Pa] and temperature [K]."""
        return cls(
            fluid=fluid,
            p=p,
            T=T,
            h=PropsSI("H", "P", p, "T", T, fluid),
            s=PropsSI("S", "P", p, "T", T, fluid),
            rho=PropsSI("D", "P", p, "T", T, fluid),
            a=PropsSI("A", "P", p, "T", T, fluid),
        )

    @classmethod
    def state_from_hs(cls, fluid: str, h: float, s: float) -> "ThermoState":
        """Build a state from enthalpy [J/kg] and entropy [J/kg/K]."""
        return cls(
            fluid=fluid,
            h=h,
            s=s,
            T=PropsSI("T", "H", h, "S", s, fluid),
            p=PropsSI("P", "H", h, "S", s, fluid),
            rho=PropsSI("D", "H", h, "S", s, fluid),
            a=PropsSI("A", "H", h, "S", s, fluid),
        )

    @classmethod
    def state_from_ps(cls, fluid: str, p: float, s: float) -> "ThermoState":
        """Build a state from pressure [Pa] and entropy [J/kg/K]."""
        return cls(
            fluid=fluid,
            p=p,
            s=s,
            T=PropsSI("T", "P", p, "S", s, fluid),
            h=PropsSI("H", "P", p, "S", s, fluid),
            rho=PropsSI("D", "P", p, "S", s, fluid),
            a=PropsSI("A", "P", p, "S", s, fluid),
        )


def h0_from_h(state, c) -> float:
    """Total enthalpy from static enthalpy and velocity: h0 = h + c**2 / 2.

    Exact for any fluid (real gas included). Units: h [J/kg], c [m/s],
        result [J/kg].

    The input state is the static state

    TODO(source): Dixon & Hall, stagnation enthalpy (chapter, equation).
    """
    if c < 0:
        raise ValueError(f"Velocity must be non-negative, got c={c} m/s")

    h0 = state.h + 0.5 * c**2
    return h0


def h_from_h0(state, c) -> float:
    """Return static enthalpy from total enthalpy and velocity: h = h0 - c**2 / 2.

    Exact for any fluid (real gas included). Units: h0 [J/kg], c [m/s],
        result [J/kg].

    The input state is the total state

    TODO(source): Dixon & Hall, static enthalpy (chapter, equation).
    """
    if c < 0:
        raise ValueError(f"Velocity must be non-negative, got c={c} m/s")

    h = state.h - 0.5 * c**2
    return h


def total_from_static(state, c) -> ThermoState:
    """Total state from static state and velocity.

    Exact for any fluid (real gas included). Units: c [m/s].

    The input state is the static state
    The output state is the total state
    Entropy remains constant, as the conversion of static to total state is
    isentropic.

    TODO(source): Dixon & Hall, total state (chapter, equation).
    """
    h0 = h0_from_h(state, c)
    return ThermoState.state_from_hs(state.fluid, h0, state.s)


def static_from_total(state, c) -> ThermoState:
    """Return the static state from the total state and velocity.

    Exact for any fluid (real gas included). Units: c [m/s].

    The input state is the total state
    The output state is the static state
    Entropy remains constant, as the conversion of total to static state is
    isentropic.

    TODO(source): Dixon & Hall, static state (chapter, equation).
    """
    h = h_from_h0(state, c)
    return ThermoState.state_from_hs(state.fluid, h, state.s)
