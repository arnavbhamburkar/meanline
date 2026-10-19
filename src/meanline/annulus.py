"""Annulus geometry.

Planned scope:
    - Hub, tip and mean radius.
    - Blockage and effective flow area.
    - Constant-radius design rules.

Annulus and blockage assumptions must be stated explicitly in every function.
SI units.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Annulus:
    """Annulus geometry representation.

    Attributes
    ----------
    r_hub : float
        Hub radius [m].
    r_tip : float
        Tip radius [m].
    blockage : float
        Blockage factor (dimensionless).

    Assumptions
    -----------
    - r_mean is the arithmetic mean of r_hub and r_tip
    - Blockage B is the fraction of the geometric flow area blocked by endwall
      boundary layers, so A_eff = A_flow (1 - B).
    - Blockage is sourced as an input
    """

    r_hub: float  # m
    r_tip: float  # m
    blockage: float  # dimensionless

    def __post_init__(self) -> None:
        """Validate the input parameters.

        Raise ValueError if any parameter is invalid.
        """
        if not (math.isfinite(self.r_hub) and self.r_hub > 0):
            raise ValueError(f"Hub radius must be positive, got r_hub={self.r_hub} m")
        if not (math.isfinite(self.r_tip) and self.r_tip > self.r_hub):
            raise ValueError(
                f"Tip radius must be finite and greater than hub radius, "
                f"got r_tip={self.r_tip} m and r_hub={self.r_hub} m"
            )
        if not 0.0 <= self.blockage < 1.0:
            raise ValueError(
                f"Blockage must satisfy 0 <= B < 1, got blockage={self.blockage}"
            )

    @property
    def r_mean(self) -> float:
        """Mean radius [m].

        Source: Dixon & Hall, Chapter 4, text after Eq. (4.24): the arithmetic
        mean r_m = (r_t + r_h) / 2 is an approximation. The precise (equal-area)
        mean radius is r_m^2 = (r_t^2 + r_h^2) / 2; the two agree for high
        hub-to-tip ratios.
        """
        return (self.r_hub + self.r_tip) * 0.5

    @property
    def A_flow(self) -> float:
        """Geometric flow area [m^2].

        Source: Dixon & Hall, Chapter 4, Eq. (4.25),
        A = pi * r_t^2 * [1 - (r_h / r_t)^2].
        """
        return math.pi * (self.r_tip**2 - self.r_hub**2)

    @property
    def A_eff(self) -> float:
        """Effective flow area [m^2], accounting for blockage."""
        return self.A_flow * (1.0 - self.blockage)


def mass_flow_rate(rho: float, annulus: Annulus, v_x: float) -> float:
    """Mass flow rate through an annulus: m_dot = rho * A_eff * v_x.

    v_x is the axial velocity component, assumed uniform across the annulus.
    Units: rho [kg/m^3], A_eff [m^2], v_x [m/s], result [kg/s].

    Source: Dixon & Hall, Chapter 1, Eq. (1.8), m_dot = rho * c * A_n, with
    c = v_x and A_n = A_eff. The book has no equation for the blockage factor.
    """
    if rho <= 0:
        raise ValueError(f"Density must be positive, got rho={rho} kg/m^3")
    if v_x <= 0:
        raise ValueError(f"Axial velocity must be positive, got v_x={v_x} m/s")

    m_dot = rho * annulus.A_eff * v_x
    return m_dot


def axial_velocity_from_mass_flow_rate(
    m_dot: float, rho: float, annulus: Annulus
) -> float:
    """Axial velocity from mass flow rate: v_x = m_dot / (rho * A_eff).

    Units: m_dot [kg/s], rho [kg/m^3], A_eff [m^2], result [m/s].

    Source: Dixon & Hall, Chapter 1, Eq. (1.8), rearranged for the velocity
    (see also Chapter 4, Eq. 4.24, for the annulus-area form).
    """
    if rho <= 0:
        raise ValueError(f"Density must be positive, got rho={rho} kg/m^3")

    v_x = m_dot / (rho * annulus.A_eff)
    return v_x
