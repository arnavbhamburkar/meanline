"""Cascade aerodynamics.

Scope:
    - Diffusion factor, de Haller number.
    - Incidence and deviation rules.

Every correlation must cite source and equation number and state its validity
range (Mach number, solidity, camber, etc.), and warn at runtime outside it.

References
----------
Dixon & Hall, diffusion factor (Lieblein): Chapter 3, Eq. (3.32) for a cascade,
Eq. (3.33) for constant axial velocity and incompressible flow, and Chapter 5,
Eq. (5.18) for a compressor rotor. Cascade geometry: Section 3.2.
"""

from dataclasses import dataclass
from math import pi


@dataclass(frozen=True)
class Cascade:
    """Cascade representation with geometric and aerodynamic parameters.

    Attributes
    ----------
    N_blades : int
        Number of blades in the cascade.
    chord : float
        Blade chord [m].
    solidity : float
        Blade solidity (dimensionless).
    """

    N_blades: int  # number of blades
    chord: float  # m
    alpha_1_dash: float  # rad, blade inlet angle at the cascade leading edge
    alpha_2_dash: float  # rad, blade exit angle at the cascade trailing edge

    pitch: float  # m, blade spacing at the mean radius
    solidity: float  # dimensionless, chord / pitch
    camber: float  # rad, camber angle

    @classmethod
    def from_geometry(
        cls,
        N_blades: int,
        chord: float,
        r_mean: float,
        alpha_1_dash: float,
        alpha_2_dash: float,
    ) -> "Cascade":
        """Build a cascade from geometric parameters.

        Parameters
        ----------
        N_blades : int
            Number of blades in the cascade.
        chord : float
            Blade chord [m].
        r_mean : float
            Mean radius [m].
        alpha_1_dash : float
            Blade inlet angle at the cascade leading edge [rad]
        alpha_2_dash : float
            Blade outlet angle at the cascade trailing edge [rad].

        Returns
        -------
        Cascade
            Cascade aerodynamics representation.
        """
        if N_blades <= 0:
            raise ValueError(
                f"Number of blades must be positive, got N_blades={N_blades}"
            )
        if chord <= 0:
            raise ValueError(f"Chord must be positive, got chord={chord} m")
        if r_mean <= 0:
            raise ValueError(f"Mean radius must be positive, got r_mean={r_mean} m")

        pitch = calculate_pitch(r_mean, N_blades)
        solidity = calculate_solidity(chord, pitch)
        camber = camber_angle(alpha_1_dash, alpha_2_dash)

        return cls(
            N_blades=N_blades,
            chord=chord,
            solidity=solidity,
            alpha_1_dash=alpha_1_dash,
            alpha_2_dash=alpha_2_dash,
            pitch=pitch,
            camber=camber,
        )


def calculate_pitch(r_mean: float, N_blades: int) -> float:
    """Calculate the pitch (spacing) between blades at the mean radius.

    r_mean = mean radius [m]
    N_blades = number of blades (integer)

    TODO(source): Dixon & Hall define the space (pitch) s in Section 3.2 but give
    no numbered equation for s = 2 * pi * r / N; it is a geometric identity.
    """
    if N_blades <= 0:
        raise ValueError(f"Number of blades must be positive, got N_blades={N_blades}")

    pitch = (2 * 3.141592653589793 * r_mean) / N_blades
    return pitch


def calculate_solidity(chord: float, pitch: float) -> float:
    """Calculate the solidity of a blade row.

    Solidity = chord / pitch

    chord = blade chord [m]
    pitch = spacing between blades [m]

    Source: Dixon & Hall, Section 3.2 (cascade geometry): space-chord ratio s/l
    and solidity sigma = l/s, defined in the text (no numbered equation).
    """
    if pitch <= 0:
        raise ValueError(f"Pitch must be positive, got pitch={pitch} m")

    solidity = chord / pitch
    return solidity


def camber_angle(alpha_1_dash: float, alpha_2_dash: float) -> float:
    """Calculate the camber angle of a blade row.

    Camber angle = alpha_1_dash - alpha_2_dash

    alpha_1_dash = blade inlet angle at the cascade leading edge [rad]
    alpha_2_dash = blade outlet angle at the cascade trailing edge [rad]

    Source: Dixon & Hall, Section 3.2 (cascade geometry): camber angle
    theta = alpha_1' - alpha_2', defined in the text (no numbered equation).
    """
    return alpha_1_dash - alpha_2_dash


def de_haller_number(v_in: float, v_out: float) -> float:
    """De Haller number: ratio of exit to inlet velocity magnitude.

    Source: Dixon & Hall, Chapter 5, Section 5.7 ("Stage Loading"): de Haller
    (1953) proposed w2/w1 >= 0.75 for satisfactory performance. This is stated
    in the text; there is no numbered equation.
    """
    if v_in == 0:
        raise ValueError(f"Velocity must be non-zero, got v_in={v_in} m/s")

    return v_out / v_in


def diffusion_factor(
    w_in: float,
    w_out: float,
    c_theta_in: float,
    c_theta_out: float,
    spacing: float,
    chord: float,
) -> float:
    """Diffusion factor of a blade row.

    D = 1 - (w_out / w_in)
        + ((c_theta_out - c_theta_in) * spacing) / (2 * w_in * chord)

    w_in = relative velocity at the rotor inlet [m/s]
    w_out = relative velocity at the rotor exit [m/s]
    c_theta_in = tangential component of absolute velocity at the rotor inlet [m/s]
    c_theta_out = tangential component of absolute velocity at the rotor exit [m/s]
    spacing = blade spacing [m]
    chord = blade chord [m]

    TODO(source): Dixon & Hall, diffusion factor (chapter 5 and equation 5.18).
    """
    if w_in == 0:
        raise ValueError(f"Velocity must be non-zero, got w_in={w_in} m/s")
    if chord <= 0:
        raise ValueError(f"Chord must be positive, got chord={chord} m")

    df = (
        1.0
        - (w_out / w_in)
        + ((c_theta_out - c_theta_in) * spacing) / (2 * w_in * chord)
    )
    return df


def turning_angle(alpha_in: float, alpha_out: float) -> float:
    """Calculate the turning angle of a blade row.

    Turning angle = alpha_in - alpha_out

    alpha_in = inlet flow angle [rad]
    alpha_out = outlet flow angle [rad]
    Both parameters depend on flow conditions (not a property of the cascade)

    Source: Dixon & Hall, Section 3.2 (cascade geometry): deflection
    epsilon = alpha_1 - alpha_2, defined in the text (no numbered equation).
    """
    return alpha_in - alpha_out


def incidence_angle(alpha_in: float, alpha_1_dash: float) -> float:
    """Calculate the incidence angle of a blade row.

    Incidence angle = alpha_in - alpha_1_dash

    alpha_in = inlet flow angle [rad] (depends on flow conditions, not a
    property of the cascade)
    alpha_1_dash = blade inlet angle at the cascade leading edge [rad]

    TODO(source): Dixon & Hall, incidence angle (chapter 3 and equation 3.1).
    """
    return alpha_in - alpha_1_dash


def deviation_angle(alpha_2_dash: float, alpha_out: float) -> float:
    """Calculate the deviation angle of a blade row.

    Deviation angle = alpha_out - alpha_2_dash

    alpha_2_dash = blade exit angle at the cascade trailing edge [rad]
    alpha_out = outlet flow angle [rad] (depends on flow conditions, not a
    property of the cascade)

    TODO(source): Dixon & Hall, deviation angle (chapter 3 and equation 3.2).
    """
    return alpha_out - alpha_2_dash


def howell_deviation(cascade: Cascade, a: float, n: float) -> float:
    """Howell deviation correlation for a cascade.

    Prediction of deviation angle based on cascade geometry and flow conditions
    Nominal incidence only, low speed cascade data. Returns the deviation angle
    in radians.

    TODO(source): Dixon & Hall, fluid deviation (chapter 3 and equations 3.43
    and 3.44).

    a = distance from the leading edge to the point of maximum camber [m]
    n = 0.5 for compressor cascade, 1 for inlet guide vane
    """
    if a / cascade.chord < 0 or a / cascade.chord > 1:
        raise ValueError(
            "Distance of maximum camber from leading edge must not be greater "
            f"than the chord, got a={a} m"
        )
    if n != 0.5 and n != 1:
        raise ValueError(
            "n must be either 0.5 (compressor cascade) or 1 (inlet guide vane), "
            f"got n={n}"
        )

    k = cascade.camber * (1 / cascade.solidity) ** n
    m_0 = 0.23 * (2 * a / cascade.chord) ** 2
    c = 180 / (500 * pi)

    if 1 - k * c <= 0:
        raise ValueError(
            "Denominator in Howell deviation correlation is zero or negative,"
            f"got k={k} and c={c}"
        )

    delta_star = k * (m_0 + c * cascade.alpha_2_dash) / (1 - k * c)
    return delta_star
