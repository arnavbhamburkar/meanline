"""Cascade aerodynamics.

Planned scope:
    - diffusion factor, de Haller number.
    - incidence and deviation rules.

Every correlation must cite source and equation number and state its validity
range (Mach number, solidity, camber, etc.), and warn at runtime outside it.

References
----------
TODO(source): diffusion-factor definition (Dixon & Hall / Cumpsty, equation).
"""

from dataclasses import dataclass


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


def calculate_pitch(r_mean, N_blades):
    """Calculate the pitch (spacing) between blades at the mean radius.

    r_mean = mean radius [m]
    N_blades = number of blades (integer)

    TODO(source): Dixon & Hall, pitch (chapter and equation).
    """
    if N_blades <= 0:
        raise ValueError(f"Number of blades must be positive, got N_blades={N_blades}")

    pitch = (2 * 3.141592653589793 * r_mean) / N_blades
    return pitch


def calculate_solidity(chord, pitch):
    """Calculate the solidity of a blade row.

    Solidity = chord / pitch

    chord = blade chord [m]
    pitch = spacing between blades [m]

    TODO(source): Dixon & Hall, solidity (chapter and equation).
    """
    if pitch <= 0:
        raise ValueError(f"Pitch must be positive, got pitch={pitch} m")

    solidity = chord / pitch
    return solidity


def camber_angle(alpha_1_dash, alpha_2_dash):
    """Calculate the camber angle of a blade row.

    Camber angle = alpha_1_dash - alpha_2_dash

    alpha_1_dash = blade inlet angle at the cascade leading edge [rad]
    alpha_2_dash = blade outlet angle at the cascade trailing edge [rad]

    TODO(source): Dixon & Hall, camber angle (chapter and equation).
    """
    return alpha_1_dash - alpha_2_dash


def de_haller_number(v_in, v_out):
    """De Haller number: ratio of exit to inlet velocity magnitude.

    TODO(source): Dixon & Hall, de Haller number (chapter and equation).
    """
    if v_in == 0:
        raise ValueError(f"Velocity must be non-zero, got v_in={v_in} m/s")

    return v_out / v_in


def diffusion_factor(w_in, w_out, c_theta_in, c_theta_out, spacing, chord):
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


def turning_angle(alpha_in, alpha_out):
    """Calculate the turning angle of a blade row.

    Turning angle = alpha_in - alpha_out

    alpha_in = inlet flow angle [rad]
    alpha_out = outlet flow angle [rad]
    Both parameters depend on flow conditions (not a property of the cascade)

    TODO(source): Dixon & Hall, turning angle (chapter and equation).
    """
    return alpha_in - alpha_out


def incidence_angle(alpha_in, alpha_1_dash):
    """Calculate the incidence angle of a blade row.

    Incidence angle = alpha_in - alpha_1_dash

    alpha_in = inlet flow angle [rad] (depends on flow conditions, not a
    property of the cascade)
    alpha_1_dash = blade inlet angle at the cascade leading edge [rad]

    TODO(source): Dixon & Hall, incidence angle (chapter 3 and equation 3.1).
    """
    return alpha_in - alpha_1_dash


def deviation_angle(alpha_2_dash, alpha_out):
    """Calculate the deviation angle of a blade row.

    Deviation angle = alpha_out - alpha_2_dash

    alpha_2_dash = blade exit angle at the cascade trailing edge [rad]
    alpha_out = outlet flow angle [rad] (depends on flow conditions, not a
    property of the cascade)

    TODO(source): Dixon & Hall, deviation angle (chapter 3 and equation 3.2).
    """
    return alpha_out - alpha_2_dash
