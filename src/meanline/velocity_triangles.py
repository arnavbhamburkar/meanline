"""Velocity triangles and stage parameters.

Planned scope:
    - Absolute and relative frames at the mean radius.
    - Euler work equation.
    - Flow coefficient, loading coefficient, degree of reaction.

Model level: mean-line, one radius. SI units.

References
----------
TODO(source): Dixon & Hall, velocity triangles and stage parameters
(chapter and equation numbers).
"""

from math import cos, sin

# All angles and swirl numbers are positive, as per Dixon & Hall.

# alpha = angle between absolute velocity and axial direction
# beta = angle between relative velocity and axial direction

# c = absolute velocity
# w = relative velocity
# u = velocity of the rotor at the mean radius
# u_1 = velocity of the rotor at the mean radius at the inlet
# u_2 = velocity of the rotor at the mean radius at the exit
# c_theta = tangential component of absolute velocity
# c_x = axial component of absolute velocity
# w_theta = tangential component of relative velocity
# w_x = axial component of relative velocity
# Suffixes _1 and _2 denote inlet and exit, respectively.

# c_x_1 = c_1 * cos(alpha_1)
# c_x_2 = c_2 * cos(alpha_2)
# w_x_1 = w_1 * cos(beta_1)
# w_x_2 = w_2 * cos(beta_2)

# c_theta_1 = c_1 * sin(alpha_1)
# c_theta_2 = c_2 * sin(alpha_2)
# w_theta_1 = w_1 * sin(beta_1)
# w_theta_2 = w_2 * sin(beta_2)


def euler_work(u_1, u_2, c_theta_1, c_theta_2):
    """Euler work equation: specific work, dh0 = U2*c_theta2 - U1*c_theta1.

    For axial machines, U1 = U2 = U, so dh0 = U*(c_theta2 - c_theta1).

    Units: u [m/s], c_theta [m/s], result [J/kg]. c_theta is positive in the
    direction of blade motion.

    TODO(source): Dixon & Hall, Euler work equation (chapter 5, equation 5.1).
    """
    work = u_2 * c_theta_2 - u_1 * c_theta_1
    return work


def axial_component(c, theta):
    """Axial component of a velocity: c_x = c * cos(theta).

    Units: c [m/s], theta [rad] measured from the axial direction.

    TODO(source): Dixon & Hall, velocity triangle (chapter, equation).
    """
    c_x = c * cos(theta)
    return c_x


def tangential_component(c, theta):
    """Tangential component of a velocity: c_theta = c * sin(theta).

    Units: c [m/s], theta [rad] measured from the axial direction.

    TODO(source): Dixon & Hall, velocity triangle (chapter, equation).
    """
    c_theta = c * sin(theta)
    return c_theta


def blade_speed_from_triangle(w_theta, c_theta):
    """Blade speed at the mean radius from triangle closure: U = c_theta + w_theta.

    Convention: c_theta is positive in the direction of blade motion and
    w_theta is positive opposite to it. Units: [m/s].

    TODO(source): Dixon & Hall, velocity triangle (chapter, equation).
    """
    u = w_theta + c_theta
    return u


def flow_coefficient(c_x, u):
    """Flow coefficient: phi = c_x / U.

    Units: c_x [m/s], u [m/s], result [-].

    TODO(source): Dixon & Hall, flow coefficient (chapter, equation).
    """
    phi = c_x / u
    return phi


def stage_loading_coefficient(work, u):
    """Stage loading coefficient: psi = Euler work / U^2.

    Units: work [J/kg], u [m/s], result [-].

    TODO(source): Dixon & Hall, stage loading coefficient (chapter, equation).
    """
    psi = work / (u**2)
    return psi


def degree_of_reaction(w_1, w_2, work):
    """Degree of reaction.

    Reaction = (static enthalpy rise across the rotor) / (stage enthalpy rise).

    Assumptions: Constant rothalpy across the rotor, constant radius, and no
    change in axial velocity across the rotor.

    With assumptions: Reaction = (w_1**2 - w_2**2) / (2 *Euler work).

    Units: w_1 [m/s], w_2 [m/s], work [J/kg], result [-].

    TODO(source): Dixon & Hall, degree of reaction (chapter 5, equation 5.20).
    """
    R = (w_1**2 - w_2**2) / (2 * work)
    return R
