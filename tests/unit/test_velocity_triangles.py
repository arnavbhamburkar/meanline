import math

import pytest

from meanline.velocity_triangles import (
    axial_component,
    blade_speed_from_triangle,
    degree_of_reaction,
    euler_work,
    flow_coefficient,
    stage_loading_coefficient,
    tangential_component,
)


def test_components_hand_calculation():
    # c = 200 m/s at alpha = 30 deg: c_x = 200*cos30 = 100*sqrt(3), c_theta = 100.
    alpha = math.radians(30.0)
    assert axial_component(200.0, alpha) == pytest.approx(100.0 * math.sqrt(3.0))
    assert tangential_component(200.0, alpha) == pytest.approx(100.0)


def test_components_recombine_to_magnitude():
    alpha = math.radians(37.0)
    c_x = axial_component(150.0, alpha)
    c_theta = tangential_component(150.0, alpha)
    assert math.hypot(c_x, c_theta) == pytest.approx(150.0)


def test_euler_work_hand_calculation():
    # Constant radius: U = 300 m/s, swirl 0 -> 100 m/s gives 300*100 = 30 kJ/kg.
    assert euler_work(300.0, 300.0, 0.0, 100.0) == pytest.approx(30_000.0)


def test_euler_work_zero_for_no_swirl_change():
    assert euler_work(250.0, 250.0, 80.0, 80.0) == pytest.approx(0.0)


def test_triangle_closure():
    # U = c_theta + w_theta with w_theta positive opposite to blade motion.
    assert blade_speed_from_triangle(w_theta=200.0, c_theta=100.0) == pytest.approx(
        300.0
    )


def test_flow_coefficient_hand_calculation():
    # c_x = 150 m/s, U = 300 m/s -> phi = 0.5.
    assert flow_coefficient(150.0, 300.0) == pytest.approx(0.5)


def test_stage_loading_coefficient_hand_calculation():
    # work = 30 kJ/kg, U = 300 m/s -> psi = 30000 / 90000 = 1/3.
    assert stage_loading_coefficient(30_000.0, 300.0) == pytest.approx(1.0 / 3.0)


def test_stage_loading_from_triangle_matches_swirl_form():
    # For constant U: psi = (c_theta2 - c_theta1) / U.
    u, c_theta_1, c_theta_2 = 300.0, 75.0, 225.0
    work = euler_work(u, u, c_theta_1, c_theta_2)
    assert stage_loading_coefficient(work, u) == pytest.approx(
        (c_theta_2 - c_theta_1) / u
    )


def test_degree_of_reaction_symmetric_triangle_is_half():
    # Symmetric (50 %) triangle: alpha1 = beta2, alpha2 = beta1, constant c_x.
    # U = 300, c_x = 150, c_theta1 = 75 -> w_theta1 = 225;
    # c_theta2 = 225 -> w_theta2 = 75.
    # work = 300*(225 - 75) = 45000.
    u, c_x = 300.0, 150.0
    c_theta_1, c_theta_2 = 75.0, 225.0
    w_theta_1 = u - c_theta_1  # triangle closure: w_theta = U - c_theta
    w_theta_2 = u - c_theta_2
    w_1 = math.hypot(c_x, w_theta_1)
    w_2 = math.hypot(c_x, w_theta_2)
    work = euler_work(u, u, c_theta_1, c_theta_2)
    # w1^2 - w2^2 = (150^2 + 225^2) - (150^2 + 75^2) = 45000 = work.
    assert degree_of_reaction(w_1, w_2, work) == pytest.approx(0.5)


def test_degree_of_reaction_zero_when_relative_speed_unchanged():
    # w1 = w2 -> no static enthalpy rise in the rotor -> R = 0 (impulse).
    assert degree_of_reaction(200.0, 200.0, 30_000.0) == pytest.approx(0.0)
