import dataclasses
import math

import pytest

from meanline.cascade import (
    Cascade,
    calculate_pitch,
    calculate_solidity,
    camber_angle,
    de_haller_number,
    deviation_angle,
    diffusion_factor,
    howell_deviation,
    incidence_angle,
    turning_angle,
)

# --- de Haller number ------------------------------------------------------


def test_de_haller_hand_calculation():
    # w_out / w_in = 150 / 200 = 0.75.
    assert de_haller_number(200.0, 150.0) == pytest.approx(0.75)


def test_de_haller_is_one_without_diffusion():
    assert de_haller_number(180.0, 180.0) == pytest.approx(1.0)


def test_de_haller_above_one_for_accelerating_flow():
    # Nozzle-like row: exit velocity larger than inlet velocity.
    assert de_haller_number(100.0, 125.0) == pytest.approx(1.25)


# --- diffusion factor ------------------------------------------------------
# Arguments: (w_in, w_out, c_theta_in, c_theta_out, s, l), with s = pitch and
# l = chord, so solidity sigma = l / s.


def test_diffusion_factor_hand_calculation_unit_solidity():
    # w_in = 200, w_out = 150, delta c_theta = 150 - 50 = 100, s = l (sigma = 1):
    # D = 1 - 0.75 + 100 / (2 * 1 * 200) = 0.25 + 0.25 = 0.5.
    df = diffusion_factor(200.0, 150.0, 50.0, 150.0, 0.05, 0.05)
    assert df == pytest.approx(0.5)


def test_diffusion_factor_hand_calculation_solidity_1p25():
    # Same velocities, pitch 0.04 m, chord 0.05 m (sigma = 1.25):
    # D = 0.25 + 100 / (2 * 1.25 * 200) = 0.25 + 0.2 = 0.45.
    df = diffusion_factor(200.0, 150.0, 50.0, 150.0, 0.04, 0.05)
    assert df == pytest.approx(0.45)


def test_diffusion_factor_zero_without_diffusion_or_turning():
    # w_out = w_in and no change in swirl -> D = 0.
    assert diffusion_factor(200.0, 200.0, 80.0, 80.0, 0.05, 0.05) == pytest.approx(0.0)


def test_diffusion_factor_reduces_to_velocity_term_without_turning():
    # No swirl change -> only 1 - w_out / w_in remains.
    df = diffusion_factor(200.0, 140.0, 60.0, 60.0, 0.05, 0.05)
    assert df == pytest.approx(1.0 - 140.0 / 200.0)


def test_diffusion_factor_depends_only_on_pitch_to_chord_ratio():
    # Scaling pitch and chord together leaves solidity, and so D, unchanged.
    small = diffusion_factor(200.0, 150.0, 50.0, 150.0, 0.02, 0.025)
    large = diffusion_factor(200.0, 150.0, 50.0, 150.0, 0.2, 0.25)
    assert small == pytest.approx(large)


def test_diffusion_factor_increases_with_solidity_reduction():
    # Fewer blades (larger pitch, lower solidity) must raise the loading term.
    dense = diffusion_factor(200.0, 150.0, 50.0, 150.0, 0.03, 0.05)
    sparse = diffusion_factor(200.0, 150.0, 50.0, 150.0, 0.06, 0.05)
    assert sparse > dense


def test_diffusion_factor_from_velocity_triangle():
    # Constant-radius rotor: U = 300, c_x = 150 m/s, c_theta1 = 75, c_theta2 = 225.
    # Relative swirl w_theta = U - c_theta, so delta w_theta = -(delta c_theta):
    # the solidity term must use the magnitude 150 m/s either way.
    u, c_x = 300.0, 150.0
    c_theta_1, c_theta_2 = 75.0, 225.0
    w_1 = math.hypot(c_x, u - c_theta_1)
    w_2 = math.hypot(c_x, u - c_theta_2)
    sigma = 1.0
    expected = (
        1.0 - w_2 / w_1 + abs((u - c_theta_1) - (u - c_theta_2)) / (2.0 * sigma * w_1)
    )
    df = diffusion_factor(w_1, w_2, c_theta_1, c_theta_2, 0.05, 0.05)
    assert df == pytest.approx(expected)


def test_de_haller_and_diffusion_factor_agree_on_velocity_ratio():
    # With no swirl change, D = 1 - de Haller number.
    w_in, w_out = 240.0, 180.0
    df = diffusion_factor(w_in, w_out, 90.0, 90.0, 0.05, 0.05)
    assert df == pytest.approx(1.0 - de_haller_number(w_in, w_out))


# --- zero-division guards --------------------------------------------------


def test_de_haller_rejects_zero_inlet_velocity():
    with pytest.raises(ValueError, match="non-zero"):
        de_haller_number(0.0, 100.0)


def test_diffusion_factor_rejects_zero_inlet_velocity():
    with pytest.raises(ValueError, match="non-zero"):
        diffusion_factor(0.0, 150.0, 50.0, 150.0, 0.05, 0.05)


def test_diffusion_factor_rejects_zero_chord():
    with pytest.raises(ValueError, match="Chord"):
        diffusion_factor(200.0, 150.0, 50.0, 150.0, 0.05, 0.0)


# --- pitch and solidity ----------------------------------------------------


def test_pitch_hand_calculation():
    # s = 2 * pi * 0.25 / 50 = pi / 100 = 0.031416 m.
    assert calculate_pitch(0.25, 50) == pytest.approx(math.pi / 100.0)


def test_pitch_times_blade_count_is_circumference():
    assert calculate_pitch(0.3, 36) * 36 == pytest.approx(2.0 * math.pi * 0.3)


def test_pitch_halves_when_blade_count_doubles():
    assert calculate_pitch(0.25, 100) == pytest.approx(0.5 * calculate_pitch(0.25, 50))


@pytest.mark.parametrize("n_blades", [0, -5])
def test_pitch_rejects_non_positive_blade_count(n_blades):
    with pytest.raises(ValueError, match="blades"):
        calculate_pitch(0.25, n_blades)


def test_solidity_hand_calculation():
    # sigma = 0.05 / 0.04 = 1.25.
    assert calculate_solidity(0.05, 0.04) == pytest.approx(1.25)


def test_solidity_from_pitch_and_chord_hand_calculation():
    # pitch = pi / 100, chord = 0.04 -> sigma = 4 / pi = 1.2732.
    pitch = calculate_pitch(0.25, 50)
    assert calculate_solidity(0.04, pitch) == pytest.approx(4.0 / math.pi)


@pytest.mark.parametrize("pitch", [0.0, -0.04])
def test_solidity_rejects_non_positive_pitch(pitch):
    with pytest.raises(ValueError, match="Pitch"):
        calculate_solidity(0.05, pitch)


def test_solidity_is_consistent_with_diffusion_factor_spacing_chord_ratio():
    # diffusion_factor uses s / l = 1 / sigma, so both views must agree.
    pitch = calculate_pitch(0.25, 50)
    chord = 0.04
    sigma = calculate_solidity(chord, pitch)
    via_geometry = diffusion_factor(200.0, 150.0, 50.0, 150.0, pitch, chord)
    via_solidity = 1.0 - 150.0 / 200.0 + (150.0 - 50.0) / (2.0 * sigma * 200.0)
    assert via_geometry == pytest.approx(via_solidity)


# --- angle definitions -----------------------------------------------------
# All angles in rad, measured from the axial direction.


def test_camber_hand_calculation():
    # 45 deg - 15 deg = 30 deg.
    assert camber_angle(math.radians(45.0), math.radians(15.0)) == pytest.approx(
        math.radians(30.0)
    )


def test_camber_is_zero_for_uncambered_blade():
    assert camber_angle(0.4, 0.4) == pytest.approx(0.0)


def test_turning_angle_hand_calculation():
    # Flow turned from 50 deg to 22 deg -> 28 deg.
    assert turning_angle(math.radians(50.0), math.radians(22.0)) == pytest.approx(
        math.radians(28.0)
    )


def test_incidence_hand_calculation():
    # Flow at 50 deg onto blade inlet angle 45 deg -> i = +5 deg.
    assert incidence_angle(math.radians(50.0), math.radians(45.0)) == pytest.approx(
        math.radians(5.0)
    )


def test_incidence_is_negative_when_flow_is_below_blade_angle():
    assert incidence_angle(math.radians(40.0), math.radians(45.0)) == pytest.approx(
        math.radians(-5.0)
    )


def test_incidence_is_zero_when_flow_is_aligned_with_blade():
    assert incidence_angle(0.7, 0.7) == pytest.approx(0.0)


def test_deviation_hand_calculation():
    # Convention as coded: delta = alpha_out - alpha_2'.
    # Blade exit angle 15 deg, flow leaves at 20 deg -> delta = +5 deg
    # (flow under-turned relative to the blade).
    # TODO(source): confirm the sign convention against Dixon & Hall.
    assert deviation_angle(math.radians(15.0), math.radians(20.0)) == pytest.approx(
        math.radians(5.0)
    )


def test_deviation_is_zero_when_flow_follows_blade():
    assert deviation_angle(0.3, 0.3) == pytest.approx(0.0)


def test_turning_equals_camber_plus_incidence_minus_deviation():
    # Identity from the definitions above:
    #   eps = (a_in - a_out) = (a_in - a1') + (a1' - a2') - (a_out - a2')
    #       = i + camber - delta.
    a_in, a_out = math.radians(50.0), math.radians(22.0)
    a1, a2 = math.radians(45.0), math.radians(15.0)
    eps = turning_angle(a_in, a_out)
    total = (
        incidence_angle(a_in, a1) + camber_angle(a1, a2) - deviation_angle(a2, a_out)
    )
    assert eps == pytest.approx(total)


# --- Cascade dataclass -----------------------------------------------------


@pytest.fixture
def cascade():
    return Cascade.from_geometry(
        N_blades=50,
        chord=0.04,
        r_mean=0.25,
        alpha_1_dash=math.radians(45.0),
        alpha_2_dash=math.radians(15.0),
    )


def test_cascade_pitch_matches_standalone_function(cascade):
    assert cascade.pitch == pytest.approx(math.pi / 100.0)


def test_cascade_solidity_hand_calculation(cascade):
    assert cascade.solidity == pytest.approx(4.0 / math.pi)


def test_cascade_camber_hand_calculation(cascade):
    assert cascade.camber == pytest.approx(math.radians(30.0))


def test_cascade_stores_inputs_unchanged(cascade):
    assert cascade.N_blades == 50
    assert cascade.chord == 0.04
    assert cascade.alpha_1_dash == pytest.approx(math.radians(45.0))
    assert cascade.alpha_2_dash == pytest.approx(math.radians(15.0))


def test_cascade_is_frozen(cascade):
    with pytest.raises(dataclasses.FrozenInstanceError):
        cascade.chord = 0.1


def test_cascade_solidity_equals_chord_over_pitch(cascade):
    assert cascade.solidity == pytest.approx(cascade.chord / cascade.pitch)


@pytest.mark.parametrize("n_blades", [0, -3])
def test_cascade_rejects_non_positive_blade_count(n_blades):
    with pytest.raises(ValueError, match="blades"):
        Cascade.from_geometry(n_blades, 0.04, 0.25, 0.8, 0.3)


@pytest.mark.parametrize("chord", [0.0, -0.04])
def test_cascade_rejects_non_positive_chord(chord):
    with pytest.raises(ValueError, match="Chord"):
        Cascade.from_geometry(50, chord, 0.25, 0.8, 0.3)


@pytest.mark.parametrize("r_mean", [0.0, -0.25])
def test_cascade_rejects_non_positive_mean_radius(r_mean):
    with pytest.raises(ValueError, match="radius"):
        Cascade.from_geometry(50, 0.04, r_mean, 0.8, 0.3)


def test_cascade_built_on_annulus_mean_radius():
    # Integration: the cascade radius comes from the annulus, as planned.
    from meanline.annulus import Annulus

    ann = Annulus(0.2, 0.3, 0.0)
    c = Cascade.from_geometry(50, 0.04, ann.r_mean, 0.8, 0.3)
    assert c.pitch == pytest.approx(2.0 * math.pi * ann.r_mean / 50)


# --- Howell nominal deviation ----------------------------------------------
# Dixon & Hall, Eqs. (3.43) and (3.44):
#   delta* = m * theta * (s/l)^n,   m = 0.23 * (2a/l)^2 + alpha_2* / 500
# with alpha_2* = alpha_2' + delta* in degrees. Reference values: Example 3.2
# (Section 3.5) and Example 5.3 (Section 5.7).


def _make_cascade(space_chord, theta_deg, alpha_2_dash_deg, n_blades=50, chord=1.0):
    """Build a Cascade with a prescribed s/l, camber and blade exit angle."""
    r_mean = space_chord * chord * n_blades / (2.0 * math.pi)
    a2 = math.radians(alpha_2_dash_deg)
    a1 = a2 + math.radians(theta_deg)
    return Cascade.from_geometry(n_blades, chord, r_mean, a1, a2)


def _howell_reference_deg(space_chord, theta_deg, a_over_l, alpha_2_dash_deg, n):
    """Independent fixed-point solution of Eqs. (3.43) and (3.44), in degrees."""
    delta = 0.0
    for _ in range(200):
        m = 0.23 * (2.0 * a_over_l) ** 2 + (alpha_2_dash_deg + delta) / 500.0
        delta = m * theta_deg * space_chord**n
    return delta


def _howell_deg(cascade, a_over_l, n):
    return math.degrees(howell_deviation(cascade, a_over_l * cascade.chord, n))


def test_howell_example_5_3_delta_star():
    # theta = 45 deg, s/l = 0.9, circular arc (a/l = 0.5), alpha_2' = -0.5 deg.
    # Book: delta* = 10.69 deg.
    c = _make_cascade(0.9, 45.0, -0.5)
    assert _howell_deg(c, 0.5, 0.5) == pytest.approx(10.69, abs=0.01)


def test_howell_example_5_3_nominal_exit_angle():
    # Book: beta_2* = alpha_2' + delta* = 10.2 deg.
    c = _make_cascade(0.9, 45.0, -0.5)
    exit_angle = math.degrees(c.alpha_2_dash) + _howell_deg(c, 0.5, 0.5)
    assert exit_angle == pytest.approx(10.2, abs=0.05)


def test_howell_example_3_2_delta_star():
    # theta = 30 deg, s/l = 1, a/l = 0.5, alpha_2' = 20 deg. The book stops its
    # iteration after two steps at 8.6 deg; the converged value is 8.617 deg.
    c = _make_cascade(1.0, 30.0, 20.0)
    assert _howell_deg(c, 0.5, 0.5) == pytest.approx(8.6, abs=0.1)
    assert _howell_deg(c, 0.5, 0.5) == pytest.approx(8.617, abs=0.001)


def test_howell_hand_calculation_closed_form():
    # k = 30 deg * 1 = 30; delta = 30 * (0.23 + 20/500) / (1 - 30/500)
    #   = 30 * 0.27 / 0.94 = 8.617 deg.
    c = _make_cascade(1.0, 30.0, 20.0)
    assert _howell_deg(c, 0.5, 0.5) == pytest.approx(30.0 * 0.27 / 0.94)


@pytest.mark.parametrize("a_over_l", [0.3, 0.4, 0.5, 0.6])
def test_howell_matches_independent_fixed_point_solution(a_over_l):
    # a/l != 0.5 is needed to expose a missing square on (2a/l).
    c = _make_cascade(0.8, 35.0, 10.0)
    expected = _howell_reference_deg(0.8, 35.0, a_over_l, 10.0, 0.5)
    assert _howell_deg(c, a_over_l, 0.5) == pytest.approx(expected)


@pytest.mark.parametrize("space_chord", [0.6, 0.9, 1.0, 1.3])
def test_howell_matches_reference_for_different_space_chord_ratios(space_chord):
    # s/l != 1 is needed to expose pitch being used in place of chord.
    c = _make_cascade(space_chord, 40.0, 5.0)
    expected = _howell_reference_deg(space_chord, 40.0, 0.5, 5.0, 0.5)
    assert _howell_deg(c, 0.5, 0.5) == pytest.approx(expected)


def test_howell_does_not_depend_on_absolute_size():
    # Only the ratios a/l and s/l enter, not the chord in metres.
    small = _make_cascade(0.9, 45.0, -0.5, chord=0.02)
    large = _make_cascade(0.9, 45.0, -0.5, chord=2.0)
    assert _howell_deg(small, 0.5, 0.5) == pytest.approx(_howell_deg(large, 0.5, 0.5))


def test_howell_result_satisfies_both_book_equations():
    # Plug delta* back into Eqs. (3.43) and (3.44): both sides must agree.
    c = _make_cascade(0.85, 38.0, 12.0)
    delta = _howell_deg(c, 0.45, 0.5)
    m = 0.23 * (2.0 * 0.45) ** 2 + (12.0 + delta) / 500.0
    assert delta == pytest.approx(m * 38.0 * 0.85**0.5)


def test_howell_deviation_is_in_radians():
    c = _make_cascade(1.0, 30.0, 20.0)
    value = howell_deviation(c, 0.5 * c.chord, 0.5)
    assert value == pytest.approx(math.radians(30.0 * 0.27 / 0.94))


def test_howell_is_zero_for_uncambered_blade():
    c = _make_cascade(0.9, 0.0, 15.0)
    assert howell_deviation(c, 0.5 * c.chord, 0.5) == pytest.approx(0.0)


def test_howell_increases_with_camber():
    low = _howell_deg(_make_cascade(0.9, 25.0, 10.0), 0.5, 0.5)
    high = _howell_deg(_make_cascade(0.9, 40.0, 10.0), 0.5, 0.5)
    assert high > low > 0.0


def test_howell_increases_with_space_chord_ratio():
    # Book: deviation increases with pitch-chord ratio and with camber.
    dense = _howell_deg(_make_cascade(0.7, 35.0, 10.0), 0.5, 0.5)
    sparse = _howell_deg(_make_cascade(1.2, 35.0, 10.0), 0.5, 0.5)
    assert sparse > dense


def test_howell_increases_with_aft_camber_position():
    # Larger a/l raises m through the (2a/l)^2 term.
    forward = _howell_deg(_make_cascade(0.9, 35.0, 10.0), 0.3, 0.5)
    aft = _howell_deg(_make_cascade(0.9, 35.0, 10.0), 0.6, 0.5)
    assert aft > forward


def test_howell_exponent_choice_is_irrelevant_at_unit_space_chord_ratio():
    # (s/l)^n = 1 for any n when s/l = 1.
    c = _make_cascade(1.0, 30.0, 20.0)
    assert _howell_deg(c, 0.5, 0.5) == pytest.approx(_howell_deg(c, 0.5, 1.0))


def test_howell_inlet_guide_vane_exponent_n_equals_one():
    # n = 1: (s/l)^1 < (s/l)^0.5 for s/l < 1, so the deviation is smaller.
    c = _make_cascade(0.8, 30.0, 10.0)
    expected = _howell_reference_deg(0.8, 30.0, 0.5, 10.0, 1.0)
    assert _howell_deg(c, 0.5, 1.0) == pytest.approx(expected)
    assert _howell_deg(c, 0.5, 1.0) < _howell_deg(c, 0.5, 0.5)


@pytest.mark.parametrize("a_over_l", [-0.1, 1.5])
def test_howell_rejects_camber_position_outside_the_chord(a_over_l):
    c = _make_cascade(0.9, 35.0, 10.0)
    with pytest.raises(ValueError, match="maximum camber"):
        howell_deviation(c, a_over_l * c.chord, 0.5)


@pytest.mark.parametrize("n", [0.0, 0.3, 0.75, 2.0, -0.5])
def test_howell_rejects_exponent_other_than_half_or_one(n):
    c = _make_cascade(0.9, 35.0, 10.0)
    with pytest.raises(ValueError, match="n must be"):
        howell_deviation(c, 0.5 * c.chord, n)


@pytest.mark.parametrize("theta_deg", [500.0, 600.0])
def test_howell_rejects_zero_or_negative_denominator(theta_deg):
    # 1 - k*c <= 0 when theta * (s/l)^n >= 500 deg, far outside real blading.
    # At s/l = 1 and theta = 500 deg the denominator is exactly zero.
    c = _make_cascade(1.0, theta_deg, 0.0)
    with pytest.raises(ValueError, match="Denominator"):
        howell_deviation(c, 0.5 * c.chord, 0.5)
