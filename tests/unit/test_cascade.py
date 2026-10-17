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
