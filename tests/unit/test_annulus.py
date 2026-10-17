import dataclasses
import math

import pytest

from meanline.annulus import (
    Annulus,
    axial_velocity_from_mass_flow_rate,
    mass_flow_rate,
)

# Reference annulus: r_hub = 0.2 m, r_tip = 0.3 m, no blockage.
#   r_mean = 0.25 m
#   A_flow = pi * (0.3^2 - 0.2^2) = 0.05 * pi m^2 = 0.15708 m^2


@pytest.fixture
def annulus():
    return Annulus(r_hub=0.2, r_tip=0.3, blockage=0.0)


# --- construction and derived geometry -------------------------------------


def test_mean_radius_hand_calculation(annulus):
    assert annulus.r_mean == pytest.approx(0.25)


def test_flow_area_hand_calculation(annulus):
    assert annulus.A_flow == pytest.approx(0.05 * math.pi)


def test_zero_blockage_gives_effective_area_equal_to_flow_area(annulus):
    assert annulus.blockage == 0.0
    assert annulus.A_eff == pytest.approx(annulus.A_flow)


def test_inputs_are_stored_unchanged(annulus):
    assert annulus.r_hub == 0.2
    assert annulus.r_tip == 0.3
    assert annulus.blockage == 0.0


def test_positional_construction_order_is_hub_tip_blockage():
    ann = Annulus(0.2, 0.3, 0.1)
    assert (ann.r_hub, ann.r_tip, ann.blockage) == (0.2, 0.3, 0.1)


def test_only_independent_quantities_are_dataclass_fields():
    # Derived values must not be stored, so they cannot go out of step.
    names = [f.name for f in dataclasses.fields(Annulus)]
    assert names == ["r_hub", "r_tip", "blockage"]


def test_flow_area_of_thin_annulus_matches_circumference_times_height():
    # r_hub = 0.499, r_tip = 0.501: A -> 2 * pi * r_mean * height = 2*pi*0.5*0.002.
    thin = Annulus(0.499, 0.501, 0.0)
    assert thin.A_flow == pytest.approx(2.0 * math.pi * 0.5 * 0.002)


def test_annulus_is_frozen(annulus):
    with pytest.raises(dataclasses.FrozenInstanceError):
        annulus.r_hub = 0.1


@pytest.mark.parametrize("name", ["r_mean", "A_flow", "A_eff"])
def test_derived_quantities_cannot_be_assigned(annulus, name):
    with pytest.raises(AttributeError):
        setattr(annulus, name, 1.0)


# --- construction checks ---------------------------------------------------


@pytest.mark.parametrize("r_hub", [0.0, -0.1, math.nan, math.inf])
def test_rejects_invalid_hub_radius(r_hub):
    with pytest.raises(ValueError, match="Hub radius"):
        Annulus(r_hub, 0.3, 0.0)


@pytest.mark.parametrize("r_tip", [0.0, -0.3, 0.2, 0.15, math.nan, math.inf])
def test_rejects_tip_not_finite_and_above_hub(r_tip):
    with pytest.raises(ValueError, match="Tip radius"):
        Annulus(0.2, r_tip, 0.0)


@pytest.mark.parametrize("blockage", [-0.01, 1.0, 1.5, math.nan])
def test_rejects_blockage_outside_zero_to_one(blockage):
    with pytest.raises(ValueError, match="[Bb]lockage"):
        Annulus(0.2, 0.3, blockage)


@pytest.mark.parametrize("blockage", [0.0, 0.05, 0.999])
def test_accepts_blockage_inside_range(blockage):
    assert Annulus(0.2, 0.3, blockage).blockage == blockage


# --- blockage --------------------------------------------------------------


def test_blockage_reduces_effective_area_only():
    blocked = Annulus(0.2, 0.3, 0.1)
    assert blocked.A_flow == pytest.approx(0.05 * math.pi)
    assert blocked.A_eff == pytest.approx(0.9 * 0.05 * math.pi)


def test_blockage_does_not_change_mean_radius():
    assert Annulus(0.2, 0.3, 0.1).r_mean == pytest.approx(0.25)


# --- mass flow rate --------------------------------------------------------


def test_mass_flow_rate_hand_calculation(annulus):
    # m_dot = 1.2 * 0.05 * pi * 150 = 9 * pi = 28.274 kg/s.
    assert mass_flow_rate(1.2, annulus, 150.0) == pytest.approx(9.0 * math.pi)


def test_mass_flow_rate_uses_effective_not_geometric_area(annulus):
    # 10 % blockage must cut the mass flow by 10 % at the same rho and v_x.
    blocked = Annulus(0.2, 0.3, 0.1)
    assert mass_flow_rate(1.2, blocked, 150.0) == pytest.approx(
        0.9 * mass_flow_rate(1.2, annulus, 150.0)
    )


def test_mass_flow_rate_scales_linearly_with_density_and_velocity(annulus):
    base = mass_flow_rate(1.0, annulus, 100.0)
    assert mass_flow_rate(2.0, annulus, 100.0) == pytest.approx(2.0 * base)
    assert mass_flow_rate(1.0, annulus, 300.0) == pytest.approx(3.0 * base)


@pytest.mark.parametrize("v_x", [0.0, -50.0])
def test_mass_flow_rate_rejects_non_positive_axial_velocity(annulus, v_x):
    with pytest.raises(ValueError, match="Axial velocity"):
        mass_flow_rate(1.2, annulus, v_x)


@pytest.mark.parametrize("rho", [0.0, -1.2])
def test_mass_flow_rate_rejects_non_positive_density(annulus, rho):
    with pytest.raises(ValueError, match="Density"):
        mass_flow_rate(rho, annulus, 150.0)


# --- axial velocity from mass flow ----------------------------------------


def test_axial_velocity_hand_calculation(annulus):
    # v_x = 9 * pi / (1.2 * 0.05 * pi) = 150 m/s.
    v_x = axial_velocity_from_mass_flow_rate(9.0 * math.pi, 1.2, annulus)
    assert v_x == pytest.approx(150.0)


def test_axial_velocity_inverts_mass_flow_rate(annulus):
    m_dot = mass_flow_rate(0.9, annulus, 187.5)
    v_x = axial_velocity_from_mass_flow_rate(m_dot, 0.9, annulus)
    assert v_x == pytest.approx(187.5)


def test_axial_velocity_rises_with_blockage_at_same_mass_flow(annulus):
    # Less open area -> the same mass flow needs a faster stream.
    blocked = Annulus(0.2, 0.3, 0.1)
    clear = axial_velocity_from_mass_flow_rate(10.0, 1.2, annulus)
    tight = axial_velocity_from_mass_flow_rate(10.0, 1.2, blocked)
    assert tight == pytest.approx(clear / 0.9)


@pytest.mark.parametrize("rho", [0.0, -1.2])
def test_axial_velocity_rejects_non_positive_density(annulus, rho):
    with pytest.raises(ValueError, match="Density"):
        axial_velocity_from_mass_flow_rate(10.0, rho, annulus)
