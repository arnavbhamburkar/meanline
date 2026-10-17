# meanline

Mean-line (1D, meridional mean radius) design and analysis of axial compressors
in Python. Work in progress: single stage first, then multi-stage.

This is a mean-line model at one radius, with annulus and blockage assumptions
stated explicitly. It is not a 3D result.

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the plan and validation strategy.

## Status

The foundation modules are implemented and unit-tested: thermodynamic state handling,
velocity triangles, Euler work, stage parameters, diffusion factor and de Haller
number. Blade-row geometry is in progress: annulus geometry with blockage, cascade geometry
(pitch, solidity, camber) and the incidence, deviation and turning definitions are
done. The deviation correlation (Howell's rule) and the blade-row model are still
to do. **Nothing is validated against published data yet**; the unit tests are
hand calculations and consistency checks. Validation against literature will live
in `tests/validation/`.

| Module | Contents |
|---|---|
| `meanline.thermo` | `ThermoState` (CoolProp real-gas state), static <-> total conversions for a given velocity |
| `meanline.velocity_triangles` | Velocity components, triangle closure, Euler work, flow coefficient, stage loading coefficient, degree of reaction |
| `meanline.annulus` | `Annulus` (hub, tip, blockage; derived mean radius, flow and effective area), mass flow and axial velocity from continuity |
| `meanline.cascade` | `Cascade` geometry (pitch, solidity, camber), incidence, deviation and turning angles, diffusion factor, de Haller number |

## Model level and assumptions

- Mean-line (1D) at a single mean radius. It says nothing about radial variation
  and is not a 3D result.
- Real-gas properties come from [CoolProp](http://www.coolprop.org); no
  ideal-gas constants are hard-coded in the model. Ideal-gas formulas appear only as
  cross-checks in the tests and the notebook.
- SI units throughout.
- Stagnation (total) states are reached isentropically: `h0 = h + c^2 / 2` at
  constant entropy, then CoolProp returns `T0`, `p0` and `rho0` from `(h0, s)`.
- `degree_of_reaction` uses the velocity form `(w1^2 - w2^2) / (2 * work)`, which
  assumes constant rothalpy across the rotor, constant radius and unchanged axial
  velocity (see its docstring).
- Equation citations are marked `TODO(source)` in the docstrings until checked
  against the cited reference (Dixon & Hall).

### Sign convention

Flow angles are measured from the axial direction. Tangential velocity `c_theta` is
positive in the direction of blade motion and `w_theta` is positive against it, so
the triangle closes as `U = c_theta + w_theta`.

## Quick example

```python
from meanline.thermo import ThermoState, total_from_static

static = ThermoState.state_from_pT("Air", p=101_325.0, T=300.0)
total = total_from_static(static, c=100.0)

print(f"T0 = {total.T:.2f} K, p0 = {total.p:.0f} Pa")  # about 304.98 K, 107333 Pa
```

A walk-through of the basic functions, with plots, is in
[notebooks/01_basics.ipynb](notebooks/01_basics.ipynb).

## Development

```
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
