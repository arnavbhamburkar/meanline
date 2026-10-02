# Project plan: meanline (axial compressor)

## Goal
A validated, documented Python mean-line tool for an axial compressor (single
stage, then multi-stage), reproducing published performance data from public
sources with an honest error table, then used for design optimization and
sensitivity studies. The compressor map it produces later feeds a turbofan
cycle model.

## References (verify every coefficient against these)
- Dixon & Hall, *Fluid Mechanics and Thermodynamics of Turbomachinery*
- Cumpsty, *Compressor Aerodynamics*
- Aungier, *Axial-Flow Compressors: A Strategy for Aerodynamic Design and Analysis*
- Saravanamuttoo et al., *Gas Turbine Theory* (cycle and matching context)
- NASA SP-36, *Aerodynamic Design of Axial-Flow Compressors* (public)
- Original NASA reports for the validation cases below; check geometry and
  available data before committing to a case.

## Validation strategy (staged, from easier to harder)
Mean-line codes can reproduce midspan behavior, but they struggle with
transonic flow and 3D effects. Validate in stages and state clearly where the
model stops being valid.

1. **Hand and textbook cases:** velocity triangles, stage loading and
   diffusion-factor examples from the references.
2. **Low-speed or subsonic stage data from public literature:** compare
   stage pressure rise and efficiency trends against published data.
3. **Stretch case:** NASA Stage 35 / Rotor 37 (public, transonic). Compare
   at midspan and discuss shock-loss modelling limits. Treat a good match as
   a bonus, and a well-explained mismatch as a valid result.

## Milestones

### Phase 1: foundations
- Thermodynamic state handling with CoolProp (total and static states)
- Velocity triangles in absolute and relative frames
- Euler work equation
- Stage parameters: flow coefficient, loading coefficient, degree of reaction
- Diffusion factor and de Haller number
- A hand-calculation test for each function

### Phase 2: blade-row geometry and cascade correlations
- Annulus and blockage handling (hub, tip, mean radius)
- Incidence and deviation rules, taken from the cited references
- Blade-row model: turning, exit flow state

### Phase 3: losses and multi-stage
- Loss mechanisms, one function per mechanism: profile loss, endwall loss,
  tip-clearance loss, shock loss (subsonic first, transonic later)
- Stage efficiency and pressure ratio
- Multi-stage stacking and work split

### Phase 4: off-design performance
- Speedlines by sweeping mass flow at constant speed
- Choke and stall (diffusion-factor / de Haller) indicators
- Compressor map plots

### Phase 5: validation
- Run the staged validation cases above
- Error table (state max and mean deviation clearly)
- Discussion of where and why the model breaks down

### Phase 6: engineering value and release
- Design sizing for a target pressure ratio and mass flow with
  `scipy.optimize.differential_evolution`
- Monte Carlo sensitivity to loss-coefficient uncertainty
- README with validation plots, notebook walkthrough, GitHub release,
  Zenodo DOI

## Extensions after the core is done
1. Turbofan cycle model that uses the compressor map (own code or pyCycle),
   compared against public engine data
2. Radial-equilibrium hub-to-tip extension (toward quasi-2D streamline curvature)
3. Axial turbine module (Ainley-Mathieson or Kacker-Okapuu family; verify sources)
4. Centrifugal compressor module and a fuel-cell air-supply and
   thermal-management model (humid air via CoolProp)
5. 2D cascade CFD validation (SU2 or OpenFOAM) as a separate repo
6. Short LaTeX technical report and a conference abstract

## Definition of done for the core project
- Tests pass in CI
- Validation plots and error tables in the README
- Every correlation cited with source and equation number, with validity range
- A reader can reproduce the validation cases with one command
