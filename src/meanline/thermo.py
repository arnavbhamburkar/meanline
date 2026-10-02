"""Thermodynamic state handling via CoolProp.

Planned scope:
    - Static and total (stagnation) state representation.
    - Static <-> total conversions for a given velocity.
    - Real-gas properties from CoolProp; no hard-coded ideal-gas constants
      unless a function is explicitly ideal-gas.

Model level: mean-line, one radius. SI units.

References
----------
TODO(source): Dixon & Hall, total/static relations (chapter and equation).
"""
