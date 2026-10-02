"""Shared pint unit registry.

All modules must use this single registry so quantities can be combined.
SI units everywhere internally; convert only at input and output.
"""

import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
