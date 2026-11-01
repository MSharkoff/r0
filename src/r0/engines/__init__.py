"""Simulation engines. Both run the same compiled model."""

import jax

# Epidemic counts span many orders of magnitude (a handful of cases in a
# population of tens of millions): double precision keeps the deterministic
# engine conservative to round-off and the stochastic engine exact.
jax.config.update("jax_enable_x64", True)

from .deterministic import simulate_deterministic  # noqa: E402
from .result import SimulationResult  # noqa: E402
from .stochastic import simulate_stochastic  # noqa: E402

__all__ = ["SimulationResult", "simulate_deterministic", "simulate_stochastic"]
