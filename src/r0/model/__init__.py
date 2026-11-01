"""Model definition: configuration, compiled structure, reproduction numbers."""

from .config import Compartment, Infection, ModelConfig, Population, Transition
from .ngm import (
    beta_from_R0,
    doubling_time,
    effective_reproduction_numbers,
    growth_rate,
    infectious_time,
    next_generation_matrix,
    reproduction_number,
)
from .structure import CompiledModel, compile_model

__all__ = [
    "CompiledModel",
    "Compartment",
    "Infection",
    "ModelConfig",
    "Population",
    "Transition",
    "beta_from_R0",
    "compile_model",
    "doubling_time",
    "effective_reproduction_numbers",
    "growth_rate",
    "infectious_time",
    "next_generation_matrix",
    "reproduction_number",
]
