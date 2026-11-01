"""R0 : simulateur de propagation d'épidémie."""

from pathlib import Path

from .engines import SimulationResult, simulate_deterministic, simulate_stochastic
from .model import CompiledModel, ModelConfig, compile_model

__version__ = "0.1.0"


def load(path: str | Path) -> CompiledModel:
    """Read a YAML configuration and compile it."""
    return compile_model(ModelConfig.from_yaml(path))


__all__ = [
    "CompiledModel",
    "ModelConfig",
    "SimulationResult",
    "compile_model",
    "load",
    "simulate_deterministic",
    "simulate_stochastic",
]
