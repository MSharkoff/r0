from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from r0 import ModelConfig, compile_model

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def seir_config(**overrides) -> dict:
    """Single-population SEIR as a plain dict, easy to tweak in tests."""
    cfg = {
        "name": "test SEIR",
        "R0": 2.5,
        "compartments": {
            "S": {},
            "E": {"duration": 3, "transitions": [{"to": "I"}]},
            "I": {"duration": 5, "infectiousness": 1.0, "transitions": [{"to": "R"}]},
            "R": {},
        },
        "population": {"age_groups": ["all"], "sizes": [1_000_000], "contacts": [[10.0]]},
        "initial": {"E": 10},
    }
    cfg.update(overrides)
    return cfg


def build(cfg: dict):
    return compile_model(ModelConfig.model_validate(cfg))


def final_size(R0: float) -> float:
    """Solution of z = 1 - exp(-R0 z) (homogeneous mixing, no waning)."""
    z = 0.5
    for _ in range(200):
        z = 1 - np.exp(-R0 * z)
    return float(z)


@pytest.fixture(scope="session")
def france():
    return compile_model(ModelConfig.from_yaml(CONFIGS / "sepiahrd_france.yaml"))
