"""Property-based tests: invariants that must hold for any valid model."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from r0 import simulate_deterministic, simulate_stochastic
from r0.model import beta_from_R0, reproduction_number
from tests.conftest import build, seir_config

models = st.fixed_dictionaries(
    {
        "R0": st.floats(0.5, 8.0),
        "latent": st.floats(0.5, 10.0),
        "infectious": st.floats(1.0, 15.0),
        "stages": st.integers(1, 4),
        "p_severe": st.floats(0.0, 1.0),
        "seed": st.integers(1, 1000),
    }
)


def _config(p):
    return seir_config(
        R0=p["R0"],
        compartments={
            "S": {},
            "E": {"duration": p["latent"], "stages": p["stages"], "transitions": [{"to": "I"}]},
            "I": {
                "duration": p["infectious"],
                "stages": p["stages"],
                "infectiousness": 1.0,
                "transitions": [{"to": "H", "probability": p["p_severe"]}, {"to": "R"}],
            },
            "H": {"duration": 7, "transitions": [{"to": "R"}]},
            "R": {},
        },
        initial={"E": p["seed"]},
    )


@settings(max_examples=15, deadline=None)
@given(models)
def test_deterministic_invariants(p):
    model = build(_config(p))
    assert np.isclose(reproduction_number(model, beta_from_R0(model)), p["R0"])
    res = simulate_deterministic(model, 60)
    assert np.allclose(res.states.sum(axis=(2, 3)), model.N.sum(), rtol=1e-9)
    assert res.states.min() > -1e-6
    assert res.inflows.min() > -1e-6


@settings(max_examples=10, deadline=None)
@given(models)
def test_stochastic_invariants(p):
    model = build(_config(p))
    res = simulate_stochastic(model, 30, replicates=2, seed=p["seed"])
    assert np.all(res.states.sum(axis=(2, 3)) == model.N.sum())
    assert res.states.min() >= 0
    assert res.inflows.min() >= 0
