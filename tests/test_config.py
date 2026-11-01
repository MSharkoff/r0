import numpy as np
import pytest
from pydantic import ValidationError

from r0 import ModelConfig, load
from tests.conftest import CONFIGS, build, seir_config


@pytest.mark.parametrize("name", ["seir.yaml", "sepiahrd_france.yaml"])
def test_example_configs_load(name):
    model = load(CONFIGS / name)
    assert model.n_states >= len(model.compartments)


def test_erlang_stages_expand_states():
    cfg = seir_config()
    cfg["compartments"]["E"]["stages"] = 4
    model = build(cfg)
    assert model.n_states == 1 + 4 + 1 + 1
    assert np.allclose(model.rate[model.state_slice("E")], 4 / 3)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["compartments"]["E"]["transitions"].__setitem__(0, {"to": "X"}), "unknown destination"),
        (lambda c: c["compartments"]["I"].pop("transitions"), "at least one transition"),
        (lambda c: c["compartments"]["R"].update(transitions=[{"to": "S"}]), "without duration"),
        (lambda c: c["compartments"]["E"]["transitions"].__setitem__(0, {"to": "E"}), "itself"),
        (
            lambda c: c["compartments"]["E"].update(
                transitions=[{"to": "I", "probability": 0.7}, {"to": "R", "probability": 0.6}]
            ),
            "exceed 1",
        ),
        (
            lambda c: c["compartments"]["E"].update(
                transitions=[{"to": "I", "probability": 0.7}, {"to": "R", "probability": 0.2}]
            ),
            "sum to 1",
        ),
        (lambda c: c["compartments"]["E"].update(transitions=[{"to": "I"}, {"to": "R"}]), "remaining"),
        (lambda c: c.update(initial={"S": 5}), "susceptible"),
        (lambda c: c.update(initial={"I": [1, 2]}), "one per age group"),
        (lambda c: c["population"].update(sizes=[0]), "positive"),
        (lambda c: c.update(infection={"susceptible": "S", "target": "Q"}), "not a compartment"),
    ],
)
def test_invalid_configs_are_rejected(mutate, message):
    cfg = seir_config()
    mutate(cfg)
    with pytest.raises(ValidationError, match=message):
        ModelConfig.model_validate(cfg)


def test_seed_larger_than_population_is_rejected():
    model = build(seir_config(initial={"E": 2_000_000}))
    with pytest.raises(ValueError, match="exceed"):
        model.initial_state()


def test_contact_matrix_is_made_reciprocal(france):
    total = france.C * france.N[:, None]
    assert np.allclose(total, total.T)


def test_age_dependent_routing(france):
    s = france.state_slice("P").stop - 1  # last stage of P
    to_ia = france.prob[s, 0]
    to_is = france.prob[s, 1]
    assert np.allclose(to_ia, [0.6, 0.45, 0.35, 0.25, 0.2])
    assert np.allclose(to_ia + to_is, 1)


def test_integer_seeding_preserves_totals(france):
    X = france.initial_state(integer=True)
    assert np.all(X == np.round(X))
    assert X[france.state_slice("E")].sum() == 100
    assert X.sum() == round(france.N.sum())
