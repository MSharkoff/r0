import numpy as np
import pytest

from r0.model import (
    beta_from_R0,
    doubling_time,
    growth_rate,
    infectious_time,
    reproduction_number,
)
from tests.conftest import build, seir_config


def test_seir_beta_closed_form():
    # R0 = beta * contacts * infectious duration
    model = build(seir_config())
    assert beta_from_R0(model) == pytest.approx(2.5 / (10 * 5))


def test_sepiahrd_infectious_time_closed_form():
    # T = 1/delta + ((1 - p) + kappa p) / gamma ; H and E are not infectious
    p, kappa, d_p, d_i = 0.4, 0.5, 2.0, 5.0
    cfg = seir_config(
        compartments={
            "S": {},
            "E": {"duration": 3, "stages": 3, "transitions": [{"to": "P"}]},
            "P": {
                "duration": d_p,
                "stages": 2,
                "infectiousness": 1.0,
                "transitions": [{"to": "Ia", "probability": p}, {"to": "Is"}],
            },
            "Is": {
                "duration": d_i,
                "stages": 3,
                "infectiousness": 1.0,
                "transitions": [{"to": "H", "probability": 0.1}, {"to": "R"}],
            },
            "Ia": {"duration": d_i, "infectiousness": kappa, "transitions": [{"to": "R"}]},
            "H": {"duration": 10, "transitions": [{"to": "D", "probability": 0.2}, {"to": "R"}]},
            "R": {},
            "D": {},
        }
    )
    model = build(cfg)
    expected = d_p + ((1 - p) + kappa * p) * d_i
    assert infectious_time(model)[0] == pytest.approx(expected)


@pytest.mark.parametrize("stages", [1, 2, 5])
def test_erlang_stages_do_not_change_R0(stages):
    cfg = seir_config()
    cfg["compartments"]["E"]["stages"] = stages
    cfg["compartments"]["I"]["stages"] = stages
    assert beta_from_R0(build(cfg)) == pytest.approx(0.05)


def test_R0_round_trip(france):
    beta = beta_from_R0(france)
    assert reproduction_number(france, beta) == pytest.approx(2.5)


def test_rt_scales_with_susceptibles_in_one_group():
    model = build(seir_config())
    beta = beta_from_R0(model)
    assert reproduction_number(model, beta, susceptible=model.N * 0.4) == pytest.approx(1.0)


def test_seir_growth_rate_closed_form():
    # (r + sigma)(r + gamma) = b sigma with b = beta * contacts = R0 * gamma
    model = build(seir_config())
    sigma, gamma = 1 / 3, 1 / 5
    b = 2.5 * gamma
    r = (-(sigma + gamma) + np.sqrt((sigma + gamma) ** 2 - 4 * (sigma * gamma - b * sigma))) / 2
    assert growth_rate(model) == pytest.approx(r)
    assert doubling_time(model) == pytest.approx(np.log(2) / r)


def test_subcritical_growth_rate_is_negative():
    model = build(seir_config(R0=0.8))
    assert growth_rate(model) < 0
    assert doubling_time(model) == float("inf")
