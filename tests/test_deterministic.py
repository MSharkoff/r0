import numpy as np
import pytest

from r0 import simulate_deterministic
from r0.model import growth_rate
from tests.conftest import build, final_size, seir_config


def test_population_is_conserved(france):
    res = simulate_deterministic(france, 300)
    totals = res.states.sum(axis=(2, 3))
    assert np.allclose(totals, france.N.sum(), rtol=1e-10)


def test_flows_match_occupancy(france):
    res = simulate_deterministic(france, 300)
    # everybody who died entered D through a daily inflow
    assert res.inflow("D").sum() == pytest.approx(res.compartment("D")[0, -1], rel=1e-8)
    # everybody infected left S
    S = res.compartment("S")[0]
    assert res.incidence().sum() == pytest.approx(S[0] - S[-1], rel=1e-8)


@pytest.mark.parametrize("R0", [1.5, 2.5, 4.0])
def test_seir_final_size_matches_theory(R0):
    model = build(seir_config(R0=R0, population={"age_groups": ["all"], "sizes": [1e9], "contacts": [[10.0]]}))
    res = simulate_deterministic(model, 1000)
    assert res.attack_rate()[0] == pytest.approx(final_size(R0), abs=1e-4)


@pytest.mark.parametrize("stages", [1, 3])
def test_final_size_holds_for_any_single_population_structure(stages):
    # The final size relation only depends on R0, not on durations, stages
    # or how the infected population is split between compartments.
    cfg = seir_config(
        population={"age_groups": ["all"], "sizes": [1e9], "contacts": [[8.0]]},
        compartments={
            "S": {},
            "E": {"duration": 3, "stages": stages, "transitions": [{"to": "P"}]},
            "P": {
                "duration": 2,
                "stages": stages,
                "infectiousness": 1.0,
                "transitions": [{"to": "Ia", "probability": 0.4}, {"to": "Is"}],
            },
            "Is": {
                "duration": 5,
                "stages": stages,
                "infectiousness": 1.0,
                "transitions": [{"to": "H", "probability": 0.1}, {"to": "R"}],
            },
            "Ia": {"duration": 5, "infectiousness": 0.5, "transitions": [{"to": "R"}]},
            "H": {"duration": 10, "transitions": [{"to": "D", "probability": 0.2}, {"to": "R"}]},
            "R": {},
            "D": {},
        },
    )
    res = simulate_deterministic(build(cfg), 1000)
    assert res.attack_rate()[0] == pytest.approx(final_size(2.5), abs=1e-4)


def test_subcritical_epidemic_fizzles():
    model = build(seir_config(R0=0.9, initial={"E": 100}))
    res = simulate_deterministic(model, 500)
    # each seed starts a subcritical chain of R0 / (1 - R0) further infections
    assert res.incidence().sum() == pytest.approx(100 * 0.9 / (1 - 0.9), rel=0.02)


def test_herd_immunity_threshold_prevents_growth():
    R0 = 2.5
    immune = (1 - 1 / R0) * 1_000_000
    model = build(seir_config(R0=R0, initial={"R": immune, "E": 10}))
    res = simulate_deterministic(model, 400)
    rt = res.rt()[0]
    assert rt[0] == pytest.approx(1.0, abs=1e-4)
    assert np.all(np.diff(rt) <= 1e-12)
    assert res.compartment("I")[0].max() < 10


def test_initial_growth_matches_linearisation(france):
    res = simulate_deterministic(france, 60)
    inc = res.incidence()[0]
    slope = np.polyfit(np.arange(20, 45), np.log(inc[20:45]), 1)[0]
    assert slope == pytest.approx(growth_rate(france), rel=0.01)


def test_rt_starts_at_R0_and_ends_below_one(france):
    rt = simulate_deterministic(france, 365).rt()[0]
    assert rt[0] == pytest.approx(2.5, rel=1e-4)
    assert rt[-1] < 1


def test_time_step_has_converged(france):
    a = simulate_deterministic(france, 200, dt=0.1)
    b = simulate_deterministic(france, 200, dt=0.05)
    assert np.allclose(a.states, b.states, rtol=1e-5, atol=1e-3)
