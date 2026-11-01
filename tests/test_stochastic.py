import numpy as np
import pytest

from r0 import simulate_deterministic, simulate_stochastic
from tests.conftest import build, seir_config


def test_population_is_conserved_exactly(france):
    res = simulate_stochastic(france, 120, replicates=3)
    assert np.all(res.states == np.round(res.states))
    assert np.all(res.states.sum(axis=(2, 3)) == round(france.N.sum()))
    assert np.all(res.states >= 0)


def test_same_seed_same_result(france):
    a = simulate_stochastic(france, 60, replicates=2, seed=7)
    b = simulate_stochastic(france, 60, replicates=2, seed=7)
    c = simulate_stochastic(france, 60, replicates=2, seed=8)
    assert np.array_equal(a.states, b.states)
    assert not np.array_equal(a.states, c.states)


def test_mean_matches_deterministic(france):
    det = simulate_deterministic(france, 300)
    sto = simulate_stochastic(france, 300, replicates=6)
    assert sto.attack_rate().mean() == pytest.approx(det.attack_rate()[0], abs=2e-3)
    peak_det = det.peak("H")[0][0]
    assert abs(np.median(sto.peak("H")[0]) - peak_det) <= 3


def test_extinction_probability_is_one_over_R0():
    # With exponential infectious periods, a single introduction dies out
    # with probability 1 / R0 (branching process with geometric offspring).
    R0 = 2.0
    model = build(
        seir_config(
            R0=R0,
            population={"age_groups": ["all"], "sizes": [5000], "contacts": [[10.0]]},
            initial={"E": 1},
        )
    )
    res = simulate_stochastic(model, 150, replicates=2000, seed=1, dt=0.25)
    extinct = np.mean(res.attack_rate() < 0.05)
    assert extinct == pytest.approx(1 / R0, abs=0.04)


def test_too_large_time_step_is_rejected():
    cfg = seir_config()
    cfg["compartments"]["E"].update(duration=1, stages=4)  # stages left at 4 per day
    with pytest.raises(ValueError, match="too large"):
        simulate_stochastic(build(cfg), 10, dt=0.5)
