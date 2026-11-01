"""Deterministic engine: the mean-field ODE system, integrated with a
fixed-step fourth-order Runge-Kutta scheme."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..model.ngm import beta_from_R0
from ..model.structure import CompiledModel
from ._common import aggregation_matrix, steps_per_day
from .result import SimulationResult


def simulate_deterministic(
    model: CompiledModel,
    days: int,
    dt: float = 0.1,
    beta: float | None = None,
) -> SimulationResult:
    """Integrate the model for ``days`` days.

    ``beta`` defaults to the value that gives the configured R0.
    """
    beta = beta_from_R0(model) if beta is None else float(beta)
    steps = steps_per_day(dt)

    A = jnp.asarray(model.linear_operator())
    B = jnp.asarray(model.inflow_operator())
    w = jnp.asarray(model.infectiousness)
    C = jnp.asarray(model.C)
    N = jnp.asarray(model.N)
    s, t, tc = model.susceptible, model.target, model.target_comp

    def deriv(X):
        lam = beta * C @ ((w @ X) / N)
        new_inf = lam * X[s]
        dX = jnp.einsum("aij,ja->ia", A, X)
        dX = dX.at[s].add(-new_inf).at[t].add(new_inf)
        dflow = jnp.einsum("acs,sa->ca", B, X).at[tc].add(new_inf)
        return dX, dflow

    def rk4(carry, _):
        X, flow = carry
        k1x, k1f = deriv(X)
        k2x, k2f = deriv(X + 0.5 * dt * k1x)
        k3x, k3f = deriv(X + 0.5 * dt * k2x)
        k4x, k4f = deriv(X + dt * k3x)
        X = X + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        flow = flow + dt / 6.0 * (k1f + 2 * k2f + 2 * k3f + k4f)
        return (X, flow), None

    flow0 = jnp.zeros((len(model.compartments), model.n_age))

    def one_day(X, _):
        (X, flow), _ = jax.lax.scan(rk4, (X, flow0), None, length=steps)
        return X, (X, flow)

    @jax.jit
    def run(X0):
        _, (Xs, flows) = jax.lax.scan(one_day, X0, None, length=days)
        return Xs, flows

    X0 = jnp.asarray(model.initial_state())
    Xs, flows = run(X0)
    agg = aggregation_matrix(model)
    traj = np.concatenate([np.asarray(X0)[None], np.asarray(Xs)], axis=0)
    states = np.einsum("cn,dnm->dcm", agg, traj)
    return SimulationResult(
        model=model,
        beta=beta,
        days=np.arange(days + 1),
        states=states[None],
        inflows=np.asarray(flows)[None],
        engine="deterministic",
    )
