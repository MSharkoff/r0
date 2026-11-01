"""Stochastic engine: a binomial chain on the same compiled model.

At each time step every individual of a state leaves it with probability
``rate * dt``, so that the number of steps spent in a stage is geometric with
mean exactly ``1 / (rate * dt)``. (Using ``1 - exp(-rate * dt)`` instead would
lengthen every stage by about ``dt / 2`` and bias R0 upwards, by several
percent for models with many Erlang stages.) Leavers are split between destinations with
sequential conditional binomial draws (a multinomial draw), and each
susceptible is infected with probability ``1 - exp(-lambda * dt)``. Counts
stay integers and the population is conserved exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..model.ngm import beta_from_R0
from ..model.structure import CompiledModel
from ._common import aggregation_matrix, steps_per_day
from .result import SimulationResult


def simulate_stochastic(
    model: CompiledModel,
    days: int,
    replicates: int = 1,
    seed: int = 0,
    dt: float = 0.1,
    beta: float | None = None,
) -> SimulationResult:
    """Run ``replicates`` independent stochastic trajectories."""
    beta = beta_from_R0(model) if beta is None else float(beta)
    steps = steps_per_day(dt)
    n, m = model.n_states, model.n_age
    n_comp = len(model.compartments)
    n_dest = model.dest.shape[1]

    w = jnp.asarray(model.infectiousness)
    C = jnp.asarray(model.C)
    N = jnp.asarray(model.N)
    if np.any(model.rate * dt > 1):
        fastest = float(model.rate.max())
        raise ValueError(
            f"dt={dt} is too large for a stage left at rate {fastest:.3g}/day; "
            f"use dt <= {1 / fastest:.3g}"
        )
    p_exit = jnp.asarray(model.rate * dt)
    prob = jnp.asarray(model.prob)  # (n, J, m)
    dest = jnp.asarray(np.maximum(model.dest, 0))
    dest_comp = jnp.asarray(np.maximum(model.dest_comp, 0))
    enters_comp = jnp.asarray(model.dest_comp >= 0)
    # The last real destination of each state takes every remaining leaver,
    # so that round-off in the conditional probabilities never loses anyone.
    has_dest = model.dest >= 0
    last = np.where(has_dest.any(axis=1), n_dest - 1 - np.argmax(has_dest[:, ::-1], axis=1), -1)
    is_last = jnp.asarray(np.arange(n_dest)[None, :] == last[:, None])  # (n, J)
    s, t, tc = model.susceptible, model.target, model.target_comp

    def step(carry, key):
        X, flow = carry
        k_inf, k_exit, k_route = jax.random.split(key, 3)
        lam = beta * C @ ((w @ X) / N)
        new_inf = jax.random.binomial(k_inf, X[s], -jnp.expm1(-lam * dt))
        exits = jax.random.binomial(k_exit, X, p_exit)

        dX = -exits
        dflow = jnp.zeros((n_comp, m))
        remaining = exits
        taken = jnp.zeros((n, m))
        route_keys = jax.random.split(k_route, n_dest)
        for j in range(n_dest):
            pj = prob[:, j, :]
            free = 1.0 - taken
            pc = jnp.where(free > 1e-12, jnp.clip(pj / jnp.where(free > 1e-12, free, 1.0), 0, 1), 0)
            pc = jnp.where(is_last[:, j][:, None], 1.0, pc)
            moved = jax.random.binomial(route_keys[j], remaining, pc)
            remaining = remaining - moved
            taken = taken + pj
            dX = dX.at[dest[:, j]].add(moved)
            dflow = dflow.at[dest_comp[:, j]].add(jnp.where(enters_comp[:, j][:, None], moved, 0))

        dX = dX.at[s].add(-new_inf).at[t].add(new_inf)
        dflow = dflow.at[tc].add(new_inf)
        return (X + dX, flow + dflow), None

    def one_day(X, key):
        keys = jax.random.split(key, steps)
        (X, flow), _ = jax.lax.scan(step, (X, jnp.zeros((n_comp, m))), keys)
        return X, (X, flow)

    def run(X0, key):
        keys = jax.random.split(key, days)
        _, (Xs, flows) = jax.lax.scan(one_day, X0, keys)
        return Xs, flows

    X0_np = model.initial_state(integer=True)
    X0 = jnp.asarray(X0_np)
    keys = jax.random.split(jax.random.key(seed), replicates)
    Xs, flows = jax.jit(jax.vmap(run, in_axes=(None, 0)))(X0, keys)

    agg = aggregation_matrix(model)
    traj = np.concatenate(
        [np.broadcast_to(X0_np, (replicates, 1, n, m)), np.asarray(Xs)], axis=1
    )
    states = np.einsum("cn,rdnm->rdcm", agg, traj)
    return SimulationResult(
        model=model,
        beta=beta,
        days=np.arange(days + 1),
        states=states,
        inflows=np.asarray(flows),
        engine="stochastic",
    )
