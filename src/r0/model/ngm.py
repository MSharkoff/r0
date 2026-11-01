"""Reproduction numbers from the next-generation matrix.

The force of infection on age group ``a`` is

    lambda_a = beta * sum_b C[a, b] * (w . X[:, b]) / N[b]

so one infected person of group ``b`` causes, early in the epidemic,
``K[a, b] = beta * C[a, b] * N[a] / N[b] * T[b]`` infections in group ``a``,
where ``T[b]`` is the infectiousness-weighted time that person spends in the
infected states. R0 is the spectral radius of ``K``.
"""

from __future__ import annotations

import numpy as np

from .structure import CompiledModel


def _transient_states(model: CompiledModel) -> np.ndarray:
    """States that are left at a positive rate, susceptible excluded."""
    mask = np.any(model.rate > 0, axis=1)
    mask[model.susceptible] = False
    return np.flatnonzero(mask)


def infectious_time(model: CompiledModel) -> np.ndarray:
    """``T[b]``: expected infectiousness-weighted days spent infected by one
    person of age group ``b`` infected now."""
    A = model.linear_operator()
    idx = _transient_states(model)
    T = np.zeros(model.n_age)
    if model.target not in idx:
        return T
    entry = np.zeros(len(idx))
    entry[np.searchsorted(idx, model.target)] = 1.0
    w = model.infectiousness[idx]
    for b in range(model.n_age):
        M = A[b][np.ix_(idx, idx)]
        tau = np.linalg.solve(-M, entry)
        T[b] = w @ tau
    return T


def next_generation_matrix(
    model: CompiledModel,
    beta: float,
    susceptible: np.ndarray | None = None,
    contact_scale: float = 1.0,
) -> np.ndarray:
    """``K[a, b]``: infections in group ``a`` caused by one infected of group
    ``b``. ``susceptible`` defaults to the whole population (R0); pass the
    current susceptible counts to get Rt."""
    S = model.N if susceptible is None else np.asarray(susceptible, dtype=float)
    T = infectious_time(model)
    return beta * contact_scale * model.C * (S[:, None] / model.N[None, :]) * T[None, :]


def spectral_radius(K: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(K))))


def beta_from_R0(model: CompiledModel, R0: float | None = None) -> float:
    """Transmission rate per contact that yields the requested R0."""
    R0 = model.config.R0 if R0 is None else R0
    rho = spectral_radius(next_generation_matrix(model, beta=1.0))
    if rho == 0:
        raise ValueError("the model has no infectious state reachable from infection")
    return R0 / rho


def reproduction_number(
    model: CompiledModel,
    beta: float,
    susceptible: np.ndarray | None = None,
    contact_scale: float = 1.0,
) -> float:
    return spectral_radius(next_generation_matrix(model, beta, susceptible, contact_scale))


def effective_reproduction_numbers(
    model: CompiledModel, beta: float, susceptible: np.ndarray
) -> np.ndarray:
    """Rt for each row of ``susceptible`` (shape ``(T, n_age)``)."""
    T = infectious_time(model)
    base = beta * model.C * T[None, :] / model.N[None, :]
    return np.array(
        [spectral_radius(base * s[:, None]) for s in np.asarray(susceptible, dtype=float)]
    )


def growth_rate(model: CompiledModel, beta: float | None = None) -> float:
    """Initial exponential growth rate ``r`` (per day), the dominant eigenvalue
    of the system linearised around the disease-free state."""
    beta = beta_from_R0(model) if beta is None else beta
    A = model.linear_operator()
    idx = _transient_states(model)
    n, m = len(idx), model.n_age
    J = np.zeros((n * m, n * m))
    t = int(np.searchsorted(idx, model.target))
    w = model.infectiousness[idx]
    for a in range(m):
        J[a * n : (a + 1) * n, a * n : (a + 1) * n] = A[a][np.ix_(idx, idx)]
        for b in range(m):
            J[a * n + t, b * n : (b + 1) * n] += beta * model.C[a, b] * model.N[a] / model.N[b] * w
    return float(np.max(np.linalg.eigvals(J).real))


def doubling_time(model: CompiledModel, beta: float | None = None) -> float:
    r = growth_rate(model, beta)
    return float(np.log(2) / r) if r > 0 else float("inf")
