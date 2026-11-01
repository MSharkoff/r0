"""Compile a :class:`ModelConfig` into arrays the engines can run.

The state of the model is a matrix ``X`` of shape ``(n_states, n_age)`` where
each row is one Erlang sub-stage of one compartment. Progression is described
by a routing table: every state ``s`` is left at rate ``rate[s, a]`` and its
leavers go to destination states ``dest[s, j]`` with probability
``prob[s, j, a]``. The same table drives the deterministic engine (as a linear
operator) and the stochastic engine (as binomial draws).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig, per_age


@dataclass(frozen=True)
class CompiledModel:
    config: ModelConfig
    compartments: tuple[str, ...]
    #: first state index and number of stages of each compartment
    offsets: tuple[int, ...]
    stages: tuple[int, ...]
    n_states: int
    n_age: int
    #: (n_states, n_age) exit rate of each state, per day
    rate: np.ndarray
    #: (n_states, n_dest) destination state, -1 for padding
    dest: np.ndarray
    #: (n_states, n_dest, n_age) routing probability
    prob: np.ndarray
    #: (n_states, n_dest) compartment entered by this route, -1 if the route
    #: stays inside the compartment (stage to stage) or is padding
    dest_comp: np.ndarray
    #: (n_states,) infectiousness weight
    infectiousness: np.ndarray
    susceptible: int  # state index
    target: int  # state index (first stage of the infection target)
    target_comp: int
    N: np.ndarray  # (n_age,)
    C: np.ndarray  # (n_age, n_age) contact matrix

    # ------------------------------------------------------------------ helpers
    def index(self, name: str) -> int:
        return self.compartments.index(name)

    def state_slice(self, name: str) -> slice:
        i = self.index(name)
        return slice(self.offsets[i], self.offsets[i] + self.stages[i])

    def linear_operator(self) -> np.ndarray:
        """Return ``A`` of shape ``(n_age, n_states, n_states)`` such that
        progression alone gives ``dX[:, a]/dt = A[a] @ X[:, a]``."""
        A = np.zeros((self.n_age, self.n_states, self.n_states))
        for s in range(self.n_states):
            A[:, s, s] -= self.rate[s]
            for j in range(self.dest.shape[1]):
                d = self.dest[s, j]
                if d >= 0:
                    A[:, d, s] += self.rate[s] * self.prob[s, j]
        return A

    def inflow_operator(self) -> np.ndarray:
        """Return ``B`` of shape ``(n_age, n_comp, n_states)`` such that
        ``B[a] @ X[:, a]`` is the rate at which individuals enter each
        compartment through progression (infection excluded)."""
        B = np.zeros((self.n_age, len(self.compartments), self.n_states))
        for s in range(self.n_states):
            for j in range(self.dest.shape[1]):
                c = self.dest_comp[s, j]
                if c >= 0:
                    B[:, c, s] += self.rate[s] * self.prob[s, j]
        return B

    def initial_state(self, integer: bool = False) -> np.ndarray:
        """Initial counts: seeded compartments go to their first stage, the
        rest of the population is susceptible. A scalar seed is spread over
        age groups in proportion to their size; with ``integer=True`` every
        count is rounded while keeping each total exact."""
        X = np.zeros((self.n_states, self.n_age))
        share = self.N / self.N.sum()
        for name, value in self.config.initial.items():
            counts = (
                np.asarray(value, dtype=float)
                if isinstance(value, list)
                else float(value) * share
            )
            if integer:
                counts = _round_preserving_total(counts)
            X[self.offsets[self.index(name)]] += counts
        N = _round_preserving_total(self.N) if integer else self.N
        X[self.susceptible] = N - X.sum(axis=0)
        if np.any(X[self.susceptible] < 0):
            raise ValueError("initial counts exceed the population of an age group")
        return X


def _round_preserving_total(x: np.ndarray) -> np.ndarray:
    """Largest-remainder rounding: integers whose sum is round(sum(x))."""
    floor = np.floor(x)
    missing = int(round(x.sum() - floor.sum()))
    order = np.argsort(-(x - floor), kind="stable")
    floor[order[:missing]] += 1
    return floor


def compile_model(config: ModelConfig) -> CompiledModel:
    names = tuple(config.compartments)
    n_age = len(config.population.age_groups)

    offsets, stages = [], []
    n_states = 0
    for name in names:
        offsets.append(n_states)
        stages.append(config.compartments[name].stages)
        n_states += config.compartments[name].stages

    n_dest = max([1] + [len(c.transitions) for c in config.compartments.values()])
    rate = np.zeros((n_states, n_age))
    dest = np.full((n_states, n_dest), -1, dtype=np.int32)
    prob = np.zeros((n_states, n_dest, n_age))
    dest_comp = np.full((n_states, n_dest), -1, dtype=np.int32)
    infectiousness = np.zeros(n_states)

    for ci, name in enumerate(names):
        comp = config.compartments[name]
        first = offsets[ci]
        infectiousness[first : first + comp.stages] = comp.infectiousness
        if comp.duration is None:
            continue
        stage_rate = comp.stages / comp.duration
        for k in range(comp.stages):
            s = first + k
            rate[s] = stage_rate
            if k < comp.stages - 1:
                dest[s, 0] = s + 1
                prob[s, 0] = 1.0
                continue
            explicit = np.zeros(n_age)
            for t in comp.transitions:
                if t.probability is not None:
                    explicit += per_age(t.probability, n_age)
            rest = np.clip(1.0 - explicit, 0.0, 1.0)
            for j, t in enumerate(comp.transitions):
                di = names.index(t.to)
                dest[s, j] = offsets[di]
                dest_comp[s, j] = di
                prob[s, j] = rest if t.probability is None else per_age(t.probability, n_age)

    sus = names.index(config.infection.susceptible)
    tgt = names.index(config.infection.target)
    return CompiledModel(
        config=config,
        compartments=names,
        offsets=tuple(offsets),
        stages=tuple(stages),
        n_states=n_states,
        n_age=n_age,
        rate=rate,
        dest=dest,
        prob=prob,
        dest_comp=dest_comp,
        infectiousness=infectiousness,
        susceptible=offsets[sus],
        target=offsets[tgt],
        target_comp=tgt,
        N=np.asarray(config.population.sizes, dtype=float),
        C=config.population.contact_matrix(),
    )
