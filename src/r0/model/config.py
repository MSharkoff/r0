"""Declarative model configuration.

A model is described by its compartments and the transitions between them,
not by code. Each compartment may have a mean duration and a number of Erlang
stages; when an individual leaves the last stage, it moves to one of the
compartment's destinations according to (possibly age-dependent)
probabilities. Infection moves individuals from the susceptible compartment
into the infection target compartment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Probability = float | list[float]

_PROB_TOL = 1e-9


class Transition(BaseModel):
    """One possible exit from a compartment.

    ``probability`` is either a scalar, a list with one value per age group,
    or ``None`` meaning "whatever probability is left" (at most one per
    compartment).
    """

    model_config = ConfigDict(extra="forbid")

    to: str
    probability: Probability | None = None


class Compartment(BaseModel):
    """A health state.

    ``duration`` is the mean time spent in the compartment, in days. A
    compartment without duration is never left by natural progression (for
    example S, or absorbing states such as R with lifelong immunity, or D).
    ``stages`` splits the compartment into Erlang sub-stages so that the
    residence time follows a gamma distribution of shape ``stages`` rather
    than an exponential one.
    """

    model_config = ConfigDict(extra="forbid")

    duration: float | None = Field(default=None, gt=0)
    stages: int = Field(default=1, ge=1)
    infectiousness: float = Field(default=0.0, ge=0)
    transitions: list[Transition] = Field(default_factory=list)
    label: str = ""

    @model_validator(mode="after")
    def _duration_and_transitions_agree(self) -> Self:
        if self.duration is None and self.transitions:
            raise ValueError("a compartment without duration cannot have transitions")
        if self.duration is not None and not self.transitions:
            raise ValueError("a compartment with a duration needs at least one transition")
        if sum(t.probability is None for t in self.transitions) > 1:
            raise ValueError("at most one transition may take the remaining probability")
        return self


class Infection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    susceptible: str = "S"
    target: str = "E"


class Population(BaseModel):
    """Age-structured population with a contact matrix.

    ``contacts[a][b]`` is the mean number of daily contacts that one person of
    age group ``a`` has with people of age group ``b``. When
    ``enforce_reciprocity`` is true the matrix is corrected so that the total
    number of contacts between two groups is the same seen from both sides
    (``C[a][b] * N[a] == C[b][a] * N[b]``).
    """

    model_config = ConfigDict(extra="forbid")

    age_groups: list[str] = Field(min_length=1)
    sizes: list[float]
    contacts: list[list[float]]
    enforce_reciprocity: bool = True

    @model_validator(mode="after")
    def _shapes(self) -> Self:
        n = len(self.age_groups)
        if len(self.sizes) != n:
            raise ValueError("sizes must have one value per age group")
        if any(s <= 0 for s in self.sizes):
            raise ValueError("population sizes must be positive")
        if len(self.contacts) != n or any(len(row) != n for row in self.contacts):
            raise ValueError("contacts must be a square matrix with one row per age group")
        if any(c < 0 for row in self.contacts for c in row):
            raise ValueError("contacts must be non-negative")
        return self

    def contact_matrix(self) -> np.ndarray:
        c = np.asarray(self.contacts, dtype=float)
        if not self.enforce_reciprocity:
            return c
        n = np.asarray(self.sizes, dtype=float)
        total = c * n[:, None]
        sym = 0.5 * (total + total.T)
        return sym / n[:, None]


class ModelConfig(BaseModel):
    """Complete description of an epidemic model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    R0: float = Field(gt=0)
    infection: Infection = Field(default_factory=Infection)
    compartments: dict[str, Compartment]
    population: Population
    initial: dict[str, Probability] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        names = set(self.compartments)
        n_age = len(self.population.age_groups)
        sus, tgt = self.infection.susceptible, self.infection.target
        for role, name in (("susceptible", sus), ("target", tgt)):
            if name not in names:
                raise ValueError(f"infection {role} '{name}' is not a compartment")
        if sus == tgt:
            raise ValueError("infection target must differ from the susceptible compartment")
        if self.compartments[sus].stages != 1:
            raise ValueError("the susceptible compartment must have a single stage")

        for name, comp in self.compartments.items():
            explicit = np.zeros(n_age)
            for t in comp.transitions:
                if t.to not in names:
                    raise ValueError(f"{name}: unknown destination '{t.to}'")
                if t.to == name:
                    raise ValueError(f"{name}: a compartment cannot transition to itself")
                if t.probability is not None:
                    p = _per_age(t.probability, n_age, f"{name}->{t.to}")
                    if np.any(p < 0) or np.any(p > 1):
                        raise ValueError(f"{name}->{t.to}: probabilities must lie in [0, 1]")
                    explicit += p
            has_rest = any(t.probability is None for t in comp.transitions)
            if np.any(explicit > 1 + _PROB_TOL):
                raise ValueError(f"{name}: transition probabilities exceed 1")
            if comp.transitions and not has_rest and np.any(np.abs(explicit - 1) > 1e-6):
                raise ValueError(f"{name}: transition probabilities must sum to 1")

        for name, value in self.initial.items():
            if name not in names:
                raise ValueError(f"initial: unknown compartment '{name}'")
            if name == sus:
                raise ValueError("initial: the susceptible compartment is computed, do not set it")
            if np.any(_per_age(value, n_age, f"initial {name}") < 0):
                raise ValueError(f"initial {name}: counts must be non-negative")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def with_updates(self, **changes) -> ModelConfig:
        """Return a validated copy with top-level fields replaced."""
        data = self.model_dump()
        data.update(changes)
        return ModelConfig.model_validate(data)


def _per_age(value: Probability, n_age: int, what: str) -> np.ndarray:
    if isinstance(value, list):
        if len(value) != n_age:
            raise ValueError(f"{what}: expected {n_age} values (one per age group)")
        return np.asarray(value, dtype=float)
    return np.full(n_age, float(value))


def per_age(value: Probability, n_age: int) -> np.ndarray:
    """Broadcast a scalar or per-age list to an array of length ``n_age``."""
    return _per_age(value, n_age, "value")
