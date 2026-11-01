"""Simulation output shared by both engines."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..model.ngm import effective_reproduction_numbers
from ..model.structure import CompiledModel


@dataclass(frozen=True)
class SimulationResult:
    """Daily trajectories.

    ``states`` has shape ``(R, D + 1, n_comp, n_age)``: replicate, day (day 0
    is the initial state), compartment (stages summed), age group.
    ``inflows`` has shape ``(R, D, n_comp, n_age)``: number of people who
    entered each compartment during each day (new infections for the
    infection target, admissions for a hospital compartment, deaths...).
    The deterministic engine returns a single replicate.
    """

    model: CompiledModel
    beta: float
    days: np.ndarray
    states: np.ndarray
    inflows: np.ndarray
    engine: str

    # --------------------------------------------------------------- accessors
    @property
    def compartments(self) -> tuple[str, ...]:
        return self.model.compartments

    @property
    def n_replicates(self) -> int:
        return self.states.shape[0]

    def compartment(self, name: str, by_age: bool = False) -> np.ndarray:
        """Occupancy of a compartment, shape ``(R, D + 1[, n_age])``."""
        x = self.states[:, :, self.model.index(name)]
        return x if by_age else x.sum(axis=-1)

    def inflow(self, name: str, by_age: bool = False) -> np.ndarray:
        """Daily entries into a compartment, shape ``(R, D[, n_age])``."""
        x = self.inflows[:, :, self.model.index(name)]
        return x if by_age else x.sum(axis=-1)

    def incidence(self, by_age: bool = False) -> np.ndarray:
        """Daily new infections."""
        return self.inflow(self.compartments[self.model.target_comp], by_age)

    def attack_rate(self) -> np.ndarray:
        """Share of the population infected over the whole run, per replicate."""
        return self.incidence().sum(axis=1) / self.model.N.sum()

    def rt(self) -> np.ndarray:
        """Effective reproduction number each day, shape ``(R, D + 1)``."""
        S = self.compartment(self.model.config.infection.susceptible, by_age=True)
        return np.stack([effective_reproduction_numbers(self.model, self.beta, s) for s in S])

    def peak(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Day and height of the peak occupancy of a compartment, per replicate."""
        x = self.compartment(name)
        return self.days[np.argmax(x, axis=1)], np.max(x, axis=1)

    # ------------------------------------------------------------------ export
    def to_csv(self, path: str | Path) -> None:
        """Write one row per replicate, day and age group."""
        ages = self.model.config.population.age_groups
        names = self.compartments
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["replicate", "day", "age_group"]
                + list(names)
                + [f"new_{n}" for n in names]
            )
            for r in range(self.n_replicates):
                for d, day in enumerate(self.days):
                    for a, age in enumerate(ages):
                        flows = (
                            self.inflows[r, d - 1, :, a]
                            if d > 0
                            else np.zeros(len(names))
                        )
                        w.writerow(
                            [r, int(day), age]
                            + [f"{v:.6g}" for v in self.states[r, d, :, a]]
                            + [f"{v:.6g}" for v in flows]
                        )
