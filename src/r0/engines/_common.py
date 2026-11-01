from __future__ import annotations

import numpy as np

from ..model.structure import CompiledModel


def steps_per_day(dt: float) -> int:
    steps = round(1.0 / dt)
    if steps < 1 or abs(steps * dt - 1.0) > 1e-9:
        raise ValueError("dt must divide one day exactly (e.g. 0.5, 0.25, 0.1)")
    return steps


def aggregation_matrix(model: CompiledModel) -> np.ndarray:
    """``(n_comp, n_states)`` matrix summing stages into compartments."""
    agg = np.zeros((len(model.compartments), model.n_states))
    for c, (o, k) in enumerate(zip(model.offsets, model.stages)):
        agg[c, o : o + k] = 1.0
    return agg
