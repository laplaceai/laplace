"""Learning layers on top of raw agent forecasts, evaluated without look-ahead.

- ``shrink``: mix an agent with the reference forecast, ``w * agent + (1 - w) * reference``, with ``w`` fitted on an
  earlier window. It measures how much of an agent is signal once its miscalibration is removed.
- ``hedge``: an online multiplicative-weights pool (the Hedge algorithm). Before each card, weights depend only on
  outcomes already settled by that card's time, so a 24-hour question can't update the weights for 24 hours.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

Dist = Mapping[str, float]
Loss = Callable[[Dist, str], float]
GRID = tuple(i / 20 for i in range(21))


def mix(a: Dist, b: Dist, w: float) -> dict[str, float]:
    return {k: w * a[k] + (1 - w) * b[k] for k in a}


def fit_shrink(agent: Sequence[Dist], reference: Sequence[Dist], outcomes: Sequence[str], loss: Loss) -> float:
    """The weight on ``agent`` in [0, 1] (grid of 0.05) that minimizes mean loss over the given cards."""

    def mean_loss(w: float) -> float:
        return sum(loss(mix(a, r, w), o) for a, r, o in zip(agent, reference, outcomes, strict=True))

    return min(GRID, key=mean_loss)


def hedge(
    members: Mapping[str, Sequence[Dist]],
    outcomes: Sequence[str],
    times: Sequence[int],
    settle_after: int,
    loss: Loss,
    eta: float = 1.0,
) -> list[dict[str, float]]:
    """Pooled forecast per card. Card ``j`` uses the losses of every card ``i`` with ``times[i] + settle_after <=
    times[j]``. Cards must be in time order."""
    names = list(members)
    cumulative = dict.fromkeys(names, 0.0)
    settled = 0
    pooled: list[dict[str, float]] = []
    for j, now in enumerate(times):
        while settled < j and times[settled] + settle_after <= now:
            for n in names:
                cumulative[n] += loss(members[n][settled], outcomes[settled])
            settled += 1
        best = min(cumulative.values())
        weights = {n: math.exp(-eta * (cumulative[n] - best)) for n in names}
        total = sum(weights.values())
        keys = members[names[0]][j].keys()
        pooled.append({k: sum(weights[n] * members[n][j][k] for n in names) / total for k in keys})
    return pooled
