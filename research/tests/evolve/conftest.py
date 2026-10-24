import numpy as np
import pytest

from laplace_research.evolve.feed import ASSETS
from laplace_research.evolve.panel import Panel

HOURS = 1_200


@pytest.fixture(scope="session")
def panel() -> Panel:
    """Synthetic prices for every feed asset: GARCH-like volatility clustering, deterministic seed."""
    rng = np.random.default_rng(7)
    n = len(ASSETS)
    var = np.full(n, 1e-4)
    r = np.empty((n, HOURS))
    for t in range(HOURS):
        r[:, t] = rng.standard_t(5, n) * np.sqrt(var * 3 / 5)
        var = 2e-6 + 0.08 * r[:, t] ** 2 + 0.9 * var
    close = 100 * np.exp(np.cumsum(r, axis=1))
    open_ = np.concatenate([close[:, :1], close[:, :-1]], axis=1)
    spread = np.abs(rng.normal(0, 0.002, (n, HOURS)))
    return Panel(
        assets=ASSETS,
        ts=np.arange(1_735_689_600, 1_735_689_600 + HOURS * 3600, 3600, dtype=np.int64),
        open=open_,
        high=np.maximum(open_, close) * (1 + spread),
        low=np.minimum(open_, close) * (1 - spread),
        close=close,
        volume=rng.uniform(10, 20, (n, HOURS)),
    )


@pytest.fixture
def history(panel: Panel, tmp_path):  # type: ignore[no-untyped-def]
    """The synthetic panel saved with its past-only history files; returns the panel path."""
    from laplace_research.evolve.history import write_history

    path = tmp_path / "panel.npz"
    panel.save(path)
    write_history(panel, path)
    return path
