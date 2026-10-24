"""Test vectors that tie the Python record format to `LaplaceLedger`: written to the contracts' test directory and
checked there by Foundry, and checked here against the code that generates them."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from eth_hash.auto import keccak

from laplace_engine.quantize import to_basis_points
from laplace_engine.record import Leaf, Tree
from laplace_research.evolve.feed import ASSETS, HORIZONS, edges, finalize
from laplace_research.evolve.islands import SEED_PROGRAM
from laplace_research.evolve.lib import laplace_probs, student_t_probs

ISSUE = 497_983  # 2026-10-23 07:00 UTC, the issue the contract tests use
PATH = Path(__file__).resolve().parents[3] / "contracts" / "test" / "vectors" / "record.json"


def leaves() -> list[Leaf]:
    """Seven forecasts (an odd count, so one node is carried up a level) from two forecasters."""
    seed = bytes(keccak(SEED_PROGRAM.encode()))
    other = bytes(keccak(b"a second forecaster"))
    questions = [(asset, horizon) for asset in range(3) for horizon in HORIZONS][:7]
    out = []
    for n, (asset, horizon) in enumerate(questions):
        sd = 0.004 * (1 + asset) * horizon**0.5
        raw = student_t_probs(sd, 3.0 + n, edges(horizon)) if n % 2 else laplace_probs(sd, edges(horizon))
        out.append(Leaf(ISSUE, seed if n % 3 else other, asset, horizon, to_basis_points(finalize(raw))))
    return out


def build() -> dict[str, Any]:
    items = leaves()
    tree = Tree.build(leaf.hash() for leaf in items)
    return {
        "issue": ISSUE,
        "root": "0x" + tree.root.hex(),
        "count": len(items),
        "assets": list(ASSETS),
        "leaves": [
            {
                "forecaster": "0x" + leaf.forecaster.hex(),
                "asset": leaf.asset,
                "horizon": leaf.horizon,
                "bps": list(leaf.bps),
                "hash": "0x" + leaf.hash().hex(),
                "proof": ["0x" + p.hex() for p in tree.proof(leaf.hash())],
            }
            for leaf in items
        ],
    }


def render() -> str:
    return json.dumps(build(), indent=1) + "\n"


def main() -> int:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(render())
    print(f"wrote {PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
