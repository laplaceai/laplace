from pathlib import Path

import pytest

from laplace_research.evolve.db import Database
from laplace_research.evolve.islands import SEED_PROGRAM
from laplace_research.evolve.mutate import MUTATORS, NoProgram, Scored, build_prompt, parse_reply


def test_parse_reply_takes_title_why_and_the_last_python_block() -> None:
    reply = "TITLE: Blend two decays\nWHY: Faster reaction.\n```python\nold\n```\ntext\n```python\nnew = 1\n```"
    code, title, why = parse_reply(reply)
    assert (code, title, why) == ("new = 1\n", "Blend two decays", "Faster reaction.")
    with pytest.raises(NoProgram):
        parse_reply("no code here")


def test_prompt_carries_hypothesis_parent_and_failures() -> None:
    parent = Scored(SEED_PROGRAM, 0.0123, {"h1": 0.01})
    prompt = build_prompt("Volatility clusters.", parent, [], ["Bad idea: ValueError"])
    assert "Volatility clusters." in prompt
    assert "+0.0123 nats" in prompt
    assert SEED_PROGRAM.strip() in prompt
    assert "Bad idea: ValueError" in prompt


def test_mutator_cost_uses_token_prices() -> None:
    m = MUTATORS["glm-5.3"]
    assert m.cost({"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}) == pytest.approx(1.4 + 4.4)


def test_database_ranks_programs_and_hides_infrastructure_failures(tmp_path: Path) -> None:
    db = Database(tmp_path / "p.sqlite")
    island = db.add_island("clustering", "Volatility clusters.")
    seed = db.add_program(island_id=island, parent_id=None, generation=0, mutator="seed", code=SEED_PROGRAM, gain=0.0)
    db.add_program(island_id=island, parent_id=seed, generation=1, mutator="m", code="a", gain=0.02)
    db.add_program(island_id=island, parent_id=seed, generation=1, mutator="m", code="b", error="ValueError: x")
    db.add_program(island_id=island, parent_id=seed, generation=1, mutator="m", code="", error="mutation: timeout")
    assert [p.gain for p in db.best(island, 5)] == [0.02, 0.0]
    assert db.recent_failures(island, 5) == ["untitled: ValueError: x"]
    assert db.has_code(island, "a")
    assert not db.has_code(island, "z")
    assert db.last_generation() == 1
