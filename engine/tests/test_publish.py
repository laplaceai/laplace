import json
from pathlib import Path

from laplace_engine.publish import FRESH, IMMUTABLE, PUBLISHED, publish_pending, publish_release

from .fakes import FakeStore


def issue(root: Path, n: int, *, committed: bool = True, published: bool = False) -> Path:
    d = root / str(n)
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{}")
    (d / "forecasts.json").write_text("{}")
    if committed:
        commitment = {"issue": n, "root": "0x1", "manifest": "0x2", "tx": f"0x{n}", "block": n, "timestamp": n * 3600}
        (d / "commitment.json").write_text(json.dumps(commitment))
    if published:
        (d / PUBLISHED).touch()
    return d


def test_publishes_committed_issues_once_and_points_latest_at_the_newest(tmp_path: Path) -> None:
    store = FakeStore()
    issue(tmp_path, 10, published=True)
    issue(tmp_path, 11)
    issue(tmp_path, 12)
    issue(tmp_path, 13, committed=False)  # sealed but never committed: not part of the record
    assert publish_pending(store, tmp_path) == [11, 12]
    assert set(store.objects) == {
        *(f"v1/issues/{n}/{f}" for n in (11, 12) for f in ("manifest.json", "forecasts.json", "commitment.json")),
        "v1/latest.json",
    }
    assert json.loads(store.objects["v1/latest.json"][0])["issue"] == 12
    assert store.objects["v1/latest.json"][2] == FRESH
    assert store.objects["v1/issues/11/manifest.json"][1:] == ("application/json", IMMUTABLE)
    assert publish_pending(FakeStore(), tmp_path) == []


def test_catching_up_on_an_older_issue_never_moves_latest_back(tmp_path: Path) -> None:
    store = FakeStore()
    issue(tmp_path, 20)
    issue(tmp_path, 21, published=True)
    assert publish_pending(store, tmp_path) == [20]
    assert "v1/latest.json" not in store.objects


def test_publishes_a_release_once(tmp_path: Path) -> None:
    release = tmp_path / "r1"
    (release / "programs").mkdir(parents=True)
    (release / "release.json").write_text("{}")
    (release / "programs" / "naive.py").write_text("def forecast(ctx):\n    return [1] * 10\n")
    store, marker = FakeStore(), tmp_path / "data" / "r1.published"
    publish_release(store, release, "r1", marker)
    assert set(store.objects) == {"v1/releases/r1/release.json", "v1/releases/r1/programs/naive.py"}
    assert store.objects["v1/releases/r1/programs/naive.py"][1] == "text/x-python; charset=utf-8"
    again = FakeStore()
    publish_release(again, release, "r1", marker)
    assert not again.objects
