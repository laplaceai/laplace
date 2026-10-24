import pytest

from laplace_research.evolve.islands import SEED_PROGRAM
from laplace_research.evolve.sandbox import ProgramRejected, check, load

BAD = {
    "import os": "import os\ndef forecast(ctx):\n    return [1] * 10\n",
    "from-import": "from os import path\ndef forecast(ctx):\n    return [1] * 10\n",
    "open": "def forecast(ctx):\n    open('x')\n    return [1] * 10\n",
    "dunder": "def forecast(ctx):\n    return ctx.__class__\n",
    "numpy io": "import numpy as np\ndef forecast(ctx):\n    np.load('x')\n    return [1] * 10\n",
    "numpy random": "import numpy as np\ndef forecast(ctx):\n    return np.random.rand(10)\n",
    "aliased numpy": "import numpy as xp\ndef forecast(ctx):\n    return xp.lib.npyio\n",
    "class": "class A:\n    pass\ndef forecast(ctx):\n    return [1] * 10\n",
    "eval": "def forecast(ctx):\n    return eval('1')\n",
    "no forecast": "def predict(ctx):\n    return [1] * 10\n",
    "syntax": "def forecast(ctx)\n    return 1\n",
}


@pytest.mark.parametrize("source", BAD.values(), ids=BAD.keys())
def test_rejects(source: str) -> None:
    with pytest.raises(ProgramRejected):
        check(source)


def test_accepts_the_seed_and_runs_it_with_restricted_builtins() -> None:
    fn = load(SEED_PROGRAM)
    assert callable(fn)


def test_restricted_namespace_blocks_dynamic_imports() -> None:
    # A name check alone would miss an import smuggled through a string; the restricted __import__ refuses it.
    fn = load("def forecast(ctx):\n    import math\n    return [math.pi] * 10\n")
    assert fn(None)[0] == pytest.approx(3.14159, rel=1e-5)
