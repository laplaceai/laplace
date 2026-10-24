import numpy as np
import pytest

from laplace_research.evolve.report import em_weights


def test_em_puts_the_weight_on_the_member_that_explains_the_data() -> None:
    p = np.array([[0.9] * 50, [0.1] * 50])
    w = em_weights(p)
    assert w[0] > 0.99


def test_em_mixes_complementary_members() -> None:
    p = np.array([[0.9, 0.1] * 50, [0.1, 0.9] * 50])
    w = em_weights(p)
    assert w == pytest.approx([0.5, 0.5], abs=1e-6)
