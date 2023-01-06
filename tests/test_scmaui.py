import os
import numpy as np
from scipy.sparse import csr_matrix

from scmaui.cli import main
from scmaui.layers import softmax1p, softmax1p0


def test_softmax1p():
    x = np.random.rand(3, 2)

    p = np.exp(x) / (1.0 + np.exp(x).sum(-1, keepdims=True))
    p0 = 1 / (1.0 + np.exp(x).sum(-1)).reshape(-1, 1)

    np.testing.assert_allclose(softmax1p(x).numpy(), p, atol=1e-7)
    np.testing.assert_allclose(softmax1p0(x).numpy(), p0, atol=1e-7)

    np.testing.assert_allclose(
        softmax1p(x).numpy().sum() + softmax1p0(x).numpy().sum(), 3.0, atol=1e-7
    )
