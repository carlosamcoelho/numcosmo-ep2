import numpy as np
from numcosmo_ep2.spt import (
    f2Kernel,
)

def test_f2_special_cases():
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])

    assert np.isclose(f2Kernel(ex, ex), 2.0)
    assert np.isclose(f2Kernel(ex, -ex), 0.0)
    assert np.isclose(f2Kernel(ex, ey), 5.0 / 7.0)