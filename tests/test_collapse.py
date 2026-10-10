import numpy as np
import pytest
from types import SimpleNamespace

from numcosmo_ep2.collapse import (
    collapseAODE,
    solveCollapseA,
    findTurnaroundEvent,
    findCollapseEvent,
    shootCollapseA,
)
from numcosmo_ep2.growth import cosmExpansion, linearGrowth


def test_collapseA_linear_limit():
    N = np.log(1e-2)

    delta = 1e-10
    deltaN = 1e-10

    omega_m0 = 0.3
    omega_lambda0 = 0.7

    _, deltaNN = collapseAODE(
        N, [delta, deltaN],
        omega_m0, omega_lambda0
    )

    a = np.exp(N)

    E2, omega_m, dlnH_dN = cosmExpansion(
        a, omega_m0, omega_lambda0
    )

    deltaNN_linear = (
        -(2.0 + dlnH_dN) * deltaN
        + 1.5 * omega_m * delta
    )

    assert np.isclose(
        deltaNN,
        deltaNN_linear,
        rtol=1e-8,
        atol=1e-18
    )


def test_collapseA_eds_linear_growth():
    ai = 1e-4
    af = 1e-2
    delta_i = 1e-8

    sol = solveCollapseA(
        ai, af, delta_i,
        omega_m0=1.0,
        omega_lambda0=0.0,
        rtol=1e-9,
        atol=1e-13
    )

    aEval = np.logspace(
        np.log10(ai),
        np.log10(af),
        50
    )

    delta = sol.sol(np.log(aEval))[0]

    # EdS linear prediction
    delta_linear = delta_i * aEval / ai

    assert np.allclose(
        delta,
        delta_linear,
        rtol=1e-5,
        atol=1e-12
    )


def test_collapseA_event():
    deltaInf = 1e6

    sol = solveCollapseA(
        ai=1e-4,
        af=1.0,
        delta_i=1e-3,
        omega_m0=1.0,
        omega_lambda0=0.0,
        deltaInf=deltaInf
    )

    collapse = findCollapseEvent(sol)

    assert sol.status == 1
    assert collapse is not None

    a_coll, y_coll = collapse
    delta_coll, deltaN_coll = y_coll

    assert 0.0 < a_coll <= 1.0

    assert np.isclose(
        delta_coll,
        deltaInf,
        rtol=1e-8
    )


def test_collapseA_no_collapse():
    sol = solveCollapseA(
        ai=1e-4,
        af=1.0,
        delta_i=1e-5,
        omega_m0=1.0,
        omega_lambda0=0.0,
        deltaInf=1e6
    )

    collapse = findCollapseEvent(sol)

    assert sol.status == 0
    assert collapse is None


def test_collapse_time_monotonic_with_delta_i():
    sol1 = solveCollapseA(
        ai=1e-4,
        af=1.0,
        delta_i=5e-4,
        omega_m0=1.0,
        omega_lambda0=0.0,
        deltaInf=1e6
    )

    sol2 = solveCollapseA(
        ai=1e-4,
        af=1.0,
        delta_i=1e-3,
        omega_m0=1.0,
        omega_lambda0=0.0,
        deltaInf=1e6
    )

    collapse1 = findCollapseEvent(sol1)
    collapse2 = findCollapseEvent(sol2)

    assert collapse1 is not None
    assert collapse2 is not None

    a1, _ = collapse1
    a2, _ = collapse2

    assert a2 < a1

# EdS automatic test (04_collapse.ipynb)
def test_shoot_collapseA_eds():
    ai = 1e-5
    ac = 1.0

    delta_i, sol = shootCollapseA(
        ai,
        ac,
        1.0,
        0.0,
        (5e-6, 5e-5)
    )

    a_coll = findCollapseEvent(sol)[0]

    assert np.isclose(a_coll, ac, rtol=0.0, atol=1e-7)

    _, D, _ = linearGrowth(
        np.array([ai, ac]),
        1.0,
        0.0,
        ai=ai
    )

    delta_c = delta_i * D[1] / D[0]

    delta_c_exact = (
        3.0 / 20.0 * (12.0 * np.pi)**(2.0 / 3.0)
    )

    assert np.isclose(
        delta_c,
        delta_c_exact,
        rtol=1e-4
    )

def test_turnaround_precedes_collapse():
    sol = solveCollapseA(
        ai=1e-4,
        af=1.0,
        delta_i=1e-3,
        omega_m0=1.0,
        omega_lambda0=0.0,
        deltaInf=1e6
    )

    turnaround = findTurnaroundEvent(sol)
    collapse = findCollapseEvent(sol)

    assert turnaround is not None
    assert collapse is not None

    a_ta, y_ta = turnaround
    a_coll, _ = collapse

    assert a_ta < a_coll

    delta_ta, deltaN_ta = y_ta

    theta_ta = -deltaN_ta / (1.0 + delta_ta)

    assert np.isclose(
        theta_ta,
        -3.0,
        rtol=1e-8,
        atol=1e-10
    )