import numpy as np
from numcosmo_ep2 import growth

def test_linear_growth_eds():
    aEval = np.logspace(-4, 0, 100)

    _, D, f = growth.linearGrowth(
        aEval, 1.0, 0.0,
        ai=1e-4,
        rtol=1e-8,
        atol=1e-12
    )

    assert np.allclose(D, aEval, rtol=1e-6, atol=1e-10)
    assert np.allclose(f, 1.0, rtol=1e-10, atol=1e-12)

def test_linear_growth_quad_eds():
    aEval = np.logspace(-4, 0, 100)

    _, D = growth.linearGrowthQuad(aEval, 1.0, 0.0)

    assert np.allclose(D, aEval, rtol=1e-8, atol=1e-10)

def test_growth_normalization():
    cosmologies = [
        (1.0, 0.0),
        (0.3, 0.7),
        (0.3, 0.0),
    ]

    for omega_m0, omega_lambda0 in cosmologies:
        _, D, _ = growth.linearGrowth(np.array([1.0]), omega_m0, omega_lambda0)
        
        assert np.isclose(D[0], 1.0)

def test_ode_quad_lcdm_agreement():
    aEval = np.logspace(-3, 0, 100)

    _, D_ode, _ = growth.linearGrowth(aEval, 0.3, 0.7, 
                                      ai=1e-4, rtol=1e-8, atol=1e-12)
    _, D_quad = growth.linearGrowthQuad(aEval, 0.3, 0.7)

    assert np.allclose(D_ode, D_quad, rtol=1e-6, atol=1e-10)

def test_ode_quad_ocdm_agreement():
    aEval = np.logspace(-3, 0, 100)

    _, D_ode, _ = growth.linearGrowth(aEval, 0.3, 0.0,
                                      ai=1e-5, rtol=1e-9, atol=1e-13)
    _, D_quad = growth.linearGrowthQuad(aEval, 0.3, 0.0)

    assert np.allclose(D_ode, D_quad, rtol=1e-5, atol=1e-10)

def test_initial_epoch_convergence_ocdm():
    aEval = np.logspace(-2, 0, 100)

    _, D_quad = growth.linearGrowthQuad(aEval, 0.3, 0.0)
    _, D_late, _ = growth.linearGrowth(aEval, 0.3, 0.0,
                                       ai=1e-2, rtol=1e-10, atol=1e-14)
    _, D_early, _ = growth.linearGrowth(aEval, 0.3, 0.0,
                                        ai=1e-4, rtol=1e-10, atol=1e-14)

    err_late = np.max(np.abs(D_late / D_quad - 1.0))
    err_early = np.max(np.abs(D_early / D_quad - 1.0))

    assert err_early < err_late