"""
Linear growth of matter density perturbations.

Implements the growing-mode solution for pressureless matter in a
matter + curvature + cosmological-constant FLRW background, neglecting
radiation. Growth is evolved using N = ln(a), with D(1) = 1.
"""

import numpy as np
from scipy.integrate import solve_ivp, quad

def cosmExpansion(a, omega_m0, omega_lambda0):
    """
    Compute background expansion quantities for matter + curvature + Lambda.

    Parameters
    ----------
    a : float or array_like
        Scale factor.
    omega_m0 : float
        Present-day matter density parameter.
    omega_lambda0 : float
        Present-day cosmological-constant density parameter.

    Returns
    -------
    E2 : float or ndarray
        Squared dimensionless Hubble function, E(a)^2 = H(a)^2 / H0^2.
    omega_m : float or ndarray
        Matter density evolution with scale factor.
    dlnH_dN : float or ndarray
        Logarithmic derivative d ln H / dN, with N = ln(a).
    """
    omega_k0 = 1.0 - omega_m0 - omega_lambda0

    E2 = omega_m0 * a**(-3) + omega_k0 * a**(-2) + omega_lambda0

    omega_m = omega_m0 * a**(-3) / E2
    omega_k = omega_k0 * a**(-2) / E2
    dlnH_dN = -1.5 * omega_m - omega_k

    return E2, omega_m, dlnH_dN
    
def growthODE(N, y, omega_m0, omega_lambda0):
    """
    Right-hand side of the linear growth equation in N = ln(a).

    Evolves the unnormalized growing solution G and its logarithmic-scale
    derivative GN = dG/dN according to

        G_NN + (2 + dlnH/dN) G_N - (3/2) Omega_m(a) G = 0,

    for a background containing pressureless matter, curvature, and a
    cosmological constant.

    Parameters
    ----------
    N : float
        Logarithmic scale factor, N = ln(a).
    y : array_like
        State vector [G, GN].
    omega_m0 : float
        Present-day matter density parameter.
    omega_lambda0 : float
        Present-day cosmological-constant density parameter.

    Returns
    -------
    list
        Derivatives [GN, GNN] with respect to N.
    """
    
    G, GN = y

    a = np.exp(N)
    E2, omega_m, dlnH_dN = cosmExpansion(a, omega_m0, omega_lambda0)
    
    GNN = -(2.0 + dlnH_dN) * GN + 1.5 * omega_m * G
    return [GN, GNN]

def linearGrowth(aEval, omega_m0, omega_lambda0,
                 ai=1e-4, rtol=1e-8, atol=1e-12):
    """
    Compute the linear growth factor and logarithmic growth rate.

    The growth equation is integrated from ai to a = 1 using N = ln(a).
    Initial conditions G(ai) = GN(ai) = ai select the asymptotic
    matter-era growing mode. The normalized growth factor is

        D(a) = G(a) / G(1),

    and the logarithmic growth rate is obtained directly from the ODE,

        f(a) = dlnD/dlna = GN / G.

    Parameters
    ----------
    aEval : array_like
        Scale factors at which the solution is evaluated. Values must
        satisfy ai <= aEval <= 1.
    omega_m0 : float
        Present-day matter density parameter.
    omega_lambda0 : float
        Present-day cosmological-constant density parameter.
    ai : float, optional
        Initial scale factor. Default is 1e-4.
    rtol : float, optional
        Relative tolerance passed to scipy.integrate.solve_ivp.
    atol : float, optional
        Absolute tolerance passed to scipy.integrate.solve_ivp.

    Returns
    -------
    G1 : float
        Unnormalized growing solution evaluated at a = 1.
    D : ndarray
        Normalized linear growth factor evaluated at aEval.
    f : ndarray
        Logarithmic growth rate evaluated at aEval.

    Raises
    ------
    ValueError
        If any requested scale factor lies outside [ai, 1].
    RuntimeError
        If the ODE integrator fails.
    """

    aEval = np.asarray(aEval, dtype=float)

    if np.any(aEval < ai) or np.any(aEval > 1.0):
        raise ValueError("aEval must satisfy ai <= aEval <= 1.")

    Ni = np.log(ai)
    Nf = 0.0

    sol = solve_ivp(
        growthODE,
        (Ni, Nf),
        [ai, ai],
        args=(omega_m0, omega_lambda0),
        rtol=rtol,
        atol=atol,
        dense_output=True
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    # Evaluate where requested
    NEval = np.log(aEval)
    G, GN = sol.sol(NEval)

    # Evaluate normalization separately at a=1
    G1 = sol.sol(0.0)[0]

    D = G / G1
    f = GN / G

    return G1, D, f

def linearGrowthQuad(aEval, omega_m0, omega_lambda0,
                     epsabs=1e-12, epsrel=1e-8):
    """
    Compute the linear growth factor using the integral growing solution.

    Evaluates

        G(a) = (5 Omega_m0 / 2) E(a)
               integral_0^a da' / [a'^3 E(a')^3],

    and returns D(a) = G(a) / G(1).

    This expression is valid for a background containing pressureless
    matter, curvature, and a smooth cosmological constant.

    Parameters
    ----------
    aEval : array_like
        Scale factors at which the solution is evaluated, with
        0 < aEval <= 1.
    omega_m0 : float
        Present-day matter density parameter.
    omega_lambda0 : float
        Present-day cosmological-constant density parameter.
    epsabs : float, optional
        Absolute tolerance passed to scipy.integrate.quad.
    epsrel : float, optional
        Relative tolerance passed to scipy.integrate.quad.

    Returns
    -------
    G1 : float
        Unnormalized integral growing solution evaluated at a = 1.
    D : ndarray
        Normalized growth factor evaluated at aEval.

    Raises
    ------
    ValueError
        If any requested scale factor lies outside (0, 1].
    """

    aEval = np.asarray(aEval, dtype=float)

    if np.any(aEval <= 0.0) or np.any(aEval > 1.0):
        raise ValueError("aEval must satisfy 0 < aEval <= 1.")

    omega_k0 = 1.0 - omega_m0 - omega_lambda0

    def E(a):
        E2 = (
            omega_m0 * a**(-3)
            + omega_k0 * a**(-2)
            + omega_lambda0
        )
        return np.sqrt(E2)

    def integrand(a):
        denom = (
            omega_m0
            + omega_k0 * a
            + omega_lambda0 * a**3
        )
        return a**1.5 / denom**1.5

    def G(a):
        I, _ = quad(
            integrand,
            0.0,
            a,
            epsabs=epsabs,
            epsrel=epsrel
        )

        return 2.5 * omega_m0 * E(a) * I

    G1 = G(1.0)
    G_eval = np.array([G(a) for a in np.ravel(aEval)]).reshape(aEval.shape)

    return G1, G_eval / G1