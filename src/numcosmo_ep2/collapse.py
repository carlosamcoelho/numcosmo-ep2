import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from numcosmo_ep2.growth import cosmExpansion, linearGrowth

# Helper functions -----------------------------------------------
def findTurnaroundEvent(sol):
    if len(sol.t_events[0]) == 0:
        return None # No turnaround
    return np.exp(sol.t_events[0][0]), sol.y_events[0][0]

def findCollapseEvent(sol):
    if len(sol.t_events[1]) == 0:
        return None # No collapse
    return np.exp(sol.t_events[1][0]), sol.y_events[1][0]

def defineEvents(form, collapseThreshold):
    form = form.upper()
    if form not in ("A", "B", "C"):
        raise ValueError("form must be 'A', 'B', or 'C'.")
    if collapseThreshold <= 0.0:
        raise ValueError("collapseThreshold must be positive.")

    # Turnaround: theta = -3
    if form == "A":
        def turnaroundEvent(N, y, *args):
            delta, deltaN = y
            theta = -deltaN / (1.0 + delta)
            return theta + 3.0

    else:  # B or C: theta is y[1]
        def turnaroundEvent(N, y, *args):
            return y[1] + 3.0

    turnaroundEvent.terminal = False
    turnaroundEvent.direction = -1.0

    # Collapse criterion
    def collapseEvent(N, y, *args):
        return y[0] - collapseThreshold
    collapseEvent.terminal = True
    collapseEvent.direction = 1.0 if form in ("A", "B") else -1.0

    return [turnaroundEvent, collapseEvent]

# Linear collapse ------------------------------------------------
def collapseLinearODE(N, y, omega_m0, omega_lambda0):
    deltaL, deltaLN = y

    a = np.exp(N)
    E2, omega_m, dlnH_dN = cosmExpansion(
        a, omega_m0, omega_lambda0
    )

    deltaLNN = (
        -(2.0 + dlnH_dN) * deltaLN
        + 1.5 * omega_m * deltaL
    )

    return [deltaLN, deltaLNN]

def solveCollapseLinear(ai, af, delta_i,
                        omega_m0, omega_lambda0,
                        rtol=1e-8, atol=1e-12):

    _, _, f_i = linearGrowth(
        np.array([ai]),
        omega_m0,
        omega_lambda0,
        ai=ai,
        rtol=rtol,
        atol=atol
    )

    y0 = [delta_i, f_i[0] * delta_i]

    sol = solve_ivp(
        collapseLinearODE,
        (np.log(ai), np.log(af)),
        y0,
        args=(omega_m0, omega_lambda0),
        rtol=rtol,
        atol=atol,
        dense_output=True
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol
    
# A collapse formulation -----------------------------------------
def collapseAODE(N, y, omega_m0, omega_lambda0):
    delta, deltaN = y
    a = np.exp(N)
    
    E2, omega_m, dlnH_dN = cosmExpansion(a, omega_m0, omega_lambda0)

    deltaNN = (
        -(2.0 + dlnH_dN) * deltaN
        + (4.0 / 3.0) * deltaN**2 / (1.0 + delta)
        + 1.5 * omega_m * delta * (1.0 + delta)
    )
    return [deltaN, deltaNN]

def solveCollapseA(ai, af, delta_i,
                   omega_m0, omega_lambda0,
                   deltaInf=1e8,
                   rtol=1e-8, atol=1e-12):

    if not (0.0 < ai < af <= 1.0):
        raise ValueError("Require 0 < ai < af <= 1.")
    if delta_i <= 0.0:
        raise ValueError("delta_i must be positive.")

    # Linear growth rate at the initial epoch
    _, _, f_i = linearGrowth(
        np.array([ai]),
        omega_m0,
        omega_lambda0,
        ai=ai,
        rtol=rtol,
        atol=atol
    )

    y0 = [delta_i, f_i[0] * delta_i]

    events = defineEvents("A", deltaInf)
    sol = solve_ivp(
        collapseAODE,
        (np.log(ai), np.log(af)),
        y0,
        args=(omega_m0, omega_lambda0),
        events=events,
        rtol=rtol,
        atol=atol,
        dense_output=True
    )

    if not sol.success:
        raise RuntimeError(sol.message)
    return sol
    
def shootCollapseA(ai, ac, omega_m0, omega_lambda0,
                   deltaBracket,
                   deltaInf=1e8,
                   rtol=1e-8, atol=1e-12,
                   xtol=1e-12):

    deltaLow, deltaHigh = deltaBracket

    if deltaLow <= 0.0 or deltaHigh <= deltaLow:
        raise ValueError("deltaBracket must satisfy 0 < low < high.")

    def objective(delta_i):

        sol = solveCollapseA(
            ai=ai,
            af=ac,
            delta_i=delta_i,
            omega_m0=omega_m0,
            omega_lambda0=omega_lambda0,
            deltaInf=deltaInf,
            rtol=rtol,
            atol=atol
        )

        collapse = findCollapseEvent(sol)

        if collapse is None:
            deltaFinal = sol.y[0, -1]
            return np.log(deltaFinal / deltaInf)

        aColl, _ = collapse
        return np.log(ac / aColl)

    delta_i = brentq(
        objective,
        deltaLow,
        deltaHigh,
        xtol=xtol
    )

    # Final trajectory with the tuned initial overdensity
    sol = solveCollapseA(
        ai=ai,
        af=ac,
        delta_i=delta_i,
        omega_m0=omega_m0,
        omega_lambda0=omega_lambda0,
        deltaInf=deltaInf,
        rtol=rtol,
        atol=atol
    )

    return delta_i, sol

# B collapse formulation -----------------------------------------
def collapseBODE(N, y, omega_m0, omega_lambda0):
    delta, theta = y
    a = np.exp(N)

    E2, omega_m, dlnH_dN = cosmExpansion(a, omega_m0, omega_lambda0)
    deltaN = -(1.0 + delta) * theta

    thetaN = (
        - (2.0 + dlnH_dN) * theta
        - theta**2 / 3.0
        - 1.5 * omega_m * delta
    )

    return [deltaN, thetaN]

def solveCollapseB(ai, af, delta_i,
                   omega_m0, omega_lambda0,
                   deltaInf=1e8,
                   rtol=1e-8, atol=1e-12):

    if not (0.0 < ai < af <= 1.0):
        raise ValueError("Require 0 < ai < af <= 1.")
    if delta_i <= 0.0:
        raise ValueError("delta_i must be positive.")

    _, _, f_i = linearGrowth(
        np.array([ai]),
        omega_m0,
        omega_lambda0,
        ai=ai,
        rtol=rtol,
        atol=atol
    )

    theta_i = -f_i[0] * delta_i / (1.0 + delta_i)
    y0 = [delta_i, theta_i]

    events = defineEvents("B", deltaInf)
    sol = solve_ivp(
        collapseBODE,
        (np.log(ai), np.log(af)),
        y0,
        args=(omega_m0, omega_lambda0),
        events=events,
        rtol=rtol,
        atol=atol,
        dense_output=True
    )

    if not sol.success:
        raise RuntimeError(sol.message)
    return sol


def shootCollapseB(ai, ac, omega_m0, omega_lambda0,
                   deltaBracket,
                   deltaInf=1e8,
                   rtol=1e-8, atol=1e-12,
                   xtol=1e-12):

    deltaLow, deltaHigh = deltaBracket

    if deltaLow <= 0.0 or deltaHigh <= deltaLow:
        raise ValueError(
            "deltaBracket must satisfy 0 < low < high."
        )

    def objective(delta_i):
        sol = solveCollapseB(
            ai=ai,
            af=ac,
            delta_i=delta_i,
            omega_m0=omega_m0,
            omega_lambda0=omega_lambda0,
            deltaInf=deltaInf,
            rtol=rtol,
            atol=atol
        )

        collapse = findCollapseEvent(sol)

        if collapse is None:
            deltaFinal = sol.y[0, -1]

            # Subcritical: delta(ac) < deltaInf
            return np.log(deltaFinal / deltaInf)

        aColl, _ = collapse

        # Supercritical: collapse happened before ac
        return np.log(ac / aColl)

    delta_i = brentq(
        objective,
        deltaLow,
        deltaHigh,
        xtol=xtol
    )

    sol = solveCollapseB(
        ai=ai,
        af=ac,
        delta_i=delta_i,
        omega_m0=omega_m0,
        omega_lambda0=omega_lambda0,
        deltaInf=deltaInf,
        rtol=rtol,
        atol=atol
    )

    return delta_i, sol

# C collapse formulation -----------------------------------------
def collapseCODE(N, y, omega_m0, omega_lambda0):
    g, theta = y
    a = np.exp(N)

    E2, omega_m, dlnH_dN = cosmExpansion(a, omega_m0, omega_lambda0)
    gN = g * (1.0 + g) * theta

    thetaN = (
        - (2.0 + dlnH_dN) * theta
        - theta**2 / 3.0
        - 1.5 * omega_m / g
    )

    return [gN, thetaN]

def solveCollapseC(ai, af, delta_i,
                   omega_m0, omega_lambda0,
                   epsilon=1e-8,
                   rtol=1e-8, atol=1e-12):

    if not (0.0 < ai < af <= 1.0):
        raise ValueError("Require 0 < ai < af <= 1.")
    if delta_i <= 0.0:
        raise ValueError("delta_i must be positive.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")

    # Linear growth rate at the initial epoch
    _, _, f_i = linearGrowth(
        np.array([ai]),
        omega_m0,
        omega_lambda0,
        ai=ai,
        rtol=rtol,
        atol=atol
    )

    theta_i = -f_i[0] * delta_i / (1.0 + delta_i)
    g_i = 1.0 / delta_i
    y0 = [g_i, theta_i]

    events = defineEvents("C", epsilon)
    sol = solve_ivp(
        collapseCODE,
        (np.log(ai), np.log(af)),
        y0,
        args=(omega_m0, omega_lambda0),
        events=events,
        rtol=rtol,
        atol=atol,
        dense_output=True
    )

    if not sol.success:
        raise RuntimeError(sol.message)
    return sol

def shootCollapseC(ai, ac, omega_m0, omega_lambda0,
                   deltaBracket,
                   epsilon=1e-8,
                   rtol=1e-8, atol=1e-12,
                   xtol=1e-12):

    deltaLow, deltaHigh = deltaBracket

    if deltaLow <= 0.0 or deltaHigh <= deltaLow:
        raise ValueError(
            "deltaBracket must satisfy 0 < low < high."
        )

    def objective(delta_i):
        sol = solveCollapseC(
            ai=ai,
            af=ac,
            delta_i=delta_i,
            omega_m0=omega_m0,
            omega_lambda0=omega_lambda0,
            epsilon=epsilon,
            rtol=rtol,
            atol=atol
        )

        collapse = findCollapseEvent(sol)

        if collapse is None:
            gFinal = sol.y[0, -1]

            # Subcritical: g(ac) > epsilon
            return np.log(epsilon / gFinal)

        aColl, _ = collapse

        # Supercritical: collapse happened before ac
        return np.log(ac / aColl)

    delta_i = brentq(
        objective,
        deltaLow,
        deltaHigh,
        xtol=xtol
    )

    sol = solveCollapseC(
        ai=ai,
        af=ac,
        delta_i=delta_i,
        omega_m0=omega_m0,
        omega_lambda0=omega_lambda0,
        epsilon=epsilon,
        rtol=rtol,
        atol=atol
    )

    return delta_i, sol

# Analysis ---------------------------------------------------------------

def virialRadiusRatio(lambda_ta):
    """
    Compute y = Rvir/Rta for the spherical-collapse virial prescription.
    """

    if lambda_ta < 0.0:
        raise ValueError("lambda_ta must be non-negative.")

    if np.isclose(lambda_ta, 0.0):
        return 0.5

    def cubic(y):
        return (
            4.0 * lambda_ta * y**3
            - 2.0 * (1.0 + lambda_ta) * y
            + 1.0
        )

    return brentq(cubic, 0.0, 0.5)

def virialDiagnostic(sol, ac,
                     omega_m0, omega_lambda0,
                     form,
                     verbose=True):

    form = form.upper()

    if form not in ("A", "B", "C"):
        raise ValueError("form must be 'A', 'B', or 'C'.")

    turnaround = findTurnaroundEvent(sol)

    if turnaround is None:
        raise RuntimeError("No turnaround event found.")

    a_ta, y_ta = turnaround

    # Reconstruct delta and theta at turnaround
    if form == "A":
        delta_ta, deltaN_ta = y_ta
        theta_ta = -deltaN_ta / (1.0 + delta_ta)

    elif form == "B":
        delta_ta, theta_ta = y_ta

    else:  # C
        g_ta, theta_ta = y_ta
        delta_ta = 1.0 / g_ta

    zeta_ta = 1.0 + delta_ta

    # rho_Lambda / rho_sphere at turnaround
    lambda_ta = (
        omega_lambda0
        * a_ta**3
        / (omega_m0 * zeta_ta)
    )

    y_vir = virialRadiusRatio(lambda_ta)

    delta_vir_m = (
        zeta_ta
        * (ac / a_ta)**3
        * y_vir**(-3)
    )

    E2_c, omega_m_c, _ = cosmExpansion(
        ac,
        omega_m0,
        omega_lambda0
    )

    delta_vir_c = omega_m_c * delta_vir_m

    if verbose:
        print("\na_ta =", a_ta)
        print("theta_ta =", theta_ta)
        print("zeta_ta =", zeta_ta)

        print("\nlambda_ta =", lambda_ta)
        print("Rvir/Rta =", y_vir)

        print("\nDelta_vir,m =", delta_vir_m)
        print("Omega_m(ac) =", omega_m_c)
        print("Delta_vir,c =", delta_vir_c)

    return {
        "a_ta": a_ta,
        "theta_ta": theta_ta,
        "zeta_ta": zeta_ta,
        "lambda_ta": lambda_ta,
        "y_vir": y_vir,
        "Delta_vir_m": delta_vir_m,
        "Delta_vir_c": delta_vir_c,
    }

def collapseDiagnostic(ac, omega_m0, omega_lambda0,
                       ai=1e-5, form="A",
                       bracket=(5e-6, 1e-4),
                       deltaInf=1e8,
                       rtol=1e-8, atol=1e-12,
                       verbose=True):

    form = form.upper()

    if form not in ("A", "B", "C"):
        raise ValueError("form must be 'A', 'B', or 'C'.")

    if form == "A":
        delta_i, sol = shootCollapseA(
            ai, ac,
            omega_m0, omega_lambda0,
            bracket,
            deltaInf=deltaInf,
            rtol=rtol, atol=atol
        )

    elif form == "B":
        delta_i, sol = shootCollapseB(
            ai, ac,
            omega_m0, omega_lambda0,
            bracket,
            deltaInf=deltaInf,
            rtol=rtol, atol=atol
        )

    else:
        delta_i, sol = shootCollapseC(
            ai, ac,
            omega_m0, omega_lambda0,
            bracket,
            epsilon=1.0 / deltaInf,
            rtol=rtol, atol=atol
        )

    collapseEvent = findCollapseEvent(sol)

    if collapseEvent is None:
        raise RuntimeError("No collapse event found.")

    a_coll, _ = collapseEvent

    # Linear counterpart evaluated at the target collapse epoch
    solL = solveCollapseLinear(
        ai, ac, delta_i,
        omega_m0, omega_lambda0
    )

    delta_c = solL.sol(np.log(ac))[0]

    virial = virialDiagnostic(
        sol,
        ac,
        omega_m0,
        omega_lambda0,
        form,
        verbose=False
    )

    result = {
        "form": form,
        "delta_i": delta_i,
        "a_coll": a_coll,
        "collapse_residual": abs(a_coll - ac) / ac,
        "delta_c": delta_c,
        **virial
    }

    if verbose:
        for key, value in result.items():
            print(f"{key} = {value}")

    return result
    

def EdS_collapse_diagnostic(ac, verbose=True,
                            ai=1e-5, form="A",
                            bracket=(5e-6, 1e-4),
                            deltaInf=1e8,
                            rtol=1e-8, atol=1e-12,):

    diag = collapseDiagnostic(
        ac=ac,
        omega_m0=1.0,
        omega_lambda0=0.0,
        ai=ai,
        form=form,
        bracket=bracket,
        deltaInf=deltaInf,
        rtol=rtol, atol=atol,
        verbose=False
    )

    delta_c_exact = 3.0/20.0 * (12.0*np.pi)**(2.0/3.0)
    a_ta_exact = ac * 2.0**(-2.0/3.0)
    zeta_ta_exact = (3.0*np.pi/4.0)**2
    delta_vir_m_exact = 18.0 * np.pi**2

    errors = {
        "a_coll": diag["collapse_residual"],

        "delta_c": abs(
            diag["delta_c"] - delta_c_exact
        ) / delta_c_exact,

        "theta_ta": abs(
            diag["theta_ta"] + 3.0
        ) / 3.0,

        "a_ta": abs(
            diag["a_ta"] - a_ta_exact
        ) / a_ta_exact,

        "zeta_ta": abs(
            diag["zeta_ta"] - zeta_ta_exact
        ) / zeta_ta_exact,

        "Delta_vir_m": abs(
            diag["Delta_vir_m"] - delta_vir_m_exact
        ) / delta_vir_m_exact
    }

    if verbose:
        print(f"--- ac = {ac}, form = {form} ---")
        for key, value in diag.items():
            print(f"{key} = {value}")

        print("\nRelative errors:")
        for key, value in errors.items():
            print(f"{key}: {value}")

    return errors