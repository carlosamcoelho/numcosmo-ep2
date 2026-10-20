import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from numcosmo_ep2.growth import cosmExpansion, linearGrowth

# Helper functions -----------------------------------------------
def findTurnaroundEvent(sol):
    """
    Extract the first turnaround event from an ODE solution.

    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Solution returned by `solve_ivp` with turnaround as event 0.

    Returns
    -------
    tuple or None
        `(a_ta, y_ta)`, where `a_ta` is the turnaround scale factor
        and `y_ta` is the state vector at the event. Returns `None`
        if no turnaround was detected.
    """
    if len(sol.t_events[0]) == 0:
        return None # No turnaround
    return np.exp(sol.t_events[0][0]), sol.y_events[0][0]

def findCollapseEvent(sol):
    """
    Extract the first collapse event from an ODE solution.

    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Solution returned by `solve_ivp` with collapse as event 1.

    Returns
    -------
    tuple or None
        `(a_coll, y_coll)`, where `a_coll` is the collapse scale
        factor and `y_coll` is the state vector at the event.
        Returns `None` if no collapse was detected.
    """
    if len(sol.t_events[1]) == 0:
        return None # No collapse
    return np.exp(sol.t_events[1][0]), sol.y_events[1][0]

def defineEvents(form, collapseThreshold):
    """
    Construct turnaround and collapse event functions.

    Turnaround is defined by theta = -3. For formulations A and B,
    collapse occurs when delta reaches `collapseThreshold`; for
    formulation C it occurs when g = 1/delta decreases to the supplied
    threshold.

    Parameters
    ----------
    form : {'A', 'B', 'C'}
        Spherical-collapse formulation.
    collapseThreshold : float
        Positive numerical threshold used to terminate collapse.

    Returns
    -------
    list
        `[turnaroundEvent, collapseEvent]` for use with `solve_ivp`.
    """
    form = form.upper()
    if form not in ("A", "B", "C"):
        raise ValueError("form must be 'A', 'B', or 'C'.")
    if collapseThreshold <= 0.0:
        raise ValueError("collapseThreshold must be positive.")

    # Turnaround: theta = -3
    # in A theta must be calculated from delta, deltaN
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

    # Collapse criterion (delta reaches a finite threshold)
    def collapseEvent(N, y, *args):
        return y[0] - collapseThreshold
    collapseEvent.terminal = True
    collapseEvent.direction = 1.0 if form in ("A", "B") else -1.0

    return [turnaroundEvent, collapseEvent]

# Linear collapse ------------------------------------------------
def collapseLinearODE(N, y, omega_m0, omega_lambda0):
    """
    Linear spherical-collapse growth equation in N = ln(a).

    The state is y = (delta_L, d delta_L/dN). The background expansion
    is obtained from `cosmExpansion`.

    Returns
    -------
    list
        `[d delta_L/dN, d² delta_L/dN²]`.
    """
    deltaL, deltaLN = y

    a = np.exp(N)
    E2, omega_m, dlnH_dN = cosmExpansion(a, omega_m0, omega_lambda0)

    deltaLNN = -(2.0 + dlnH_dN) * deltaLN + 1.5 * omega_m * deltaL

    return [deltaLN, deltaLNN]

def solveCollapseLinear(ai, af, delta_i,
                        omega_m0, omega_lambda0,
                        rtol=1e-8, atol=1e-12):
    """
    Integrate the linear overdensity associated with a collapse solution.

    Growing-mode initial conditions are imposed using the logarithmic
    growth rate f(ai), so that delta_L,N(ai) = f(ai) delta_i.

    Parameters
    ----------
    ai, af : float
        Initial and final scale factors.
    delta_i : float
        Initial linear overdensity.
    omega_m0, omega_lambda0 : float
        Present-day matter and cosmological-constant density parameters.
    rtol, atol : float
        Relative and absolute ODE tolerances.

    Returns
    -------
    scipy.integrate.OdeResult
        Dense-output solution in N = ln(a).
    """
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
    """
    Nonlinear spherical-collapse system in formulation A.

    The state is y = (delta, delta_N), where delta_N = d delta/dN.
    The second-order density equation is evolved directly.

    Returns
    -------
    list
        `[delta_N, delta_NN]`.
    """
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
    """
    Integrate spherical collapse using formulation A.

    Growing-mode initial conditions use
    delta_N(ai) = f(ai) delta_i. Integration stops when the nonlinear
    overdensity reaches `deltaInf`; turnaround is recorded separately.

    Parameters
    ----------
    ai, af : float
        Initial and final scale factors.
    delta_i : float
        Initial nonlinear overdensity.
    omega_m0, omega_lambda0 : float
        Background cosmological parameters.
    deltaInf : float
        Finite overdensity used as the numerical collapse threshold.
    rtol, atol : float
        ODE tolerances.

    Returns
    -------
    scipy.integrate.OdeResult
        Nonlinear collapse trajectory with event information.
    """
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
    """
    Tune the initial overdensity so formulation A collapses at `ac`.

    A Brent root search is performed over `deltaBracket`. Subcritical
    trials are ranked by their final overdensity relative to `deltaInf`,
    while supercritical trials are ranked by their collapse epoch.

    Returns
    -------
    delta_i : float
        Tuned initial overdensity.
    sol : scipy.integrate.OdeResult
        Final nonlinear trajectory obtained with the tuned value.
    """
    deltaLow, deltaHigh = deltaBracket

    if deltaLow <= 0.0 or deltaHigh <= deltaLow:
        raise ValueError("deltaBracket must satisfy 0 < low < high.")

    def objective(delta_i):
        # Solves the A formulation
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

        # Finds the collapse event
        collapse = findCollapseEvent(sol)

        if collapse is None: # No collapse happened
            deltaFinal = sol.y[0, -1]
            # returns a continuous delta-based value
            # for the function optimizer (so that it
            # increases to zero when deltaFinal=deltaInf)
            return np.log(deltaFinal / deltaInf)

        # If there was an event, find its a of collapse
        aColl, _ = collapse
        # returns an aColl-based value for the function
        # optimizer (such that it decreases from zero when
        # aColl gets bigger than the desired ac).
        return np.log(ac / aColl)

    # Solves the delta_i optimization
    delta_i = brentq(objective, deltaLow, deltaHigh, xtol=xtol)

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
    """
    Nonlinear spherical-collapse system in formulation B.

    The state is y = (delta, theta), where theta is the dimensionless
    velocity divergence. The continuity and velocity-divergence
    equations are evolved as a coupled first-order system.

    Returns
    -------
    list
        `[delta_N, theta_N]`.
    """
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
    """
    Integrate spherical collapse using formulation B.

    Growing-mode initial conditions use

        theta_i = -f(ai) delta_i / (1 + delta_i).

    Collapse is terminated when delta reaches `deltaInf`.

    Returns
    -------
    scipy.integrate.OdeResult
        Nonlinear solution with turnaround and collapse events.
    """
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
    """
    Tune the initial overdensity so formulation B collapses at `ac`.

    Uses Brent's method over `deltaBracket` and returns the tuned
    overdensity together with the corresponding nonlinear trajectory.

    Returns
    -------
    delta_i : float
        Tuned initial overdensity.
    sol : scipy.integrate.OdeResult
        Final formulation-B collapse solution.
    """
    deltaLow, deltaHigh = deltaBracket

    if deltaLow <= 0.0 or deltaHigh <= deltaLow:
        raise ValueError(
            "deltaBracket must satisfy 0 < low < high."
        )

    # 'objective' follows the same logic from the A formulation
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
    """
    Nonlinear spherical-collapse system in formulation C.

    The state is y = (g, theta), with g = 1/delta. Collapse therefore
    corresponds to g -> 0 rather than delta -> infinity.

    Returns
    -------
    list
        `[g_N, theta_N]`.
    """
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
    """
    Integrate spherical collapse using the inverse-overdensity variable.

    The initial state is g_i = 1/delta_i and
    theta_i = -f(ai) delta_i/(1 + delta_i). Numerical collapse is
    defined by g reaching the positive threshold `epsilon`.

    Parameters
    ----------
    epsilon : float
        Finite threshold approximating the singular limit g -> 0.

    Returns
    -------
    scipy.integrate.OdeResult
        Formulation-C collapse trajectory with event information.
    """
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
    """
    Tune the initial overdensity so formulation C collapses at `ac`.

    Brent's method is applied to the initial overdensity. For
    subcritical trajectories the objective uses g(ac)/epsilon; for
    trajectories collapsing early it uses the collapse epoch.

    Returns
    -------
    delta_i : float
        Tuned initial overdensity.
    sol : scipy.integrate.OdeResult
        Final formulation-C trajectory.
    """
    deltaLow, deltaHigh = deltaBracket

    if deltaLow <= 0.0 or deltaHigh <= deltaLow:
        raise ValueError(
            "deltaBracket must satisfy 0 < low < high."
        )

    # 'objective' follows the same logic from the A formulation
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
    Compute the virial-to-turnaround radius ratio.

    Solves the cubic virial relation

        4 lambda_ta y^3 - 2(1 + lambda_ta)y + 1 = 0,

    for the physical root y = Rvir/Rta in [0, 1/2]. The EdS limit
    lambda_ta = 0 gives y = 1/2.

    Parameters
    ----------
    lambda_ta : float
        Ratio of cosmological-constant to matter density inside the
        perturbation at turnaround.

    Returns
    -------
    float
        Virial radius ratio Rvir/Rta.
    """

    if lambda_ta < 0.0:
        raise ValueError("lambda_ta must be non-negative.")

    if np.isclose(lambda_ta, 0.0):
        return 0.5

    def cubic(y):
        return 4.0 * lambda_ta * y**3 - 2.0 * (1.0 + lambda_ta) * y + 1.0

    return brentq(cubic, 0.0, 0.5)

def virialDiagnostic(sol, ac,
                     omega_m0, omega_lambda0,
                     form,
                     verbose=True):
    """
    Compute turnaround and virial quantities from a collapse trajectory.

    The turnaround state is reconstructed according to the chosen
    formulation. The routine then evaluates the virial-radius ratio and
    overdensities relative to the mean matter and critical densities.

    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Nonlinear collapse trajectory containing a turnaround event.
    ac : float
        Target collapse scale factor.
    omega_m0, omega_lambda0 : float
        Background cosmological parameters.
    form : {'A', 'B', 'C'}
        Collapse formulation used to generate `sol`.
    verbose : bool
        Print derived quantities if True.

    Returns
    -------
    dict
        Turnaround scale factor, velocity divergence, turnaround
        overdensity, lambda_ta, Rvir/Rta, Delta_vir,m and Delta_vir,c.
    """
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
    lambda_ta = omega_lambda0 * a_ta**3 / (omega_m0 * zeta_ta)

    y_vir = virialRadiusRatio(lambda_ta)

    delta_vir_m = zeta_ta * (ac / a_ta)**3 * y_vir**(-3)

    E2_c, omega_m_c, _ = cosmExpansion(ac, omega_m0, omega_lambda0)

    delta_vir_c = omega_m_c * delta_vir_m

    if verbose: # Prints the diagnostic values
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
    """
    Run the complete spherical-collapse diagnostic for one cosmology.

    The initial overdensity is tuned so nonlinear collapse occurs at
    `ac`. The corresponding linear solution gives the collapse threshold
    delta_c, while the nonlinear trajectory supplies turnaround and
    virial quantities.

    Parameters
    ----------
    ac : float
        Target collapse scale factor.
    omega_m0, omega_lambda0 : float
        Present-day cosmological density parameters.
    ai : float
        Initial scale factor.
    form : {'A', 'B', 'C'}
        Nonlinear formulation.
    bracket : tuple
        Brent-search bracket for the initial overdensity.
    deltaInf : float
        Finite overdensity defining numerical collapse in A/B; its
        inverse is used as the g threshold in C.
    rtol, atol : float
        Numerical integration tolerances.
    verbose : bool
        Print the diagnostic dictionary if True.

    Returns
    -------
    dict
        Tuned initial overdensity, collapse epoch and residual, linear
        threshold delta_c, and turnaround/virial quantities.
    """
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
    solL = solveCollapseLinear(ai, ac, delta_i, 
                               omega_m0, omega_lambda0)

    delta_c = solL.sol(np.log(ac))[0]

    virial = virialDiagnostic(sol, ac, omega_m0, omega_lambda0,
                              form, verbose=False)

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
    """
    Validate a numerical collapse solution against exact EdS results.

    Runs `collapseDiagnostic` for Omega_m0=1 and Omega_Lambda0=0 and
    compares the numerical solution with the analytic EdS benchmarks

        delta_c = (3/20)(12 pi)^(2/3),
        a_ta = ac 2^(-2/3),
        zeta_ta = (3 pi/4)^2,
        Delta_vir,m = 18 pi^2,
        theta_ta = -3.

    Returns
    -------
    dict
        Relative errors in collapse epoch, delta_c, theta_ta,
        turnaround epoch, turnaround overdensity and virial overdensity.
    """
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
        "delta_c": abs(diag["delta_c"] - delta_c_exact) / delta_c_exact,
        "theta_ta": abs(diag["theta_ta"] + 3.0) / 3.0,
        "a_ta": abs(diag["a_ta"] - a_ta_exact) / a_ta_exact,
        "zeta_ta": abs(diag["zeta_ta"] - zeta_ta_exact) / zeta_ta_exact,
        "Delta_vir_m": abs(diag["Delta_vir_m"] - delta_vir_m_exact) / delta_vir_m_exact
    }

    if verbose:
        print(f"--- ac = {ac}, form = {form} ---")
        for key, value in diag.items():
            print(f"{key} = {value}")
        print("\nRelative errors:")
        for key, value in errors.items():
            print(f"{key}: {value}")
    return errors