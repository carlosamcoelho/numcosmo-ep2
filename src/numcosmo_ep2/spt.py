import numpy as np
import gc
from numcosmo_ep2 import lpt

def f2Kernel(k1, k2):
    """
    Evaluate the symmetrized EdS second-order density kernel F2.

    Parameters
    ----------
    k1, k2 : array_like
        Input wavevectors with final axis of length 3.

    Returns
    -------
    F2 : ndarray
        Symmetrized second-order kernel.
    """
    k1 = np.asarray(k1)
    k2 = np.asarray(k2)

    n1 = np.linalg.norm(k1, axis=-1)
    n2 = np.linalg.norm(k2, axis=-1)

    if np.any(n1 == 0.0) or np.any(n2 == 0.0):
        raise ValueError("F2 is undefined for zero-length input modes.")

    mu = np.sum(k1 * k2, axis=-1) / (n1 * n2)

    return (
        5.0 / 7.0
        + 0.5 * mu * (n1 / n2 + n2 / n1)
        + 2.0 / 7.0 * mu**2
    )

def dealiasMask(kx, ky, kz, kNy, fraction=2.0/3.0):
    """
    Construct a Cartesian spectral band-limit mask.

    A Fourier mode is retained only if each Cartesian component satisfies

        |k_i| <= fraction * kNy.

    With the default ``fraction=2/3``, this implements the component-wise
    2/3 truncation used to control aliasing in the pseudo-spectral
    evaluation of quadratic products.

    Parameters
    ----------
    kx, ky, kz : ndarray
        Cartesian Fourier-wavevector components. The arrays may use
        broadcast-compatible shapes, as returned by ``lpt.fourierGrid``.
    kNy : float
        Nyquist wavenumber of the mesh.
    fraction : float, optional
        Fraction of the Nyquist wavenumber retained in each Cartesian
        direction. Default is 2/3.

    Returns
    -------
    mask : ndarray of bool
        Boolean mask with the broadcast shape of ``kx``, ``ky``, and
        ``kz``. ``True`` denotes retained Fourier modes.

    Notes
    -----
    This is a Cartesian cutoff, not a spherical cutoff in |k|. A retained
    mode may therefore have |k| > fraction * kNy if all of its individual
    Cartesian components remain below the cutoff.
    """
    kcut = fraction * kNy

    return (
        (np.abs(kx) <= kcut)
        & (np.abs(ky) <= kcut)
        & (np.abs(kz) <= kcut)
    )

def secondOrderDensity(delta1_k, N, L, dealias=True, dealiasing_frac=2.0/3.0):
    """
    Construct the second-order SPT density using the EdS F2 kernel.

    Uses the equivalent real-space decomposition

        delta2 = (17/21) delta1^2
                 - Psi1 . grad(delta1)
                 + (2/7) sij sij.

    Parameters
    ----------
    delta1_k : ndarray
        Physical Fourier coefficients in NumPy rFFT layout.
    N : int
        Mesh size per dimension.
    L : float
        Periodic box side length.
    dealias : bool, optional
        If True, apply the spectral band-limit before the quadratic
        products and to the resulting second-order Fourier field.
    dealiasing_frac : float, optional
        Cartesian cutoff as a fraction of the Nyquist wavenumber.
        Default is 2/3.

    Returns
    -------
    delta2_k : ndarray
        Physical Fourier coefficients of the second-order density.
    delta2_x : ndarray
        Real-space second-order density.
    """
    dx = L / N

    _, _, kx, ky, kz, k2, _ = lpt.fourierGrid(N, L)

    delta_k = delta1_k.copy()

    kNy = np.pi * N / L

    if dealias:
        mask = dealiasMask(
            kx, ky, kz,
            kNy,
            fraction=dealiasing_frac
        )
        delta_k *= mask

    inv_k2 = np.zeros_like(k2)
    nz = k2 > 0.0
    inv_k2[nz] = 1.0 / k2[nz]

    # NumPy FFT normalization
    delta_fft = delta_k / dx**3

    delta_x = np.fft.irfftn(delta_fft, s=(N, N, N), axes=(0, 1, 2))

    # ---- gradient of delta1
    grad = np.empty((3, N, N, N))

    for i, ki in enumerate((kx, ky, kz)):
        grad[i] = np.fft.irfftn(
            1j * ki * delta_fft,
            s=(N, N, N),
            axes=(0, 1, 2)
        )

    # ---- linear displacement
    psi = np.empty_like(grad)

    for i, ki in enumerate((kx, ky, kz)):
        psi[i] = np.fft.irfftn(
            1j * ki * inv_k2 * delta_fft,
            s=(N, N, N),
            axes=(0, 1, 2)
        )

    # ---- tidal contraction sij sij
    tidal2 = np.zeros_like(delta_x)

    kvals = (kx, ky, kz)

    for i in range(3):
        for j in range(3):
            projector = kvals[i] * kvals[j] * inv_k2
            if i == j:
                projector = projector - 1.0 / 3.0

            sij = np.fft.irfftn(
                projector * delta_fft,
                s=(N, N, N),
                axes=(0, 1, 2)
            )

            tidal2 += sij**2

    local = (17.0 / 21.0) * delta_x**2
    advection = -np.sum(psi * grad, axis=0)
    tidal = (2.0 / 7.0) * tidal2

    # Quadratic field before output de-aliasing
    delta2_raw_x = local + advection + tidal

    delta2_k = dx**3 * np.fft.rfftn(delta2_raw_x, axes=(0, 1, 2))

    # For the 2/3 rule, discard the contaminated output band as well.
    if dealias:
        delta2_k *= mask

    # The density contrast has no zero mode.
    delta2_k[0, 0, 0] = 0.0

    delta2_x = np.fft.irfftn(
        delta2_k / dx**3,
        s=(N, N, N),
        axes=(0, 1, 2)
    )

    return delta2_k, delta2_x

def shellAverageCrossPower(deltaA_k, deltaB_k, kmag,
                           L, N, bins):
    """
    Estimate the isotropic cross-power spectrum by shell averaging.

    The estimator uses

        P_AB(k) = Re[deltaA_k deltaB_k*] / V,

    with V = L^3, and accounts for the multiplicity of modes omitted
    by the NumPy rFFT half-spectrum. Both Fourier fields must use the
    physical normalization <|delta_k|^2> = V P(k).

    Parameters
    ----------
    deltaA_k, deltaB_k : ndarray
        Physical Fourier coefficients of the two fields, in NumPy rFFT
        layout with shape (N, N, N//2 + 1).
    kmag : ndarray
        Magnitude of the Fourier wavevector on the same rFFT grid.
    L : float
        Periodic box side length.
    N : int
        Mesh size per dimension.
    bins : array_like
        Wavenumber bin edges used for shell averaging.

    Returns
    -------
    kbin : ndarray
        Mean wavenumber in each non-empty shell.
    Pcross : ndarray
        Shell-averaged real cross-power spectrum.
    Nmodes : ndarray
        Effective number of full-spectrum Fourier modes in each shell.

    Notes
    -----
    For statistically isotropic real fields, the ensemble cross-spectrum
    is real. The real part is therefore retained explicitly. In a finite
    realization, this estimator can fluctuate around its ensemble
    expectation.
    """
    V = L**3

    Pmode = (
        np.real(deltaA_k * np.conj(deltaB_k))
        / V
    )

    wz = np.full(N // 2 + 1, 2.0)
    wz[0] = 1.0

    if N % 2 == 0:
        wz[-1] = 1.0

    weights = np.broadcast_to(
        wz[None, None, :],
        kmag.shape
    )

    mask = kmag > 0.0

    k = kmag[mask]
    p = Pmode[mask]
    w = weights[mask]

    Nmodes, _ = np.histogram(k, bins=bins, weights=w)
    Psum, _ = np.histogram(k, bins=bins, weights=w * p)
    ksum, _ = np.histogram(k, bins=bins, weights=w * k)

    valid = Nmodes > 0

    return (
        ksum[valid] / Nmodes[valid],
        Psum[valid] / Nmodes[valid],
        Nmodes[valid]
    )

def smoothedHigherOrderStats(delta_k, kmag, R, N, L):
    """
    Measure one-point higher-order statistics after Gaussian smoothing.

    The Fourier-space field is smoothed with

        W_R(k) = exp[-(k R)^2 / 2],

    transformed to real space, and recentered before its moments are
    evaluated.

    Parameters
    ----------
    delta_k : ndarray
        Physical Fourier coefficients of the density contrast in NumPy
        rFFT layout.
    kmag : ndarray
        Magnitude of the Fourier wavevector on the same rFFT grid.
    R : float
        Gaussian smoothing radius, in units inverse to those of ``kmag``.
    N : int
        Mesh size per dimension.
    L : float
        Periodic box side length.

    Returns
    -------
    variance : float
        Variance of the smoothed density field,
        <delta_R^2>.
    m3 : float
        Third central moment of the smoothed field,
        <delta_R^3>.
    gamma1 : float
        Standardized skewness,

            gamma1 = <delta_R^3> / <delta_R^2>^(3/2).

    S3 : float
        Reduced skewness,

            S3 = <delta_R^3> / <delta_R^2>^2.

    Notes
    -----
    The returned ``gamma1`` and ``S3`` assume a non-zero smoothed
    variance. For Gaussian initial conditions, their ensemble expectation
    is zero, although individual finite-volume realizations need not
    return exactly zero.
    """
    dx = L / N

    W = np.exp(-0.5 * (kmag * R)**2)

    delta_R = np.fft.irfftn(
        delta_k * W / dx**3,
        s=(N, N, N),
        axes=(0, 1, 2)
    )

    delta_R -= delta_R.mean()

    variance = np.mean(delta_R**2)
    m3 = np.mean(delta_R**3)

    sigma = np.sqrt(variance)
    gamma1 = m3 / variance**1.5
    S3 = m3 / variance**2

    return variance, m3, gamma1, S3


def runHigherOrderEnsemble(N, L, seeds, D_spt,
                           classPower, R_values, 
                           dealiasing_frac=2.0/3.0):
    """
    Measure smoothed higher-order statistics over paired SPT realizations.

    For each random seed, a Gaussian linear field is generated, rescaled
    to the fixed growth amplitude ``D_spt``, and spectrally band-limited.
    The corresponding second-order SPT field is then constructed from the
    same realization. Statistics are measured for both ``delta1`` and
    ``delta1 + delta2`` at every requested smoothing radius.

    Parameters
    ----------
    N : int
        Mesh size per dimension.
    L : float
        Periodic box side length.
    seeds : sequence of int
        Random seeds defining the Gaussian realizations.
    D_spt : float
        Linear growth amplitude applied identically to every realization.
        It is kept fixed so that all seeds represent the same physical
        epoch.
    classPower : callable
        Function returning the reference linear power spectrum P(k).
    R_values : array_like
        Gaussian smoothing radii.
    dealiasing_frac : float, optional
        Cartesian spectral cutoff as a fraction of the Nyquist
        wavenumber. Default is 2/3.

    Returns
    -------
    result : dict
        Dictionary containing ``R`` and arrays of shape
        ``(n_realizations, n_R)`` for the Gaussian and SPT variance,
        third moment, standardized skewness, and reduced skewness.
        Keys are ``var_g``, ``m3_g``, ``gamma_g``, ``S3_g``,
        ``var_spt``, ``m3_spt``, ``gamma_spt``, and ``S3_spt``.

    Notes
    -----
    Gaussian and SPT measurements are paired realization by realization.
    This makes differences such as

        Delta S3 = S3_spt - S3_g

    substantially less sensitive to finite-volume fluctuations common to
    both fields.
    """
    dx = L / N
    kNy = np.pi * N / L

    _, _, kx, ky, kz, k2, kmag = lpt.fourierGrid(N, L)

    dealias_mask = dealiasMask(
        kx,
        ky,
        kz,
        kNy,
        fraction=dealiasing_frac
    )

    nseed = len(seeds)
    nR = len(R_values)

    result = {
        "R": np.asarray(R_values),
        "var_g": np.empty((nseed, nR)),
        "m3_g": np.empty((nseed, nR)),
        "gamma_g": np.empty((nseed, nR)),
        "S3_g": np.empty((nseed, nR)),
        "var_spt": np.empty((nseed, nR)),
        "m3_spt": np.empty((nseed, nR)),
        "gamma_spt": np.empty((nseed, nR)),
        "S3_spt": np.empty((nseed, nR)),
    }

    for iseed, seed in enumerate(seeds):

        # Gaussian field at reference epoch
        delta_ref_k, delta_ref_x = lpt.gaussianField(
            N,
            L,
            classPower,
            seed=seed
        )

        # Real-space reference field is not needed here
        del delta_ref_x

        # Same physical epoch for every realization
        delta1_k = D_spt * delta_ref_k * dealias_mask

        delta1_k[0, 0, 0] = 0.0

        del delta_ref_k

        # Second-order SPT field.
        # secondOrderDensity should use the same 2/3 de-aliasing
        # convention for its quadratic output.
        delta2_k, delta2_x = secondOrderDensity(
            delta1_k,
            N,
            L,
            dealias=True,
            dealiasing_frac=dealiasing_frac
        )

        del delta2_x

        delta12_k = delta1_k + delta2_k

        for iR, R in enumerate(R_values):

            (
                result["var_g"][iseed, iR],
                result["m3_g"][iseed, iR],
                result["gamma_g"][iseed, iR],
                result["S3_g"][iseed, iR]
            ) = smoothedHigherOrderStats(
                delta1_k,
                kmag,
                R,
                N,
                L
            )

            (
                result["var_spt"][iseed, iR],
                result["m3_spt"][iseed, iR],
                result["gamma_spt"][iseed, iR],
                result["S3_spt"][iseed, iR]
            ) = smoothedHigherOrderStats(
                delta12_k,
                kmag,
                R,
                N,
                L
            )

        del delta1_k, delta2_k, delta12_k

        gc.collect()

        if (iseed + 1) % 5 == 0:
            print(f"{iseed + 1}/{nseed} realizations")

    return result

def oneSptDiagnostic(N, L, seed, D, classPower, R_values, 
                     dealiasing_frac=2.0/3.0, kmax_frac=0.3):
    """
    Run the main SPT diagnostics for one Gaussian realization.

    A Gaussian density field is generated at a fixed growth amplitude,
    the second-order SPT correction is constructed, and both power-spectrum
    and smoothed higher-order statistics are measured.

    Parameters
    ----------
    N : int
        Mesh size per dimension.
    L : float
        Periodic box side length.
    seed : int
        Random seed of the Gaussian realization.
    D : float
        Linear growth amplitude applied to the reference Gaussian field.
    classPower : callable
        Function returning the reference linear power spectrum P(k).
    R_values : array_like
        Gaussian smoothing radii used for the higher-order statistics.
    dealiasing_frac : float, optional
        Cartesian spectral cutoff used in constructing the quadratic SPT
        field. Default is 2/3.
    kmax_frac : float, optional
        Maximum analyzed wavenumber expressed as a fraction of the
        Nyquist wavenumber. Default is 0.3.

    Returns
    -------
    result : dict
        Dictionary containing mesh metadata, shell-averaged spectra
        ``P11``, ``P22``, ``P12``, and ``Ptotal``, and arrays of
        ``S3`` and ``gamma1`` for the Gaussian and SPT fields.

    Notes
    -----
    ``Ptotal`` is the power spectrum of the truncated field
    ``delta1 + delta2``. It must not be identified with the complete
    one-loop SPT prediction, which additionally contains the P13
    contribution.
    """

    dx = L / N
    V = L**3

    _, _, kx, ky, kz, k2, kmag = lpt.fourierGrid(N, L)

    kNy = np.pi * N / L
    kf = 2.0 * np.pi / L

    dealias_mask = dealiasMask(
        kx,
        ky,
        kz,
        kNy,
        fraction=dealiasing_frac
    )

    # Gaussian reference field
    delta_ref_k, _ = lpt.gaussianField(
        N,
        L,
        classPower,
        seed=seed
    )

    # Same physical epoch in every test
    delta1_k = D * delta_ref_k * dealias_mask
    delta1_k[0, 0, 0] = 0.0

    # Second-order field
    delta2_k, _ = secondOrderDensity(
        delta1_k,
        N,
        L,
        dealiasing_frac=dealiasing_frac
    )

    delta12_k = delta1_k + delta2_k

    # ------------------------------------------------------------
    # Power spectra
    # ------------------------------------------------------------

    kmax = kmax_frac * kNy

    bins = np.arange(
        0.5 * kf,
        kmax + 1.5 * kf,
        kf
    )

    kbin, P11, Nmodes = lpt.shellAveragePower(delta1_k, kmag, L, N, bins)
    _, P22, _ = lpt.shellAveragePower(delta2_k, kmag, L, N, bins)
    _, Ptotal, _ = lpt.shellAveragePower(delta12_k, kmag, L, N, bins)
    _, P12, _ = shellAverageCrossPower(delta1_k, delta2_k, kmag, L, N, bins)

    # ------------------------------------------------------------
    # Higher-order statistics
    # ------------------------------------------------------------

    S3_g = []
    S3_spt = []

    gamma_g = []
    gamma_spt = []

    for R in R_values:
        var, m3, gamma1, S3 = smoothedHigherOrderStats(delta1_k, kmag, R, N, L)
        S3_g.append(S3)
        gamma_g.append(gamma1)

        var, m3, gamma1, S3 = smoothedHigherOrderStats(delta12_k, kmag, R, N, L)
        S3_spt.append(S3)
        gamma_spt.append(gamma1)

    return {
        "N": N,
        "L": L,
        "dx": dx,
        "kNy": kNy,
        "kbin": kbin,
        "P11": P11,
        "P22": P22,
        "P12": P12,
        "Ptotal": Ptotal,
        "S3_g": np.asarray(S3_g),
        "S3_spt": np.asarray(S3_spt),
        "gamma_g": np.asarray(gamma_g),
        "gamma_spt": np.asarray(gamma_spt),
    }

def runSptDiagnosticEnsemble(config, seeds, D, classPower,
                             R_values, dealiasing_frac=2.0/3.0):
    """
    Run the single-realization SPT diagnostics over an ensemble.

    Parameters
    ----------
    config : dict
        Simulation configuration containing at least the keys ``"N"``
        and ``"L"``.
    seeds : sequence of int
        Random seeds defining the Gaussian realizations.
    D : float
        Fixed linear growth amplitude used for every realization.
    classPower : callable
        Function returning the reference linear power spectrum P(k).
    R_values : array_like
        Gaussian smoothing radii used for higher-order statistics.
    dealiasing_frac : float, optional
        Cartesian spectral cutoff as a fraction of the Nyquist
        wavenumber. Default is 2/3.

    Returns
    -------
    runs : list of dict
        One result dictionary per seed, with each entry having the format
        returned by ``oneSptDiagnostic``.

    Notes
    -----
    Keeping ``D`` fixed across seeds and configurations is important when
    comparing realization scatter, mesh resolution, or box size at the
    same physical epoch.
    """
    runs = []

    for i, seed in enumerate(seeds):
        runs.append(
            oneSptDiagnostic(
                N=config["N"],
                L=config["L"],
                seed=seed,
                D=D,
                classPower=classPower,
                R_values=R_values,
                dealiasing_frac=dealiasing_frac
            )
        )

        print(
            f"N={config['N']:3d}, "
            f"L={config['L']:5.0f}: "
            f"{i + 1}/{len(seeds)}"
        )
    return runs