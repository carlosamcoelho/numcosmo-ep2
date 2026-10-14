import numpy as np
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
    Construct a Cartesian spectral de-aliasing mask.
    """
    kcut = fraction * kNy

    return (
        (np.abs(kx) <= kcut)
        & (np.abs(ky) <= kcut)
        & (np.abs(kz) <= kcut)
    )

def secondOrderDensity(delta1_k, N, L, dealias=True):
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
    dealias : bool
        Apply a 2/3 spectral input filter before quadratic products.

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
            fraction=2.0/3.0
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
