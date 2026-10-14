import numpy as np

from numcosmo_ep2 import lpt, spt


def _physical_rfft(field, L):
    N = field.shape[0]
    dx = L / N
    return dx**3 * np.fft.rfftn(field, axes=(0, 1, 2))


def _direct_delta2_from_f2(delta1_x, L):
    """
    Small-grid reference implementation of the discrete F2 convolution.
    """
    N = delta1_x.shape[0]
    dx = L / N
    V = L**3

    delta1_full = dx**3 * np.fft.fftn(delta1_x, axes=(0, 1, 2))

    k1d = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

    delta2_full = np.zeros_like(delta1_full, dtype=complex)

    for kout in np.ndindex((N, N, N)):

        total = 0.0j

        for qidx in np.ndindex((N, N, N)):

            pidx = tuple((kout[d] - qidx[d]) % N for d in range(3))

            q = np.array([k1d[qidx[0]], k1d[qidx[1]], k1d[qidx[2]]])
            p = np.array([k1d[pidx[0]], k1d[pidx[1]], k1d[pidx[2]]])

            if np.linalg.norm(q) == 0.0 or np.linalg.norm(p) == 0.0:
                continue
            total += spt.f2Kernel(q, p) * delta1_full[qidx] * delta1_full[pidx]

        delta2_full[kout] = total / V

    delta2_k = delta2_full[:, :, :N // 2 + 1].copy()
    delta2_k[0, 0, 0] = 0.0

    return delta2_k


def test_f2_special_configurations():
    kx = np.array([1.0, 0.0, 0.0])
    ky = np.array([0.0, 1.0, 0.0])

    # Equal and parallel
    np.testing.assert_allclose(spt.f2Kernel(kx, kx), 2.0)

    # Equal and perpendicular
    np.testing.assert_allclose(spt.f2Kernel(kx, ky), 5.0 / 7.0)

    # Equal and anti-parallel
    np.testing.assert_allclose(spt.f2Kernel(kx, -kx), 0.0,atol=1e-15)


def test_second_order_matches_direct_f2_convolution():
    # Odd N avoids Nyquist derivative ambiguities.
    # The input contains only |n_i| <= 1 modes, so its
    # quadratic products remain inside the represented band.
    N = 7
    L = 2.0 * np.pi
    dx = L / N

    x = np.arange(N) * dx

    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    delta1_x = 0.30 * np.cos(X) + 0.20 * np.cos(Y) + 0.15 * np.sin(X + Z)

    delta1_k = _physical_rfft(delta1_x, L)
    delta1_k[0, 0, 0] = 0.0

    delta2_k, _ = spt.secondOrderDensity(delta1_k, N, L, dealias=False)
    delta2_direct_k = _direct_delta2_from_f2(delta1_x, L)

    np.testing.assert_allclose(delta2_k, delta2_direct_k, rtol=1e-12, atol=1e-12)


def test_second_order_is_quadratic_and_zero_mean():
    N = 9
    L = 90.0

    rng = np.random.default_rng(1234)

    delta1_x = rng.normal(size=(N, N, N))
    delta1_x -= delta1_x.mean()
    delta1_k = _physical_rfft(delta1_x, L)

    delta1_k[0, 0, 0] = 0.0
    delta2_k, delta2_x = spt.secondOrderDensity(delta1_k, N, L)

    factor = 1.7

    scaled_k, _ = spt.secondOrderDensity(factor * delta1_k, N, L)

    np.testing.assert_allclose(scaled_k, factor**2 * delta2_k, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(delta2_k[0, 0, 0], 0.0, atol=1e-14)
    np.testing.assert_allclose(delta2_x.mean(), 0.0, atol=1e-14)


def test_cross_power_reduces_to_auto_power():
    N = 8
    L = 80.0

    rng = np.random.default_rng(4321)

    field = rng.normal(size=(N, N, N))
    field -= field.mean()

    delta_k = _physical_rfft(field, L)

    _, _, _, _, _, _, kmag = lpt.fourierGrid(N, L)

    kf = 2.0 * np.pi / L
    kNy = np.pi * N / L

    bins = np.arange(0.5 * kf, kNy + 1.5 * kf, kf)

    k_auto, P_auto, Nm_auto = lpt.shellAveragePower(delta_k, kmag, L, N, bins)
    k_cross, P_cross, Nm_cross = spt.shellAverageCrossPower(delta_k, delta_k, kmag, L, N, bins)

    np.testing.assert_allclose(k_cross, k_auto)
    np.testing.assert_allclose(P_cross, P_auto)
    np.testing.assert_allclose(Nm_cross, Nm_auto)


def test_smoothed_stats_of_single_cosine_have_zero_skewness():
    N = 16
    L = 100.0
    dx = L / N

    x = np.arange(N) * dx

    field = np.cos(2.0 * np.pi * x / L)[:, None, None]
    field = np.broadcast_to(field, (N, N, N)).copy()

    delta_k = _physical_rfft(field,L)

    _, _, _, _, _, _, kmag = lpt.fourierGrid(N, L)

    variance, m3, gamma1, S3 = spt.smoothedHigherOrderStats(delta_k, kmag, R=5.0, N=N, L=L)

    assert variance > 0.0

    np.testing.assert_allclose(m3, 0.0, atol=1e-14)
    np.testing.assert_allclose(gamma1, 0.0, atol=1e-14)
    np.testing.assert_allclose(S3, 0.0, atol=1e-13)