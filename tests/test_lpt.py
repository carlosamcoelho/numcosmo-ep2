import numpy as np
import pytest

from numcosmo_ep2.lpt import (
    fourierGrid,
    gaussianField,
    shellAveragePower,
    removeNyquistModes,
    zaDisplacementFourier,
    zaParticles,
    displacementToReal,
    uniformLattice,
    toyPower,
    cicWindow,
    cicDeposit,
)

def test_fourier_grid():
    N = 16
    L = 100.0

    kxy, kz, kx, ky, kz3, k2, kmag = fourierGrid(N, L)

    assert kxy.shape == (N,)
    assert kz.shape == (N // 2 + 1,)
    assert k2.shape == (N, N, N // 2 + 1)
    assert kmag.shape == k2.shape

    kf = 2.0 * np.pi / L
    kNy = np.pi * N / L

    assert np.isclose(kxy[1], kf)
    assert np.isclose(kz[1], kf)
    assert np.isclose(kz[-1], kNy)

    assert k2[0, 0, 0] == 0.0
    assert np.count_nonzero(k2 == 0.0) == 1

def test_physical_fft_roundtrip():
    N = 16
    L = 100.0
    dx = L / N

    delta_k, delta_x = gaussianField(N, L, toyPower, seed=1234)
    delta_k_recovered = dx**3 * np.fft.rfftn(delta_x)

    assert np.allclose(delta_k_recovered, delta_k, rtol=1e-12, atol=1e-12)

def test_gaussian_zero_mode_and_mean():
    N = 16
    L = 100.0

    delta_k, delta_x = gaussianField(N, L, toyPower, seed=1234)

    assert delta_k[0, 0, 0] == 0.0
    assert abs(delta_x.mean()) < 1e-12

def test_gaussian_seed_reproducibility():
    args = (16, 100.0, toyPower)

    dk1, dx1 = gaussianField(*args, seed=123)
    dk2, dx2 = gaussianField(*args, seed=123)
    dk3, dx3 = gaussianField(*args, seed=456)

    assert np.array_equal(dk1, dk2)
    assert np.array_equal(dx1, dx2)
    assert not np.array_equal(dk1, dk3)

def test_rfft_mode_weights():
    N = 16

    wz = np.full(N // 2 + 1, 2.0)
    wz[0] = 1.0
    wz[-1] = 1.0

    total_modes = N**2 * wz.sum()

    assert total_modes == N**3

def test_shell_average_constant_power():
    N = 16
    L = 100.0
    V = L**3

    _, _, _, _, _, _, kmag = fourierGrid(N, L)

    P0 = 7.5

    delta_k = np.sqrt(V * P0) * np.ones(kmag.shape, dtype=complex)
    delta_k[0, 0, 0] = 0.0

    kf = 2.0 * np.pi / L
    kNy = np.pi * N / L

    bins = np.arange(0.5 * kf, kNy + 1.5 * kf, kf)
    _, Pbin, Nmodes = shellAveragePower(delta_k, kmag, L, N, bins)

    assert np.all(Nmodes > 0)
    assert np.allclose(Pbin, P0)

def test_remove_nyquist_modes():
    N = 16

    shape = (N, N, N // 2 + 1)
    delta_k = np.ones(shape, dtype=complex)

    filtered = removeNyquistModes(delta_k)

    assert np.all(filtered[N//2, :, :] == 0.0)
    assert np.all(filtered[:, N//2, :] == 0.0)
    assert np.all(filtered[:, :, -1] == 0.0)

    assert filtered[0, 0, 0] == 0.0

def test_za_divergence():
    N = 16
    L = 100.0
    D1 = 0.25

    delta_k, _ = gaussianField(N, L, toyPower, seed=123)

    _, _, kx, ky, kz, k2, _ = fourierGrid(N, L)

    psi_k, delta_used = zaDisplacementFourier(delta_k, kx, ky, kz, k2, D1)

    minus_div_k = -1j * (kx * psi_k[0] + ky * psi_k[1] + kz * psi_k[2])
    target = D1 * delta_used
    residual = np.linalg.norm(minus_div_k - target) / np.linalg.norm(target)

    assert residual < 1e-12

def test_za_curl():
    N = 16
    L = 100.0
    D1 = 0.25

    delta_k, _ = gaussianField(N, L, toyPower, seed=123)
    _, _, kx, ky, kz, k2, _ = fourierGrid(N, L)

    psi_k, delta_used = zaDisplacementFourier(delta_k, kx, ky, kz, k2, D1)

    curl_x = 1j * (ky * psi_k[2] - kz * psi_k[1])
    curl_y = 1j * (kz * psi_k[0] - kx * psi_k[2])
    curl_z = 1j * (kx * psi_k[1] - ky * psi_k[0])

    curl_norm = np.sqrt(np.linalg.norm(curl_x)**2 + np.linalg.norm(curl_y)**2 + np.linalg.norm(curl_z)**2)
    reference = np.linalg.norm(D1 * delta_used)

    assert curl_norm / reference < 1e-12

def test_za_mean_displacement():
    N = 16
    L = 100.0
    D1 = 0.25

    delta_k, _ = gaussianField(N, L, toyPower, seed=123)
    _, _, kx, ky, kz, k2, _ = fourierGrid(N, L)

    psi_k, _ = zaDisplacementFourier(delta_k, kx, ky, kz, k2, D1)
    psi_x = displacementToReal(psi_k, N, L)
    mean = psi_x.mean(axis=(1, 2, 3))

    assert np.all(np.abs(mean) < 1e-12)

def test_za_single_cosine_mode():
    N = 32
    L = 100.0
    D1 = 0.3
    eps = 1e-2

    dx = L / N

    x = np.arange(N) * dx

    # Choose a mode safely below Nyquist
    nmode = 2
    k = 2.0 * np.pi * nmode / L

    delta_1d = eps * np.cos(k * x)
    delta_x = np.broadcast_to(delta_1d[:, None, None], (N, N, N)).copy()

    # physical Fourier coefficients
    delta_k = dx**3 * np.fft.rfftn(delta_x)

    _, _, kx, ky, kz, k2, _ = fourierGrid(N, L)

    psi_k, _ = zaDisplacementFourier(delta_k, kx, ky, kz, k2, D1)
    psi_x = displacementToReal(psi_k, N, L)

    expected_x = (-D1 * eps / k * np.sin(k * x))
    expected_x = np.broadcast_to(expected_x[:, None, None], (N, N, N))

    assert np.allclose(psi_x[0], expected_x, rtol=1e-11, atol=1e-12)
    assert np.max(np.abs(psi_x[1])) < 1e-12
    assert np.max(np.abs(psi_x[2])) < 1e-12

def test_uniform_lattice_bounds_and_shape():
    N = 8
    L = 100.0

    q = uniformLattice(N, L)

    assert q.shape == (N, N, N, 3)
    assert np.all(q >= 0.0)
    assert np.all(q < L)

def test_cic_uniform_lattice():
    N = 8
    L = 100.0

    q = uniformLattice(N, L)
    delta = cicDeposit(q.reshape(-1, 3), L, N)

    assert np.allclose(delta, 0.0, atol=1e-14)

def test_za_particle_wrapping():
    q = np.array([[[[[0.1, 0.2, 9.9]]]]]).reshape(1, 1, 1, 3)

    psi = np.array([[[[-0.2, 0.0, 0.3]]]]).reshape(1, 1, 1, 3)

    x, _ = zaParticles(q, psi, L=10.0, Hconf_i=1.0, f_i=1.0)

    assert np.all(x >= 0.0)
    assert np.all(x < 10.0)

def test_za_velocity_normalization():
    rng = np.random.default_rng(123)

    psi = rng.normal(size=(4, 4, 4, 3))
    q = np.zeros_like(psi)

    Hconf = 0.7
    f = 0.9

    _, v = zaParticles(q, psi, L=100.0, Hconf_i=Hconf, f_i=f)

    assert np.allclose(v, Hconf * f * psi)

def test_cic_window_zero_mode():
    N = 16
    L = 100.0

    _, _, kx, ky, kz, _, _ = fourierGrid(N, L)

    W = cicWindow(kx, ky, kz, L/N)

    assert np.isclose(W[0, 0, 0], 1.0)
    assert np.all(W > 0.0)
    assert np.all(W <= 1.0)