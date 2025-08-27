import numpy as np
from scipy.fft import ifft
from scipy.signal import csd, welch


def compute_cpsd_matrix(x, y, Ts, nperseg=64, return_onesided=False):
    n_signal_x = len(x)
    n_signal_y = len(y)
    fs = 1 / Ts

    freqs, _ = csd(y[0], y[0], fs=fs, nperseg=64, return_onesided=return_onesided)

    p_xy = np.zeros((len(freqs), n_signal_x, n_signal_y,), dtype=complex)

    for i in range(n_signal_x):
        for j in range(n_signal_y):
            f, p_xy[:, i, j] = csd(x[i], y[j], fs=fs, nperseg=nperseg, return_onesided=return_onesided)

    return p_xy


def compute_power_cepstrum(p_xy):
    n_freq, n_x, n_y = p_xy.shape

    c = np.zeros(n_freq)

    tr = np.real(np.trace(p_xy, axis1=1, axis2=2)) / n_x
    eps = 1e-15 * tr
    eye_matrix = np.eye(n_x)
    for k in range(n_freq):
        matrix = p_xy[k] + eps[k] * eye_matrix
        sign, logdet = np.linalg.slogdet(matrix)
        c[k] = logdet

    return np.real(c)


def compute_cepstrum_transfer_function(u, y, eps):

    cpsd_uu = compute_cpsd_matrix(u, u, 1)
    cpsd_uy = compute_cpsd_matrix(u, y, 1)
    cpsd_yu = np.swapaxes(np.conj(cpsd_uy), 1, 2)
    prod_cross_powers = cpsd_uy @ cpsd_yu

    pc_1 = compute_power_cepstrum(prod_cross_powers)
    pc_2 = compute_power_cepstrum(cpsd_uu)

    return np.real(ifft(pc_1 - 2*pc_2))


def compute_cepstral_distance(u_1, y_1, u_2, y_2, eps=0.0):
    c_h_1 = compute_cepstrum_transfer_function(u_1, y_1, eps)
    c_h_2 = compute_cepstrum_transfer_function(u_2, y_2, eps)

    k = np.arange(len(c_h_1))
    diff = c_h_1 - c_h_2
    distance = np.sum(k * diff ** 2)

    return distance
