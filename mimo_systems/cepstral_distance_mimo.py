import numpy as np
from scipy.fft import ifft
from scipy.signal import csd


def compute_cpsd_matrix(x, y, Ts, return_onesided=False):
    n_signal_x = len(x)
    n_signal_y = len(y)
    fs = 1 / Ts

    freqs, _ = csd(y[0], y[0], fs=fs, nperseg=64, return_onesided=return_onesided)

    p_xy = np.zeros((len(freqs), n_signal_x, n_signal_y,), dtype=complex)

    for i in range(n_signal_x):
        for j in range(n_signal_y):
            f, p_xy[:, i, j] = csd(x[i], y[j], fs=fs, nperseg=64, return_onesided=return_onesided)

    return p_xy


def compute_power_cepstrum(p_xy):
    n_freq = len(p_xy)

    c = np.zeros(n_freq)
    for k in range(n_freq):
        m = p_xy[k, :, :]
        det_m = np.linalg.det(p_xy[k, :, :])
        tr = np.real(np.trace(p_xy[k])) / p_xy.shape[1]
        eps = 1e-13 * tr
        # eps = 0
        det_eps = np.linalg.det(p_xy[k, :, :]+eps*np.eye(p_xy.shape[1]))
        c[k] = np.log(np.abs(np.linalg.det(p_xy[k, :, :]+eps*np.eye(p_xy.shape[1]))))

    return np.real(c)


def compute_cepstrum_transfer_function(u, y, eps):

    cpsd_uu = compute_cpsd_matrix(u, u, 1)
    cpsd_uy = compute_cpsd_matrix(u, y, 1)
    cpsd_yu = compute_cpsd_matrix(y, u, 1)
    prod_cross_powers = cpsd_uy @ cpsd_yu

    pc_1 = compute_power_cepstrum(prod_cross_powers)
    pc_2 = compute_power_cepstrum(cpsd_uu)
    # is_inf_1 = np.isinf(pc_1)
    # is_inf_2 = np.isinf(pc_2)
    # pc_1[is_inf_1] = 0
    # pc_2[is_inf_2] = 0
    return np.real(ifft(pc_1 - 2*pc_2))


def compute_cepstral_distance(u_1, y_1, u_2, y_2, eps=0.0):
    c_h_1 = compute_cepstrum_transfer_function(u_1, y_1, eps)
    c_h_2 = compute_cepstrum_transfer_function(u_2, y_2, eps)

    k = np.arange(len(c_h_1))
    diff = c_h_1 - c_h_2
    distance = np.sum(k * diff ** 2)

    return distance
