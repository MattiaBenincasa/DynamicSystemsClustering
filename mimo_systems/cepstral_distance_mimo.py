import numpy as np
from scipy.fft import ifft
from scipy.signal import csd
from cepstral_distance_siso import extended_cepstral_distance


def compute_cpsd_matrix(x, y, Ts, nperseg=256, return_onesided=False):
    n_signal_x = len(x)
    n_signal_y = len(y)
    fs = 1 / Ts

    freqs, _ = csd(y[0], y[0], fs=fs, nperseg=nperseg, return_onesided=return_onesided)

    p_xy = np.zeros((len(freqs), n_signal_x, n_signal_y,), dtype=complex)

    for i in range(n_signal_x):
        for j in range(n_signal_y):
            f, p_xy[:, i, j] = csd(x[i], y[j], fs=fs, nperseg=nperseg, return_onesided=return_onesided)

    return p_xy


def compute_power_cepstrum(p_xy, regularized):
    n_freq, n_x, n_y = p_xy.shape
    c = np.zeros(n_freq)

    if regularized:
        tr = np.real(np.trace(p_xy, axis1=1, axis2=2)) / n_x
        eps = 1e-12 * tr
    else:
        eps = np.zeros(n_freq)

    eye_matrix = np.eye(n_x)
    for k in range(n_freq):
        matrix = p_xy[k] + eps[k] * eye_matrix
        sign, logdet = np.linalg.slogdet(matrix)
        c[k] = logdet

    return np.real(c)


def compute_cepstrum_transfer_function(u, y, regularized):

    cpsd_uu = compute_cpsd_matrix(u, u, 1)
    cpsd_uy = compute_cpsd_matrix(u, y, 1)
    cpsd_yu = np.swapaxes(np.conj(cpsd_uy), 1, 2)
    prod_cross_powers = cpsd_uy @ cpsd_yu

    pc_1 = compute_power_cepstrum(prod_cross_powers, regularized)
    pc_2 = compute_power_cepstrum(cpsd_uu, regularized)

    return np.real(ifft(pc_1 - 2*pc_2))


def compute_cepstral_distance(u_1, y_1, u_2, y_2, regularized=False):
    c_h_1 = compute_cepstrum_transfer_function(u_1, y_1, regularized)
    c_h_2 = compute_cepstrum_transfer_function(u_2, y_2, regularized)

    k = np.arange(len(c_h_1)-1)
    diff = c_h_1[1:] - c_h_2[1:]
    distance = np.sum(k * diff ** 2)

    return distance


# cepstral distance with single input active. SISO distance can be used
def mimo_distance_single_input_active(u_1, outputs_1, u_2, outputs_2):
    distance = 0
    n_outputs = len(outputs_1)

    for i in range(n_outputs):
        distance += extended_cepstral_distance(u_1, outputs_1[i], u_2, outputs_2[i])

    return distance/n_outputs
