import numpy as np
from scipy.fft import ifft
from scipy.signal import csd


def compute_cpsd_matrix(x, y, Ts, return_onesided=False):
    n_signal_x = len(x)
    n_signal_y = len(y)
    fs = 1 / Ts

    _, temp_csd = csd(y[0], y[0], fs=fs, return_onesided=return_onesided)
    n_freq = len(temp_csd)
    p_xy = np.zeros((n_freq, n_signal_x, n_signal_y,), dtype=complex)

    for i in range(n_signal_x):
        for j in range(n_signal_y):
            f, p_xy[:, i, j] = csd(x[i], y[j], fs=fs, return_onesided=return_onesided)

    return p_xy


def compute_power_cepstrum(p_xy, eps):
    n_freq = len(p_xy)

    c = np.zeros(n_freq)
    for k in range(n_freq):
        m = p_xy[k, :, :]
        det_m = np.linalg.det(p_xy[k, :, :])
        det_eps = np.linalg.det(p_xy[k, :, :]+eps*np.eye(p_xy.shape[1]))
        c[k] = np.log(np.abs(np.linalg.det(p_xy[k, :, :]+eps*np.eye(p_xy.shape[1]))))

    return np.real(ifft(c))


def compute_cepstrum_transfer_function(u, y, eps):

    cpsd_uu = compute_cpsd_matrix(u, u, 1)
    cpsd_uy = compute_cpsd_matrix(u, y, 1)
    cpsd_yu = compute_cpsd_matrix(y, u, 1)

    prod_cross_powers = cpsd_uy @ cpsd_yu

    return compute_power_cepstrum(prod_cross_powers, eps) - 2*compute_power_cepstrum(cpsd_uu, eps)


def compute_cepstral_distance(u_1, y_1, u_2, y_2, eps=0.0):
    c_h_1 = compute_cepstrum_transfer_function(u_1, y_1, eps)
    c_h_2 = compute_cepstrum_transfer_function(u_2, y_2, eps)

    distance = 0
    length = len(c_h_1)
    for k in range(length):
        difference = c_h_1[k] - c_h_2[k]
        distance += k*difference*difference

    return distance


def compute_distance_matrix_fMRI(subject_ids, inputs, outputs, eps=1e-12):
    n = len(subject_ids)

    d = np.zeros((n, n))

    for i in range(n):
        if i == 1:
            break
        subject_id_i = subject_ids[i]
        u_1, y_1 = inputs[subject_id_i], outputs[subject_id_i]
        for j in range(i+1, n):
            print(f'i: {i} j: {j}')
            subject_id_j = subject_ids[j]
            u_2, y_2 = inputs[subject_id_j], outputs[subject_id_j]
            distance = compute_cepstral_distance(u_1, y_1, u_2, y_2, eps)
            d[i, j] = distance
            d[j, i] = distance

    np.save('distance_matrix.npy', d)
    return d
