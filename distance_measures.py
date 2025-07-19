from scipy.signal import welch
from scipy.fft import ifft
import numpy as np


def compute_in_out_cepstral(u, y):
    f_u, Puu = welch(u)
    f_y, Pyy = welch(y)

    # cepstral coefficients
    c_u = np.real(ifft(np.log(Puu)))
    c_y = np.real(ifft(np.log(Pyy)))

    return c_u, c_y


def extended_cepstral_distance(u_1, y_1, u_2, y_2):
    c_u_1, c_y_1 = compute_in_out_cepstral(u_1, y_1)
    c_u_2, c_y_2 = compute_in_out_cepstral(u_2, y_2)

    length = len(c_u_1)
    distance = 0
    for k in range(1, length):
        distance += k*((c_y_1[k] - c_u_1[k])-(c_y_2[k] - c_u_2[k]))**2

    return distance


def extended_cepstral_distance_mimo(u_1, y_1, u_2, y_2):
    n_inputs, k = u_1.shape
    n_outputs, _ = y_2.shape

    distance = 0

    for i in range(n_inputs):
        for j in range(n_outputs):
            distance += extended_cepstral_distance(u_1[i], y_1[j], u_2[i], y_2[j])

    return distance


def compute_distance_matrix(subject_ids, inputs, outputs, distance_function):
    n = len(subject_ids)

    d = np.zeros((n, n))

    for i in range(n):
        subject_id_i = subject_ids[i]
        u_1, y_1 = inputs[subject_id_i], outputs[subject_id_i]
        for j in range(i+1, n):
            print(f'i: {i} j: {j}')
            subject_id_j = subject_ids[j]
            u_2, y_2 = inputs[subject_id_j], outputs[subject_id_j]
            distance = distance_function(u_1, y_1, u_2, y_2)
            d[i, j] = distance
            d[j, i] = distance

    np.save('distance_matrix.npy', d)
    return d
