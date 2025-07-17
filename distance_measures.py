from scipy.signal import welch
from scipy.fft import ifft
import numpy as np


def compute_in_out_cepstral(u, y):
    f_u, Puu = welch(u[:, 0])
    f_y, Pyy = welch(y[:, 0])

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
    n_outputs, _ = y_1.shape

    distance = 0

    for i in range(n_inputs):
        for j in range(n_outputs):
            distance += extended_cepstral_distance(u_1[i].reshape(-1, 1), y_1[j].reshape(-1, 1),
                                                   u_2[i].reshape(-1, 1), y_2[j].reshape(-1, 1))

    return distance
