from scipy.signal import welch
from scipy.fft import ifft
import numpy as np


def compute_in_out_cepstral(u, y, nperseg):
    f_u, Puu = welch(u, nperseg=nperseg)
    f_y, Pyy = welch(y, nperseg=nperseg)

    # cepstral coefficients
    c_u = np.real(ifft(np.log(Puu)))
    c_y = np.real(ifft(np.log(Pyy)))

    return c_u, c_y


def extended_cepstral_distance(u_1, y_1, u_2, y_2):
    c_u_1, c_y_1 = compute_in_out_cepstral(u_1, y_1, len(u_1)/4)
    c_u_2, c_y_2 = compute_in_out_cepstral(u_2, y_2, len(u_1)/4)

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
