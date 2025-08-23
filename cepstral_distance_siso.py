from scipy.signal import welch
from scipy.fft import ifft
import numpy as np


def compute_power_cepstrum(u, y, window_size):
    f_u, Puu = welch(u, nperseg=window_size, return_onesided=True)
    f_y, Pyy = welch(y, nperseg=window_size, return_onesided=True)

    return np.real(ifft(np.log(Pyy/Puu)))


def extended_cepstral_distance(u_1, y_1, u_2, y_2):
    window_size = np.minimum(len(u_1), np.maximum(256, int(np.floor(len(u_1) / 4))))
    powerceps_1 = compute_power_cepstrum(u_1, y_1, window_size)
    powerceps_2 = compute_power_cepstrum(u_2, y_2, window_size)

    weights = np.arange(len(powerceps_1))

    return np.dot(weights, np.square(powerceps_1-powerceps_2))
