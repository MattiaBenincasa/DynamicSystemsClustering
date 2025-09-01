import numpy as np
import scipy.signal as sps
from matplotlib import pyplot as plt


def power_cepstrum(u, y, N, fs):
    windowsize = np.minimum(N, np.maximum(256, int(np.floor(N/4))))
    f, Puu = sps.welch(u, fs, nperseg=windowsize, return_onesided=True)
    f, Pyy = sps.welch(y, fs, nperseg=windowsize, return_onesided=True)
    powerceps = np.fft.irfft(np.log(Pyy/Puu))

    return powerceps


def poles_zeros_cepstrum(poles, zeros, weights):
    # stable case

    weights = np.arange(len(weights))
    th_cepstrum = np.zeros(len(weights))

    for k in range(1, len(weights)):
        th_cepstrum[k] = np.real(np.sum(np.power(poles, np.abs(k))/np.abs(k)) - np.sum(np.power(zeros, np.abs(k))/np.abs(k)))

    return th_cepstrum


def poles_zeros_norm(poles, zeros):

    if poles[0] >= 1:
        zeros_minimum = [1 / x for x in zeros]
        poles_minimum = [1 / x for x in poles]
    else:
        zeros_minimum = zeros
        poles_minimum = poles

    distpoleszeros = np.sum(np.log((1 - np.outer(poles_minimum, np.conj(zeros_minimum))) ** 2)) - np.sum(
        np.log(1 - np.outer(poles_minimum, np.conj(poles_minimum)))) - np.sum(
        np.log(1 - np.outer(zeros_minimum, np.conj(zeros_minimum))))
    return distpoleszeros


def poles_and_zeros_distance(poles_1, zeros_1, poles_2, zeros_2):
    distance = 0
    for k in range(1, 64):
        sum_ = np.real(np.sum(np.power(zeros_1, np.abs(k))/np.abs(k)) - np.sum(np.power(poles_1, np.abs(k))/np.abs(k)) -
                       np.sum(np.power(zeros_2, np.abs(k))/np.abs(k)) + np.sum(np.power(poles_2, np.abs(k))/np.abs(k)))
        distance += k * sum_ * sum_

    return distance


def compare_th_data_cepstrum(n_samples, title_plots):
    np.random.seed(seed=0)

    fs = 100     # sampling frequency

    zeros = [0.8, 0.6, 0]
    poles = [0.9, 0.7, 0.4]
    k = 1

    sys = sps.ZerosPolesGain(zeros, poles, k, dt=1 / fs)  # Initialize the system
    u = np.random.randn(n_samples)
    # x_0 = np.random.normal(0, 200, size=3)
    # noise = 100*np.random.normal(10, 100, size=2**6)
    ty, y = sps.dlsim(sys, u)
    y = y[:, 0]

    powerceps = power_cepstrum(u, y, n_samples, fs)  # Estimate the power cepstral coefficients of the underlying system.

    powercepslength = int(np.floor(powerceps.shape[0] / 2))  # This is the length of one side of the power cepstrum.
    weights = np.arange(0, powercepslength)  # Initialize the weights

    th_cepstrum = poles_zeros_cepstrum(poles, zeros, weights)   # Calultate real power cepstrum coefficients of the system.
    plt.title(title_plots)
    plt.plot(th_cepstrum[1:30], label='Real cepstrum')
    plt.plot(powerceps[1:30, ], linestyle='dashed', label="Estimated cepstrum")
    plt.xlabel("Numero coefficienti")
    plt.ylabel("Valore cepstrum")
    plt.legend()
    plt.savefig(f'../th_cepstrum_plots/{title_plots}.png', dpi=300)
    plt.show()


compare_th_data_cepstrum(2**12, "Confronto cepstrum sistema SISO 2^12 campioni")
compare_th_data_cepstrum(2**6, "Confronto cepstrum sistema SISO 2^6 campioni")
