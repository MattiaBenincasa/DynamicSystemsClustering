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


def compare_th_data_cepstrum():
    np.random.seed(seed=0)

    N = 2 ** 14  # N is the length of the time series. The longer it is, the more reliable the estimates of the spectra will be.
    fs = 100     # sampling frequency

    zeros = [0.8, 0.6, 0]
    poles = [0.9, 0.7, 0.4]
    k = 1

    sys = sps.ZerosPolesGain(zeros, poles, k, dt=1 / fs)  # Initialize the system
    u = np.random.randn(N)
    ty, y = sps.dlsim(sys, u)
    y = y[:, 0]

    powerceps = power_cepstrum(u, y, N, fs)  # Estimate the power cepstral coefficients of the underlying system.

    powercepslength = int(np.floor(powerceps.shape[0] / 2))  # This is the length of one side of the power cepstrum.
    weights = np.arange(0, powercepslength)  # Initialize the weights

    th_cepstrum = poles_zeros_cepstrum(poles, zeros, weights)   # Calultate real power cepstrum coefficients of the system.
    plt.title("Confronto cepstrum sistema SISO")
    plt.plot(th_cepstrum[1:30], label='Real cepstrum')
    plt.plot(powerceps[1:30, ], linestyle='dashed', label="Estimated cepstrum")
    plt.legend()
    plt.show()


compare_th_data_cepstrum()
