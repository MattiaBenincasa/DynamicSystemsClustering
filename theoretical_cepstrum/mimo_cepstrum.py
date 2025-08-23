import numpy as np
import control.matlab as ml
from siso_cepstrum import poles_zeros_cepstrum
import control
from matplotlib import pyplot as plt
from mimo_systems.cepstral_distance_mimo import compute_cepstrum_transfer_function
from control import forced_response


def generate_stable_mimo_system():

    mimo_sys = ml.drss(20, 148, 6)
    poles = control.poles(mimo_sys)
    zeros = control.zeros(mimo_sys)

    while np.abs(zeros).any() >= 1:
        mimo_sys = ml.drss(20, 148, 6)
        poles = control.poles(mimo_sys)
        zeros = control.zeros(mimo_sys)

    return mimo_sys, poles, zeros


def comparing_th_data_cepstrum():

    N = 2 ** 8

    weights = np.arange(N)
    mimo_sys, poles, zeros = generate_stable_mimo_system()
    th_cepstrum = poles_zeros_cepstrum(poles, zeros, weights)

    # data-based cepstrum
    u = np.random.normal(0, 0.7, size=(6, len(weights)))
    t = np.arange(u.shape[1])
    t, y = forced_response(mimo_sys, timepts=t, inputs=u)
    powerceps = compute_cepstrum_transfer_function(u, y, eps=0)
    plt.title("Confronto cepstrum MIMO 6 input - 148 output - 2^8 campioni")
    plt.plot(th_cepstrum[1:30], label="Real cepstrum")
    plt.plot(powerceps[1:30], linestyle='dashed', label="Estimated cepstrum")
    plt.legend()
    plt.show()


comparing_th_data_cepstrum()
