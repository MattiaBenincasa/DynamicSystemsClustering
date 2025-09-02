import numpy as np
import control.matlab as ml
from theoretical_cepstrum.siso_cepstrum import poles_zeros_cepstrum, poles_zeros_norm
import control
from matplotlib import pyplot as plt
from mimo_systems.cepstral_distance_mimo import compute_cepstrum_transfer_function
from control import forced_response


def generate_stable_mimo_system(n_inputs, n_outputs, n_states):

    mimo_sys = ml.drss(n_states, n_outputs, n_inputs)
    poles = control.poles(mimo_sys)

    # transmission zeros
    zeros = control.zeros(mimo_sys)

    while np.abs(zeros).any() >= 1:
        mimo_sys = ml.drss(n_states, n_outputs, n_inputs)
        poles = control.poles(mimo_sys)
        zeros = control.zeros(mimo_sys)

    return mimo_sys, poles, zeros


def comparing_th_data_cepstrum(n_samples, n_states, n_inputs, n_outputs, title_plots):
    np.random.seed(seed=42)
    weights = np.arange(n_samples)
    mimo_sys, poles, zeros = generate_stable_mimo_system(n_inputs, n_outputs, n_states)
    th_cepstrum = poles_zeros_cepstrum(poles, zeros, weights)

    # data-based cepstrum
    u = np.random.normal(0, 0.7, size=(n_inputs, n_samples))
    t = np.arange(u.shape[1])
    t, y = forced_response(mimo_sys, timepts=t, inputs=u)
    powerceps = compute_cepstrum_transfer_function(u, y, regularized=False)
    plt.title(title_plots)
    plt.plot(th_cepstrum[1:30], label="Real cepstrum")
    plt.plot(powerceps[1:30], linestyle='dashed', label="Estimated cepstrum")
    plt.xlabel("Numero coefficienti")
    plt.ylabel("Valore cepstrum")
    plt.legend()
    plt.savefig(f'../th_cepstrum_plots/{title_plots}.png', dpi=300)
    plt.show()


# for this simulation it is recommended to set nperseg=4096 ((2^14)/4) in compute_cpsd_matrix() cepstral_distance_mimo.py
# comparing_th_data_cepstrum(2**14, 3, 3, 3, "Confronto cepstrum MIMO 3 input - 3 output")
# comparing_th_data_cepstrum(2**14, 20, 6, 148, "Confronto cepstrum MIMO 6 input - 148 output")
