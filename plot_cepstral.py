from electric_circuits.electric_circuits import generate_white_noise_signal, generate_sinusoidal_signal, simulate_circuit_on_multiple_input
from electric_circuits.test_clustering import init_circuits
from distance_measures import compute_distance_matrix
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt


def avg_ignore_diagonal_elements(square_matrix):
    non_diag_elements = []
    n = len(square_matrix)
    for i in range(n):
        for j in range(n):
            if i != j:
                non_diag_elements.append(square_matrix[i, j])

    return mean(non_diag_elements)


def plot_cepstral_distance():
    n_samples = 2**14
    n_signals = 5

    sys_1, sys_2 = init_circuits()

    # input_white_noise = generate_white_noise_signal(n_samples, 0, 0.6, n_signals)
    input_white_noise = generate_sinusoidal_signal(n_samples, n_signals)
    outputs_white_noise = {'system_1': simulate_circuit_on_multiple_input(input_white_noise, sys_1),
                           'system_2': simulate_circuit_on_multiple_input(input_white_noise, sys_2)}

    print('Distance matrix different circuits')
    dm_different = compute_distance_matrix((input_white_noise, outputs_white_noise['system_1']),
                                           (input_white_noise, outputs_white_noise['system_2']))
    print(dm_different)
    print(f'Mean value: {dm_different.mean()}')
    print('------------------------------------------------')
    print('Distance matrix same circuits')
    dm_same = compute_distance_matrix((input_white_noise, outputs_white_noise['system_1']),
                                      (input_white_noise, outputs_white_noise['system_1']))
    print(dm_same)
    print(f'Mean value: {dm_same.mean()}')
    print('------------------------------------------------')

    sigmas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 1, 10, 20, 30]
'''
    outputs_white_noise_sigmas = {}
    # change sigma
    for sigma in sigmas:
        outputs_white_noise_sigmas[sigma] = {'system_1': np.add(outputs_white_noise['system_1'],
                                                                np.random.normal(0, sigma, size=(n_signals, n_samples))),
                                             'system_2': np.add(outputs_white_noise['system_2'],
                                                                np.random.normal(0, sigma, size=(n_signals, n_samples)))}

    same = []   # (input, output) from the same circuit
    different = []  # (input, output) from different circuit

    for sigma in sigmas:
        same.append(compute_distance_matrix((input_white_noise, outputs_white_noise_sigmas[sigma]['system_1']),
                                            (input_white_noise, outputs_white_noise_sigmas[sigma]['system_1'])).mean())
        different.append(compute_distance_matrix((input_white_noise, outputs_white_noise_sigmas[sigma]['system_1']),
                                            (input_white_noise, outputs_white_noise_sigmas[sigma]['system_2'])).mean())

    w, x = 0.4, np.arange(len(sigmas))
    plt.bar(x-w/2, same, w, label='same dynamic')
    plt.bar(x+w/2, different, w, label='different dynamic')

    plt.xticks(x, [str(sigma) for sigma in sigmas])
    plt.ylabel('distance')
    plt.legend()
    plt.show()'''

