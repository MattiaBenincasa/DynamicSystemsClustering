from electric_circuits.electric_circuits import (generate_discrete_lti_circuit,
                                                 generate_white_noise_signal,
                                                 simulate_circuit_on_multiple_input,
                                                 generate_sinusoidal_signal,
                                                 simulate_circuit_on_multi_input_with_x0)
from clustering import generate_dataset_circuit, k_means, compute_and_plot_conf_matrix
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from matplotlib import pyplot as plt


def init_circuits():
    # first electric circuit
    R1 = 100
    L11 = 20
    L12 = 60
    C1 = 50

    # second electric circuit
    R2 = 100
    L21 = 200
    L22 = 160
    C2 = 75

    sys_1 = generate_discrete_lti_circuit(R1, L11, L12, C1)
    sys_2 = generate_discrete_lti_circuit(R2, L21, L22, C2)

    return sys_1, sys_2


def test_two_circuits_clustering():
    # simulation setup parameters
    n_samples = 2**10
    n_input_signals = 100

    sys_1, sys_2 = init_circuits()

    # generate white noise inputs
    inputs_noise = generate_white_noise_signal(n_samples, 0, 0.6, n_input_signals)

    # simulate circuits on white noise signal
    outputs_noise = {'system_1': simulate_circuit_on_multiple_input(inputs_noise, sys_1),
                     'system_2': simulate_circuit_on_multiple_input(inputs_noise, sys_2)}

    # generate sinusoidal signal
    inputs_sinusoidal = generate_sinusoidal_signal(n_samples, n_input_signals)

    # simulate circuits on sinusoidal signal
    outputs_sinusoid = {'system_1': simulate_circuit_on_multiple_input(inputs_sinusoidal, sys_1),
                        'system_2': simulate_circuit_on_multiple_input(inputs_sinusoidal, sys_2)}

    dataset_noise, true_clusters_noise = generate_dataset_circuit(inputs_noise, outputs_noise['system_1'],
                                                                  outputs_noise['system_2'])

    dataset_sinusoid, true_clusters_sinusoid = generate_dataset_circuit(inputs_sinusoidal, outputs_sinusoid['system_1'],
                                                                        outputs_sinusoid['system_2'])
    centroids_noise, predicted_clusters_noise = k_means(dataset_noise, 2)
    centroids_sinusoid, predicted_clusters_sinusoid = k_means(dataset_sinusoid, 2)
    print(f'Two circuits n_samples: {n_samples} n_input_signals: {n_input_signals}:')
    print(f'ARI index white noise input: {adjusted_rand_score(true_clusters_noise, predicted_clusters_noise)}')
    print(f'ARI index sinusoidal input: {adjusted_rand_score(true_clusters_sinusoid, predicted_clusters_sinusoid)}')
    print('----------------------------------------------')
    compute_and_plot_conf_matrix(true_clusters_noise, predicted_clusters_noise, 'noise signal clustering')
    compute_and_plot_conf_matrix(true_clusters_sinusoid, predicted_clusters_sinusoid, 'sinusoidal signal clustering')


def test_increasing_noise_intensity():

    n_samples = 2**12
    n_input_signals = 50

    sys_1, sys_2 = init_circuits()
    sigmas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 1]
    ARIs = []
    for sigma in sigmas:
        # inputs_noise = generate_white_noise_signal(n_samples, 0, 0.6, n_input_signals)
        inputs_noise = generate_sinusoidal_signal(n_samples, n_input_signals)
        outputs_white_noise = {'system_1': np.add(simulate_circuit_on_multiple_input(inputs_noise, sys_1),
                                                  np.random.normal(0, sigma, size=(n_input_signals, n_samples))),
                               'system_2': np.add(simulate_circuit_on_multiple_input(inputs_noise, sys_2),
                                                  np.random.normal(0, sigma, size=(n_input_signals, n_samples)))}

        dataset_noise, true_clusters_noise = generate_dataset_circuit(inputs_noise, outputs_white_noise['system_1'],
                                                                      outputs_white_noise['system_2'])

        centroids_noise, predicted_clusters_noise = k_means(dataset_noise, 2)
        ARIs.append(adjusted_rand_score(true_clusters_noise, predicted_clusters_noise))
        print(f'sigma {sigma} computed')

    x = np.arange(len(sigmas))
    plt.bar(x, ARIs, 0.4, label='ARI index')
    plt.xticks(x, [str(sigma) for sigma in sigmas])
    plt.xlabel('standard deviation measure error')
    plt.ylabel('ARI')
    plt.title('sinusoid in - different standard deviation')
    plt.legend()
    plt.show()

    # print(f'Two circuits n_samples: {n_samples} n_input_signals: {n_input_signals}:')
    # print(f'ARI index white noise input: {adjusted_rand_score(true_clusters_noise, predicted_clusters_noise)}')
    # print('----------------------------------------------')
    # compute_and_plot_conf_matrix(true_clusters_noise, predicted_clusters_noise, 'noise signal clustering')


def test_clustering_with_different_initial_conditions(sigma):
    #   clustering of 100 (u, y) signal pairs: 50 from system 1 and 50 from system 2.
    #   Within the group of 50 signals, there are 25 outputs generated with x(0) = 0
    #   and 25 generated with x!=0.
    #   Different initial conditions are generated from normal distribution with sigma
    #   standard deviation

    n_samples = 2**12
    sys_1, sys_2 = init_circuits()
    inputs = generate_white_noise_signal(n_samples, 0, 0.6, 50)
    x0_1 = np.random.normal(0, sigma, size=(3, 25))
    x0_2 = np.random.normal(0, sigma, size=(3, 25))
    outputs_1_no_x0 = simulate_circuit_on_multiple_input(inputs[:25], sys_1)
    outputs_2_no_x0 = simulate_circuit_on_multiple_input(inputs[:25], sys_2)
    outputs_1_with_x0 = simulate_circuit_on_multi_input_with_x0(inputs[25:50], sys_1, x0_1)
    outputs_2_with_x0 = simulate_circuit_on_multi_input_with_x0(inputs[25:50], sys_2, x0_2)

    outputs_1 = outputs_1_no_x0 + outputs_1_with_x0
    outputs_2 = outputs_2_no_x0 + outputs_2_with_x0

    dataset, true_clusters = generate_dataset_circuit(inputs, outputs_1, outputs_2)
    centroids, predicted_clusters = k_means(dataset, 2)
    print(adjusted_rand_score(true_clusters, predicted_clusters))
