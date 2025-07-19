from electric_circuits.electric_circuits import (generate_discrete_lti_circuit,
                                                 generate_white_noise_signal,
                                                 multi_input_circuit_simulation,
                                                 generate_sinusoidal_signal)
from clustering import generate_dataset_circuit, k_means, compute_and_plot_conf_matrix
from sklearn.metrics.cluster import adjusted_rand_score


def test_two_circuits_clustering():
    # simulation setup parameters
    n_samples = [2**6, 2**8, 2**10, 2**12, 2**14, 2**16]
    n_input_signals = 100

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

    # generate white noise inputs
    inputs_noise = generate_white_noise_signal(n_samples[2], n_input_signals)

    # simulate circuits on white noise signal
    outputs_noise = {'system_1': multi_input_circuit_simulation(inputs_noise, sys_1),
                     'system_2': multi_input_circuit_simulation(inputs_noise, sys_2)}

    # generate sinusoidal signal
    inputs_sinusoidal = generate_sinusoidal_signal(n_samples[2], n_input_signals)

    # simulate circuits on sinusoidal signal
    outputs_sinusoid = {'system_1': multi_input_circuit_simulation(inputs_sinusoidal, sys_1),
                        'system_2': multi_input_circuit_simulation(inputs_sinusoidal, sys_2)}

    dataset_noise, true_clusters_noise = generate_dataset_circuit(inputs_noise, outputs_noise['system_1'],
                                                                  outputs_noise['system_2'])

    dataset_sinusoid, true_clusters_sinusoid = generate_dataset_circuit(inputs_sinusoidal, outputs_sinusoid['system_1'],
                                                                        outputs_sinusoid['system_2'])
    centroids_noise, predicted_clusters_noise = k_means(dataset_noise, 2)
    centroids_sinusoid, predicted_clusters_sinusoid = k_means(dataset_sinusoid, 2)

    print(f'ARI index white noise input: {adjusted_rand_score(true_clusters_noise, predicted_clusters_noise)}')
    print(f'ARI index sinusoid+noise input: {adjusted_rand_score(true_clusters_sinusoid, predicted_clusters_sinusoid)}')
    compute_and_plot_conf_matrix(true_clusters_noise, predicted_clusters_noise, 'noise signal clustering')
    compute_and_plot_conf_matrix(true_clusters_sinusoid, predicted_clusters_sinusoid, 'sinusoidal signal clustering')

