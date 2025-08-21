from electric_circuits.electric_circuits import (generate_discrete_lti_circuit,
                                                 generate_white_noise_signal,
                                                 simulate_circuit_on_multiple_input,
                                                 generate_sinusoidal_signal,
                                                 simulate_circuit_on_multi_input_with_x0,
                                                 generate_multiple_multi_sin_waves,
                                                 simulate_circuit_on_multiple_inputs_with_output_noise)
from clustering import generate_dataset_circuit, k_means, compute_and_plot_conf_matrix
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from electric_circuits.utils_test_cepstral_distance import (compute_avg_distances,
                                                            plot_distance_comparison,
                                                            plot_mean_and_std, compute_mean_and_std)


def init_circuits(fs=1):
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

    sys_1 = generate_discrete_lti_circuit(R1, L11, L12, C1, fs=fs)
    sys_2 = generate_discrete_lti_circuit(R2, L21, L22, C2, fs=fs)

    return sys_1, sys_2


def test_two_circuits_clustering(sys_1, sys_2, n_samples, r, inputs):

    ari_with_different_length = {}
    distance_with_different_lengths_same_system = {}
    distance_with_different_lengths_different_systems = {}

    for n in n_samples:
        ari = []
        distance_same = []
        distance_different = []
        for i in range(r):

            # simulate circuits on multiple inputs
            outputs = {'system_1': simulate_circuit_on_multiple_input(inputs[n][i], sys_1),
                       'system_2': simulate_circuit_on_multiple_input(inputs[n][i], sys_2)}

            # compute avg distance between timeseries
            dist_avg_different, dist_avg_same = compute_avg_distances((inputs[n][i], outputs['system_1']),
                                                                      (inputs[n][i], outputs['system_2']))
            distance_same.append(dist_avg_same)
            distance_different.append(dist_avg_different)

            dataset, true_clusters = generate_dataset_circuit(inputs[n][i], outputs['system_1'], outputs['system_2'])
            centroids, predicted_clusters = k_means(dataset, 2)
            ari.append(adjusted_rand_score(true_clusters, predicted_clusters))
            print("computed")
        ari_with_different_length[n] = ari
        distance_with_different_lengths_same_system[n] = distance_same
        distance_with_different_lengths_different_systems[n] = distance_different
        print(f'{n} computed')

    plot_distance_comparison(compute_mean_and_std(distance_with_different_lengths_same_system),
                             compute_mean_and_std(distance_with_different_lengths_different_systems),
                             title="Confronto delle distanze",
                             x_label="lunghezza serie temporali",
                             y_label="distanza")

    plot_mean_and_std(compute_mean_and_std(ari_with_different_length),
                      title="Clustering di serie temporali di lunghezza diversa",
                      x_label="lunghezza serie temporali",
                      y_label="indice ARI")


def test_increasing_noise_intensity_clustering(sys_1, sys_2, snr_values, r, inputs):

    ari_with_different_snr = {}
    distance_with_different_snr_same_system = {}
    distance_with_different_snr_different_system = {}

    for snr in snr_values:
        ari = []
        distance_same = []
        distance_different = []
        for i in range(r):

            # simulate circuits on multiple inputs
            outputs = {'system_1': simulate_circuit_on_multiple_inputs_with_output_noise(inputs[snr][i], sys_1, snr),
                       'system_2': simulate_circuit_on_multiple_inputs_with_output_noise(inputs[snr][i], sys_2, snr)}

            dataset_noise, true_clusters_noise = generate_dataset_circuit(inputs[snr][i], outputs['system_1'],
                                                                          outputs['system_2'])

            # compute avg distance between timeseries
            dist_avg_different, dist_avg_same = compute_avg_distances((inputs[snr][i], outputs['system_1']),
                                                                      (inputs[snr][i], outputs['system_2']))
            distance_same.append(dist_avg_same)
            distance_different.append(dist_avg_different)

            centroids_noise, predicted_clusters_noise = k_means(dataset_noise, 2)
            ari.append(adjusted_rand_score(true_clusters_noise, predicted_clusters_noise))
            print(f'SNR {snr} computed')
        ari_with_different_snr[snr] = ari
        distance_with_different_snr_same_system[snr] = distance_same
        distance_with_different_snr_different_system[snr] = distance_different

    plot_distance_comparison(compute_mean_and_std(distance_with_different_snr_same_system),
                             compute_mean_and_std(distance_with_different_snr_different_system),
                             title="Confronto delle distanze",
                             x_label="Rumore di misura SNR (dB)",
                             y_label="distanza")

    plot_mean_and_std(compute_mean_and_std(ari_with_different_snr),
                      title="Clustering di serie temporali con errori di misura in output",
                      x_label="Rumore di misura SNR (dB)",
                      y_label="indice ARI")


def test_clustering_with_different_initial_conditions(sys_1, sys_2, sigma, n_samples, r, inputs, n_inputs):
    # random initial conditions
    x0 = np.random.normal(0, sigma, size=(3, n_inputs))
    ari_with_different_length = {}
    distance_with_different_lengths_same_system = {}
    distance_with_different_lengths_different_systems = {}
    for n in n_samples:
        ari = []
        distance_same = []
        distance_different = []
        for i in range(r):
            outputs_1 = simulate_circuit_on_multi_input_with_x0(inputs[n][i], sys_1, x0)
            outputs_2 = simulate_circuit_on_multi_input_with_x0(inputs[n][i], sys_2, x0)

            # compute avg distance between timeseries
            dist_avg_different, dist_avg_same = compute_avg_distances((inputs[n][i], outputs_1),
                                                                      (inputs[n][i], outputs_2))
            distance_same.append(dist_avg_same)
            distance_different.append(dist_avg_different)

            dataset, true_clusters = generate_dataset_circuit(inputs[n][i], outputs_1, outputs_2)
            centroids, predicted_clusters = k_means(dataset, 2)
            ari.append(adjusted_rand_score(true_clusters, predicted_clusters))
            print("computed")
        print(f'{n} computed')
        ari_with_different_length[n] = ari
        distance_with_different_lengths_same_system[n] = distance_same
        distance_with_different_lengths_different_systems[n] = distance_different

    plot_distance_comparison(compute_mean_and_std(distance_with_different_lengths_same_system),
                             compute_mean_and_std(distance_with_different_lengths_different_systems),
                             title="Confronto delle distanze",
                             x_label="lunghezza serie temporali",
                             y_label="distanza")

    plot_mean_and_std(compute_mean_and_std(ari_with_different_length),
                      title="Clustering con condizioni iniziali diverse da zero",
                      x_label="lunghezza serie temporali",
                      y_label="indice ARI")


def setup_and_execute_tests():
    fs = 100
    sys_1, sys_2 = init_circuits(fs)
    n_samples = [2**10, 2**12, 2**14]
    r = 3

    # white noise inputs
    white_noise_inputs = {}

    for n in n_samples:
        white_noise_inputs[n] = [generate_white_noise_signal(n, 0, 0.6, 50) for _ in range(r)]

    # test_two_circuits_clustering(sys_1, sys_2, n_samples, r, white_noise_inputs)

    snr_values = [35, 30, 25, 15]
    white_noise_inputs_snr = {}
    for snr in snr_values:
        white_noise_inputs_snr[snr] = [generate_white_noise_signal(n_samples[0], 0, 0.6, 50) for _ in range(len(snr_values))]
    test_increasing_noise_intensity_clustering(sys_1, sys_2, snr_values, r, white_noise_inputs_snr)
    # test_clustering_with_different_initial_conditions(sys_1, sys_2, 1, n_samples, r, white_noise_inputs, 50)
    # sinusoidal inputs
    sinusoidal_inputs = {}

    for n in n_samples:
        sinusoidal_inputs[n] = [generate_sinusoidal_signal(n, fs=fs, n_signals=50) for _ in range(r)]

    # test_two_circuits_clustering(sys_1, sys_2, n_samples, r, sinusoidal_inputs)

    # multisine waves inputs
    multisine_waves = {}
    for n in n_samples:
        multisine_waves[n] = [generate_multiple_multi_sin_waves(n, 50, fs) for _ in range(r)]

    # test_two_circuits_clustering(sys_1, sys_2, n_samples, r, multisine_waves)
