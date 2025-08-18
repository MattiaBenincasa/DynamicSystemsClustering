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
from matplotlib import pyplot as plt
from electric_circuits.utils_test_cepstral_distance import compute_avg_distances, plot_distance_comparison


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
    # simulation parameters setup
    n_samples = [2**10, 2**12]
    sys_1, sys_2 = init_circuits()
    r = 1  # repetitions for each test with a specific length
    ari_with_different_length = {}
    distance_with_different_lengths_same_system = {}
    distance_with_different_lengths_different_systems = {}
    for n_sample in n_samples:
        ari = []
        distance_same = []
        distance_different = []
        for _ in range(r):
            # generate input signals
            inputs = generate_white_noise_signal(n_sample, 0, 0.6, 50)
            # inputs = generate_multiple_multi_sin_waves(n_sample, 50)

            # simulate circuits on white noise signal
            outputs = {'system_1': simulate_circuit_on_multiple_input(inputs, sys_1),
                       'system_2': simulate_circuit_on_multiple_input(inputs, sys_2)}

            # compute avg distance between timeseries
            dist_avg_different, dist_avg_same = compute_avg_distances((inputs, outputs['system_1']),
                                                                      (inputs, outputs['system_2']))
            distance_same.append(dist_avg_same)
            distance_different.append(dist_avg_different)

            dataset, true_clusters = generate_dataset_circuit(inputs, outputs['system_1'], outputs['system_2'])
            centroids, predicted_clusters = k_means(dataset, 2)
            ari.append(adjusted_rand_score(true_clusters, predicted_clusters))
            print("computed")
        ari_with_different_length[n_sample] = ari
        distance_with_different_lengths_same_system[n_sample] = distance_same
        distance_with_different_lengths_different_systems[n_sample] = distance_different
        print(f'{n_sample} computed')

    plot_distance_comparison(compute_mean_and_std(distance_with_different_lengths_same_system),
                             compute_mean_and_std(distance_with_different_lengths_different_systems),
                             title="Confronto delle distanze",
                             x_label="lunghezza serie temporali",
                             y_label="distanza")

    plot_mean_and_std(compute_mean_and_std(ari_with_different_length),
                      title="Clustering di serie temporali di lunghezza diversa",
                      x_label="lunghezza serie temporali",
                      y_label="indice ARI")


def test_increasing_noise_intensity_clustering(n_samples):
    n_input_signals = 50
    sys_1, sys_2 = init_circuits()
    # different SNR in dB
    snr_values = [30, 20, 15]
    ari_with_different_snr = {}
    distance_with_different_snr_same_system = {}
    distance_with_different_snr_different_system = {}
    r = 1  # repetitions for each test with a specific length
    for snr in snr_values:
        ari = []
        distance_same = []
        distance_different = []
        for _ in range(r):
            inputs_noise = generate_white_noise_signal(n_samples, 0, 0.6, n_input_signals)
            # inputs_noise = generate_sinusoidal_signal(n_samples, n_input_signals)
            outputs_white_noise = {'system_1': simulate_circuit_on_multiple_inputs_with_output_noise(inputs_noise, sys_1, snr),
                                   'system_2': simulate_circuit_on_multiple_inputs_with_output_noise(inputs_noise, sys_2, snr)}

            dataset_noise, true_clusters_noise = generate_dataset_circuit(inputs_noise, outputs_white_noise['system_1'],
                                                                          outputs_white_noise['system_2'])

            # compute avg distance between timeseries
            dist_avg_different, dist_avg_same = compute_avg_distances((inputs_noise, outputs_white_noise['system_1']),
                                                                      (inputs_noise, outputs_white_noise['system_2']))
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


def test_clustering_with_different_initial_conditions(sigma):
    n_inputs_signals = 50
    n_samples = [2**8, 2**10, 2**12, 2**14, 2**16]
    sys_1, sys_2 = init_circuits()
    r = 3   # repetitions for each test with a specific length

    # random initial conditions
    x0 = np.random.normal(0, sigma, size=(3, n_inputs_signals))
    ari_with_different_length = {}
    distance_with_different_lengths_same_system = {}
    distance_with_different_lengths_different_systems = {}
    for n in n_samples:
        ari = []
        distance_same = []
        distance_different = []
        for _ in range(r):
            inputs = generate_white_noise_signal(n, 0, 0.6, n_inputs_signals)
            outputs_1 = simulate_circuit_on_multi_input_with_x0(inputs, sys_1, x0)
            outputs_2 = simulate_circuit_on_multi_input_with_x0(inputs, sys_2, x0)

            # compute avg distance between timeseries
            dist_avg_different, dist_avg_same = compute_avg_distances((inputs, outputs_1),
                                                                      (inputs, outputs_2))
            distance_same.append(dist_avg_same)
            distance_different.append(dist_avg_different)

            dataset, true_clusters = generate_dataset_circuit(inputs, outputs_1, outputs_2)
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


def compute_mean_and_std(ari_values):
    results = {}
    for key, ari_values in ari_values.items():
        mean = np.mean(ari_values)
        dev_std = np.std(ari_values)
        results[key] = {
            'mean': mean,
            'dev_std': dev_std
        }

    return results


def plot_mean_and_std(results, title, x_label, y_label):
    keys = list(results.keys())
    means = [result['mean'] for result in results.values()]
    dev_std = [result['dev_std'] for result in results.values()]

    lower_errors = []
    upper_errors = []
    for m, s in zip(means, dev_std):
        lower = max(0, m - s)
        upper = min(1, m + s)
        lower_errors.append(m - lower)
        upper_errors.append(upper - m)

    asymmetric_errors = [lower_errors, upper_errors]

    plt.figure(figsize=(8, 6))
    plt.bar(
        [str(k) for k in keys],
        means,
        capsize=5,
        width=0.6,
        color='skyblue',
        ecolor='red'
    )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
