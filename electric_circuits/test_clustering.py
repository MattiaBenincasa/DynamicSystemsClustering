from electric_circuits.electric_circuits import (generate_discrete_lti_circuit,
                                                 generate_white_noise_signal,
                                                 simulate_circuit_on_multiple_input,
                                                 generate_sinusoidal_signal,
                                                 simulate_circuit_on_multi_input_with_x0,
                                                 generate_multiple_multi_sin_waves,
                                                 simulate_circuit_on_multiple_inputs_with_output_noise)
from clustering import generate_dataset_circuit, k_means
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
from electric_circuits.utils_test_cepstral_distance import (compute_avg_distances,
                                                            plot_distance_comparison,
                                                            plot_ari_indices, compute_mean_and_std,
                                                            save_distance_results_into_latex_table)
from cepstral_distance_siso import extended_cepstral_distance


def compute_distance_matrix(dataset):
    length = len(dataset)
    dm = np.zeros((length, length))

    for i in range(length):
        for j in range(i+1, length):
            dist = extended_cepstral_distance(dataset[i][0], dataset[i][1], dataset[j][0], dataset[j][1])
            dm[i, j] = dm[j, i] = dist
        print(f"column {i} computed")

    return dm


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


def test_two_circuits_clustering(sys_1, sys_2, n_samples, r, inputs, label_plot):

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
            dm = compute_distance_matrix(dataset)
            model_km = KMedoids(n_clusters=2, metric="precomputed", random_state=0, max_iter=1000)
            predicted_clusters = model_km.fit_predict(dm)
            ari.append(adjusted_rand_score(true_clusters, predicted_clusters))
            print(f"test {i} computed")
        ari_with_different_length[n] = ari
        distance_with_different_lengths_same_system[n] = distance_same
        distance_with_different_lengths_different_systems[n] = distance_different
        print(f'{n} computed')
    results_same = compute_mean_and_std(distance_with_different_lengths_same_system)
    results_different = compute_mean_and_std(distance_with_different_lengths_different_systems)
    plot_distance_comparison(results_same,
                             results_different,
                             title=f"{label_plot} - Confronto delle distanze",
                             x_label="lunghezza serie temporali",
                             y_label="valore distanza")

    save_distance_results_into_latex_table(results_same, results_different, f"{label_plot} - Confronto delle distanze")

    plot_ari_indices(compute_mean_and_std(ari_with_different_length),
                     title=f"{label_plot} - Valutazione ARI clustering",
                     x_label="lunghezza serie temporali",
                     y_label="indice ARI")


def test_increasing_noise_intensity_clustering(sys_1, sys_2, snr_values, r, inputs, label_plot):

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

            dm = compute_distance_matrix(dataset_noise)
            model_km = KMedoids(n_clusters=2, metric="precomputed", random_state=0, max_iter=1000)
            predicted_clusters_noise = model_km.fit_predict(dm)
            ari.append(adjusted_rand_score(true_clusters_noise, predicted_clusters_noise))
            print(f'SNR {snr} computed - test n. {i}')
        ari_with_different_snr[snr] = ari
        distance_with_different_snr_same_system[snr] = distance_same
        distance_with_different_snr_different_system[snr] = distance_different
    results_same = compute_mean_and_std(distance_with_different_snr_same_system)
    results_different = compute_mean_and_std(distance_with_different_snr_different_system)
    plot_distance_comparison(results_same,
                             results_different,
                             title=f"{label_plot} - Confronto delle distanze",
                             x_label="Rumore di misura SNR (dB)",
                             y_label="valore distanza")

    save_distance_results_into_latex_table(results_same, results_different, f"{label_plot} - Confronto delle distanze")

    plot_ari_indices(compute_mean_and_std(ari_with_different_snr),
                     title=f"{label_plot} - Valutazione ARI clustering",
                     x_label="Rumore di misura SNR (dB)",
                     y_label="indice ARI")


def test_clustering_with_different_initial_conditions(sys_1, sys_2, intensity_scale, n_samples, r, inputs, n_inputs, label_plot):
    # random initial conditions
    x0 = intensity_scale*np.random.normal(0, 0.4, size=(3, n_inputs))
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
            dm = compute_distance_matrix(dataset)
            model_km = KMedoids(n_clusters=2, metric="precomputed", random_state=0, max_iter=1000)
            predicted_clusters = model_km.fit_predict(dm)
            ari.append(adjusted_rand_score(true_clusters, predicted_clusters))
            print(f"test {i} computed - {n} samples")
        print(f'{n} computed')
        ari_with_different_length[n] = ari
        distance_with_different_lengths_same_system[n] = distance_same
        distance_with_different_lengths_different_systems[n] = distance_different
    results_same = compute_mean_and_std(distance_with_different_lengths_same_system)
    results_different = compute_mean_and_std(distance_with_different_lengths_different_systems)
    plot_distance_comparison(results_same,
                             results_different,
                             title=f"{label_plot} - Confronto delle distanze",
                             x_label="lunghezza serie temporali",
                             y_label="valore distanza")
    save_distance_results_into_latex_table(results_same, results_different, title=f"{label_plot} - Confronto delle distanze")
    plot_ari_indices(compute_mean_and_std(ari_with_different_length),
                     title=f"{label_plot} - Valutazione ARI clustering",
                     x_label="lunghezza serie temporali",
                     y_label="indice ARI")


def test_1():
    fs = 50
    sys_1, sys_2 = init_circuits(fs)
    n_samples = [2 ** 6, 2 ** 8, 2**10, 2**12, 2**14, 2**16]
    r = 10

    white_noise_inputs = {}

    for n in n_samples:
        white_noise_inputs[n] = [generate_white_noise_signal(n, 0, 0.6, 100) for _ in range(r)]

    test_two_circuits_clustering(sys_1, sys_2, n_samples, r, white_noise_inputs, label_plot="Test 1")


def test_2():
    fs = 50
    sys_1, sys_2 = init_circuits(fs)
    n_samples = [2 ** 6, 2**8, 2 ** 10, 2 ** 12, 2**14, 2**16]
    r = 10
    f1 = 1  # Hz
    f2 = 2  # Hz
    f3 = 5  # Hz

    sinusoidal_signals = {}

    for n in n_samples:
        sinusoidal_signals[n] = [generate_sinusoidal_signal(n, f1, fs=fs, n_signals=30) +
                                 generate_sinusoidal_signal(n, f2, fs=fs, n_signals=40) +
                                 generate_sinusoidal_signal(n, f3, fs=fs, n_signals=30) for _ in range(r)]

    test_two_circuits_clustering(sys_1, sys_2, n_samples, r, sinusoidal_signals, label_plot="Test 2")


def test_3():
    fs = 50
    sys_1, sys_2 = init_circuits(fs)
    n_samples = [2 ** 6, 2**8, 2**10, 2**12, 2**14, 2**16]
    r = 10
    f = 2  # Hz

    input_signals = {}

    for n in n_samples:
        input_signals[n] = [generate_sinusoidal_signal(n, f, fs=fs, n_signals=50) +
                            generate_white_noise_signal(n, 0, 0.6, n_signals=50) +
                            generate_multiple_multi_sin_waves(n, fs=fs, n_signals=50) for _ in range(r)]

    test_two_circuits_clustering(sys_1, sys_2, n_samples, r, input_signals, label_plot="Test 3")


def test_4():
    fs = 50
    sys_1, sys_2 = init_circuits(fs)
    n_samples = [2 ** 12, 2 ** 16]
    snr = [30, 25, 20, 15, 10]
    r = 10

    input_signals = {}

    for n in n_samples:
        for sn in snr:
            input_signals[sn] = [generate_white_noise_signal(n, 0, 0.7, n_signals=100) for _ in range(r)]

        test_increasing_noise_intensity_clustering(sys_1, sys_2, snr, r, input_signals, f'Test 4 - {n} campioni')


def test_5():
    fs = 50
    sys_1, sys_2 = init_circuits(fs)
    n_samples = [2 ** 10, 2**12, 2 ** 14]
    intensity_scale = [1, 10, 100]
    r = 10

    white_noise_inputs = {}

    for n in n_samples:
        white_noise_inputs[n] = [generate_white_noise_signal(n, 0, 0.6, 100) for _ in range(r)]

    for scale in intensity_scale:
        test_clustering_with_different_initial_conditions(sys_1, sys_2, scale, n_samples, r, white_noise_inputs, 100, f'Test 5 - k = {scale}')
