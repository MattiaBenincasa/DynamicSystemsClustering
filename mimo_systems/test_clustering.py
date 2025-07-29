from mimo_systems.mimo_system import (get_two_mimo_systems,
                                      generate_white_noise_signal,
                                      simulate_system_on_multiple_input)
from clustering import generate_dataset_mimo_systems, k_means_mimo
from sklearn.metrics.cluster import adjusted_rand_score


def test_clustering_two_mimo_systems():
    sys_1, sys_2 = get_two_mimo_systems()

    inputs = generate_white_noise_signal(2 ** 14, 2, 0, 1, 50)
    outputs_1 = simulate_system_on_multiple_input(inputs, sys_1)
    outputs_2 = simulate_system_on_multiple_input(inputs, sys_2)
    print('Data computed')

    input_ds, output_ds, true_clusters = generate_dataset_mimo_systems(inputs, outputs_1, outputs_2)
    print('Dataset created -> kmeans starts')
    centroids_in, centroids_out, predicted_clusters = k_means_mimo(input_ds, output_ds, 2, tol=1e-5)
    print(f'ARI index: {adjusted_rand_score(true_clusters, predicted_clusters)}')
