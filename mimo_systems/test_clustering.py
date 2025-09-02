from mimo_systems.mimo_system import (generate_white_noise_signal,
                                      simulate_system_on_multiple_input)
from clustering import generate_dataset_mimo_systems
from sklearn_extra.cluster import KMedoids
from sklearn.metrics.cluster import adjusted_rand_score
from fMRI_data.test_clustering import compute_distance_matrix
from theoretical_cepstrum.mimo_cepstrum import generate_stable_mimo_system
import numpy as np


def test_clustering_two_mimo_systems():
    np.random.seed(5)

    sys_1, poles_1, zeros_1 = generate_stable_mimo_system(2, 10, 10)
    sys_2, poles_2, zeros_2 = generate_stable_mimo_system(2, 10, 10)

    inputs = generate_white_noise_signal(2 ** 14, 2, 0, 1, 50)
    outputs_1 = simulate_system_on_multiple_input(inputs, sys_1)
    outputs_2 = simulate_system_on_multiple_input(inputs, sys_2)
    print('Data computed')

    input_ds, output_ds, true_clusters = generate_dataset_mimo_systems(inputs, outputs_1, outputs_2)
    dm = compute_distance_matrix(input_ds, output_ds)
    model_km = KMedoids(n_clusters=2, metric="precomputed", random_state=0, max_iter=1000)
    predicted_clusters = model_km.fit_predict(dm)
    print(f'ARI index: {adjusted_rand_score(true_clusters, predicted_clusters)}')
