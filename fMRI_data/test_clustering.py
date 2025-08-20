import random

from fMRI_data.fMRI_data_generator import (generate_estimated_system,
                                           generate_input_from_visual_cue_times,
                                           simulate_estimated_statespace_system)
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from mimo_systems.power_cepstrum import compute_cepstral_distance
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import linkage, fcluster


def compute_distance_matrix(dataset_in, dataset_out):
    length = len(dataset_in)

    dm = np.zeros((length, length))
    for i in range(length):
        print(f'riga {i}')
        for j in range(i+1, length):
            print(f'colonna {j}')
            dist = compute_cepstral_distance(dataset_in[i], dataset_out[i], dataset_in[j], dataset_out[j], eps=1e-15)
            dm[i, j] = dm[j, i] = dist

    return dm


def test_clustering():

    # id_systems = (100206, 100610, 101006, 101309, 101915, 510326, 517239, 520228, 524135, 525541)
    id_systems = (100206, 520228)
    systems = []

    for id_system in id_systems:
        systems.append(generate_estimated_system(id_system))

    # generate 20 different inputs from 3 different visual cues

    visual_cues = [
        [12, 33, 54, 75, 96, 138, 159, 180, 221, 242],
        [10, 34, 54, 78, 97, 140, 159, 178, 222, 244],
        [12, 32, 54, 75, 96, 140, 159, 180, 220, 242]]

    inputs = np.zeros((10, 6, 284), dtype=float)

    for i in range(10):
        inputs[i] = generate_input_from_visual_cue_times(visual_cues[0])

    # for i in range(10, 15):
    #    inputs[i] = generate_input_from_visual_cue_times(visual_cues[1])

    # for i in range(15, 20):
    #     inputs[i] = generate_input_from_visual_cue_times(visual_cues[2])

    # 20 outputs for 20 systems
    outputs = np.zeros((len(systems), 10, 148, 284))

    for i in range(len(systems)):
        for j in range(10):
            outputs[i, j] = simulate_estimated_statespace_system(systems[i], inputs[j])[1]

    # dataset creation
    dataset_in = np.tile(inputs, (len(systems), 1, 1))
    dataset_out = np.zeros((10*len(systems), 148, 284))

    k = 0
    for i in range(len(systems)):
        for j in range(10):
            dataset_out[k] = outputs[i, j]
            k += 1

    clusters_numbers = np.arange(len(systems))
    true_clusters = np.repeat(clusters_numbers, 10)
    permutation = np.random.permutation(len(dataset_in))

    # apply permutation to dataset in e out and cluster labels
    true_clusters = true_clusters[permutation]
    dataset_in = dataset_in[permutation]
    dataset_out = dataset_out[permutation]

    # execute clustering
    # cen_in, cen_out, predicted_clusters = k_means_mimo(dataset_in, dataset_out, 2)
    dm = compute_distance_matrix(dataset_in, dataset_out)
    # print(np.max(dm))
    # print(np.min(dm))
    model = KMedoids(n_clusters=len(systems), metric="precomputed", random_state=0, max_iter=1000)
    predicted_clusters = model.fit_predict(dm)
    print(f"ARI: {adjusted_rand_score(true_clusters, predicted_clusters)}")
