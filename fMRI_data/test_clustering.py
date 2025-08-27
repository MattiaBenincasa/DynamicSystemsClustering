import random

from fMRI_data.fMRI_data_generator import (generate_estimated_system,
                                           generate_input_from_visual_cue_times,
                                           simulate_estimated_statespace_system,
                                           generate_input_from_another_input, get_systems_with_different_norms,
                                           compute_norm_for_all_systems, get_distant_systems)
import numpy as np
from sklearn.metrics.cluster import adjusted_rand_score
from mimo_systems.cepstral_distance_mimo import compute_cepstral_distance
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering


def compute_distance_matrix(dataset_in, dataset_out):
    length = len(dataset_in)

    dm = np.zeros((length, length))
    for i in range(length):
        print(f'riga {i}')
        for j in range(i+1, length):
            print(f'colonna {j}')
            dist = compute_cepstral_distance(dataset_in[i], dataset_out[i], dataset_in[j], dataset_out[j], eps=1e-14)
            dm[i, j] = dm[j, i] = dist

    return dm


def test_clustering():
    # np.random.seed(seed=42)
    # random.seed(42)
    # id con dinamiche uguali 100206, 100610, 101006, 517239, 520228, 524135, 525541, 667056
    # id_systems = (100206, 101309, 756055) # 695768
    norms = compute_norm_for_all_systems()
    id_systems = get_systems_with_different_norms(8, 45, norms).keys()
    # id_systems = (100206, 667056)
    #id_systems = get_distant_systems(10, 7)
    data_per_cluster = 20
    systems = []

    for id_system in id_systems:
        systems.append(generate_estimated_system(id_system))

    # generate 20 different inputs from 3 different visual cues

    visual_cues = [
        [12, 33, 54, 75, 96, 138, 159, 180, 221, 242],
        [10, 34, 54, 78, 97, 140, 159, 178, 222, 244],
        [12, 32, 54, 75, 96, 140, 159, 180, 220, 242]]

    inputs = np.zeros((data_per_cluster, 6, 284), dtype=float)
    base_input = generate_input_from_visual_cue_times(visual_cues[0])
    for i in range(data_per_cluster):
        inputs[i] = generate_input_from_visual_cue_times(visual_cues[0])
        # inputs[i] = generate_input_from_another_input(base_input)

    # for i in range(10, 15):
    #    inputs[i] = generate_input_from_visual_cue_times(visual_cues[1])

    # for i in range(15, 20):
    #     inputs[i] = generate_input_from_visual_cue_times(visual_cues[2])

    # 20 outputs for 20 systems
    outputs = np.zeros((len(systems), data_per_cluster, 148, 284))

    for i in range(len(systems)):
        for j in range(data_per_cluster):
            outputs[i, j] = simulate_estimated_statespace_system(systems[i], inputs[j])[1]

    # dataset creation
    dataset_in = np.tile(inputs, (len(systems), 1, 1))
    dataset_out = np.zeros((data_per_cluster*len(systems), 148, 284))

    k = 0
    for i in range(len(systems)):
        for j in range(data_per_cluster):
            dataset_out[k] = outputs[i, j]
            k += 1

    clusters_numbers = np.arange(len(systems))
    true_clusters = np.repeat(clusters_numbers, data_per_cluster)
    permutation = np.random.permutation(len(dataset_in))

    # apply permutation to dataset in e out and cluster labels
    true_clusters = true_clusters[permutation]
    dataset_in = dataset_in[permutation]
    dataset_out = dataset_out[permutation]

    # execute clustering
    dm = compute_distance_matrix(dataset_in, dataset_out)
    model = KMedoids(n_clusters=len(systems), metric="precomputed", random_state=0, max_iter=1000)
    predicted_clusters_km = model.fit_predict(dm)
    agg_clustering = AgglomerativeClustering(n_clusters=len(systems), metric='precomputed', linkage='complete')
    predicted_clusters_agg = agg_clustering.fit_predict(dm)
    print(f"KMedoids ARI: {adjusted_rand_score(true_clusters, predicted_clusters_km)}")
    print(f"Agglomerative ARI: {adjusted_rand_score(true_clusters, predicted_clusters_agg)}")
