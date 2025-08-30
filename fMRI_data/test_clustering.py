from fMRI_data.fMRI_data_generator import (generate_estimated_system,
                                           generate_input_from_visual_cue_times,
                                           simulate_estimated_statespace_system,
                                           get_systems_with_different_norms,
                                           compute_norm_for_all_systems,
                                           generate_input_with_single_channel_active)
import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from mimo_systems.cepstral_distance_mimo import compute_cepstral_distance, mimo_distance_single_input_active
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt


def compute_distance_matrix(dataset_in, dataset_out):
    length = len(dataset_in)

    dm = np.zeros((length, length))
    for i in range(length):
        print(f'riga {i}')
        for j in range(i+1, length):
            print(f'colonna {j}')
            dist = compute_cepstral_distance(dataset_in[i], dataset_out[i], dataset_in[j], dataset_out[j], regularized=True)
            dm[i, j] = dm[j, i] = dist

    return dm


def compute_distance_matrix_single_input_active(dataset_in, dataset_out):
    length = len(dataset_in)

    dm = np.zeros((length, length))
    for i in range(length):
        channel_active_i = np.nonzero(dataset_in[i])[0][0]
        print(f'riga {i}')
        for j in range(i + 1, length):
            channel_active_j = np.nonzero(dataset_in[j])[0][0]
            dist = mimo_distance_single_input_active(dataset_in[i][channel_active_i], dataset_out[i], dataset_in[j][channel_active_j], dataset_out[j])
            dm[i, j] = dm[j, i] = dist

    return dm


def test_clustering():
    # id con dinamiche uguali 100206, 100610, 101006, 517239, 520228, 524135, 525541, 667056
    # id_systems = (100206, 101309, 756055) # 695768
    norms = compute_norm_for_all_systems()
    id_systems = get_systems_with_different_norms(8, 50, norms).keys()
    # id_systems = (100206, 667056)
    # id_systems = get_distant_systems(10, 7)
    data_per_cluster = 10
    systems = []

    for id_system in id_systems:
        systems.append(generate_estimated_system(id_system))

    # generate 20 different inputs from 3 different visual cues

    visual_cues = [
        [12, 33, 54, 75, 96, 138, 159, 180, 221, 242],
        [10, 34, 54, 78, 97, 140, 159, 178, 222, 244],
        [12, 32, 54, 75, 96, 140, 159, 180, 220, 242]]

    inputs = np.zeros((data_per_cluster, 6, 284), dtype=float)
    for i in range(data_per_cluster):
        inputs[i] = generate_input_from_visual_cue_times(visual_cues[0])

    execute_and_evaluate_clustering(systems, data_per_cluster, inputs)


def test_clustering_single_input_activated():
    id_systems = (100206, 101309, 756055, 100610, 525541, 667056, 101006, 517239)

    systems = []

    for id_system in id_systems:
        systems.append(generate_estimated_system(id_system))

    visual_cues = [
        [12, 33, 54, 75, 96, 138, 159, 180, 221, 242],
        [10, 34, 54, 78, 97, 140, 159, 178, 222, 244, 260],
        [12, 32, 54, 75, 96, 140, 159, 180, 220, 242, 256],
        [15, 31, 50, 72, 90, 130, 153, 172, 220, 260],
        [10, 27, 45, 72, 93, 120, 153, 170, 230, 270],
        #[7, 21, 44, 68, 91, 130, 155, 178, 220, 245],
        #[11, 35, 50, 78, 102, 142, 160, 188, 231, 255],
        #[15, 30, 56, 80, 105, 135, 158, 185, 210, 240, 265],
        #[9, 28, 49, 75, 95, 125, 148, 173, 205, 235],
        #[18, 42, 63, 85, 110, 145, 165, 190, 225, 250, 270],
        #[13, 33, 58, 81, 108, 138, 162, 185, 215, 245],
        #[20, 45, 65, 90, 115, 140, 165, 195, 230, 255],
        #[10, 25, 50, 75, 100, 130, 150, 175, 200, 225, 250],
        #[14, 38, 60, 88, 112, 137, 164, 192, 218, 242, 268],
        #[22, 48, 70, 95, 125, 150, 175, 205, 235, 260]
    ]

    inputs = np.zeros((len(visual_cues), 6, 284), dtype=float)
    data_per_cluster = len(visual_cues)

    for i in range(data_per_cluster):
        n_channel = 5
        inputs[i] = generate_input_with_single_channel_active(n_channel, visual_cues[i])

    execute_and_evaluate_clustering(systems, data_per_cluster, inputs)


def execute_and_evaluate_clustering(systems, data_per_cluster, inputs):
    outputs = np.zeros((len(systems), data_per_cluster, 148, 284))

    for i in range(len(systems)):
        for j in range(data_per_cluster):
            outputs[i, j] = simulate_estimated_statespace_system(systems[i], inputs[j])[1]

    # dataset creation
    dataset_in = np.tile(inputs, (len(systems), 1, 1))
    dataset_out = np.zeros((data_per_cluster * len(systems), 148, 284))

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
    dm = compute_distance_matrix_single_input_active(dataset_in, dataset_out)

    # KMedoids
    model_km = KMedoids(n_clusters=len(systems), metric="precomputed", random_state=0, max_iter=1000)
    predicted_clusters_km = model_km.fit_predict(dm)
    print(f"KMedoids ARI: {adjusted_rand_score(true_clusters, predicted_clusters_km)}")
    silhouette_kmd = silhouette_score(dm, predicted_clusters_km, metric='precomputed')
    print(f"Silhouette Score per KMedoids: {silhouette_kmd}")

    # Agglomerative
    agg_clustering = AgglomerativeClustering(n_clusters=len(systems), metric='precomputed', linkage='complete')
    predicted_clusters_agg = agg_clustering.fit_predict(dm)
    print(f"Agglomerative ARI: {adjusted_rand_score(true_clusters, predicted_clusters_agg)}")
    silhouette_agg = silhouette_score(dm, predicted_clusters_agg, metric='precomputed')

    print(f"Silhouette Score per Agglomerative: {silhouette_agg}")
    cm = confusion_matrix(true_clusters, predicted_clusters_km)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion matrix Agg.")
    plt.show()
