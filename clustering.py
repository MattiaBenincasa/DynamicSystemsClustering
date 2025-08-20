import numpy as np
from distance_measures import extended_cepstral_distance
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mimo_systems.power_cepstrum import compute_cepstral_distance


def generate_dataset_circuit(inputs, outputs_1, outputs_2):
    data_sys_1 = [(u, y) for u, y in zip(inputs, outputs_1)]
    data_sys_2 = [(u, y) for u, y in zip(inputs, outputs_2)]
    dataset = np.array(data_sys_1 + data_sys_2)
    true_clusters = np.array([0]*len(inputs) + [1]*len(inputs))
    permutation = np.random.permutation(len(dataset))
    return dataset[permutation], true_clusters[permutation]


def assign_clusters(dataset, centroids):
    clusters = []

    for point in dataset:
        distances = [extended_cepstral_distance(point[0], point[1], centroid[0], centroid[1]) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)

    return np.array(clusters)


def update_centroids(dataset, clusters, k):
    new_centroids = []
    for i in range(k):
        cluster_points = dataset[clusters == i]
        if len(cluster_points) == 0:
            print('Empty cluster')
        else:
            new_centroid = cluster_points.mean(axis=0)
            new_centroids.append(new_centroid)
    return np.array(new_centroids)


def k_means(dataset, k, max_iters=500, tol=1e-8):
    centroids_indices = np.random.choice(len(dataset), size=k, replace=False)
    centroids = [dataset[i] for i in centroids_indices]
    clusters = []

    for i in range(max_iters):
        clusters = assign_clusters(dataset, centroids)
        new_centroids = update_centroids(dataset, clusters, k)
        if np.all(np.abs(new_centroids - centroids) < tol):
            break
        centroids = new_centroids
    return centroids, clusters


def compute_and_plot_conf_matrix(true_labels, pred_labels, title_plot=None):
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion matrix: {title_plot}")
    plt.show()


# CLUSTERING ADAPTED FOR MIMO DATASET

def generate_dataset_mimo_systems(inputs, outputs_1, outputs_2):
    inputs = [u.T for u in inputs]
    inputs_ds = np.array(inputs + inputs)
    outputs_ds = np.array(outputs_1 + outputs_2)
    true_clusters = np.array([0]*len(inputs) + [1]*len(inputs))
    permutation = np.random.permutation(len(inputs_ds))
    return inputs_ds[permutation], outputs_ds[permutation], true_clusters[permutation]


def assign_clusters_mimo(inputs, outputs, cen_in, cen_out, k):
    clusters = []
    n_points = len(inputs)
    for i in range(n_points):
        distances = [compute_cepstral_distance(inputs[i], outputs[i], cen_in[j], cen_out[j], eps=1e-14) for j in range(k)]
        cluster = np.argmin(distances)
        clusters.append(cluster)

    return np.array(clusters)


def update_centroids_mimo(inputs, outputs, clusters, k):
    new_centroids_in = []
    new_centroids_out = []
    for i in range(k):
        cluster_points_in = inputs[clusters == i]
        cluster_points_out = outputs[clusters == i]
        if len(cluster_points_in) == 0 or len(cluster_points_out) == 0:
            print('Empty cluster')
        else:
            new_cen_in = cluster_points_in.mean(axis=0)
            new_cen_out = cluster_points_out.mean(axis=0)
            new_centroids_in.append(new_cen_in)
            new_centroids_out.append(new_cen_out)
    return np.array(new_centroids_in), np.array(new_centroids_out)


def k_means_mimo(inputs, outputs, k, max_iters=500, tol=1e-6):
    centroids_indices = np.random.choice(len(inputs), size=k, replace=False)
    cen_in = [inputs[i] for i in centroids_indices]
    cen_out = [outputs[i] for i in centroids_indices]
    clusters = []

    for i in range(max_iters):
        print(f'Iteration: {i}')
        clusters = assign_clusters_mimo(inputs, outputs, cen_in, cen_out, k)
        new_cen_in, new_cen_out = update_centroids_mimo(inputs, outputs, clusters, k)
        if np.all(np.abs(new_cen_in - cen_in) < tol) and np.all(np.abs(new_cen_out - cen_out) < tol):
            break
        cen_in = new_cen_in
        cen_out = new_cen_out
    return cen_in, cen_out, clusters
