import numpy as np
from distance_measures import extended_cepstral_distance
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
