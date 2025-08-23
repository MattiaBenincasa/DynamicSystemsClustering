from statistics import mean
import numpy as np
from matplotlib import pyplot as plt

from cepstral_distance_siso import extended_cepstral_distance


def compute_distance_matrix(in_out_1, in_out_2):
    n = len(in_out_1[0])
    m = len(in_out_2[0])
    dm = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            dm[i, j] = extended_cepstral_distance(in_out_1[0][i], in_out_1[1][i], in_out_2[0][j], in_out_2[1][j])

    return dm


def avg_ignore_diagonal_elements(square_matrix):
    non_diag_elements = []
    n = len(square_matrix)
    for i in range(n):
        for j in range(n):
            if i != j:
                non_diag_elements.append(square_matrix[i, j])

    return mean(non_diag_elements)


# this function return average distance between time series
# from same system and different systems
def compute_avg_distances(in_out_1, in_out_2):
    dm_different = compute_distance_matrix(in_out_1, in_out_2)
    dm_same_1 = compute_distance_matrix(in_out_1, in_out_1)

    # compute mean values
    avg_different = dm_different.mean()
    avg_same = avg_ignore_diagonal_elements(dm_same_1)

    return avg_different, avg_same


def plot_distance_comparison(results_same, results_different, title, x_label, y_label):
    keys = list(results_same.keys())

    means_same = [result['mean'] for result in results_same.values()]
    dev_std_same = [result['dev_std'] for result in results_same.values()]
    means_different = [result['mean'] for result in results_different.values()]
    dev_std_different = [result['dev_std'] for result in results_different.values()]

    w, x = 0.4, np.arange(len(keys))
    plt.bar(x - w / 2, means_same, w, yerr=dev_std_same, capsize=5,  ecolor="red", label='same dynamic')
    plt.bar(x + w / 2, means_different, w, yerr=dev_std_different, capsize=5, ecolor="red", label='different dynamic')
    plt.xticks(x, [str(key) for key in keys])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


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