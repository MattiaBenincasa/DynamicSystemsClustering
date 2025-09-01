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
    n = len(square_matrix)
    diag_elements = ~np.eye(n, dtype=bool)
    return np.mean(square_matrix[diag_elements])


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
    bars_same = plt.bar(x - w / 2, means_same, w, yerr=dev_std_same, capsize=5,  ecolor="red", label='stessa dinamica')
    bars_different = plt.bar(x + w / 2, means_different, w, yerr=dev_std_different, capsize=5, ecolor="red", label='dinamica diversa')
    plt.xticks(x, [str(key) for key in keys])

    for i in range(len(bars_same)):
        dis = bars_same[i].get_height()

        plt.text(
            bars_same[i].get_x() + bars_same[i].get_width() / 6,
            dis,
            f'{dis:.3f}',
            ha='center',
            va='bottom'
        )

    for bar in bars_different:
        dis = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 6,
            dis,
            f'{dis:.3f}',
            ha='center',
            va='bottom'
        )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(f'result_plots/{title}.png', dpi=300)
    plt.show()


def save_distance_results_into_latex_table(results_same, results_different, title):
    keys = list(results_same.keys())
    means_same = [result['mean'] for result in results_same.values()]
    dev_std_same = [result['dev_std'] for result in results_same.values()]
    means_different = [result['mean'] for result in results_different.values()]
    dev_std_different = [result['dev_std'] for result in results_different.values()]

    with open(f"distance_results_latex_tables/{title}", "w") as f:
        f.write('\\begin{tabular}{ |c|c|c| }\n')
        f.write('\\hline\n')
        f.write('numero campioni & d(stessa dinamica) & d(dinamiche diverse) \\\\\n')
        f.write('\\hline\n')

        for i in range(len(keys)):
            same_dynamic = f"${means_same[i]:.3f}\\pm{dev_std_same[i]:.3f}$"
            different_dynamic = f"${means_different[i]:.3f}\\pm{dev_std_different[i]:.3f}$"
            f.write(f"{keys[i]} & {same_dynamic} & {different_dynamic} \\\\\n")

        f.write('\\hline\n')
        f.write('\\end{tabular}')


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


def plot_ari_indices(results, title, x_label, y_label):
    keys = list(results.keys())
    means = [result['mean'] for result in results.values()]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        [str(k) for k in keys],
        means,
        capsize=5,
        width=0.6,
        color='skyblue',
    )

    for bar in bars:
        ari = bar.get_height()

        if ari < 1.0:
            label = f'{ari:.2f}'
        else:
            label = ''

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            ari,
            label,
            ha='center',
            va='bottom'
        )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'result_plots/{title}.png', dpi=300)
    plt.show()
