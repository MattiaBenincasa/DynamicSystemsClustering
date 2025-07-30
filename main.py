import numpy as np
from electric_circuits.test_clustering import test_two_circuits_clustering, test_increasing_noise_intensity, test_clustering_with_different_initial_conditions
from fMRI.init_data import load_data, preprocess_output_data
from plot_cepstral import plot_cepstral_distance, test_different_initial_conditions
from mimo_systems.power_cepstrum import compute_cepstral_distance, compute_distance_matrix_fMRI
from mimo_systems.mimo_system import compute_distance_between_mimo_systems
from mimo_systems.test_clustering import test_clustering_two_mimo_systems

# SISO -> circuit
# test_two_circuits_clustering()
# test_increasing_noise_intensity()
# plot_cepstral_distance()
# test_different_initial_conditions()
test_clustering_with_different_initial_conditions(1.5)
# MIMO System
# compute_distance_between_mimo_systems()
# test_clustering_two_mimo_systems()

# fMRI data
# subject_ids, outputs, inputs, heart, resp = load_data()
# outputs_rf = preprocess_output_data(subject_ids, outputs, heart, resp)

# u_1 = inputs['100206']
# u_2 = inputs['102311']
# y_1 = outputs_rf['100206']
# y_2 = outputs_rf['102311']

# print(f'Filtered data {compute_cepstral_distance(u_1, y_1, u_2, y_2, eps=1e-12)}')
# print(f'Non Filtered data {compute_cepstral_distance(u_1, outputs["100206"], u_2, outputs["102311"], eps=1e-12)}')

# d = compute_distance_matrix_fMRI(subject_ids, inputs, outputs_rf)
# d = np.load('distance_matrix.npy')
# print(d[:, 0])  # these are the distance of the first ID from all the others IDs

# print(f'Max value: {np.max(d[:, 0])}')
# print(f'Min value: {np.min(d[1:, 0])}')
