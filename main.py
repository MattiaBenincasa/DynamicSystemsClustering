from electric_circuits.test_clustering import test_two_circuits_clustering, test_increasing_noise_intensity
from fMRI.init_data import load_data, preprocess_data
from distance_measures import extended_cepstral_distance_mimo
from plot_cepstral import plot_cepstral_distance
from mimo_systems.power_cepstrum import compute_cepstral_distance
from mimo_systems.mimo_system import compute_distance_between_mimo_systems
from mimo_systems.test_clustering import test_clustering_two_mimo_systems

# SISO -> circuit
# test_two_circuits_clustering()
# test_increasing_noise_intensity()
# plot_cepstral_distance()

# MIMO System
# compute_distance_between_mimo_systems()
# test_clustering_two_mimo_systems()

# fMRI data
# subject_ids, outputs, inputs, heart, resp = load_data()
# outputs_rf = preprocess_data(outputs['100206'], heart['100206'], resp['100206'])

# u_1 = inputs['100206']
# u_2 = inputs['102311']
# y_1 = outputs['100206']
# y_2 = outputs['102311']

# print(extended_cepstral_distance_mimo(u_1, y_1, u_2, y_2))
# print(compute_cepstral_distance(u_1, y_1, u_2, y_2))
