import numpy as np
from electric_circuits.test_clustering import setup_and_execute_tests
from mimo_systems.cepstral_distance_mimo import compute_cepstral_distance
from mimo_systems.mimo_system import compute_distance_between_mimo_systems
from mimo_systems.test_clustering import test_clustering_two_mimo_systems
from cepstral_distance_siso import extended_cepstral_distance

import time
import control as ct
from theoretical_cepstrum.siso_cepstrum import poles_and_zeros_distance
from fMRI_data.fMRI_data_generator import (
    generate_estimated_system,
    simulate_estimated_statespace_system,
    generate_input_from_visual_cue_times,
    generate_input_from_another_input,
    generate_input_with_single_channel_active,
    compute_norm_for_all_systems,
    plot_cepstrum_fMRI_systems,
    get_systems_with_different_norms,
    get_distant_systems,
)
from fMRI_data.test_clustering import test_clustering

# SISO -> circuit
setup_and_execute_tests()

# MIMO System
# compute_distance_between_mimo_systems()
# test_clustering_two_mimo_systems()

sys_1 = generate_estimated_system("100206")
sys_2 = generate_estimated_system("756055")

# generate different inputs
in_1 = generate_input_from_visual_cue_times([12, 33, 54, 75, 96, 138, 159, 180, 221, 242])
in_2 = generate_input_from_visual_cue_times([12, 33, 54, 75, 96, 138, 159, 180, 221, 242])
# in_1 = generate_input_with_single_channel_active(1, [12, 33, 54, 75, 96, 138, 159, 180, 221, 242])
# in_2 = generate_input_with_single_channel_active(1, [15, 35, 57, 81, 112, 141, 163, 186, 224, 252])
# print(f'Determinante: {np.linalg.det(in_1)}')
# simulate sys_1 and sys_2 on in_1
t_1, y_sim_1 = simulate_estimated_statespace_system(sys_1, in_1)
t_2, y_sim_2 = simulate_estimated_statespace_system(sys_2, in_1)

# simulate sys_1 on in_2 to have a different output for sys_1
t_3, y_sim_3 = simulate_estimated_statespace_system(sys_1, in_2)
start_time = time.time()
print(f'Different system: {compute_cepstral_distance(in_1, y_sim_1, in_1, y_sim_2, eps=0)}')
end_time = time.time()
print(f'Same system: {compute_cepstral_distance(in_1, y_sim_1, in_2, y_sim_3, eps=0)}')

print(f"Execution time: {end_time-start_time} sec")
print("--------------------------------------------------------")
dist_different = np.zeros(148)
dist_same = np.zeros(148)


# fill different systems distance
'''for j in range(148):
    dist_different[j] = extended_cepstral_distance(in_1[1], y_sim_1[j], in_1[1], y_sim_2[j])

# fill same systems distance
for j in range(148):
    dist_same[j] = extended_cepstral_distance(in_1[1], y_sim_1[j], in_2[1], y_sim_3[j])

difference = dist_different-dist_same
significant_regions = np.where(difference > 80)
print(f'Significant regions: {significant_regions}')
print(dist_different[significant_regions])
print(dist_same[significant_regions])

norms = compute_norm_for_all_systems()
systems = get_systems_with_different_norms(5, 80, norms)
print(systems)'''

poles_1, zeros_1 = ct.poles(sys_1), ct.zeros(sys_1)
poles_2, zeros_2 = ct.poles(sys_2), ct.zeros(sys_2)
distance_from_p_z = poles_and_zeros_distance(poles_1, zeros_1, poles_2, zeros_2)
# print(f"Distance with poles and zeros: {distance_from_p_z}")
# plot_cepstrum_fMRI_systems(in_1, y_sim_1, "100206")
# plot_cepstrum_fMRI_systems(in_2, y_sim_3, "100206")
test_clustering()

# systems = get_distant_systems(10, 10)
# print(systems.keys())
