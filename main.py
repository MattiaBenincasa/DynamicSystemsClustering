import numpy as np
from electric_circuits.test_clustering import test_two_circuits_clustering, test_increasing_noise_intensity_clustering, test_clustering_with_different_initial_conditions
from fMRI.init_data import load_data, preprocess_output_data
from input_analysis import count_1s_in_input, find_differences, group_equal_matrices
from mimo_systems.power_cepstrum import compute_cepstral_distance
from mimo_systems.mimo_system import compute_distance_between_mimo_systems
from mimo_systems.test_clustering import test_clustering_two_mimo_systems
import time

from fMRI_data.fMRI_data_generator import (
    generate_estimated_system,
    simulate_estimated_statespace_system,
    generate_input_from_visual_cue_times,
)
from fMRI_data.test_clustering import test_clustering

# SISO -> circuit
# test_two_circuits_clustering()
# test_increasing_noise_intensity_clustering(2**14)
# test_clustering_with_different_initial_conditions(1)

# MIMO System
# compute_distance_between_mimo_systems()
# test_clustering_two_mimo_systems()

sys_1 = generate_estimated_system("100206")
sys_2 = generate_estimated_system("756055")

# generate different inputs
in_1 = generate_input_from_visual_cue_times([12, 33, 54, 75, 96, 138, 159, 180, 221, 242])
in_2 = generate_input_from_visual_cue_times([12, 33, 54, 75, 96, 138, 159, 180, 221, 242])

# print(f'Determinante: {np.linalg.det(in_1)}')
# simulate sys_1 and sys_2 on in_1
t_1, y_sim_1 = simulate_estimated_statespace_system(sys_1, in_1)
t_2, y_sim_2 = simulate_estimated_statespace_system(sys_2, in_1)

# simulate sys_1 on in_2 to have a different output for sys_1
t_3, y_sim_3 = simulate_estimated_statespace_system(sys_1, in_2)
start_time = time.time()
print(f'Different system: {compute_cepstral_distance(in_1, y_sim_1, in_1, y_sim_2, eps=1e-15)}')
end_time = time.time()
print(f'Same system: {compute_cepstral_distance(in_1, y_sim_1, in_2, y_sim_3, eps=1e-15)}')

print(f"Execution time: {end_time-start_time} sec")

test_clustering()
