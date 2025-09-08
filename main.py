import numpy as np
from electric_circuits.test_clustering import test_1, test_2, test_3, test_4, test_5
from mimo_systems.cepstral_distance_mimo import compute_cepstral_distance, mimo_distance_single_input_active
from mimo_systems.mimo_system import compute_distance_between_mimo_systems
from mimo_systems.test_clustering import test_clustering_two_mimo_systems
from cepstral_distance_siso import extended_cepstral_distance

import time
import control as ct
from theoretical_cepstrum.siso_cepstrum import poles_and_zeros_distance, poles_zeros_norm
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
    print_distance_between_systems
)
from fMRI_data.test_clustering import test_clustering, test_clustering_single_input_activated

# SISO -> circuit
# test_1()
# test_2()
# test_3()
# test_4()
# test_5()
# MIMO System
# compute_distance_between_mimo_systems()
# test_clustering_two_mimo_systems()
# print_distance_between_systems("100610", "667056", [12, 33, 54, 75, 96, 138, 159, 180, 221, 242], 10)
# print_distance_between_systems("100206", "101309", [12, 33, 54, 75, 96, 138, 159, 180, 221, 242], 10)
# test_clustering(6, 50, 5)
# test_clustering(6, 50, 10)
# test_clustering(6, 50, 20)
# test_clustering(8, 35, 5)
# test_clustering(8, 35, 10)
# test_clustering(8, 35, 20)
# test_clustering(10, 25, 5)
# test_clustering(10, 25, 10)
# test_clustering(10, 25, 20)
# test_clustering(None, None, 15)
# test_clustering_single_input_activated()
