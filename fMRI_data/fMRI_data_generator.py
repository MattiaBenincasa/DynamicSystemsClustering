import control
import numpy as np
import csv
import random
from control import ss, forced_response
from matplotlib import pyplot as plt
from theoretical_cepstrum.siso_cepstrum import poles_zeros_cepstrum, poles_and_zeros_distance, poles_zeros_norm
from mimo_systems.cepstral_distance_mimo import compute_cepstrum_transfer_function


def load_fMRI_matrix(subject_id):

    with open(f'fMRI_data/matrix_estimated/{subject_id}/Aest.csv', 'r') as f:
        reader = csv.reader(f)
        A = list(reader)

    with open(f'fMRI_data/matrix_estimated/{subject_id}/Best.csv', 'r') as f:
        reader = csv.reader(f)
        B = list(reader)

    with open(f'fMRI_data/matrix_estimated/{subject_id}/Cest.csv', 'r') as f:
        reader = csv.reader(f)
        C = list(reader)

    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)

    return A, B, C,


def generate_estimated_system(subject_id):
    Aest, Best, Cest = load_fMRI_matrix(subject_id)

    p = Cest.shape[0]
    m = Best.shape[1]
    return ss(Aest, Best, Cest, np.zeros((p, m)), True)


def simulate_estimated_statespace_system(estimate_system, u):
    t = np.arange(u.shape[1])
    t, y = forced_response(estimate_system, timepts=t, inputs=u)
    return t, y


def generate_input_from_visual_cue_times(cue_times, input_channels=6, length=284):
    visual_cue = np.zeros((input_channels, length))
    for cue_time in cue_times:
        random_channel = random.randint(1, 5)
        visual_cue[0, cue_time] = 1
        visual_cue[random_channel, cue_time+3] = 1

    return visual_cue


# other possible functions for generating data
def generate_variable_inputs_1(cue_times, input_channels=6, length=284):
    visual_cue = np.zeros((input_channels, length))
    for cue_time in cue_times:
        random_channel = random.randint(1, 5)
        random_cue_time = random.randint(1, 3)
        visual_cue[0, cue_time] = 1
        visual_cue[random_channel, random_cue_time] = 1

    return visual_cue


def generate_variable_inputs_2(input_channels=6, length=284):
    inputs = np.zeros((input_channels, length))
    for i in range(input_channels):
        for t in range(length):
            if random.random() < 0.05:
                inputs[i, t] = 1

    return inputs


def generate_input_from_another_input(inputs):
    new_inputs = inputs.copy()

    one_indices = np.argwhere(new_inputs[1:5, :] == 1)
    indices = random.choice(one_indices)
    new_inputs[indices[0]+1, indices[1]] = 0
    new_inputs[indices[0]+1, indices[1]+3] = 1

    return new_inputs


def generate_input_with_single_channel_active(n_channel, cues):
    inputs = np.zeros((6, 284))

    for cue in cues:
        inputs[n_channel, cue] = 1

    return inputs


# norms and distances
def compute_norm_for_all_systems():
    norms = {}

    with open("fMRI_data/unrelated_subjects_final.txt", "r") as f:
        subject_ids = [line.strip() for line in f if line.strip()]

    for subject_id in subject_ids:
        system = generate_estimated_system(subject_id)
        norms[subject_id] = poles_zeros_norm(control.poles(system), control.zeros(system))

    return norms


def get_systems_with_different_norms(n_systems, delta, norms):
    items = sorted(norms.items(), key=lambda x: x[1])
    systems = {}
    k0, v0 = items[0]
    systems[k0] = v0
    for k, v in items:
        if all(abs(v - existing) > delta for existing in systems.values()):
            systems[k] = v
        if len(systems) == n_systems:
            break

    return systems


def get_distant_systems(distance, n_systems):
    systems_selected = {"100206": generate_estimated_system("100206")}

    with open("fMRI_data/unrelated_subjects_final.txt", "r") as f:
        subject_ids = [line.strip() for line in f if line.strip()]

    for subject_id in subject_ids:
        system_to_check = generate_estimated_system(subject_id)
        poles, zeros = control.poles(system_to_check), control.zeros(system_to_check)
        if all(poles_and_zeros_distance(poles, zeros, control.poles(system), control.zeros(system)) > distance for system in systems_selected.values()):
            systems_selected[subject_id] = system_to_check

        if len(systems_selected) == n_systems:
            break

    return systems_selected


def plot_cepstrum_fMRI_systems(u, y, id_system):
    sys_1 = generate_estimated_system(id_system)

    poles, zeros = control.poles(sys_1), control.zeros(sys_1)
    weights = np.arange(64)
    th_cepstrum = poles_zeros_cepstrum(poles, zeros, weights)
    estimated_cepstrum = compute_cepstrum_transfer_function(u, y, eps=0)

    plt.title(f"Cepstrum sistema {id_system}")
    plt.plot(th_cepstrum[1:30], label='Real cepstrum')
    plt.plot(estimated_cepstrum[1:30, ], linestyle='dashed', label="Estimated cepstrum")
    plt.legend()
    plt.show()
