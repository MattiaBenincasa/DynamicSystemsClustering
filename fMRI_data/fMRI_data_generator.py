import numpy as np
import csv
import random
from control import ss, forced_response
from copy import deepcopy


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

