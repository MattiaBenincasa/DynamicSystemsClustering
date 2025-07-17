import numpy as np
from numpy.linalg import pinv
from scipy.signal import remez, filtfilt


def rearrange_inputs(inputs):
    visual_cue = np.zeros((1, 284))
    cue_times = [12, 33, 54, 75, 96, 138, 159, 180, 221, 242]
    visual_cue[0, cue_times] = 1

    inputs = np.vstack((inputs, visual_cue))

    # rearrange inputs
    new_order = [5, 3, 2, 1, 4, 0]
    return inputs[new_order]


def load_data():
    L = 284
    with open("fMRI/unrelated_subjects_final.txt", "r") as f:
        subject_ids = [line.strip() for line in f if line.strip()]

    outputs = {}
    inputs = {}
    heart = {}
    resp = {}

    for subject_id in subject_ids:
        outputs[subject_id] = np.loadtxt(f'fMRI/fMRI_data/bold_{subject_id}.txt', dtype=float, delimiter=',')[:, :L]
        inputs_ = np.loadtxt(f'fMRI/fMRI_data/cues_{subject_id}.txt', dtype=float, delimiter=',')[:, :L]
        inputs[subject_id] = rearrange_inputs(inputs_)
        heart[subject_id] = np.loadtxt(f'fMRI/fMRI_data/heart_{subject_id}.txt', dtype=float)[:L]
        resp[subject_id] = np.loadtxt(f'fMRI/fMRI_data/resp_{subject_id}.txt', dtype=float)[:L]

    return subject_ids, outputs, inputs, heart, resp


def preprocess_data(outputs, heart, resp):
    # physiological signals
    U_P = np.vstack([heart, resp])

    # physiologically regressed outputs
    outputs_r = outputs @ (np.eye(284) - pinv(U_P) @ U_P)

    fs = 1.0
    numtaps = 51  # order + 1 -> 50 + 1

    bands = [0.0, 0.04, 0.06, 0.12, 0.15, 0.5]

    desired = [0, 1, 0]
    weights = [10, 1, 10]

    fir_coeff = remez(numtaps, bands, desired, weight=weights, fs=fs)

    outputs_rf = filtfilt(fir_coeff, [1.0], outputs_r, axis=1)

    return outputs_rf
