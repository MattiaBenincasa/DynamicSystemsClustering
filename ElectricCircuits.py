import numpy as np
from scipy.signal import cont2discrete, dlsim


# The circuit topology implemented here follows the configuration described in:
# O. Lauwers, B. De Moor, "A time series distance measure for efficient clustering of input/output
# signals by their underlying dynamics", see Figure 1.
def generate_discrete_lti_circuit(R, L1, L2, C):
    # system matrices
    a = np.array([[0, 1 / L2, 0],
                  [-1 / C, 0, -1 / C],
                  [0, 1 / L1, -R / L1]])

    b = np.array([[0],
                  [1 / C],
                  [0]])

    c = np.array([[0, 1, -R]])

    d = np.array([[0]])

    return cont2discrete((a, b, c, d), dt=0.01)


def simulate_circuit_on_white_noise(circuit, n_samples):
    # white noise
    u = np.random.normal(0, 1, size=(n_samples, 1))
    # System simulation
    return dlsim(circuit, u), u
