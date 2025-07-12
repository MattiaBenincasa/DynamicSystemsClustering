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

    return cont2discrete((a, b, c, d), dt=1)


def generate_white_noise_signal(n_samples):
    return 100*np.random.normal(0, 0.6, size=(n_samples, 1))


def generate_sinusoidal_signal(n_samples):
    f = 5
    fs = 800
    Ts = 1 / fs
    n = np.arange(n_samples)
    noise = np.random.normal(0, 0.2, size=n_samples)
    return (10 * np.sin(2 * np.pi * f * n * Ts) + noise).reshape(-1, 1)


def multiple_circuit_simulation(n_input_signals, sys_1, sys_2, n_samples, generate_input_signal):
    inputs = []
    out_sys_1 = []
    out_sys_2 = []

    for _ in range(n_input_signals):
        u = generate_input_signal(n_samples)
        inputs.append(u)

        # simulate first system
        tout_1, y_1, x_1 = dlsim(sys_1, u)
        out_sys_1.append(y_1)

        # simulate second system
        tout_2, y_2, x_2 = dlsim(sys_2, u)
        out_sys_2.append(y_2)

    outputs = {'system_1': out_sys_1, 'system_2': out_sys_2}

    return inputs, outputs
