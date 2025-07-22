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


def generate_white_noise_signal(n_samples, mean, sigma, n_signals=1):
    if n_signals > 1:
        inputs = []
        for i in range(n_signals):
            inputs.append(100*np.random.normal(mean, sigma, size=n_samples))
        return inputs
    elif n_signals == 1:
        return 100*np.random.normal(mean, sigma, size=n_samples)


def generate_sinusoidal_signal(n_samples, n_signals=1):
    f = 5
    fs = 800
    Ts = 1 / fs
    n = np.arange(n_samples)
    # noise = np.random.normal(0, 0.2, size=n_samples)
    if n_signals > 1:
        inputs = []
        for i in range(n_signals):
            inputs.append(10 * np.sin(2 * np.pi * f * n * Ts))
        return inputs
    elif n_signals == 1:
        return 10 * np.sin(2 * np.pi * f * n * Ts)


# simulate one circuit on more inputs
def multi_input_circuit_simulation(input_signals, circuit):
    output_signals = []

    for input_signal in input_signals:
        _, y, _ = dlsim(circuit, input_signal)
        output_signals.append(y.reshape(-1))

    return output_signals


# simulate different circuits on the same input signals
def multi_circuits_simulation_on_same_inputs(circuits, input_signals):
    n_circuits = len(circuits)
    outputs = np.array(n_circuits)

    for i in range(n_circuits):
        outputs[i] = multi_input_circuit_simulation(input_signals, circuits[i])

    return outputs
