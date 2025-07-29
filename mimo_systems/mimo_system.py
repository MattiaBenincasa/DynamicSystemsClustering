from scipy.signal import cont2discrete, dlti, dlsim
import numpy as np

from mimo_systems.power_cepstrum import compute_cepstral_distance


def create_discrete_mimo_system(A, B, C, D, dt=1.0):
    sys = cont2discrete((A, B, C, D), dt)
    return dlti(*sys[:4])


def get_two_mimo_systems():
    A1 = np.array([
        [-1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0, -2.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0, -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0, -3.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0, -1.5,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0, -2.5,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.8,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -3.5,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -2.8,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.5],
    ])

    B1 = np.array([
        [ 1.0,  0.0],
        [ 0.0,  1.0],
        [ 0.5,  0.2],
        [-0.3,  1.2],
        [ 0.8, -0.1],
        [ 0.1,  0.6],
        [ 0.9,  0.1],
        [ 0.0, -0.3],
        [ 1.1,  0.0],
        [ 0.2, -0.7]
    ])

    C1 = np.eye(10)

    D1 = np.zeros((10, 2))

    A2 = np.array([
        [-0.8,  0.1,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0],
        [-0.2, -1.5,  0.3,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0, -0.4, -2.0,  0.2,  0.0,  0.0,  0.6,  0.0,  0.0,  0.0],
        [ 0.0,  0.0, -0.1, -1.2,  0.4,  0.0,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0, -0.3, -1.8,  0.1,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0, -0.2, -2.5,  0.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0, -1.5, -0.9,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.1, -2.3,  0.0,  0.4],
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.3,  0.0],
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.4,  0.0, -1.2],
    ])

    B2 = np.array([
        [ 0.7, -0.1],
        [-0.2,  0.9],
        [ 1.0,  0.3],
        [ 0.4, -0.8],
        [-0.1,  0.5],
        [ 0.6,  0.2],
        [ 0.2,  0.0],
        [ 1.4, -0.2],
        [-1.0,  0.2],
        [ 0.3, -1.2],
    ])

    C2 = np.array([
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.2, 0.0, 0.0, 0.4, 0.0, 0.0, 1.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 1.0],

    ])

    D2 = np.array([
        [0.1, 0.0],
        [0.0, 0.2],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.4, 0.0],
        [0.0, 0.0]
    ])

    return (create_discrete_mimo_system(A1, B1, C1, D1),
            create_discrete_mimo_system(A2, B2, C2, D2))


def generate_white_noise_signal(n_samples, n_inputs, mean, sigma, n_signals=1):
    if n_signals > 1:
        inputs = []
        for i in range(n_signals):
            inputs.append(np.random.normal(mean, sigma, size=(n_samples, n_inputs)))
        return inputs
    elif n_signals == 1:
        return np.random.normal(mean, sigma, size=(n_samples, n_inputs))


def generate_sinusoidal_signal(n_samples, freq, n_signals=1):
    # two inputs
    fs = 800
    Ts = 1 / fs
    n = np.arange(n_samples)
    if n_signals > 1:
        inputs = []
        for i in range(n_signals):
            u1 = 10 * np.sin(2 * np.pi * freq[0] * n * Ts) + np.random.normal(0, 0.2, size=n_samples)
            u2 = 2 * np.sin(2 * np.pi * freq[1] * n * Ts + np.pi/4) + np.random.normal(0, 0.6, size=n_samples)
            inputs.append(np.vstack((u1, u2)).T)
        return inputs
    elif n_signals == 1:
        u1 = 10 * np.sin(2 * np.pi * freq[0] * n * Ts) + np.random.normal(0, 0.2, size=n_samples)
        u2 = 2 * np.sin(2 * np.pi * freq[1] * n * Ts + np.pi / 4) + np.random.normal(0, 0.6, size=n_samples)
        return np.vstack((u1, u2)).T


def simulate_system_on_multiple_input(input_signals, system):
    output_signals = []

    for input_signal in input_signals:
        _, y, _ = dlsim(system, input_signal)
        output_signals.append(y.T)

    return output_signals


def compute_distance_between_mimo_systems():
    sys_1, sys_2 = get_two_mimo_systems()

    u_1 = generate_white_noise_signal(2**14, 2, 0, 1)
    u_2 = generate_white_noise_signal(2**14, 2, 0, 1)

    freq_1 = (1.5, 5)
    freq_2 = (0.6, 0.3)

    # u_1 = generate_sinusoidal_signal(2 ** 14, freq_1)
    # u_2 = generate_sinusoidal_signal(2 ** 14, freq_2)

    _, y_11, _ = dlsim(sys_1, u_1)
    _, y_21, _ = dlsim(sys_2, u_1)
    _, y_12, _ = dlsim(sys_1, u_2)
    _, y_22, _ = dlsim(sys_2, u_2)

    print(f'Distance between time series generated by different systems: {compute_cepstral_distance(u_1.T, y_11.T, u_1.T, y_21.T)}')
    print(f'Distance between time series generated by the same systems: {compute_cepstral_distance(u_1.T, y_11.T, u_2.T, y_12.T)}')

