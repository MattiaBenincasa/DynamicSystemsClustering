import numpy as np
from scipy.signal import cont2discrete, dlsim
from matplotlib import pyplot as plt


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
            inputs.append(10 * np.sin(2 * np.pi * f * n * Ts) + np.random.normal(0, 0.2, size=n_samples))
        return inputs
    elif n_signals == 1:
        return 10 * np.sin(2 * np.pi * f * n * Ts) + np.random.normal(0, 0.2, size=n_samples)


def generate_multi_sine_wave(n_samples):
    # sampling parameters
    fs = 400
    Ts = 1 / fs
    n = np.arange(n_samples)

    n_sinusoid = 3
    max_amp = 5.0
    min_amp = 0.5
    max_freq = 0.01
    min_freq = 0.001
    min_phase = 0
    max_phase = 2 * np.pi

    sinusoidal_signals = []

    for sinusoid in range(n_sinusoid):
        sinusoidal_signals.append({
            'amplitude': np.random.uniform(min_amp, max_amp),
            'freq': np.random.uniform(min_freq, max_freq),
            'phase': np.random.uniform(min_phase, max_phase)
        })

    multi_sine = np.zeros(n_samples)

    for sinusoid in sinusoidal_signals:
        amplitude = sinusoid['amplitude']
        freq = sinusoid['freq']
        fase = sinusoid['phase']
        multi_sine += amplitude * np.sin(2 * np.pi * freq * n * Ts + fase) + np.random.normal(0, 0.1, size=n_samples)

    return multi_sine


def generate_multiple_multi_sin_waves(n_samples, n_signals):
    if n_signals > 1:
        signals = []
        for _ in range(n_signals):
            signals.append(generate_multi_sine_wave(n_samples))
        return signals
    else:
        return generate_multi_sine_wave(n_samples)


# simulate one circuit on more inputs
def simulate_circuit_on_multiple_input(input_signals, circuit):
    output_signals = []

    for input_signal in input_signals:
        _, y, _ = dlsim(circuit, input_signal)
        output_signals.append(y.reshape(-1))

    return output_signals


def simulate_circuit_on_multi_input_with_x0(input_signals, circuit, x0):
    output_signals = []

    for i in range(len(input_signals)):
        _, y, _ = dlsim(circuit, input_signals[i], x0=x0[:, i])
        output_signals.append(y.reshape(-1))

    return output_signals


def simulate_circuit_on_multiple_inputs_with_output_noise(input_signals, circuit, snr):
    clean_outputs = simulate_circuit_on_multiple_input(input_signals, circuit)
    noisy_output = []
    for output in clean_outputs:
        noisy_output.append(generate_noise_from_snr(output, snr))

    return noisy_output


def generate_noise_from_snr(signal, snr):
    power_signal = np.mean(signal**2)
    power_noise = power_signal / (10 ** (snr / 10))
    noise = np.random.normal(0, np.sqrt(power_noise), size=signal.size)
    return signal + noise
