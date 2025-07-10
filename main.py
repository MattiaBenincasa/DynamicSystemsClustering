import numpy as np
from ElectricCircuits import generate_discrete_lti_circuit, simulate_circuit_on_white_noise
from matplotlib import pyplot as plt

# simulation parameters
n_samples = 1000

# electric circuit
R = 100
L1 = 20
L2 = 60
C = 50

circuit = generate_discrete_lti_circuit(R, L1, L2, C)

# System simulation
results = simulate_circuit_on_white_noise(circuit, n_samples)
tout, y, x = results[0]
u = results[1]

# input/output plot
time_steps = np.arange(n_samples)
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(time_steps, u)
plt.title(f'Segnale di Ingresso (White Noise)')
plt.subplot(2, 1, 2)
plt.plot(time_steps, y)
plt.title('Output del Sistema')
plt.tight_layout()
plt.show()
