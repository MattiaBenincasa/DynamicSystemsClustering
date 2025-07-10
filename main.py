import numpy as np
from ElectricCircuits import generate_discrete_lti_circuit, generate_white_noise_signal, multiple_circuit_simulation
from matplotlib import pyplot as plt

# simulation parameters
n_samples = 1000
n_input_signals = 200

# first electric circuit
R1 = 100
L11 = 20
L12 = 60
C1 = 50

# second electric circuit
R2 = 100
L21 = 200
L22 = 160
C2 = 75


sys_1 = generate_discrete_lti_circuit(R1, L11, L12, C1)
sys_2 = generate_discrete_lti_circuit(R2, L21, L22, C2)

inputs, outputs = multiple_circuit_simulation(n_input_signals, sys_1, sys_2, n_samples, generate_white_noise_signal)

# input/output plot
time_steps = np.arange(n_samples)
plt.plot(time_steps, outputs['system_1'][0], label='system_1')
plt.plot(time_steps, outputs['system_2'][0], label='system_2')
plt.show()
