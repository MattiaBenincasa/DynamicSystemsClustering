from electric_circuits import generate_discrete_lti_circuit, generate_white_noise_signal, multiple_circuit_simulation
from clustering import generate_dataset_circuit, k_means
from sklearn.metrics.cluster import adjusted_rand_score
from distance_measures import extended_cepstral_distance

# simulation parameters
n_samples = 2**10
n_input_signals = 100

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
dataset, true_clusters = generate_dataset_circuit(inputs, outputs['system_1'], outputs['system_2'])
centroids, predicted_clusters = k_means(dataset, 2)

print(f'ARI index: {adjusted_rand_score(true_clusters, predicted_clusters)}')


#for i in range(50):
#    print(f"Distanza cepstral segnale {i}: {extended_cepstral_distance(inputs[i], outputs['system_1'][i], inputs[i], outputs['system_2'][i])}")
