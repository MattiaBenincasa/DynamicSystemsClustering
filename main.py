from electric_circuits.test_clustering import test_two_circuits_clustering
from fMRI.init_data import load_data, preprocess_data
from distance_measures import extended_cepstral_distance_mimo, compute_distance_matrix


test_two_circuits_clustering()

'''
subject_ids, outputs, inputs, heart, resp = load_data()
outputs_rf = preprocess_data(outputs['100206'], heart['100206'], resp['100206'])

u_1 = inputs['100206']
u_2 = inputs['102311']
y_1 = outputs['100206']
y_2 = outputs['102311']

# print(extended_cepstral_distance_mimo(u_1, y_1, u_2, y_2))
print(compute_distance_matrix(subject_ids, inputs, outputs, extended_cepstral_distance_mimo))'''
