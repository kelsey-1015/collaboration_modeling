import numpy as np

""" Total number of participating parties """
num_party = 4

""" Total number of scopes"""
num_scope = 3

""" Initialize the collaboration matrices in each group"""
matrix_data = np.zeros((num_party, num_party))
matrix_algorithm = np.zeros((num_party, num_party))
matrix_result = np.zeros((num_party, num_party))

value_check_table = np.array([[1, 1, 2], [1, 2, 2], [2, 2, 2], [1, 2, 1]])
arche_index = np.array(['Archetype IV', 'Archetype V', 'Archetype VI OR VII','Archetype VIII'])


str_no_match = "Sorry! No archetype is perfectly matched in current database!"

# input_matrix = [matrix_data, matrix_algorithm, matrix_result]
# input_matrix = np.stack(input_matrix, axis=-1)

""" Initialize the 3D collaboration matrices with scope order [data, algo, result]"""
