import numpy as np
import variants as v
from distance_calculation import minimum_distance
from itertools import product

""" Generate the input collaboration matrix randomly with limitation and select the archtype in minimum distance """

input_matrix_data = np.zeros((v.num_party, v.num_party))
input_matrix_algorithm = np.zeros((v.num_party, v.num_party))
input_matrix_result = np.zeros((v.num_party, v.num_party))

input_matrix = [input_matrix_data, input_matrix_algorithm, input_matrix_result]
input_matrix = np.stack(input_matrix, axis=-1)

num_variant = v.num_party * v.num_party - v.num_party   # The diagonal element in matrix is always constant

col_value = [0, 1, 2]

scope_arrangement = {"DATA": 0, "ALGORITHM": 1, "RESULT": 2} # into variable file later


def distance_distribution(scope):

    count_arch = np.zeros(8)

    count = 0

    for col_value_list in product(col_value, repeat=num_variant):

        col_value_list = list(col_value_list)
        for index, value in enumerate(col_value_list):
            if index % (v.num_party + 1) == 0:
                col_value_list.insert(index, 0)

        col_value_list.append(0)
        col_value_matrix = np.reshape(col_value_list, (v.num_party, v.num_party))

        input_matrix[:, :, scope_arrangement[scope]] = col_value_matrix

        arch_indices = minimum_distance(input_matrix)[2]

        for a in arch_indices:

            # data_file = open("archetype_result.txt", 'w')
            # data_file.write(str(a)+",")

            # data_file.close()

            count_arch[a] += 1

        count += 1

        if count % 10000 == 0:

            print(count)

    return count_arch


data_distribution = distance_distribution("RESULT")
# final_result = open("Distribution.txt", 'a')
# final_result.write(str(data_distribution))
# final_result.close()
print(data_distribution)





