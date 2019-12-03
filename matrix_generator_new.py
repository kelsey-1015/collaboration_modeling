import numpy as np
import variants as v
from distance_calculation import minimum_distance_3d
from itertools import *

# number of elements in input collaboration matrix that generates randomly -- (total number - fixed diagonal elements)
num_variant_total = num_variant = (v.num_party * v.num_party - v.num_party) * v.num_scope
# define the number of non-zeros values in the input matrix -- reduce search space
# current collaboration value under definition
col_value = [1, 2]

N_nonzero = 1
N_party = v.num_party
num_archetype = 8

count = 0
# A list to count the number of each archetypes that being counted
count_arch = np.zeros(num_archetype)
# list of distance which is larger than 16
count_lager_than_range = 0

# Firstly loop over the various locations of non-zero values
for index in combinations(range(num_variant_total), N_nonzero):
    index = list(index)

    # for specific(chosen) non-zero values positions, loop over different value level combinations
    for col_value_list in product(col_value, repeat=N_nonzero):

        # The following code block works on a specific input collaboration matrix
        matrix_gene = np.zeros((N_party, N_party, 3))
        input_list = np.zeros(num_variant_total)
        # insert specif value combination into predetermined locations
        col_value_list = list(col_value_list)
        input_list[index] = col_value_list
        # all the elements are defined and reshape it into 3D matrix
        # split into 3 scopes as 3 individual lists
        matrix_scope = np.split(input_list, 3)

        count += 1

        # loop over each lists for each scope matrix
        for j, m in enumerate(matrix_scope):

            # insert zero values in diagonal locations using element index loop
            for i, v in enumerate(m):
                if i % (N_party + 1) == 0:
                    m = np.insert(m, i, 0)
            m = np.append(m, 0)
            # reshape list into a matrix
            m = m.reshape(N_party, N_party)
            # insert 2D matrix in a 3D matrix
            matrix_gene[:, :, j] = m

        # calculate distance and select archetype with each random col matrix input
        arch_selected_indices = minimum_distance_3d(matrix_gene)[2]
        distance = minimum_distance_3d(matrix_gene)[0]

        distance_file = open("distance.txt", 'a')
        distance_file.write(str(distance) + ",")
        distance_file.close()

        # sort the calculated results into bins
        for a in arch_selected_indices:

            count_arch[a] += 1
            # save data in as an output file
            data_file = open("archetype.txt", 'a')
            data_file.write(str(a)+",")
            data_file.close()

        if distance >= 16:
            count_lager_than_range += 1

        if count % 50000 == 0:
            print(count)

final_result = open("Distribution.txt", 'a')
final_result.write(str(count_arch))
final_result.close()

print(count_arch)
print(count_lager_than_range)



# print(count)
    # for col_value_list in product(col_value, repeat=2):

