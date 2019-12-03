import re
import ast
import numpy as np


""" Total number of participating parties """
num_party = 4

""" Total number of scopes"""
num_scope = 3

""" Initialize the collaboration matrices in each group"""
matrix_data = np.zeros((num_party, num_party))
matrix_algorithm = np.zeros((num_party, num_party))
matrix_result = np.zeros((num_party, num_party))


def pattern_generator(cl_before, cl_last):
    """ The description matrices of existing 8 archetypes in database share some similarites,
    Only one row or column contains non-zero values with following patterns.e.g. [a,a,[b,c]],
    The purpose of this function is just to reduce the effort for input each matrix by number """
    cl_pattern = cl_before * np.ones((num_party - 2))
    cl_pattern = np.append(cl_pattern, cl_last)
    return cl_pattern


''' Archetype I '''
archetype_1 = np.zeros((num_party, num_party, num_scope))
# Data Scope
cl_info_list = pattern_generator(2, [0, 0])
archetype_1[:, -2, 0] = cl_info_list


''' Archetype II'''
archetype_2 = np.zeros((num_party, num_party, num_scope))
# Data Scope
cl_info_list = pattern_generator(2, [0, 0])
archetype_2[:, -1, 0] = cl_info_list

# Algorithm Scope
cl_info_list = pattern_generator(0, [2, 0])
archetype_2[:, -1, 1] = cl_info_list


''' Archetype III'''
archetype_3 = np.zeros((num_party, num_party, num_scope))
# Algorithm Scope
cl_info_list = pattern_generator(1, [0, 0])
archetype_3[-2, :, 1] = cl_info_list
# IR Scope
cl_info_list = pattern_generator(2, [0, 0])
archetype_3[:, -2, 2] = cl_info_list


''' Archetype IV'''
archetype_4 = np.zeros((num_party, num_party, num_scope))
# Data Scope
cl_info_list = pattern_generator(1, [0, 0])
archetype_4[:, -1, 0] = cl_info_list
# Algorithm Scope
cl_info_list = pattern_generator(0, [1, 0])
archetype_4[:, -1, 1] = cl_info_list
# IR Scope
cl_info_list = pattern_generator(0, [0, 2])
archetype_4[:, -2, 2] = cl_info_list


''' Archetype V'''
archetype_5 = np.zeros((num_party, num_party, num_scope))
# Data Scope
cl_info_list = pattern_generator(1, [0, 0])
archetype_5[:, -1, 0] = cl_info_list
# Algorithm Scope
cl_info_list = pattern_generator(0, [2, 0])
archetype_5[:, -1, 1] = cl_info_list
# IR Scope
cl_info_list = pattern_generator(0, [0, 2])
archetype_5[:, -2, 2] = cl_info_list


''' Archetype VI'''
archetype_6 = np.zeros((num_party, num_party, num_scope))
# Data Scope
cl_info_list = pattern_generator(2, [0, 0])
archetype_6[:, -1, 0] = cl_info_list

# Algorithm Scope
cl_info_list = pattern_generator(0, [2, 0])
archetype_6[:, -1, 1] = cl_info_list
# IR Scope
cl_info_list = pattern_generator(0, [0, 2])
archetype_6[:, -2, 2] = cl_info_list


''' Archetype VII'''
archetype_7 = np.zeros((num_party, num_party, num_scope))
# Data Scope
cl_info_list = pattern_generator(2, [0, 0])
archetype_7[:, -1, 0] = cl_info_list
# Algorithm Scope
cl_info_list = pattern_generator(0, [2, 0])
archetype_7[:, -1, 1] = cl_info_list
# IR Scope
cl_info_list = pattern_generator(0, [0, 2])
archetype_7[:, -2, 2] = cl_info_list

''' Archetype VIII'''
archetype_8 = np.zeros((num_party, num_party, num_scope))
# Data Scope
cl_info_list = pattern_generator(1, [0, 0])
archetype_8[:, -1, 0] = cl_info_list
# Algorithm Scope
cl_info_list = pattern_generator(0, [2, 0])
archetype_8[:, -1, 1] = cl_info_list
# IR Scope
cl_info_list = pattern_generator(0, [0, 1])
archetype_8[:, -2, 2] = cl_info_list


""" A dictionary containing archetype name and corresponding description matrix"""

archetype_db = {"Archetype I": archetype_1,
                "Archetype II": archetype_2,
                "Archetype III": archetype_3,
                "Archetype IV": archetype_4,
                "Archetype V": archetype_5,
                "Archetype VI": archetype_6,
                "Archetype VII": archetype_7,
                "Archetype VIII": archetype_8
                }


""" Define the weights for hamming distance calculation """
weight = np.array([2, 1])

archetype_str_list = np.array(['Archetype I', 'Archetype II', 'Archetype III',
                               'Archetype IV', 'Archetype V','Archetype VI',
                               'Archetype VII', 'Archetype VIII'])

def get_dic(collaboration_model):

    """ Get the tuple containing non_zero elements from sparse collaboration matrix,
    reinstall these information with a dictionary with format:
    Each dictionaly entry: "coordinate":relationship_tuple
    "coordinate" = [source_index, target_index] && relationship_tuple = [cl_data,cl_algorithm,cl_result]]"""

    matrix_sum = np.sum(collaboration_model, axis=2)
    rows, cols = np.nonzero(matrix_sum)

    coordinate = np.stack((rows, cols), axis=1)
    coordinate = np.array([str(cc) for cc in coordinate])
    relationship_tuple = collaboration_model[rows, cols, :]
    relationship_dictionary = dict(zip(coordinate, relationship_tuple))
    return relationship_dictionary


def weighted_hamming_distance(relationship_tuple_1, relationship_tuple_2, weight_vector=weight):

    """ calculate weighted hamming distance between two corresponding collaboration relationship tuples"""

    distance = 0

    for cl_1, cl_2 in zip(relationship_tuple_1, relationship_tuple_2):

        if cl_1 != cl_2:
            if cl_1 == 0 or cl_2 == 0:
                distance += weight_vector[0]
            else:
                distance += weight_vector[1]
    return distance


def difference_matrix_generator(dic_1, dic_2):

    """ Generate 2D difference matrix between two archetypes"""

    matrix_diffs = np.zeros((num_party, num_party))
    reference_zero = np.zeros(num_scope)

    for i in dic_1:
        if i in dic_2.keys():
            diffs = weighted_hamming_distance(dic_1[i], dic_2[i])
        else:
            diffs = weighted_hamming_distance(dic_1[i], reference_zero)
        index = re.sub('\s+', ',', i)
        index = np.array(ast.literal_eval(index))
        matrix_diffs[index[0], index[1]] = diffs
    for j in dic_2:
        if j in dic_1.keys():
            continue
        else:
            diffs = weighted_hamming_distance(dic_2[j], reference_zero)
            index = re.sub('\s+', ',', j)
            index = np.array(ast.literal_eval(index))
            matrix_diffs[index[0], index[1]] = diffs
    return matrix_diffs


def minimum_calculation(collaboration_model):

    """ This function input the customer-required collaboration model as a 3D matrix;
    Calculate its distance to all archetypes in database and select the optimum one with corresponding distance"""

    distance_list = []
    cm_dic = get_dic(collaboration_model)

    for k1, arch in archetype_db.items():
        arch_dic = get_dic(arch)
        distance = np.sum(difference_matrix_generator(cm_dic, arch_dic))
        distance_list.append(distance)

    mini_value = min(distance_list)
    mini_indices = [i for i, x in enumerate(distance_list) if x == mini_value]
    re_arch = list(archetype_str_list[mini_indices])
    return mini_value, re_arch, mini_indices


def generate_distance_matrix():

    """ This function calculates mutual distance between archetypes, used for check code modification """

    mutual_distance_matrix = np.zeros((8, 8))

    c1 = 0

    for k1, v1 in archetype_db.items():

        c2 = 0
        for k2, v2 in archetype_db.items():
            m1 = get_dic(v1)
            m2 = get_dic(v2)
            distance = np.sum(difference_matrix_generator(m1, m2))
            mutual_distance_matrix[c1][c2] = distance
            c2 += 1
        c1 += 1

    return mutual_distance_matrix


if __name__ == "__main__":
    print(generate_distance_matrix())