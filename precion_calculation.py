
import numpy as np
import csv
from collections import defaultdict
from itertools import *


""" This script calcultes the distance from CR and one specific archetype and output the distance information in csv 
file"""


""" Initialize the collaboration matrices in each group"""
num_party = 4
num_scope = 3
matrix_data = np.zeros((num_party, num_party))
matrix_algorithm = np.zeros((num_party, num_party))
matrix_result = np.zeros((num_party, num_party))

"""Effective entries in a given 3D collaboration matrix: total number - diagonal entries"""
num_effective = (num_party * num_party - num_party) * num_scope

"""Collaboration levels under definition"""
cl_level = [1, 2]

"""Initialize mapping function from collaboration matrix into DMP personas, dependent with deployment
Currently [d,d,a,dmp]"""
mapping_dic = {0: 11, 1: 11, 2: 22, 3: 33}

file_name = "cr_archetype_5.csv"


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

#
# ''' Archetype VII'''
# archetype_7 = np.zeros((num_party, num_party, num_scope))
# # Data Scope
# cl_info_list = pattern_generator(2, [0, 0])
# archetype_7[:, -1, 0] = cl_info_list
# # Algorithm Scope
# cl_info_list = pattern_generator(0, [2, 0])
# archetype_7[:, -1, 1] = cl_info_list
# # IR Scope
# cl_info_list = pattern_generator(0, [0, 2])
# archetype_7[:, -2, 2] = cl_info_list

''' Archetype VII'''
archetype_7 = np.zeros((num_party, num_party, num_scope))
# Data Scope
cl_info_list = pattern_generator(1, [0, 0])
archetype_7[:, -1, 0] = cl_info_list
# Algorithm Scope
cl_info_list = pattern_generator(0, [2, 0])
archetype_7[:, -1, 1] = cl_info_list
# IR Scope
cl_info_list = pattern_generator(0, [0, 1])
archetype_7[:, -2, 2] = cl_info_list

""" List of archetype description matrices and list of nonzero elements in each archetype description matrix """
ref_CR = np.zeros([num_party, num_party, num_scope])
ref_num_nonzero = 0
ref_nz_dic = {0: 0}

archetype_db = [ref_CR, archetype_1, archetype_2, archetype_3, archetype_4, archetype_5, archetype_6, archetype_7]
num_nonzero_archetypes = [ref_num_nonzero, 2, 3, 4, 4, 4, 4, 4, 4]

archetype_nz_list = [ref_nz_dic, {1: 2, 4: 2}, {2: 2, 5: 2, 20: 2}, {18: 1, 19: 1, 25: 2, 28: 2},
                     {2: 1, 5: 1, 20: 1, 35: 2}, {2: 1, 5: 1, 20: 2, 35: 2}, {2: 2, 5: 2, 20: 2, 35: 2},
                     {2: 2, 5: 2, 20: 2, 35: 2}, {2: 1, 5: 1, 20: 2, 35: 1}]

""" The subset of archetype trust models a DMP under evaluation equips"""
dmp_archetype_index = [1, 2, 3, 4, 5, 6, 7, 8]

""" Define the weights for hamming distance calculation """
hamming_weights = [2, 1]
""" Affordable distance"""
d_a = 6


def print_matrix_by_scope(matrix):

    for i in range(num_scope):
        print("scope", i)
        print(matrix[:, :, i])


def max_num_nonzero(archetype_index, d_a):
    """Calculate the maximum non_zero elemnents in input CR so that a distance less than d_a can be achieved;
    For optimization"""
    max_weight = max(hamming_weights)
    num_nonzero_archetype = num_nonzero_archetypes[archetype_index]
    if d_a % 2 == 0:
        num_nonzero_cr_max = int(d_a/max_weight+num_nonzero_archetype)
    else:
        num_nonzero_cr_max = int((d_a+1) / max_weight + num_nonzero_archetype)
    return num_nonzero_cr_max, num_nonzero_archetype


def map_index_to_role(coordinate_matrix, dic=mapping_dic):
    """This function represent the [source, target] of a relationship with DMP personas instead of
    matrix index. Part of pre-processing. For simplity, '11' --> data provider; '22'--> algorithm provider;
    '33' --> DMP
    """
    for k in dic:
        coordinate_matrix[coordinate_matrix == k] = dic[k]
    return coordinate_matrix


def get_dic(collaboration_model):

    """ This function outputs a dict containing key collaboration information
    Key: [source, pair] --> mapping index into participating roles
    Value: list[cl_1, cl_2, cl_3], merge relationships if they have same [source, target] in terms of participating
    roles

"""

    matrix_sum = np.sum(collaboration_model, axis=2)
    rows, cols = np.nonzero(matrix_sum)

    coordinate = np.stack((rows, cols), axis=1)
    coordinate = map_index_to_role(coordinate)
    coordinate = np.array([str(cc) for cc in coordinate])
    relationship_tuple = collaboration_model[rows, cols, :]

    relationship_dictionary = defaultdict(list)
    for key, value in zip(coordinate, relationship_tuple):
        relationship_dictionary[key].append(value)

    relationship_dictionary = dict(relationship_dictionary)
    return relationship_dictionary


def weighted_hamming_distance(relationship_tuple_1, relationship_tuple_2):

    """ calculate weighted hamming distance between two corresponding collaboration relationship tuples"""

    distance = 0

    for cl_1, cl_2 in zip(relationship_tuple_1, relationship_tuple_2):

        if cl_1 != cl_2:
            if cl_1 == 0 or cl_2 == 0:
                distance += hamming_weights[0]
            else:
                distance += hamming_weights[1]
    return distance


def pre_processing(relationship_list_1, relationship_list_2):
    """ Input two list of relationship tuples under same [source, target] in participating role manner;
    Output the WHD distance with optimal deployements"""

    # Reassign two list input according to their length-- TO DO LATER, CURRENT JUST ASSIGN BY TURN
    if len(relationship_list_1) >= len(relationship_list_2):
        list_larger = relationship_list_1
        list_smaller = relationship_list_2
    else:
        list_larger = relationship_list_2
        list_smaller = relationship_list_1

    combi_list = (list(zip(list_smaller, p)) for p in permutations(list_larger))
    # first loop over different combinations, no index_requred
    distance_combi_list = []
    for combi in combi_list:
        distance_combi = 0
        for relationship_pair in combi:
            distance_pairwise = weighted_hamming_distance(relationship_pair[0], relationship_pair[1])
            distance_combi += distance_pairwise
        distance_combi_list.append(distance_combi)

    return min(distance_combi_list)


def pre_processing_ref_zero(relationship_list):
    """This function calculates distance when the [source, target] pair only exists in one CR instead of both;
    Potentially its WHD with all zero lists"""

    ref_zero = [0]* num_scope
    distance_combi = 0
    for r in relationship_list:
        distance_pairwise = weighted_hamming_distance(r, ref_zero)
        distance_combi += distance_pairwise
    return distance_combi


def difference_list_generator(dic_1, dic_2):

    """ Generate list of hamming distance; use list instead of matrix, but lost information about source/destination
    INCLUDE PRE-PROCESSING_VERSION_1"""

    list_diffs = []

    for i in dic_1:
        if i in dic_2.keys():
            # calculate the distance with various combinations
            distance = pre_processing(dic_1[i], dic_2[i])
        else:
            distance = pre_processing_ref_zero(dic_1[i])

        list_diffs.append(distance)

    for j in dic_2:
        if j in dic_1.keys():
            continue
        else:
            distance = pre_processing_ref_zero(dic_2[j])

        list_diffs.append(distance)
    return list_diffs


def mutual_distance(collaboration_model, archetype):
    """ This function calculates mutual distance between customer-defined collaboration model and an existing
    archetype --- MODIFIED_VERSION1"""
    cm_dic = get_dic(collaboration_model)
    archetype_dic = get_dic(archetype)
    distance = np.sum(difference_list_generator(cm_dic, archetype_dic))
    return distance


def minimum_calculation(collaboration_model):

    """ This function input the customer-required collaboration model as a 3D matrix;
    Calculate its distance to all archetypes in database and select the optimum one with corresponding distance"""

    distance_list = []

    for arch in archetype_db:
        distance = mutual_distance(collaboration_model, arch)
        distance_list.append(distance)
    distance_list = distance_list[1:]
    # mini_value = min(distance_list)
    # mini_indices = [i for i, x in enumerate(distance_list) if x == mini_value]
    # mini_indice = mini_indices[0]
    distance_ref = [100]
    distance_list = distance_ref+ distance_list
    return distance_list


def precision(collaboration_request, archetype_set):
    # archetype_set = [1, 2, 3, 4, 5]
    distance_list = minimum_calculation(collaboration_request)
    distance_list = np.array(distance_list)
    distance_list_subset = distance_list[archetype_set]
    # print(distance_list_subset)
    mini_distance = min(distance_list_subset)
    # print(mini_distance)

    precision = 1 - mini_distance/float(d_a)
    return precision
    # print(precision)


def flexibility(num_hard):
    return(1 - num_hard/float(num_effective))


if __name__ == "__main__":
    """ Initialize collaboration request I"""
    collaboration_request_1 = np.zeros((num_party, num_party, num_scope))
    # Generate the distance list for archetype I
    collaboration_request_1[1, 3, 0] = 2

    collaboration_request_1[2, 0, 1] = 2
    collaboration_request_1[2, 3, 1] = 2

    collaboration_request_1[0, 2, 2] = 2
    collaboration_request_1[3, 2, 2] = 2

    """ Initialize collaboration request II"""
    collaboration_request_2 = np.zeros((num_party, num_party, num_scope))

    collaboration_request_2[0, 2, 0] = 1
    collaboration_request_2[1, 2, 0] = 2

    # print_matrix_by_scope(collaboration_request_1)
    dmp_1 = [1, 2, 3, 4, 7]
    dmp_2 = [1, 2, 3, 5, 7]
    dmp_3 = [1, 2, 3, 5, 6]
    dmp_4 = [1, 3, 4, 5, 7]
    dmp_5 = [2, 3, 4, 6, 7]
    DMP = [dmp_1, dmp_2, dmp_3, dmp_4, dmp_5]

    # print_matrix_by_scope(collaboration_request_2)
    # for dmp in DMP:
    #     print(precision(collaboration_request_2, dmp))
    print(flexibility(34))























