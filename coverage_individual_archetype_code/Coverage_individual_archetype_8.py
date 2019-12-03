
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

file_name = "cr_archetype_8.csv"


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

""" List of archetype description matrices and list of nonzero elements in each archetype description matrix """
ref_CR = np.zeros([num_party, num_party, num_scope])
ref_num_nonzero = 0
ref_nz_dic = {0: 0}

archetype_db = [ref_CR, archetype_1, archetype_2, archetype_3, archetype_4, archetype_5, archetype_6, archetype_7, archetype_8]
num_nonzero_archetypes = [ref_num_nonzero, 2, 3, 4, 4, 4, 4, 4, 4]

archetype_nz_list = [ref_nz_dic, {1: 2, 4: 2}, {2: 2, 5: 2, 20: 2}, {18: 1, 19: 1, 25: 2, 28: 2},
                     {2: 1, 5: 1, 20: 1, 35: 2}, {2: 1, 5: 1, 20: 2, 35: 2}, {2: 2, 5: 2, 20: 2, 35: 2},
                     {2: 2, 5: 2, 20: 2, 35: 2}, {2: 1, 5: 1, 20: 2, 35: 1}]

""" The subset of archetype trust models a DMP under evaluation equips"""
dmp_archetype_index = [1, 2, 3, 4, 5, 6, 7, 8]

""" Define the weights for hamming distance calculation """
hamming_weights = [2, 1]


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

    mini_value = min(distance_list)
    mini_indices = [i for i, x in enumerate(distance_list) if x == mini_value]
    return mini_value, mini_indices


def coverage_subset_limitation(num_nonzero, archetype_index, num_nonzero_in):

    distance_list = []

    # TODO calculate further for non_zero_dic for each archetype
    archetype_nz_dic = archetype_nz_list[archetype_index]
    n_a = num_nonzero_archetypes[archetype_index]
    # loop over effective entries outside the non_zero area of an archetype
    num_effective_o = num_effective - n_a
    # the number of non_zero elements in outside area equals (original - N_in)
    num_nonzero_o = num_nonzero - num_nonzero_in

    # outer loop --> loop over all possible combinations in outer area
    for nonzero_index_o in combinations(range(num_effective_o), num_nonzero_o):
        nonzero_index_o = list(nonzero_index_o)

        # Inner loop -- in outer area, determine each collaboration level for each non-zero elements
        # with fixed positions
        for cl_level_list in product(cl_level, repeat=num_nonzero_o):
            # From here Within this loop, a specific customer-defined CR model is determined
            # without all_zero diagonal elements and inner area elements

            cl_level_list = list(cl_level_list)
            cr_matrix_gene = np.zeros((num_party, num_party, num_scope))

            effective_list_o = np.zeros(num_effective_o)
            effective_list_o[nonzero_index_o] = cl_level_list
            # insert the inner areas elements --> get the entire efficient list without reshape and diagonal elements
            if n_a == num_nonzero_in:
                effective_list_o = list(effective_list_o)
                for i in archetype_nz_dic:
                    effective_list_o.insert(i, archetype_nz_dic[i])
            else:
                print("N_a not equal to N_in!")

            # Get the entire efficient list with length = num_effective(36)
            effective_list = np.array(effective_list_o)

            # CR identification id defined as the indexes of non_zero elements in list without diagonal elements
            cr_identification_index = list(np.nonzero(effective_list)[0])
            cr_indetification_value = list(np.int_(effective_list[cr_identification_index]))
            cr_identification = cr_identification_index + cr_indetification_value


            # print(cr_identification_index, type(cr_identification_index))
            # print(cr_indetification_value, type(cr_indetification_value))
            # print(cr_identification, type(cr_identification))

            effective_list_scope = np.split(effective_list, num_scope)

            # loop over each lists for each scope matrix

            # insert diagonal elements and reshape
            for j, m in enumerate(effective_list_scope):
                m = list(m)
                for i in range(num_party):
                    m.insert(i*5, 0)
                m = np.reshape(m, (num_party, num_party))
                cr_matrix_gene[:, :, j] = m

            # print_matrix_by_scope(cr_matrix_gene)
            ''' Calculate the distance for a given CR and archetype model'''
            d = mutual_distance(cr_matrix_gene, archetype_db[archetype_index])
            distance_list.append(d)
            # write the distance into a csv file

            with open(file_name, mode='a') as cr_distance_file:
                cr_distance_writer = csv.writer(cr_distance_file, delimiter=';')
                cr_distance_writer.writerow([cr_identification, d])

    return distance_list


def coverage_subset(num_nonzero, archetype_index):

    distance_list = []

    for nonzero_index in combinations(range(num_effective), num_nonzero):
        nonzero_index = list(nonzero_index)

        # Inner loop -- determine each collaboration level for each non-zero elements with fixed positions
        for cl_level_list in product(cl_level, repeat=num_nonzero):
            '''Within this loop, a specific customer-defined CR model is determined.'''
            cl_level_list = list(cl_level_list)
            # this list used to identity cr information in csv file
            cr_identification = nonzero_index + cl_level_list
            cr_matrix_gene = np.zeros((num_party, num_party, num_scope))

            effective_list = np.zeros(num_effective)
            effective_list[nonzero_index] = cl_level_list

            # Reshape the list into a 3D matrix
            effective_list_scope = np.split(effective_list, num_scope)

            # loop over each lists for each scope matrix

            # try a simpler way to insert diagonal elements and reshape
            for j, m in enumerate(effective_list_scope):
                m = list(m)
                for i in range(num_party):
                    m.insert(i*5, 0)
                m = np.reshape(m, (num_party, num_party))
                cr_matrix_gene[:, :, j] = m

            ''' Calculate the distance for a given CR and archetype model'''
            d = mutual_distance(cr_matrix_gene, archetype_db[archetype_index])
            distance_list.append(d)
            # write the distance into a csv file
            with open(file_name, mode='a') as cr_distance_file:
                cr_distance_writer = csv.writer(cr_distance_file, delimiter=';')
                cr_distance_writer.writerow([cr_identification, d])

    return distance_list


def distance_list_generator(archetype_index, d_a, n_limit):

    cr_to_archetype_list = []

    # # get the range of non_zero number
    # n_max_cr, n_a = max_num_nonzero(archetype_index, d_a)
    # # the CRs with n_c = n_c_max can be calculated out without simulation
    n_cr_list = list(range(0, n_limit+1))

    for n in n_cr_list:
        coverage_subset(n, archetype_index)
        # else:
        #     coverage_subset_limitation(n, archetype_index, n_a)

    return 0


if __name__ == "__main__":

    # Generate the distance list for archetype I
    distance_list_generator(8, 6, 5)
    final_result = open("archetype_finished.txt", 'a')
    final_result.write('Archetype_1_Completed!\n')
    final_result.close()


























