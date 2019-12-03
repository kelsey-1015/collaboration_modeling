import numpy as np
import variants as v

""" Generate Standard Database whose dimension equals to num-party """


def pattern_generator(cl_before, cl_last):
    """ The description matrices of existing 8 archetypes in database share some similarites,
    Only one row or column contains non-zero values with following patterns.e.g. [a,a,[b,c]],
    The purpose of this function is just to reduce the effort for input each matrix by number """
    cl_pattern = cl_before * np.ones((v.num_party - 2))
    cl_pattern = np.append(cl_pattern, cl_last)
    return cl_pattern


''' Database for current archetype'''

''' Archetype I '''
archetype_1 = np.zeros((v.num_party, v.num_party, v.num_scope))
# Data Scope
cl_info_list = pattern_generator(2, [0, 0])
archetype_1[:, -2, 0] = cl_info_list


''' Archetype II'''
archetype_2 = np.zeros((v.num_party, v.num_party, v.num_scope))
# Data Scope
cl_info_list = pattern_generator(2, [0, 0])
archetype_2[:, -1, 0] = cl_info_list

# Algorithm Scope
cl_info_list = pattern_generator(0, [2, 0])
archetype_2[:, -1, 1] = cl_info_list


''' Archetype III'''
archetype_3 = np.zeros((v.num_party, v.num_party, v.num_scope))
# Algorithm Scope
cl_info_list = pattern_generator(1, [0, 0])
archetype_3[-2, :, 1] = cl_info_list
# IR Scope
cl_info_list = pattern_generator(2, [0, 0])
archetype_3[:, -2, 2] = cl_info_list


''' Archetype IV'''
archetype_4 = np.zeros((v.num_party, v.num_party, v.num_scope))
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
archetype_5 = np.zeros((v.num_party, v.num_party, v.num_scope))
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
archetype_6 = np.zeros((v.num_party, v.num_party, v.num_scope))
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
archetype_7 = np.zeros((v.num_party, v.num_party, v.num_scope))
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
archetype_8 = np.zeros((v.num_party, v.num_party, v.num_scope))
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


