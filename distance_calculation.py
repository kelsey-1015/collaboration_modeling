
import variants as v
from archetype_database import archetype_db

""" This script calculates distance between the input and each candidate from the database. And give back the one 
with minimum distance -- most similarity"""


""" Define hamming distance weights.l;/"""
weight = np.array([2, 1])

# '''3D matrix for customer defined collaboration matrix'''
# input_matrix = [v.matrix_data, v.matrix_algorithm, v.matrix_result]
# input_matrix = np.stack(input_matrix, axis=-1)

""" Used for change index to actual str representation"""
archetype_str_list = np.array(['Archetype I', 'Archetype II', 'Archetype III',
                               'Archetype IV', 'Archetype V','Archetype VI',
                               'Archetype VII', 'Archetype VIII'])


def get_dic(matrix):

    """ get the collaboration tuple from sparse collaboration matrix, [source, target, collaboration for each row]"""
    matrix_sum = np.sum(matrix, axis=2)
    rows, cols = np.nonzero(matrix_sum)
    codi = np.stack((rows, cols), axis=1)
    codi_str = np.array([str(cc) for cc in codi])
    dcp_nonzero = matrix[rows, cols, :]
    dcp_dictionary = dict(zip(codi_str, dcp_nonzero))
    return dcp_dictionary


def weighted_hamming_distance(list_1, list_2, weight_vector=weight):

    """ calculate weighted hamming distance between two collaboration relationship tuples"""

    distance = 0
    for n1, n2 in zip(list_1, list_2):
        if n1 != n2:
            if n1 == 0 or n2 == 0:
                distance += weight_vector[0]
            else:
                distance += weight_vector[1]
    return distance


def difference_matrix_generator(dic_1, dic_2):

    """ Generate 2D difference matrix between two archetypes"""

    matrix_diffs = np.zeros((v.num_party, v.num_party))
    reference_zero = np.zeros(v.num_scope)
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


def generate_distance_matrix():

    """ Generate Distance matrix of all possible combinations in current archetype database and """

    distance_matrix = np.zeros((8, 8))

    c1 = 0

    for k1, v1 in archetype_db.items():

        c2 = 0
        for k2, v2 in archetype_db.items():
            m1 = get_dic(v1)
            m2 = get_dic(v2)
            distance = np.sum(difference_matrix_generator(m1, m2))
            distance_matrix[c1][c2] = distance
            c2 += 1
        c1 += 1

    return distance_matrix


def minimum_distance(matrix_1, matrix_2, matrix_3):

    """ Compute the distance between requirement and archetypes in database and return those archetype names with
    minimum distance"""

    distance_list = []

    col_matrix_3d = [matrix_1, matrix_2, matrix_3]
    col_matrix_3d = np.stack(col_matrix_3d, axis=-1)

    input_dic = get_dic(col_matrix_3d)

    for k1, arch in archetype_db.items():
        arch_dic = get_dic(arch)
        distance = np.sum(difference_matrix_generator(input_dic, arch_dic))
        distance_list.append(distance)

    mini_value = min(distance_list)
    mini_indices = [i for i, x in enumerate(distance_list) if x == mini_value]
    re_arch = list(archetype_str_list[mini_indices])
    return mini_value, re_arch, mini_indices


def minimum_distance_3d(col_matrix_3d):

    """ Compute the distance between requirement and archetypes in database and return those archetype names with
    minimum distance -- INPUT as a 3D matrix"""

    distance_list = []

    input_dic = get_dic(col_matrix_3d)

    for k1, arch in archetype_db.items():
        arch_dic = get_dic(arch)
        distance = np.sum(difference_matrix_generator(input_dic, arch_dic))
        distance_list.append(distance)

    mini_value = min(distance_list)
    mini_indices = [i for i, x in enumerate(distance_list) if x == mini_value]
    re_arch = list(archetype_str_list[mini_indices])
    return mini_value, re_arch, mini_indices


if __name__ == "__main__":
    distance_matrix_standard = np.array([[0, 14, 18, 16, 16, 16, 16, 16],
                                         [14,  0, 20,  6,  5,  2,  2,  5],
                                         [18, 20,  0, 22, 22, 22, 22, 22],
                                         [16,  6, 22,  0,  1,  4,  4,  2],
                                         [16,  5, 22,  1,  0,  3,  3,  1],
                                         [16,  2, 22,  4,  3,  0,  0,  4],
                                         [16,  2, 22,  4,  3,  0,  0,  4],
                                         [16,  5, 22,  2,  1,  4,  4,  0]])
    print(generate_distance_matrix())
    # print(np.array_equal(distance_matrix_standard, generate_distance_matrix()))


