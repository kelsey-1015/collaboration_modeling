import numpy as np
import variants as v
# import distance_calculation as dis


""" This script is used to select the archetype with perfect match. It catches the signatures of each archetype and 
can recognize those with various dimensions of"""


# check table for different collaboration values
# A search table with all the possible collaboration values for the current arch variants
# value_check_table = np.array([[1, 1, 2], [1, 2, 2], [2, 2, 2], [1, 2, 1]])
# arche_index = np.array(['Archetype IV',
#                        'Archetype V', 'Archetype VI OR VII','Archetype VIII'])
#
#
# str_no_match = "Sorry! No archetype is perfectly matched in current database!"


def evaluation_method_1(matrix_scope, rowwise, matrix_dim, trust_value):

    matrix_eval = matrix_scope[: matrix_dim - 1, : matrix_dim - 1]  # delete the last column and last row
    num_eval = matrix_dim - 1  # dimension of the evaluation matrix

    matrix_index = np.count_nonzero(matrix_eval, axis=rowwise)  # array indicating the number of non-zero value in each column
    check_1 = (np.count_nonzero(matrix_index) == 1)  # check only one column contains nonzero values

    num_zero = np.count_nonzero(matrix_eval == 0)
    num_value = np.count_nonzero(matrix_eval == trust_value)
    check_2_zero = (num_zero == (num_eval * num_eval - num_eval + 1))
    check_2_value = (num_value == (num_eval - 1))
    check_dia = np.all(matrix_eval.diagonal() == 0)

    check = check_1 and check_2_zero and check_2_value and check_dia
    return check


def evaluation_method_2(matrix_1, matrix_2):

    # separate matrices into light grey and blue area
    matrix_eval_1 = matrix_1[:-1, -1]
    matrix_eval_2 = matrix_2[:-1, -1]

    # check light grey area for first matrix has required pattern -- all zero values
    matrix_index_col = np.count_nonzero(matrix_1, axis=0)
    matrix_index_row = np.count_nonzero(matrix_1, axis=1)
    matrix_index = np.append(matrix_index_col[:-1], matrix_index_row[-1])
    flag_1 = (not np.any(matrix_index))

    # check the DMP column has required pattern for individual matrix
    flag_2 = (np.unique(matrix_eval_1).size == 2) and (np.count_nonzero(matrix_eval_1 == 0) == 1)
    # Flag indicating the correct format for each matrix
    flag_each_1 = (flag_1 and flag_2)

    # check light grey area for first matrix has required pattern -- all zero values
    matrix_index_col = np.count_nonzero(matrix_2, axis=0)
    matrix_index_row = np.count_nonzero(matrix_2, axis=1)
    matrix_index = np.append(matrix_index_col[:-1], matrix_index_row[-1])
    flag_1 = (not np.any(matrix_index))

    # check the DMP column has required pattern for individual matrix
    flag_2 = (np.unique(matrix_eval_2).size == 2) and (np.count_nonzero(matrix_eval_2 == 0) == v.num_party-2)
    # Flag indicating the correct format for each matrix
    flag_each_2 = (flag_1 and flag_2)

    # check the non-zero elements of two matrices do not overlap
    eval_index_1 = np.where(matrix_eval_1 == 0)[0]
    eval_index_2 = np.where(matrix_eval_2 == 0)[0]
    eval_index = np.append(eval_index_1, eval_index_2)
    flag_joint = (np.unique(eval_index).size == v.num_party - 1)

    return flag_joint and flag_each_1 and flag_each_2


def location_transpose(matrix_1, matrix_2):

    matrix_loc_1 = (matrix_1 == 0)

    matrix_loc_2 = (matrix_2 == 0)

    flag_trans_loc = np.array_equal(matrix_loc_1, matrix_loc_2.transpose())

    return flag_trans_loc


def get_col_values(matrix):
    matrix_eval = np.unique(matrix)
    return np.sort(matrix_eval)[-1]


def mapping_values(value_1, value_2, value_3):
    """ Select the archetype due to different collaboration level combinations"""
    value = np.array([value_1, value_2, value_3])
    index_table = np.all(value == v.value_check_table, axis=1)
    if np.any(index_table):  # if there is a match in the table
        index = np.where(index_table)[0]
        return v.arche_index[index]

    else:
        return v.str_no_match


def algorithm_calc(matrix_data, matrix_algo, matrix_result):

    # global output_str
    # flag that the collaboration algorithm nonzero
    flag_data = np.any(matrix_data)
    flag_algorithm = np.any(matrix_algo)
    flag_result = np.any(matrix_result)

    if not flag_algorithm:        # if algorithm matrix contains all zero
        if flag_result:           # if algo=0; result!=0;
            output_str = v.str_no_match
        else:  # if algo=0, result=0
            if evaluation_method_1(matrix_data, 0, v.num_party, 2):  # if algo=0, result=0, data fulfill the requirements
                output_str = "Archetype I"
            else:                                           # if algo=0, result!=0, data not match the requirement
                output_str = v.str_no_match
    else:               # if algo !=0

        if not flag_data:   # if data=0
            check_1 = evaluation_method_1(matrix_result, 0, v.num_party, 2)
            check_2 = evaluation_method_1(matrix_algo, 1, v.num_party, 1)
            check_3 = location_transpose(matrix_result, matrix_algo)
            if check_1 and check_2 and check_3:
                output_str = "Archetype III"
            else:
                output_str = v.str_no_match
        else:  # if algo!=0,data!=0
            if evaluation_method_2(matrix_data, matrix_algo):
                # if algo!=0,data!=0, matrix_algo and matrix_data fulfill conditions
                if not flag_result:   # if matrix_result =0
                    value_data = get_col_values(matrix_data)
                    value_algo = get_col_values(matrix_algo)

                    if value_data == 2 and value_algo == 2:
                        output_str = 'Archetype II'
                    else:
                        output_str = v.str_no_match

                else:  # if result!=0

                    if location_transpose(matrix_algo, matrix_result):

                        value_data = get_col_values(matrix_data)
                        value_algo = get_col_values(matrix_algo)
                        value_result = get_col_values(matrix_result)
                        result_table = mapping_values(value_data, value_algo, value_result)
                        output_str = result_table

                    else:
                        output_str = v.str_no_match

            else:    # if algo!=0,data!=0,matrix_algo and matrix_data do not full

                output_str = v.str_no_match

    return output_str

#
# def arch_select():
#     r = algorithm_calc(v.matrix_data, v.matrix_algorithm, v.matrix_result)
#     if r != v.str_no_match:
#         r = r + 'Perfect match'
#         return r
#     else:
#         return dis.minimum_distance(dis.input_matrix)[0]



""" Only run this function when directly run, rather than as import
if __name__ == '__main__':
    algorithm_calc(v.matrix_data, v.matrix_algorithm, v.matrix_result)

"""









