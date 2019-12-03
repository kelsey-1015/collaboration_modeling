import numpy as np
import math
import itertools
import operator
import plot
import statistics
import matplotlib.pyplot as plt

""" This script calcultes the distance from CR and one specific archetype and output the distance information in csv 
file"""


""" Initialize the collaboration matrices in each group"""
num_party = 4
num_scope = 3
matrix_data = np.zeros((num_party, num_party))
matrix_algorithm = np.zeros((num_party, num_party))
matrix_result = np.zeros((num_party, num_party))

"""Effective entr `ies in a given 3D collaboration matrix: total number - diagonal entries"""
num_effective = (num_party * num_party - num_party) * num_scope

"""Collaboration levels under definition"""
cl_level = [1, 2]

"""Initialize mapping function from collaboration matrix into DMP personas, dependent with deployment
Currently [d,d,a,dmp]"""
mapping_dic = {0: 11, 1: 11, 2: 22, 3: 33}

# basic primitives of all 7 archetypes
primitive_1 = {'[11, 22]': [2, 0, 0]} # achetype I, without intermediate infrastructure
primitive_2 = {'[11, 33]': [2, 0, 0], '[22, 33]': [0, 2, 0]} # achetype II, with I, without input all zero inputs
primitive_3 = {'[22, 11]': [0, 1, 0], '[11, 22]': [0, 0, 2]} # achetype III, without intermediate infrastructure
primitive_4 = {'[11, 33]': [1, 0, 0], '[22, 33]': [0, 1, 0], '[33, 22]': [0, 0, 2]} # achetype IV
primitive_5 = {'[11, 33]': [1, 0, 0], '[22, 33]': [0, 2, 0], '[33, 22]': [0, 0, 2]} # achetype V
primitive_6 = {'[11, 33]': [2, 0, 0], '[22, 33]': [0, 2, 0], '[33, 22]': [0, 0, 2]} # achetype VI/VII
primitive_8 = {'[11, 33]': [1, 0, 0], '[22, 33]': [0, 2, 0], '[33, 22]': [0, 0, 1]} # achetype VIII


ref_index = 0
archetype_indexes = [1, 2, 3, 4, 5, 6, 8]
# primitives without usage of intermediate infrastructure
primitives_index_OI = [1, 3]
primitives_index_I = [2, 4, 5, 6, 8]
ref_primitive = {}
primitive_db =[ref_primitive, primitive_1, primitive_2, primitive_3, primitive_4, primitive_5, primitive_6, primitive_6,
               primitive_8]


def distribution_generator():
    """ Generate combination of non_negative integers with sum == sum"""
    elements = range(num_party+1)
    distribution_list = []
    for i in itertools.product(elements, repeat=3):
        i = list(i)
        if sum(i) == num_party and i[0] != 0 and i[1] != 0:
            distribution_list.append(list(i))
    return distribution_list


def combination_caculation(n, k):

    """ Take k from n, n denotes number of avaiable primitives, k is number of participating in constructed
     archetype """
    # n>=1
    if n < 0:
        return('INVALID NEGATIVE NUMBER')
    elif n == 0:
        return 0
    else:
        r = math.factorial(n+k-1)/(math.factorial(k)*math.factorial(n-1))
    return int(r)


def extensibility_individual(archetype_index_set):
    # get the number of original archetypes before extension
    N_A_o = len(archetype_index_set)
    # Initial the number of original archetypes before extension
    N_A_e = 0
    distribution_list = distribution_generator()

    for distribution in distribution_list:
        persona_distribution = distribution

        primitive_set = []
        primitive_set_OI = []
        primitive_set_I = []

        # calcute the number of primitivis required under current persons distribution -- HOW MANY POSITIONS
        N_p_r = max(persona_distribution)


        # when no intermediate infrastructure involves
        if persona_distribution[-1] == 0:
            # only use primitives without infrastructure
            for index in archetype_index_set:
                if index in primitives_index_OI:
                    primitive_set.append(primitive_db[index])

            N_p_a = len(primitive_set) # number of available primitives -- How MANY TYPE OF BALLS AVAILABLE
            if N_p_a == 0:
                N_A_e_i = 0
            else:
                N_A_e_i = combination_caculation(N_p_a, N_p_r)

        else:  # persona_distribution[-1] != 0, at least one primitive inclusing infrastructure take part in
            for index in archetype_index_set:
                if index in primitives_index_OI:
                    primitive_set_OI.append(primitive_db[index])
                else:
                    primitive_set_I.append(primitive_db[index])
                primitive_set.append(primitive_db[index])
            N_p_a = len(primitive_set)
            N_p_a_oi = len(primitive_set_OI)
            if len(primitive_set_I) == 0: # no infrasture primitive available, no construction feasible
                N_A_e_i = 0
            else: # total number of archetypes with extension - those contructed without I
                N_A_e_total = combination_caculation(N_p_a, N_p_r )

                N_A_e_oi = combination_caculation(N_p_a_oi, N_p_r)
                N_A_e_i = N_A_e_total - N_A_e_oi
        #
        # if N_A_e_i != 0:
        #     print(distribution, N_A_e_i)
        N_A_e = N_A_e + N_A_e_i
    extensibility = 1 - float(N_A_o)/float(N_A_e)
    # extensibility = round(extensibility * 10000) / 10000
    return extensibility
    # return N_A_e


def extensibility_box_plot():
    """ This function generates necessary results for box plots """
    # iterate over archetyoe_set_size, which equavalent to primitive size in our case
    e_archetype_size_all = []
    for archetype_set_size in range(1, len(archetype_indexes)+1):
        # print(archetype_set_size)
        e_archetype_size_specific = []
        for archetype_index_set in itertools.combinations(archetype_indexes, archetype_set_size):
            archetype_index_set = list(archetype_index_set)
            # print(archetype_index_set)
            e = extensibility_individual(archetype_index_set)
            e_archetype_size_specific.append(e)
        e_archetype_size_all.append(e_archetype_size_specific)
    return e_archetype_size_all


def extensibility_dic_generator():
    dict_extensibility_information = {}
    for archetype_set_size in range(1, 8):
        # print("set_size", archetype_set_size)
        extensibility_fixed_setsize = extensibility_data[archetype_set_size-1]
        # print(extensibility_fixed_setsize)
        archetype_set_index = 0
        for archetype_set in itertools.combinations(archetype_indexes, archetype_set_size):
            archetype_set = list(archetype_set)
            dict_extensibility_information[str(archetype_set)] = extensibility_fixed_setsize[archetype_set_index]
            archetype_set_index += 1
    return dict_extensibility_information


if __name__ == "__main__":
    extensibility_data = extensibility_box_plot()
    extensibility_dict = extensibility_dic_generator()
    A = extensibility_dict
    print(A["[1, 2, 3, 6, 8]"])
    # newA = dict(sorted(list(A.items()), key=operator.itemgetter(1), reverse=True)[:10])
    # print(newA)
    # print(extensibility_data)
    # average_list =[]
    # std_list = []
    # for extensibility_list in extensibility_data:
    #     print(extensibility_list)
    #     average_list.append(sum(extensibility_list)/float(len(extensibility_list)))
    #     if len(extensibility_list) ==1:
    #         std_list.append(0)
    #     else:
    #         std_list.append(statistics.stdev(extensibility_list))
    # print(average_list)
    # print(std_list)
    # # plot.plot_multiple_data_sets(average_list, std_list, "Mean", 'Standard Deviation', plot.set_size_plot_label)
    # plot.normal_plot(average_list, plot.set_size_plot_label, "Average", "Archetype Set Size", 'average.png')
    # plot.normal_plot(std_list, plot.set_size_plot_label, "Standard Deviation", "Archetype Set Size", 'std.png')

    # print(extensibility_dic_generator())
    # print(distribution_generator())
    # fig = plt.figure()
    # plt.boxplot(extensibility_data, patch_artist=True, labels=['1', '2', '3', '4', '5', '6', '7'])
    # fig.suptitle('extensibility over different archetype set', fontsize=20)
    # plt.xlabel('Archetype set size', fontsize=16)
    # plt.ylabel('extensibility', fontsize=16)
    # plt.grid()
    # plt.show()
    # fig.savefig('extensibility.png')
    # archetype_input_index =[4]
    # print(extensibility_individual(archetype_input_index))