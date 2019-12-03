import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import operator

num_cl = 3
num_efficient = 36
total_num_CR = num_cl**num_efficient

standard_mutual_distance_matrix = [[0, 10, 12, 12, 12, 12, 12, 12],
                                    [10, 0, 14, 5, 4, 2, 2, 4],
                                    [12, 14, 0, 16, 16, 16, 16, 16],
                                    [12, 5, 16, 0, 1, 3, 3, 2],
                                    [12, 4, 16, 1, 0, 2, 2, 1],
                                    [12, 2, 16, 3, 2, 0, 0, 3],
                                    [12, 2, 16, 3, 2, 0, 0, 3],
                                    [12, 4, 16, 2, 1, 3, 3, 0]]

""" Covered number of under various archetype set"""
archetype_size_1 = [145772, 153724, 229424, 82174, 82174, 82174, 82174]

archetype_size_2 = [297715, 373430, 227320, 227320, 227320, 227320, 382432, 225551, 220079, 218031,
                    220079, 311462, 311462, 311462, 311462, 133871, 139591, 139416, 139070, 133871,
                    139591]

archetype_size_3 = [524683, 369263, 363794, 361748, 363794, 454846, 454846, 454846, 454846, 278775,
                    284248, 284100, 283748, 278775, 284248, 454171, 448711, 446663, 448711, 268215,
                    271703, 273696, 269207, 271712, 269728, 363115, 368791, 368624, 368270, 363115,
                    368791, 187215, 182080, 193208, 187215]

archetype_size_4 = [596143, 590686, 588640, 590686, 411907, 415178, 417170, 412703, 415187, 413203,
                    506257, 511690, 511546, 511190, 506257, 511690, 331655, 326744, 337410, 331655,
                    496835, 500287, 502280, 497795, 500300, 498316, 313855, 316360, 319848, 317352,
                    416379, 411288, 422336, 416379, 235360]
archetype_size_5 = [638787, 642022, 644014, 639551, 642035, 640051, 457330, 459814, 463085, 460610,
                    559061, 554190, 564780, 559061, 379562, 542439, 544944, 548396, 545904, 362000,
                    464488]

archetype_size_6 = [684174, 686658, 689893, 687422, 505237, 606932, 590548]
archetype_size_7 = [732045]


archetype_size_1 = np.array(archetype_size_1)/total_num_CR
archetype_size_2 = np.array(archetype_size_2)/total_num_CR
archetype_size_3 = np.array(archetype_size_3)/total_num_CR
archetype_size_4 = np.array(archetype_size_4)/total_num_CR
archetype_size_5 = np.array(archetype_size_5)/total_num_CR
archetype_size_6 = np.array(archetype_size_6)/total_num_CR
archetype_size_7 = np.array(archetype_size_7)/total_num_CR

coverage_data_ref = 0
coverage_data = [coverage_data_ref, archetype_size_1, archetype_size_2, archetype_size_3, archetype_size_4,
                 archetype_size_5, archetype_size_6, archetype_size_7]

archetypes = [1, 2, 3, 4, 5, 6, 8]


def box_plot(input_data, plot_label):
    """ generate boxplot from data; input data as nested list = [sublist1, sublist2, ...., sublist n];
    Each subset provides data for each box
     """

    fig = plt.figure()
    plt.boxplot(input_data, patch_artist=True, labels=plot_label)
    fig.suptitle('coverage over different archetype set', fontsize=20)
    plt.xlabel('Archetype set size', fontsize=16)
    plt.ylabel('Covered num of CMs', fontsize=16)
    plt.grid()
    plt.show()
    fig.savefig('Coverage_absolute_values.png')


def bar_plot(input_data, x_label, y_label, file_name):
    """ Input data --> list"""

    fig = plt.figure()
    y_pos = np.arange(len(x_label))
    plt.bar(y_pos, input_data, align='center')
    plt.xticks(y_pos, x_label)
    plt.ylabel(y_label)
    # plt.title('Programming language usage')
    plt.show()
    fig.savefig(file_name)


def normal_plot(input_data, x_label, y_label, file_name):
    fig = plt.figure()
    y_pos = np.arange(len(x_label))
    plt.plot(y_pos, input_data, 'bo', y_pos, input_data, 'r--')
    plt.xticks(y_pos, x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.draw()
    plt.show()
    fig.savefig(file_name)


def coverage_dic_generator():
    dict_coverage_information = {}
    for archetype_set_size in range(2, 8):
        # print("set_size", archetype_set_size)
        coverage_fixed_setsize = coverage_data[archetype_set_size]
        archetype_set_index = 0
        for archetype_set in combinations(archetypes, archetype_set_size):
            archetype_set = list(archetype_set)
            dict_coverage_information[str(archetype_set)] = coverage_fixed_setsize[archetype_set_index]
            archetype_set_index += 1
    return dict_coverage_information


def distance_dic_generator():

    archetype_set_all = []
    # a dictionary whose key is archetype set, and value is list of corresponding mutual distances
    dict_distance_information = {}
    for archetype_set_size in range(2, 8):
        archetype_set_specific_size =[]

        for archetype_set in combinations(archetypes, archetype_set_size):
            archetype_set = list(archetype_set)
            # print("archetype_set:", archetype_set)
            # list for all mutual distances within a specific archetype set
            distance_archetype_set = []

            for mutual_combi in combinations(archetype_set, 2):
                mutual_combi = list(mutual_combi)
                source_index = mutual_combi[0] - 1
                target_index = mutual_combi[1] - 1
                mutual_distance = standard_mutual_distance_matrix[source_index][target_index]
                distance_archetype_set.append(mutual_distance)

            dict_distance_information[str(archetype_set)] = distance_archetype_set
            archetype_set_specific_size.append(archetype_set)

        archetype_set_all.append(archetype_set_specific_size)

    return dict_distance_information


def parse_distance_dict(input_dict):
    for key, value in input_dict.items():
        # print(value)
        average = sum(value)/float(len(value))
        input_dict[key] = average
        # print(average)
    return input_dict


def dict_to_plot(input_dict, archetype_set_size, y_label, filename):
    label_size_2 = []
    value_size_2 = []
    for key, value in input_dict.items():
        key = eval(key)
        if len(key) == archetype_set_size:
            label_size_2.append(str(key))
            value_size_2.append(float(value))
    normal_plot(value_size_2, label_size_2, y_label, filename)


def dict_to_plot_double(input_dict_1, input_dict_2, archetype_set_size, file_name):

    """ This function is mainly used for plotting the relationships with various archetype size """
    x_labels = []
    coverage_values = []
    average_values = []

    # parse dictionary with coverage
    for key, value in input_dict_1.items():
        key = eval(key)
        if len(key) == archetype_set_size:
            x_labels.append(str(key))
            coverage_values.append(float(value))

    # parse dictionary with mutual distance information -- average
    for key, value in input_dict_2.items():
        key = eval(key)
        if len(key) == archetype_set_size:
            average_values.append(float(value))

    fig = plt.figure()

    x_pos = np.arange(len(x_labels))

    ax1 = fig.add_subplot(111, label="coverage")
    ax2 = fig.add_subplot(111, label="average", frame_on=False)

    ax1.plot(x_pos, coverage_values, 'r--', x_pos, coverage_values, 'ro')
    ax1.set_ylabel("Coverage", color="r")
    ax1.set_ylim([0, 2*max(coverage_values)])
    ax1.set_xticks([])
    ax1.tick_params(axis='y', colors="r")
    ax1.legend(['Coverage'], loc=4)

    ax2.plot(x_pos, average_values, 'b--', x_pos, average_values, 'bo')
    ax2.yaxis.tick_right()
    ax2.set_ylabel("Average of mutual distance", color="b")
    ax2.yaxis.set_label_position('right')
    ax2.set_xticks([])
    ax2.set_ylim([-1.4*max(average_values), 1.4*max(average_values)])
    ax2.tick_params(axis='y', colors="b")
    ax2.legend(['Average mutual distance'], loc=1)

    # sparsely distributed x labels for better display
    x_pos_sparse = x_pos[0::5]
    x_labels = np.array(x_labels)
    x_labels_sparse = x_labels[x_pos_sparse]

    plt.xticks(x_pos_sparse, x_labels_sparse)
    plt.xlabel("various archetype sets of size 7")
    plt.grid()

    plt.show()
    # plt.draw()
    fig.savefig(file_name)


if __name__ == '__main__':

    coverage_dict = coverage_dic_generator()
    A = coverage_dict
    # newA = dict(sorted(list(A.items()), key=operator.itemgetter(1), reverse=True)[:30])
    # print(newA)
    archetype = '[1, 3, 4, 5, 8]'
    print(A[archetype])
    # dict_to_plot(coverage_dict, 2, "coverage", "coverage_size_2.png")

    # distance_list_dic = distance_dic_generator()
    # distance_average_dic = parse_distance_dict(distance_list_dic)
    # # print(distance_average_dic)
    # # dict_to_plot(distance_average_dic, 2, "average of mutual distance", "average_size_2.png")
    #
    # dict_to_plot_double(coverage_dict, distance_list_dic, 7, "average_coverage_size_7.png")
