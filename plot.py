import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
""" This scripts plots data"""
num_cl = 3
num_efficient =36
total_num_CR = num_cl**num_efficient
d_a = 4
# number of
# total_num_CR = 8952874705
# total_num_CR = 1206803025
# total_num_CR = 138299985
archetype_size_1_d_6 = [145772, 153724, 229424, 82174, 82174, 82174, 82174]


archetype_size_2_d_6  = [298974, 374718, 227448, 227448, 227448, 227448, 382620, 228222, 229670, 233914, 
                    229670, 311542, 311542, 311542, 311542, 157731, 149393, 149178, 162740, 157731, 
                    149393]

archetype_size_3_d_6  = [527394, 373192, 374427, 378666, 374427, 456342, 456342, 456342, 456342, 302753, 
                    294186, 293974, 307542, 302753, 294186, 457078, 458522, 462754, 458522, 297803, 
                    293457, 292738, 308252, 301178, 294905, 387063, 378721, 378514, 392052, 387063, 
                    378721, 223613, 218310, 208664, 223613]

archetype_size_4_d_6  = [601576, 602807, 607034, 602807, 442524, 437946, 437233, 452537, 445687, 439181, 
                    531611, 523044, 522836, 536384, 531611, 523044, 368175, 363084, 352994, 368175, 
                    526635, 522273, 521566, 537048, 530002, 523717, 361701, 358008, 350240, 365076, 
                    452905, 447622, 437968, 452905, 276728]
archetype_size_5_d_6  = [670884, 666294, 665589, 680865, 674039, 667525, 505962, 502483, 494269, 509125, 
                    596997, 591922, 581828, 596997, 421052, 590493, 586820, 579036, 593860, 414442, 
                    506008]

archetype_size_6_d_6  = [734286, 730823, 722597, 737441, 558467, 649862, 643222]
archetype_size_7_d_6  = [786779]



archetype_size_1_d_4 = [6764, 7021, 16454, 5474, 5474, 5474, 5474]

archetype_size_2_d_4 =[13777, 23210, 12230, 12230, 12230, 12230, 23467, 12244, 12235, 12495, 12235, 
                    21928, 21928, 21928, 21928, 10746, 10452, 10065, 10539, 10746, 10452]

archetype_size_3_d_4 =[30215, 18996, 18983, 19243, 18983, 28676, 28676, 28676, 28676, 17498, 17200, 16813, 17287, 17498, 17200, 28690, 28681, 28941, 28681, 17260, 17222, 16699, 17300, 
                    17314, 17213, 27200, 26906, 26519, 26993, 27200, 26906, 15319, 15138, 14791, 15319]
archetype_size_4_d_4 =[35434, 35421, 35681, 35421, 24008, 23966, 23443, 24040, 24058, 23953, 33944, 33646, 
                        33259, 33733, 33944, 33646, 22063, 21886, 21531, 22063, 33706, 33668, 33145, 33746, 
                        33760, 33659, 21833, 21582, 21425, 21887, 31773, 31592, 31245, 31773, 19463]
archetype_size_5_d_4 =[40446, 40404, 39881, 40478, 40496, 40391, 28573, 28326, 28161, 28623, 38509, 38332, 
                        37977, 38509, 26203, 38279, 38028, 37871, 38333, 25907, 35917]
archetype_size_6_d_4 =[45011, 44764, 44599, 45061, 32643, 42649, 42353]

archetype_size_7_d_4 =[49081]
# total_num_CR = sum(archetype_size_7)

archetype_size_1_d_6 = np.array(archetype_size_1_d_6 )/total_num_CR
archetype_size_2_d_6 = np.array(archetype_size_2_d_6 )/total_num_CR
archetype_size_3_d_6 = np.array(archetype_size_3_d_6 )/total_num_CR
archetype_size_4_d_6 = np.array(archetype_size_4_d_6 )/total_num_CR
archetype_size_5_d_6 = np.array(archetype_size_5_d_6 )/total_num_CR
archetype_size_6_d_6 = np.array(archetype_size_6_d_6 )/total_num_CR
archetype_size_7_d_6 = np.array(archetype_size_7_d_6 )/total_num_CR

coverage_data_d_6 = [archetype_size_1_d_6 , archetype_size_2_d_6 , archetype_size_3_d_6 , archetype_size_4_d_6 ,
                 archetype_size_5_d_6 , archetype_size_6_d_6 , archetype_size_7_d_6]


archetype_size_1_d_4 = np.array(archetype_size_1_d_4 )/total_num_CR
archetype_size_2_d_4 = np.array(archetype_size_2_d_4 )/total_num_CR
archetype_size_3_d_4 = np.array(archetype_size_3_d_4 )/total_num_CR
archetype_size_4_d_4 = np.array(archetype_size_4_d_4 )/total_num_CR
archetype_size_5_d_4 = np.array(archetype_size_5_d_4 )/total_num_CR
archetype_size_6_d_4 = np.array(archetype_size_6_d_4 )/total_num_CR
archetype_size_7_d_4 = np.array(archetype_size_7_d_4 )/total_num_CR

coverage_data_d_4 = [archetype_size_1_d_4 , archetype_size_2_d_4 , archetype_size_3_d_4 , archetype_size_4_d_4 ,
                 archetype_size_5_d_4 , archetype_size_6_d_4 , archetype_size_7_d_4]
set_size_plot_label = ['1', '2', '3', '4', '5', '6', '7']


Archetype_label = ["I", "II", 'III', 'IV', 'V', 'VI', 'VII']


def box_plot(input_data, plot_label):
    """ generate boxplot from data; input data as nested list = [sublist1, sublist2, ...., sublist n];
    Each subset provides data for each box
     """

    fig = plt.figure()
    plt.boxplot(input_data, patch_artist=True, labels=plot_label, color = 'b')
    fig.suptitle('Coverage over different archetype sets', fontsize=20)
    plt.xlabel('Archetype set size', fontsize=16)
    plt.ylabel('Covered num of CMs', fontsize=16)
    plt.grid()
    plt.show()
    fig.savefig('Coverage_absolute_values.png')


def box_plot_multiple(input_data_1, input_data_2, plot_label):
    """ generate boxplot from data; input data as nested list = [sublist1, sublist2, ...., sublist n];
    Each subset provides data for each box
     """
    positions0 = [1,3,5,7,9, 11, 13]
    positions1=[2,4,6,8,10,12,14]
    fig = plt.figure()
    bp0=plt.boxplot(input_data_1,patch_artist=True, labels=plot_label)
    bp1=plt.boxplot(input_data_2, patch_artist=True, labels=plot_label)
    for box in bp0['boxes']:
        # change outline color
        box.set(color='red', linewidth=2)
        box.set(facecolor = 'blue' )

    for box in bp1['boxes']:
        # change outline color
        box.set(color='green', linewidth=2)
        box.set(facecolor = 'red' )
    # fig.suptitle('Coverage over different archetype sets', fontsize=20)
    plt.xlabel('Supported Archetype Set Size', fontsize=14)
    plt.ylabel('Coverage of DMPs', fontsize=14)
    plt.legend([bp0["boxes"][0], bp1["boxes"][0]], ["D_a =6", "D_a =4"], loc='upper left')
    plt.grid()
    plt.show()
    fig.savefig('Coverage_absolute_values.eps', format='eps', dpi=1000)


def bar_plot(input_data, x_label, y_label, file_name):
    """ Input data --> list"""

    fig = plt.figure()
    y_pos = np.arange(len(x_label))
    plt.bar(y_pos, input_data, align='center')
    plt.xticks(y_pos, x_label)
    plt.ylabel(y_label)
    # plt.title('Programming language usage')
    plt.show()
    plt.grid()
    fig.savefig(file_name)


def bar_plot_multiple(input_data_1, input_data_2, x_label, y_label, file_name):
    """ Input data --> list"""

    fig = plt.figure()
    y_pos = np.arange(len(x_label))

    Distance = ["D_a =6", "D_a =4"]
    
    plt.bar(y_pos, input_data_1, color = 'b', width = 0.5)
    plt.bar(y_pos, input_data_2, color = 'r', width = 0.5)

    plt.xticks(y_pos, x_label)
    plt.ylabel(y_label)
    plt.legend(Distance, loc =1)


    plt.show()

    

def normal_plot(input_data, x_labels, y_label, x_label, file_name):
    fig = plt.figure()
    y_pos = np.arange(len(x_labels))
    plt.plot(y_pos, input_data, 'bo', y_pos, input_data, 'r--')
    plt.xticks(y_pos, x_labels)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid()
    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'r--')
    # plt.axis([0, 6, 0, 20])
    plt.show()


def plot_multiple_data_sets(input_data_1, input_data_2, y_label_1, y_label_2, x_labels):
    fig = plt.figure()

    x_pos = np.arange(len(x_labels))

    ax1 = fig.add_subplot(111, label="coverage")
    ax2 = fig.add_subplot(111, label="average", frame_on=False)

    ax1.plot(x_pos, input_data_1, 'r--', x_pos, input_data_1, 'ro')
    ax1.set_ylabel(y_label_1, color="r")
    ax1.set_ylim([0, 2*max(input_data_1)])
    ax1.set_xticks([])
    ax1.tick_params(axis='y', colors="r")
    ax1.legend([y_label_1], loc=4)

    ax2.plot(x_pos, input_data_2, 'b--', x_pos, input_data_2, 'bo')
    ax2.yaxis.tick_right()
    ax2.set_ylabel(y_label_2, color="b")
    ax2.yaxis.set_label_position('right')
    ax2.set_xticks([])
    ax2.set_ylim([-1.4*max(input_data_2), 1.4*max(input_data_2)])
    ax2.tick_params(axis='y', cors="blue")
    ax2.legend([y_label_2], loc=1)

    # sparsely distributed x labels for better display
    # x_pos_sparse = x_pos[0::5]
    # x_labels = np.array(x_labels)
    # x_labels_sparse = x_labels[x_pos_sparse]

    plt.xticks(x_pos, x_labels)
    plt.xlabel("Archetype Set Size")
    
    plt.grid()

    plt.show()
    # plt.draw()
    # fig.savefig(file_name)


if __name__ == '__main__':
    # box_plot_multiple(coverage_data_d_6, coverage_data_d_4, set_size_plot_label)
    #
    # archetype_size_1_d_6 = np.array(archetype_size_1_d_6 )/total_num_CR
    # archetype_size_1_d_4 = np.array(archetype_size_1_d_4 )/total_num_CR
    bar_plot_multiple(archetype_size_1_d_6, archetype_size_1_d_4, Archetype_label, "Coverage", "Individual_coverage_compare" )