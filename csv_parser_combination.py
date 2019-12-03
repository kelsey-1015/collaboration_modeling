import csv
from itertools import combinations

""" This script deals with csv file with specific archetype and non_zero number;
Extract all CRs with distance less than d_a into a list;"""
# for N_c = [0, 7], the index of each element is its corresponding non-zero numbers
collaboration_id_lenghths = [0, 2, 4, 6, 8, 10, 12, 14]
# initial the non_zero number of processed csv file -- layer
N_c = 4
d_a = 6
collaboration_id_lenghth = collaboration_id_lenghths[N_c]
# initialize the size of each archetype set
archetype_set_size = 2
# archetype set, Archetype VI and VII are always identical
archetypes = [1, 2, 3, 4, 5, 6, 8]

input_path = 'N_04/'

output_file_name = "coverage_combi_2_N_04.csv"
# print(output_file_name)

for archetype_set in combinations(archetypes, archetype_set_size):
    cr_list = []
    for archetype_number in archetype_set:
            filename = 'A_' + str(archetype_number) + '_N_0_4.csv'
            processed_filename = input_path+filename
            print(processed_filename)
            with open(processed_filename) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')
                # initialize a list to instore cr_ids that meet the requirements
                # just for _0_4 case
                for row in csv_reader:
                    if int(row[1]) <= d_a:
                        cr_id = eval(row[0])
                        if cr_id not in cr_list:
                            cr_list.append(cr_id)

    with open(output_file_name, mode='a') as cr_distance_file:
        cr_distance_writer = csv.writer(cr_distance_file, delimiter=';')
        cr_distance_writer.writerow([archetype_set, len(cr_list)])
