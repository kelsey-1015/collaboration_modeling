import csv
from itertools import combinations

""" This script deals with csv file with specific archetype and non_zero number;
Extract all CRs with distance less than d_a into a list;"""

d_a = 4
archetypes = [1, 2, 3, 4, 5, 6, 8]

input_filename = 'Filter_6_A_1.csv'

output_filename = "Filter_4_A_1.csv"

with open(input_filename, 'r') as f_in, open(output_filename, 'w') as f_out:
    writer = csv.writer(f_out, delimiter=' ')
    for row in csv.reader(f_in, delimiter=' '):
        if int(row[1]) <= d_a:
            writer.writerow(row)

