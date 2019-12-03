import csv
import subprocess
from itertools import combinations


archetypes = [1, 2, 3, 4, 5, 6, 8]
coverage_setsize_list=[]
for archetype_set_size in range(1, 8):
    covered_num_list =[]
    cmd =[]
    for archetype_set in combinations(archetypes, archetype_set_size):

        file_name_list =[]
        for archetype_index in archetype_set:
            file_name = 'Filter_3_A_' + str(archetype_index) + '.csv'
            file_name_list.append(file_name)

        # print(file_name_list[0])
        cmd = ' '.join(file_name_list)
        # print(cmd)
        cmd = 'cat '+cmd + ' > '+'tmp.csv'
        # print(cmd)
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        cmd = "sort -u tmp.csv -o tmp.csv"
        # print(cmd)
        process = subprocess.Popen(cmd, shell=True)
        process.wait()
        cmd = "cat tmp.csv |wc -l"
        process = subprocess.Popen(cmd, shell=True,stdout=subprocess.PIPE)
        process.wait()
        out = process.communicate()[0].decode()

        covered_num_list.append(int(out))
    print(len(covered_num_list))
    final_result = open("coverage_archetype_size_old.txt", 'a')
    final_result.write('Archetype set size: '+str(archetype_set_size)+'\n')
    final_result.write(str(covered_num_list)+'\n')

    final_result.close()


