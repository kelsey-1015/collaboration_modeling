import itertools

num_effective = 36
cl_level = [2, 1]


def loop_time_calculator(num_nz, num_v=num_effective):
    """ Out number of possible customer requests with a fixed number of non-zero values
    """
    outerloop_num = len(list(itertools.combinations(range(num_v), num_nz)))
    interloop_num = len(list(itertools.product(cl_level, repeat=num_nz)))
    return outerloop_num*interloop_num


def total_number_list_normal(N_c_nz_max):
    """ This number gives the total cr number with an upper limit of non_zero values: N_c_nz_max"""
    total_number_list = []
    for n in range(N_c_nz_max+1):
        print(n)
        x = loop_time_calculator(n)
        total_number_list.append(x)
    return total_number_list


if __name__ == "__main__":
    # print(loop_time_calculator(1, 36))
    cr_list = total_number_list_normal(6)
    total_1 = sum(cr_list)
    # print(total_1)
    total_2 = loop_time_calculator(4, 32)
    # print(total_number_list_normal())
    print(total_1+total_2)


