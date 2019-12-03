import tkinter as tk
from tkinter import simpledialog
import numpy as np
import variants as v
import selection_algorithm as selc
import distance_calculation as dis

'''This GUI inputs customer-required collaboration matrix and output the optimum archetype with closet distance'''

root = tk.Tk()
root.title('Archetype Selection')

v.num_party = int(simpledialog.askstring("Query", "How many parties are participating?"))

""" Refresh the dimension of collaboration matrix"""
v.matrix_data = np.zeros((v.num_party, v.num_party))
v.matrix_algorithm = np.zeros((v.num_party, v.num_party))
v.matrix_result = np.zeros((v.num_party, v.num_party))

master = tk.Frame(root)
master.pack()

test_pad = tk.Frame(root)
test_pad.pack()

S = tk.Scrollbar(test_pad)
T = tk.Text(test_pad, height=5, width=40)
S.pack(side=tk.RIGHT, fill=tk.Y)
T.pack(side=tk.LEFT, fill=tk.Y)
S.config(command=T.yview)
T.config(yscrollcommand=S.set)


def on_press(matrix_scope):

    input_data = []
    for row in rows_outside:
        for col in row:
            input_data.append(int(col.get()))
    input_data = np.array(input_data)
    input_data = input_data.reshape(v.num_party, v.num_party)
    if matrix_scope == 'data':
        v.matrix_data = input_data
        print("initialize data matrix: \n", v.matrix_data)
        text_str = "You have input collaboration matrix in data scope: \n"+np.array_str(v.matrix_data)
        T.delete('1.0', tk.END)
        T.insert(tk.END, text_str)
    elif matrix_scope == 'algorithm':
        v.matrix_algorithm = input_data
        print("initialize algorithm matrix: \n", v.matrix_algorithm)
        text_str = "You have input collaboration matrix in algorithm scope: \n"+np.array_str(v.matrix_algorithm)
        T.delete('1.0', tk.END)
        T.insert(tk.END, text_str)
    elif matrix_scope == 'result':
        v.matrix_result = input_data
        print("initialize  result matrix: \n", v.matrix_result)
        text_str = "You have input collaboration matrix in output result scope: \n"+np.array_str(v.matrix_result)
        T.delete('1.0', tk.END)
        T.insert(tk.END, text_str)
    else:
        T.delete('1.0', tk.END)
        T.insert(tk.END, "Sorry! No archetype is perfectly matched in current database!")


def make_format(upper_space):
    """Arrange an entry table for 2D collaboration matrix input'"""
    rows = []
    for i in range(v.num_party):
        cols = []
        for j in range(v.num_party):
            e = tk.Entry(master, relief=tk.RIDGE, width=10)
            e.grid(row=i+upper_space, column=j, sticky=tk.NSEW)
            e.insert(tk.END, '0')
            cols.append(e)
        rows.append(cols)
    return rows


def clean_up():
    for row in rows_outside:
        for col in row:
            col.delete(0, tk.END)
            col.insert(tk.END, '0')
    T.delete('1.0', tk.END)
    T.insert(tk.END, "Please input another matrix or get the result!")


def select_archetype():

    r = selc.algorithm_calc(v.matrix_data, v.matrix_algorithm, v.matrix_result)
    if r != v.str_no_match:
        result_str = r + ' -- Perfect match!'
    else:
        archetype = dis.minimum_distance(v.matrix_data, v.matrix_algorithm, v.matrix_result)[1]
        distance = dis.minimum_distance(v.matrix_data, v.matrix_algorithm, v.matrix_result)[0]
        result_str = str(archetype) + " with distance:" + str(distance)

    T.delete('1.0', tk.END)
    T.insert(tk.END, result_str)


def reset():
    v.matrix_data = np.zeros((v.num_party, v.num_party))
    v.matrix_algorithm = np.zeros((v.num_party, v.num_party))
    v.matrix_result = np.zeros((v.num_party, v.num_party))
    for row in rows_outside:
        for col in row:
            col.delete(0, tk.END)
            col.insert(tk.END, '0')
    T.delete('1.0', tk.END)


rows_outside = make_format(0)

tk.Button(master, text='NEW TABLE', width=16, justify=tk.LEFT, command=clean_up)\
    .grid(row=0, column=v.num_party+1)
tk.Button(master, text='SUBMIT_DATA', width=16, justify=tk.LEFT, command=lambda: on_press('data'))\
    .grid(row=1, column=v.num_party+1)
tk.Button(master, text='SUBMIT_ALGO', width=16, justify=tk.LEFT, command=lambda: on_press('algorithm'))\
    .grid(row=2, column=v.num_party+1)
tk.Button(master, text='SUBMIT_RESULT', width=16, justify=tk.LEFT, command=lambda: on_press('result'))\
    .grid(row=3, column=v.num_party+1)

tk.Button(master, text='RUN', width=16, justify=tk.LEFT, command=select_archetype).grid(row=0, column=v.num_party+2)
tk.Button(master, text='RESET', width=16, justify=tk.LEFT, command=reset).grid(row=3, column=v.num_party+2)


root.mainloop()

