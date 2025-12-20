# -*- ecoding: utf-8 -*-
# @Project : EMNH
# @Author: Hangyu Lou
# @Time: 2024/9/3 16:58 

import os
import numpy as np
import pandas as pd
import copy
import ast


def read_Kp_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        obj_num = ast.literal_eval(lines[0].strip())
        item_num = ast.literal_eval(lines[1].strip())
        capcity = ast.literal_eval(lines[2].strip())
        values = ast.literal_eval(lines[3].strip())
        weight = ast.literal_eval(lines[4].strip())

        # 只对重量进行归一化，不对值进行归一化
        normalize_weight = [item/capcity for item in weight]
    return normalize_weight, values

path = r'D:\Paper Data\paper 3\instance\KP_benchmark'
for file in [f for f in os.listdir(path) if f.endswith('.dat')]:
    file_path = path + f'/{file}'
    print(file_path)
    weight, values = read_Kp_file(file_path)
    weight = np.array(weight).reshape(len(weight),1)
    values = np.array(values).reshape(-1,len(values))
    merged_data = np.concatenate((weight,values),axis=1)
    # Save to CSV
    output_file_path = file_path[:-4] + '.txt'
    np.savetxt(output_file_path, merged_data)
    print('修改完毕!')

