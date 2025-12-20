# -*- ecoding: utf-8 -*-
# @Project : EMNH
# @Author: Hangyu Lou
# @Time: 2024/11/23 16:27
import numpy as np


def read_cvrp_data(file_path):
    with open(file_path, 'r') as f:
        # 跳过前四行，读取数据并转换为 np.array
        return np.array([list(map(float, line.split())) for line in f.readlines()[4:] if line.strip() and len(line.split()) == 3])

 

# 调用函数并打印结果
filename = r"C:\Users\Hangyu\Desktop\GitHub\Test Data\Random Test Set\MOCVRP\MOCVRP_20_random_instance1.txt"  # 替换为你的文件名
data = read_cvrp_data(filename)
print(data)
