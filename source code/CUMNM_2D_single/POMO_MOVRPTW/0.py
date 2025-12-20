# -*- ecoding: utf-8 -*-
# @Project : EMNH
# @Author: Hangyu Lou
# @Time: 2024/11/23 17:04 
import os
import pandas
import numpy as np

def read_movrptw(file_path):
    def read_capacity(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # 遍历每行，查找 "CAPACITY" 关键字所在行
        for i, line in enumerate(lines):
            if 'CAPACITY' in line:  # 找到 CAPACITY 所在行
                # 确保后面至少有一行数据
                if i + 1 < len(lines):
                    # 获取下一行并提取两个数值
                    next_line = lines[i + 1].split()
                    if len(next_line) >= 2:  # 确保有两个数值
                        return float(next_line[1])  # 返回两个数值

    data = np.loadtxt(file_path, skiprows=8)[:,1:]
    capacity = read_capacity(file_path)
    data[:,2] /= capacity
    return data


for name in os.listdir(r'C:\Users\Hangyu\Desktop\VRPTW-Solomon(56)'):
    if name.split('.')[-1] == 'txt':
        # file_path = rf"C:\Users\Hangyu\Desktop\VRPTW-Solomon(56)\{name}"
        file_path = rf"C:\Users\Hangyu\Desktop\VRPTW-Solomon(56)\C101.txt"
        print(file_path)
        data = read_movrptw(file_path)

        np.savetxt(rf'C:\Users\Hangyu\Desktop\VRPTW-Solomon(56)\normal\{name}', data)
        print(name)
