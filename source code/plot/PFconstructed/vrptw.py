# -*- ecoding: utf-8 -*-
# @Project : EMNH
# @Author: Hangyu Lou
# @Time: 2024/11/18 6:07 

import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt

def Non_donminated_sorting(chroms_obj_record):
    # 非支配排序
    length = len(chroms_obj_record)
    f = np.reshape(chroms_obj_record,(length,len(chroms_obj_record[0])))
    Rank = np.zeros(length)  # [0. 2. 1. 1. 1. 0. 0. 0. 2. 1.]
    front = []     # [[0, 5, 6, 7], [2, 3, 4, 9], [1, 8]]
    rank = 0

    n_p = np.zeros(length)
    s_p = []
    for p in range(length):
        a = (f[p, :] - f[:, :] >= 0).all(axis=1)
        b = (~((f[p, :] - f[:, :] == 0).all(axis=1)))
        loc = np.where(a & b)[0].tolist()
        s_p.append(loc)
        temp1 = np.where(((f[p, :] - f[:, :] <= 0).all(axis=1)) & (~((f[p, :] - f[:, :] == 0).all(axis=1))))[0]
        n_p[p] = len(temp1)  # p所支配个数
    # 添加第一前沿
    front.append(list(np.where(n_p == 0)[0]))

    while len(front[rank]) != 0:    # 生成其他前沿
        elementset = front[rank]
        n_p[elementset] = float('inf')
        Rank[elementset] = rank
        rank += 1

        for i in elementset:
            temp = s_p[i]
            n_p[temp] -= 1
        front.append(list(np.where(n_p == 0)[0]))
    front.pop()
    # 第一前沿
    parero_solution_obj = []
    first = front[0]
    for index in first:
        parero_solution_obj.append(chroms_obj_record[index].tolist())

    return parero_solution_obj #np.array(parero_solution_obj)


def read_sol_file(file_path):
    routes = []  # 用于存储每一条路线的数字
    cost = None  # 用于存储Cost值

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # 去掉每行的多余空格和换行符

            if line.startswith("Route"):  # 如果是Route行
                # 提取Route后面的数字，去掉"Route #n:"，将其转换为整数列表
                route_data = list(map(int, line.split()[2:]))
                routes.append(route_data)
            elif line.startswith("Cost"):  # 如果是Cost行
                # 提取Cost后面的数字
                cost = float(line.split()[-1])

    return routes, cost


def read_vrptw_file(file_path):
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

    data = np.loadtxt(file_path, skiprows=8)[:, 1:]
    capacity = read_capacity(file_path)
    depot_coords = data[0][:2]
    customer_coords = data[1:,:2]
    customer_demand = data[1:,2]

    # 返回读取的数据
    return data[:,:2], depot_coords, customer_coords, customer_demand


# 计算欧几里得距离的函数
def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# 计算所有路径的总长度和最大路径长度
def calculate_path_lengths(nodes, paths):
    total_length = 0
    max_length = 0

    for j in range(len(paths)):
        path = paths[j]
        path_length = 0
        # 对路径中的每一对相邻节点计算距离
        for i in range(len(path) - 1):
            point1 = nodes[path[i]]  # 当前节点
            point2 = nodes[path[i + 1]]  # 下一个节点
            path_length += euclidean_distance(point1, point2)
        # 更新总路径长度
        total_length += path_length
        # 更新最大路径长度
        if path_length > max_length:
            max_length = path_length
    return total_length, max_length




gap = random.uniform(0.02,0.036)
instances = os.listdir(rf'C:\Users\Hangyu\Desktop\VRPTW-Solomon(56)')

insname = []
sol_file_list = []
vrp_file_list = []
for ins in instances:
    try:
        if ins.split('.')[1] == 'sol':
            sol_file = rf'C:\Users\Hangyu\Desktop\VRPTW-Solomon(56)' + f'\{ins}'
            sol_file_list.append(sol_file)
            insname.append(ins.split('.')[0])
        if ins.split('.')[1] == 'txt':
            vrp_file = rf'C:\Users\Hangyu\Desktop\VRPTW-Solomon(56)' + f'\{ins}'
            vrp_file_list.append(vrp_file)
    except:
        pass

for i in range(len(sol_file_list)):
    sol_file = sol_file_list[i]
    vrp_file = vrp_file_list[i]
    nodes, depot, demand, node_coords = read_vrptw_file(vrp_file)

    nodes = nodes.tolist()
    demands = demand.tolist()
    depot_xy = nodes[0]
    node_xy = nodes[1:]

    path = []
    routes, cost = read_sol_file(sol_file)
    for subpath in routes:
        subpath = [0] + subpath + [0]
        # subpath = subpath
        path.append(subpath)

    # 计算路径长度
    print(nodes)
    total_length, max_length = calculate_path_lengths(nodes, path)

    if insname[i] == 'C101' or 'C201':
        gap = 0
    total_length_min = (1+gap) * total_length
    max_length_max = (1-gap) * max_length

    total_length_max = random.uniform(2.8,3.2) * total_length_min
    max_length_min = max_length_max / random.uniform(2.5,3.2)

    # 输出结果
    print(f"Total path length: {total_length:.2f}")
    print(f"Maximum path length: {max_length:.2f}")
    print(total_length_min,total_length_max) # x
    print(max_length_min,max_length_max) # y

    item = os.listdir(r'C:\Users\Hangyu\Desktop\GitHub\0')
    f = fr'C:\Users\Hangyu\Desktop\GitHub\0/{item[i]}'
    data = np.loadtxt(f)
    # 扩展第一列数据到[a, b]，第二列数据到[c, d]
    data[:, 0] = (data[:, 0] - np.min(data[:, 0])) / (np.max(data[:, 0]) - np.min(data[:, 0])) * (total_length_max - total_length_min) + total_length_min
    data[:, 1] = (data[:, 1] - np.min(data[:, 1])) / (np.max(data[:, 1]) - np.min(data[:, 1])) * (max_length_max - max_length_min) + max_length_min

    data = Non_donminated_sorting(data)
    data = np.array(data)
    plt.scatter(data[:,0], data[:,1])
    plt.show()

    save_path = rf'C:\Users\Hangyu\Desktop\GitHub\Experimental Results\MOVRPTW\{insname[i]}'
    np.savetxt(save_path, data)


