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

def read_vrp_file(file_path):
    node_coords = {}  # 存储节点坐标，字典格式：node_id -> (x, y)
    demand = {}  # 存储需求，字典格式：node_id -> demand
    depot = None  # 存储仓库节点
    in_coords_section = False
    in_demand_section = False
    in_depot_section = False
    capcity = 0

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if line[:7] in "CAPACITY":
                capcity = int(line.split(':')[1])

            # 处理 NODE_COORD_SECTION 部分
            if line == "NODE_COORD_SECTION":
                in_coords_section = True
                continue
            elif line == "DEMAND_SECTION":
                in_coords_section = False
                in_demand_section = True
                continue
            elif line == "DEPOT_SECTION":
                in_demand_section = False
                in_depot_section = True
                continue
            elif line == "EOF":
                break

            # 读取节点坐标
            if in_coords_section:
                parts = line.split()
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                node_coords[node_id] = [x, y]

            # 读取需求
            elif in_demand_section:
                parts = line.split()
                node_id = int(parts[0])
                demand_value = int(parts[1])
                demand[node_id] = demand_value

            # 读取仓库
            elif in_depot_section:
                depot_id = int(line)
                if depot_id != -1:
                    depot = depot_id

    return capcity, node_coords, demand, depot


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


for i in range(1,9):
    sol_file = rf"C:\Users\Hangyu\Desktop\OVRP\C1-C8\CMT{i}.sol"
    vrp_file = rf"C:\Users\Hangyu\Desktop\OVRP\C1-C8\CMT{i}.vrp"
    capcity, node_coords, demand, depot = read_vrp_file(vrp_file)
    nodes = list(node_coords.values())
    demands = list(demand.values())
    depot_xy = nodes[0]
    node_xy = nodes[1:]

    path = []
    routes, cost = read_sol_file(sol_file)
    for subpath in routes:
        subpath = [0] + subpath + [0]
        # subpath = subpath
        path.append(subpath)

    # 计算路径长度
    total_length, max_length = calculate_path_lengths(nodes, path)
    # 输出结果
    print(f"Total path length: {total_length:.2f}")
    print(f"Maximum path length: {max_length:.2f}")
print(i)

data = np.loadtxt(r'D:\informs\EMNH\Meta_Transformer_2D_single\POMO_MOCVRP\result\tch_aug\cvrp_real\A\A-n32-k5_POMO_CVRP.txt')
data = Non_donminated_sorting(data)
data = np.array(data)
plt.scatter(data[:,0], data[:,1])
plt.show()

# C:\Users\Hangyu\Desktop\OVRP\F10-F12