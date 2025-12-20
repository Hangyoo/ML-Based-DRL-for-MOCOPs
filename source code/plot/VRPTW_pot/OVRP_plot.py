import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math 

import re

# 用于提取坐标的函数
def extract_coordinates(file_path):
    coordinates = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 读取每一行，并寻找符合坐标格式的行
    for line in lines:
        # 使用正则表达式匹配坐标相关的行
        match = re.match(r'\s*(\d+)\s+(\d+)\s+(\d+)', line)  # 匹配行：客户号、X坐标、Y坐标
        if match:
            cust_no = int(match.group(1))   # 客户编号
            x_coord = int(match.group(2))   # X坐标
            y_coord = int(match.group(3))   # Y坐标
            coordinates.append([x_coord, y_coord])

    return coordinates




# 计算欧几里得距离的函数
def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


# 计算所有路径的总长度和最大路径长度
def calculate_path_lengths(nodes, paths):
    total_length = 0
    max_length = 0

    # 遍历每一条路径
    # capcitys = [capcity] * len(paths)
    for j in range(len(paths)):
        path = paths[j]
        path_length = 0
        # 对路径中的每一对相邻节点计算距离
        for i in range(len(path) - 1):
            point1 = nodes[path[i]]  # 当前节点
            point2 = nodes[path[i + 1]]  # 下一个节点
            path_length += euclidean_distance(point1, point2)
            # capcitys[j] -= demands[path[i + 1]]
        # 更新总路径长度
        total_length += path_length

        # 更新最大路径长度
        if path_length > max_length:
            max_length = path_length
    # print(capcitys)
    return total_length, max_length


# 输出节点坐标
# print("Node Coordinates:")
# for node_id, (x, y) in node_coords.items():
#     print(f"Node {node_id}: ({x}, {y})")

# # 输出需求
# print("\nDemand:")
# for node_id, demand_value in demand.items():
#     print(f"Node {node_id}: {demand_value}")

# 使用该函数获取坐标
file_path = r"C:\Users\Hangyu\Desktop\VRPTW-Solomon(56)\C101.txt"  # 将文件路径替换为实际路径
# file_path = r"C:\Users\Hangyu\Desktop\VRPTW-Solomon(56)\C201.txt"  # 将文件路径替换为实际路径
nodes = extract_coordinates(file_path)

depot_xy = nodes[0]
node_xy = nodes[1:]

plt.rcParams['figure.figsize']=(8,8)

print(nodes)

# 绘制点和Depot点
for node in node_xy:
    plt.scatter(node[0],node[1],marker='o',color='#875853',s=50)

# # 根据路径绘制曲线


C101 = [[5, 3, 7, 8, 10, 11, 9, 6, 4, 2, 1, 75 ],
[13, 17, 18, 19, 15, 16, 14, 12],
[20, 24, 25, 27, 29, 30, 28, 26, 23, 22, 21, ],
[32, 33, 31, 35, 37, 38, 39, 36, 34, ],
[43, 42, 41, 40, 44, 45,46,48,51, 52,50, 49, 47, ],
[57, 55, 54, 53, 56, 58, 60, 59, ],
[67, 65, 63, 62, 74, 72, 61, 64, 68, 66, 69, ],
[81, 78, 76, 71, 70, 73, 77, 79, 80, ],
[90, 87, 86, 83, 82, 84, 85, 88, 89, 91, ],
[98, 96, 95, 94, 92, 93, 97, 100, 99 ]]
path_ = C101

# C201 = [[20, 22, 24, 27, 30, 29, 6, 32, 33, 31, 35, 37, 38, 39, 36, 34, 28, 26, 23, 18, 19, 16, 14, 12, 15, 17, 13, 25, 9, 11, 10, 8, 21],
#         [67, 63, 62, 74, 72, 61, 64, 66, 69, 68, 65, 49, 55, 54, 53, 56, 58, 60, 59, 57, 40, 44, 46, 45, 51, 50, 52, 47, 43, 42, 41, 48],
#         [93, 5, 75, 2, 1, 99, 100, 97, 92, 94, 95, 98, 7, 3,4, 89, 91, 88, 84, 86, 83, 82, 85, 76, 71, 70, 73, 80, 79, 81, 78, 77, 96, 87, 90]]
# path_ = C201

path = []
for subpath in path_:
    subpath = [0] + subpath + [0]
    # subpath = subpath
    path.append(subpath)
print(path)



# 计算路径长度
total_length, max_length = calculate_path_lengths(nodes, path)
# 输出结果
print(f"Total path length: {total_length:.2f}")
print(f"Maximum path length: {max_length:.2f}")

for subpath in path:
    plt.plot([nodes[item][0] for item in subpath],[nodes[item][1] for item in subpath],linewidth=3, zorder=1)
    # 标记点的位置
    # for i, txt in enumerate(subpath):
    #     plt.annotate(txt, (nodes[txt][0], nodes[txt][1]), fontsize=15)
plt.xlabel('X  coordinate',fontsize=23)
plt.ylabel('Y  coordinate',fontsize=23)
plt.tick_params(labelsize=23,pad=4)
plt.scatter(depot_xy[0],depot_xy[1],marker='*',color='r',s=100, zorder=2)
plt.show()



