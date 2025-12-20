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
file_path = r"C:\Users\Hangyu\Desktop\CVRP\A\A-n80-k10.vrp"  # 将文件路径替换为实际路径
# file_path = r"C:\Users\Hangyu\Desktop\CVRP\X\X-n393-k38.vrp"  # 将文件路径替换为实际路径
nodes = extract_coordinates(file_path)

depot_xy = nodes[0]
node_xy = nodes[1:]

plt.rcParams['figure.figsize']=(8,8)

print(nodes)

# 绘制点和Depot点
for node in node_xy:
    plt.scatter(node[0],node[1],marker='o',color='#875853',s=50)

# # 根据路径绘制曲线


An80 = [[1, 7, 21, 40], [10, 63, 11, 24, 6, 23], [13, 74, 60, 39, 3, 77, 51], [17, 31, 27, 59, 5, 44, 12, 62], [29, 20, 75, 57, 19, 26, 35, 65, 69, 56, 47, 15, 33, 64], [30, 78, 61, 43,16,68, 8, 37, 2, 34], [38, 72, 54, 9, 55, 41, 25, 46], [42, 53, 66, 67, 36, 73, 49], [52, 28, 79, 48, 18, 14, 71], [58, 32, 4, 22, 45, 50, 76, 70]]

path_ = An80

# Xn502 = [[73, 49, 126, 71, 96, 164, 139, 146, 90, 330, 120, 125], [220, 154, 369, 20, 9, 61, 309, 355, 264], [105, 65, 137, 1, 58, 129, 84, 174, 162, 179, 127], [158, 4, 184, 29, 78, 104, 67, 68, 145, 2, 177], [244, 8, 169, 141, 19, 46, 152, 323, 349], [128, 136, 69, 130, 3, 180, 74, 23, 110, 62, 40], [34, 16,32,39,60,28, 43, 147, 193, 358, 170], [85, 196, 157, 53, 108, 5, 42, 98, 33, 22, 181], [64, 161, 163, 112, 31, 99, 41, 173, 144, 106], [192, 187, 133, 10, 30, 119, 44, 11, 183, 190], [138, 194, 191, 77, 118, 166, 59, 91, 218, 238], [111, 195, 189, 159, 168, 75, 107, 182, 15, 35], [82, 57, 117, 36, 156, 37, 114, 97, 6, 26], [352, 100, 52, 17, 115, 7, 83, 150, 50, 89], [143, 124, 38, 95, 176, 79, 94, 14, 25, 178, 172], [81, 295, 72, 345, 237, 63, 249, 360, 344, 103], [140, 290, 246,116,167,24,142,339,383, 342, 92], [322, 209, 223, 308, 175, 48, 280, 240, 102], [54, 368, 109, 288, 211, 390, 255, 362, 18, 326, 317], [198, 337, 364, 357, 363, 371, 121, 225, 243, 236], [123, 343, 372, 221, 206, 272, 389, 86, 212], [252, 347, 283, 66, 353, 235,251, 227,274], [287, 289, 13, 80, 365, 377, 356, 135, 228, 286], [354, 261, 373, 333, 265, 263, 233, 348, 374, 201, 260], [165, 270, 266, 324, 276, 153, 335, 216, 375, 385], [70, 229, 380, 241, 370, 262, 204, 391, 232, 222, 329], [321, 325, 282, 340, 277, 367, 328, 259, 200, 307], [171, 27, 214, 56, 199, 284, 281, 294, 250, 301, 185], [258, 388, 210, 314, 302, 245, 205, 318, 392, 296, 273], [299, 134, 239, 376, 313, 215, 379, 275, 278, 292], [336, 148, 316, 382, 304, 45, 188, 350, 319, 268, 257, 226, 122], [21, 253, 310, 279, 224, 242, 303, 315, 160, 366], [155, 55, 76, 346, 306, 88, 297, 149, 300], [101, 361, 341, 51, 217, 327, 151, 113, 132, 386], [271, 12, 381, 285, 197, 298, 131, 202, 254, 256, 311], [334, 305, 248, 203, 207, 234, 378, 332, 231, 213], [384, 359, 351, 87, 387, 230, 47, 247, 293, 269, 93], [320, 331, 267, 219, 312, 291, 186, 208, 338]]
#
# path_ = Xn502

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



