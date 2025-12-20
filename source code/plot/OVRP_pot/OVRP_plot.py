import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import math

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
def calculate_path_lengths(nodes, paths, demands):
    total_length = 0
    max_length = 0

    # 遍历每一条路径
    capcitys = [capcity] * len(paths)
    for j in range(len(paths)):
        path = paths[j]
        path_length = 0
        # 对路径中的每一对相邻节点计算距离
        for i in range(len(path) - 1):
            point1 = nodes[path[i]]  # 当前节点
            point2 = nodes[path[i + 1]]  # 下一个节点
            path_length += euclidean_distance(point1, point2)
            capcitys[j] -= demands[path[i + 1]]
        # 更新总路径长度
        total_length += path_length
        print(path_length)
        # 更新最大路径长度
        if path_length > max_length:
            max_length = path_length
    print(capcitys)
    return total_length, max_length


# 调用函数读取文件并输出结果
file_path = r"C:\Users\Hangyu\Desktop\OVRP\C1-C8\CMT3.vrp"
# file_path = r"C:\Users\Hangyu\Desktop\OVRP\F10-F12\F-n72-k4.vrp"
capcity, node_coords, demand, depot = read_vrp_file(file_path)


# 输出节点坐标
# print("Node Coordinates:")
# for node_id, (x, y) in node_coords.items():
#     print(f"Node {node_id}: ({x}, {y})")

# # 输出需求
# print("\nDemand:")
# for node_id, demand_value in demand.items():
#     print(f"Node {node_id}: {demand_value}")

# 输出仓库位置
nodes = list(node_coords.values())
demands = list(demand.values())
print(demand)

print(f"\nDepot: {depot}")
num_nodes = 50
depot_xy = nodes[0]
node_xy = nodes[1:]

plt.rcParams['figure.figsize']=(8,8)

print(nodes)

# 绘制点和Depot点
for node in node_xy:
    plt.scatter(node[0],node[1],marker='o',color='#875853',s=50)

# # 根据路径绘制曲线
routec1 = [[46, 5, 49, 10, 39, 33, 45, 15, 44, 37, 12],
           [11, 2, 29, 21, 50, 16, 34, 30, 9, 38],
           [32, 1, 22, 20, 35, 36, 3, 28, 31, 26, 8],
           [27, 48, 23, 7, 43, 24, 25, 14, 6],
           [47, 4, 17, 42, 19, 40, 41, 13, 18]]
path_ = routec1


# routec4 = [[56, 146, 149, 4, 111, 66, 41, 94, 19, 64, 42, 92, 137, 147, 17],
#            [102, 6, 132, 98, 58, 95, 25, 133, 110, 18, 139], [5, 103, 12],
#            [46, 57, 23, 69, 7, 61, 114, 99, 43, 86, 97, 24, 96, 14, 68],
#            [32, 51, 22, 101, 3, 59, 20, 131, 83, 2, 100, 11],
#            [47, 55, 134, 67, 13, 136, 40, 88, 93, 65, 107, 44, 108],
#            [76, 49, 10, 54, 39, 89, 117, 75, 105, 30, 104, 9, 38],
#            [119, 1, 120, 80, 70, 28, 116, 121, 115, 36, 85, 35, 84, 128, 29, 129],
#            [144, 109, 143, 135, 141, 150, 87, 148, 142, 145, 63],
#            [27, 138, 48, 112, 26, 113, 140, 82, 31, 8, 60, 81, 77],
#            [37, 52, 15, 45, 91, 72, 33, 73, 106, 125, 124, 122, 123, 71, 90],
#            [78, 126, 16, 127, 53, 21, 79, 74, 34, 50, 130, 118, 62]]
#
# path_ = routec4

routec3 =[[92, 37, 98, 100, 91, 16, 86, 38, 44, 14, 42, 43, 15, 57, 2, 58,0],
          [6, 96, 99, 59, 93, 85, 61, 45,17,84, 5, 60, 89,0],
          [0,27, 69, 1, 70, 30, 20, 66, 32, 90, 63, 10, 62, 88, 31],
          [21, 72, 75, 56,  23, 39, 67,41, 22, 74, 73, 40,0],
          [18, 83, 8, 46, 47, 36, 49, 64, 11, 19, 48, 82, 7, 52,0],
          [94, 95, 97,87, 13,0],
          [28, 12, 80, 68, 29, 24, 54, 55, 25, 4,26, 53,0],
          [76, 77, 3, 79, 78, 34, 35, 65, 71, 9, 51, 81, 33, 50,0]]
path_ = routec3
#
#
# F11 = [[54, 55, 41, 57, 56, 39, 68, 40, 38, 37, 69, 67, 66, 65, 64, 62, 63, 59, 58, 61, 60, 34, 31, 32,0],
#        [36, 11, 1, 15, 14],
#        [35, 33, 18, 19, 2, 17, 16,13, 12, 71, 6, 10, 8, 7, 9, 4, 3, 5,0],
#        [23, 26, 24, 25, 49, 51, 70, 50, 47, 48, 52, 45, 53, 46, 44, 43, 42, 27, 28, 22, 21, 30, 29, 20,0]]
# path_ = F11




# C5 = [[0,194, 158, 192, 184, 190, 43, 199, 197, 136, 1, 191, 196, 66],
#       [0,16, 159, 104, 183, 23, 116, 62, 185, 89, 137, 91, 141, 22, 186],
#       [0,93, 156, 114, 142, 113, 42, 68, 143, 41, 90, 115, 160, 53, 198, 195],
#       [0,6, 61, 96, 105, 33, 193, 112, 2, 157],
#       [0,86, 101, 28, 64, 94, 140, 121, 82, 173, 21, 172, 139, 54, 152],
#       [0,125, 59, 5, 103, 88, 37, 138, 36, 155, 47, 48, 30],
#       [0,45, 29, 79, 15, 154, 124, 20, 166, 122, 174, 171, 120],
#       [0,175, 46, 65, 167, 99, 179, 27, 58, 111, 4, 87],
#       [0,34, 176, 8, 102, 178, 78, 19, 70, 128, 123, 13, 83, 153, 98],
#       [0,149, 51,  108, 69, 7, 132, 180,14, 133, 177, 35],
#       [0,50, 169, 119, 38, 165, 164, 85, 134, 84, 170, 11, 52, 150],
#       [0,126, 17, 76, 40, 130, 12, 168, 81, 26, 60, 127],
#       [9, 109, 39, 57, 189, 131, 80, 10, 77, 129, 71, 100,0],
#       [0,187, 97, 161, 72,  110, 25, 56, 118,32, 106, 44, 55, 3],
#       [0,75, 162, 31, 163, 148, 92, 135, 145, 73, 146, 18, 147, 181],
#       [0,95, 151, 117, 63, 107, 24, 144, 74, 49, 182, 67, 188]]
# path_ = C5

path = []
for subpath in path_:
    subpath = [0] + subpath + [0]
    # subpath = subpath
    path.append(subpath)
print(path)

print('demands',demands)


# 计算路径长度
total_length, max_length = calculate_path_lengths(nodes, path,demands)
# 输出结果
print(f"Total path length: {total_length:.2f}")
print(f"Maximum path length: {max_length:.2f}")

# 使用默认颜色循环
colors = plt.cm.tab10(np.linspace(0, 1, len(path)))  # 你可以替换为其它的颜色集（如 'tab20' 等）

for i, subpath in enumerate(path):
    x, y = [nodes[item][0] for item in subpath], [nodes[item][1] for item in subpath]

    # 从颜色列表中获取颜色
    color = colors[i % len(colors)]  # 保证颜色循环

    # 绘制路径，确保颜色一致
    plt.plot(x[:2], y[:2], '-.', linewidth=3, zorder=1, color=color)
    plt.plot(x[-2:], y[-2:], '-.', linewidth=3, zorder=1, color=color)
    plt.plot(x[1:-1], y[1:-1], linewidth=3, zorder=1, color=color)
    # 标记点的位置
    # for i, txt in enumerate(subpath):
    #     plt.annotate(txt, (nodes[txt][0], nodes[txt][1]), fontsize=15)
plt.xlabel('X  coordinate',fontsize=23)
plt.ylabel('Y  coordinate',fontsize=23)
plt.tick_params(labelsize=23,pad=4)
plt.scatter(depot_xy[0],depot_xy[1],marker='*',color='r',s=100, zorder=2)
plt.show()



