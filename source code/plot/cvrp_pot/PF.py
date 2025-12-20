# -*- ecoding: utf-8 -*-
# @Project : EMNH
# @Author: Hangyu Lou
# @Time: 2024/11/15 10:46 

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


# 调用函数读取文件并输出结果
# a_n80 = r"C:\Users\Hangyu\Desktop\CVRP\A\A-n80-k10.sol"
# file_path = a_n80  # 替换为你的文件路径
# routes, cost = read_sol_file(file_path)

xn502 = r"C:\Users\Hangyu\Desktop\CVRP\X\X-n393-k38.sol"
file_path = xn502  # 替换为你的文件路径
routes, cost = read_sol_file(file_path)


# 输出结果
print("Routes:", routes)
print("Cost:", cost)
