# -*- ecoding: utf-8 -*-
# @Project : EMNH
# @Author: Hangyu Lou
# @Time: 2024/9/8 20:36 
import numpy as np

# 从文本中提取的距离数据
lower_row_data = [
     9,    14,    21,    23,    22,    25,    32,    36,    38,    42,
    50,    52,     5,    12,    22,    21,    24,    31,    35,    37,
    41,    49,    51,     7,    17,    16,    23,    26,    30,    36,
    36,    44,    46,    10,    21,    30,    27,    37,    43,    31,
    37,    39,    19,    28,    25,    35,    41,    29,    31,    29,
     9,    10,    16,    22,    20,    28,    30,     7,    11,    13,
    17,    25,    27,    10,    16,    10,    18,    20,     6,     6,
    14,    16,    12,    12,    20,     8,    10,    10
]

def create_full_matrix(n, lower_row_data):
    matrix = np.zeros((n, n), dtype=int)
    index = 0
    for i in range(1, n):
        for j in range(i):
            matrix[i, j] = matrix[j, i] = lower_row_data[index]
            index += 1
    return matrix

# 生成距离矩阵
n = 13  # 总共有13个点，包括depot
distance_matrix = create_full_matrix(n, lower_row_data)

print("完整的距离矩阵:")
print(distance_matrix)
