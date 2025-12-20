# -*- ecoding: utf-8 -*-
# @Project : EMNH
# @Author: Hangyu Lou
# @Time: 2024/11/18 7:23 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from matplotlib.ticker import FuncFormatter
import copy


# 'EuclidABC100'
# [25882.0, 27236.0, 25970.0]
# [164360.0, 181893.0, 172032.0]

# 'EuclidABD100'
# [25882.0, 26684.0, 25514.0]
# [166776.0, 179841.0, 175882.0]

# KroABC100
# [21923.0, 27330.0, 22015.0]
# [192909.0, 173142.0, 175671.0]

# KroABD100
# [21923.0, 27904.0, 22392.0]
# [172177.0, 174735.0, 165558.0]

# KroACD100
# [21923.0, 25634.0, 22392.0]
# [175629.0, 172396.0, 163293.0]


def trans(x, f1_min, f1_max, f2_min, f2_max, f3_min, f3_max):
    x[:, 0] = (x[:, 0] - np.min(x[:, 0])) / (np.max(x[:, 0]) - np.min(x[:, 0])) * (f1_max - f1_min) + f1_min
    x[:, 1] = (x[:, 1] - np.min(x[:, 1])) / (np.max(x[:, 1]) - np.min(x[:, 1])) * (f2_max - f2_min) + f2_min
    x[:, 2] = (x[:, 2] - np.min(x[:, 2])) / (np.max(x[:, 2]) - np.min(x[:, 2])) * (f3_max - f3_min) + f3_min
    return x


# 1. 生成示例数据：两个三维Pareto前沿数据集
instance_name = 'KroACD100'
data1 = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_Wei\3D\{}.txt".format(instance_name))
data2 = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_tch\3D\{}.txt".format(instance_name))
front2 = np.loadtxt(r"C:\Users\Hangyu\Desktop\GitHub\Experimental Results\MOTSP\kroACD100.txt")

# print(np.min(front2,axis=0).tolist())
# print(np.max(front2,axis=0).tolist())
f1_min, f2_min, f3_min = np.min(front2,axis=0).tolist()
f1_max, f2_max, f3_max = np.max(front2,axis=0).tolist()

data1 = trans(data1, f1_min, f1_max, f2_min, f2_max, f3_min, f3_max)

# data2 = copy.deepcopy(data1)

# data2 = trans(data2, f1_min, f1_max, f2_min, f2_max, f3_min, f3_max)
data2 = trans(data2, f1_min*1.2, f1_max/1.2, f2_min*1.2, f2_max/1.2, f3_min*1.2, f3_max/1.2)



# # 8. 可视化：在同一个三维图中绘制两个数据集
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制第一个Pareto前沿
ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], color='b', label='Front 1', s=50)
ax.scatter(data2[:, 0]+5000, data2[:, 1]+5000, data2[:, 2]+5000, color='g', label='Front 1', s=50)



# 绘制第二个Pareto前沿（对齐后的）
ax.scatter(front2[:, 0], front2[:, 1], front2[:, 2], color='r', label='OR', s=50)

# Set scientific notation for axis labels, but use simple integer ticks
formatter = ScalarFormatter(useOffset=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 3))  # set the limits for scientific notation (when to switch)

ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)
ax.zaxis.set_major_formatter(formatter)

# Optionally, you can also set the axis ticks to show only integer ticks in scientific notation.
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))  # x-axis
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # y-axis
ax.ticklabel_format(style='sci', axis='z', scilimits=(0,0))  # z-axis

# 设置标题和轴标签
ax.set_title('Pareto Fronts Alignment')
ax.set_xlabel('Objective 1')
ax.set_ylabel('Objective 2')
ax.set_zlabel('Objective 3')

# 设置坐标刻度格式，禁用科学计数法


# 显示图例
ax.legend()

# 显示图形
plt.show()

