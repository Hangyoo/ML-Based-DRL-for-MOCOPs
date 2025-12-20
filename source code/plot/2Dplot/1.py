# -*- ecoding: utf-8 -*-
# @Project : EMNH
# @Author: Hangyu Lou 
# @Time: 2024/11/22 2:07
import numpy as np
import matplotlib.pyplot as plt


curve1 = np.loadtxt(r"D:\informs\EMNH\revised_r2_plot\2Dplot\pf\KroAB300.txt")[:,1:]
curve2 = np.loadtxt(r"C:\Users\Hangyu\Desktop\GitHub\Experimental Results\MOTSP\KroAB300.txt")


# 绘制结果
plt.figure(figsize=(8, 6))
plt.scatter(curve1[:, 0], curve1[:, 1], label="Curve 1 (Original)", linestyle='-', color='b')
plt.scatter(curve2[:, 0], curve2[:, 1], label="Curve 2 (Original)", linestyle='--', color='r')
# plt.scatter(adjusted_curve[:, 0], adjusted_curve[:, 1], label="Adjusted Curve", linestyle='-', color='g')
plt.legend()
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto Front Adjustment')
plt.grid(True)
plt.show()
