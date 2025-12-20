# -*- ecoding: utf-8 -*-
# @Project : EMNH
# @Author: Hangyu Lou
# @Time: 2024/9/2 20:58 

import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.patches import ConnectionPatch
from Meta_Transformer_2D_reptile.utils.Nondominate_sort import Non_donminated_sorting

plt.rcParams['figure.figsize']=(7,7)
fig, axes = plt.subplots()

# 调整坐标轴刻度大小
axes.legend(fontsize=13,framealpha=0.15,edgecolor='k',loc=1)
axes.set_xlabel('$f_1$',fontsize=13)
axes.set_ylabel('$f_2$',fontsize=13)

# DRL-MOA over
data = np.loadtxt(r"D:\Paper Data\paper 3\Supplementary experiments\obj\mokp\2D.FUN.MLNMCO.MOKP100_tch.txt")
data = Non_donminated_sorting(data)
data1 = np.array(data)
axes.scatter(data1[:,0],data1[:,1], marker="v", s=50, color='darkmagenta', label = 'our')

# LI-Transfer  over
data = np.loadtxt(r"D:\informs\EMNH\Meta_Transformer_2D_reptile\POMO_EACH_DECODER\result\tch_aug\kp\None100_each_decoder.txt")
data = Non_donminated_sorting(data)
data = np.array(data)
axes.scatter(data[:,0],data[:,1], marker="x", s=80,label = 'POMO')

# 调整坐标轴刻度大小
axes.legend(fontsize=13,framealpha=0.15,edgecolor='k',loc=1)
axes.set_xlabel('$f_1$',fontsize=13)
axes.set_ylabel('$f_2$',fontsize=13)

axes.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')  # 采用科学计数法
axes.tick_params(labelsize=14,pad=5)
# plt.title('KroAB100',fontsize=14)
plt.show()