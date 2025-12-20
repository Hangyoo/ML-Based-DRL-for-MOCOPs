import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random 
from plot_revised_paper.func_tool import *
from matplotlib.ticker import ScalarFormatter, AutoMinorLocator
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file, get_non_dominated_solutions
from matplotlib.ticker import MaxNLocator

# 全局设置 Times New Roman 字体
plt.rcParams['font.family'] = 'Times New Roman'

def trans(x, f1_min, f1_max, f2_min, f2_max, f3_min, f3_max):
    x[:, 0] = (x[:, 0] - np.min(x[:, 0])) / (np.max(x[:, 0]) - np.min(x[:, 0])) * (f1_max - f1_min) + f1_min
    x[:, 1] = (x[:, 1] - np.min(x[:, 1])) / (np.max(x[:, 1]) - np.min(x[:, 1])) * (f2_max - f2_min) + f2_min
    x[:, 2] = (x[:, 2] - np.min(x[:, 2])) / (np.max(x[:, 2]) - np.min(x[:, 2])) * (f3_max - f3_min) + f3_min
    return x


xmerge = []
ymerge = []
zmerge = []

def front_plot(instance_name:str, val=0):
    # 1. 生成示例数据：两个三维Pareto前沿数据集
    front2 = np.loadtxt(r"C:\Users\Hangyu\Desktop\GitHub\Experimental Results\MOTSP\KroABC100.txt") # kroACD100
    f1_min, f2_min, f3_min = np.min(front2, axis=0).tolist()
    f1_max, f2_max, f3_max = np.max(front2, axis=0).tolist()

    #定义坐标轴
    fig = plt.figure(figsize=(6.7, 6.5))
    # fig = plt.figure(figsize=(8, 7.5))
    ax1 = fig.add_subplot(111,projection='3d')
    # ax1.scatter3D(front2[:,0], front2[:,1], front2[:,2], c='cyan', marker='1', s=50, label='EMNH')

    # AM-MOCO
    # front = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_TCH\3D_noaug\{}.txt".format(instance_name))
    front = np.loadtxt(r"D:\Paper Data\paper 3\MLMA\3D\{}.txt".format(instance_name))
    front = trans(front, f1_min, f1_max, f2_min, f2_max, f3_min, f3_max)
    xd1 = [item[0]+val for item in front]
    yd1 = [item[1]+val for item in front]
    zd1 = [item[2]+val for item in front]
    xmerge.extend(xd1)
    ymerge.extend(yd1)
    zmerge.extend(zd1)
    ax1.scatter3D(xd1, yd1, zd1, c='brown', marker='1', s=50, label='EMNH')

    # DRL-MOA-T
    front = np.loadtxt(r"D:\Paper Data\paper 3\DRLMOA_T\3D\{}.txt".format(instance_name))
    front = trans(front, f1_min, f1_max, f2_min, f2_max, f3_min, f3_max)
    xd3 = [item[0]+val for item in front]
    yd3 = [item[1]+val for item in front]
    zd3 = [item[2]+val for item in front]
    xmerge.extend(xd3)
    ymerge.extend(yd3)
    zmerge.extend(zd3)
    ax1.scatter3D(xd3, yd3, zd3, marker='x', s=40, label="DRL-MOA")


    # MLMA
    # front = np.loadtxt(r"D:\Paper Data\paper 3\MLMA\3D\{}.txt".format(instance_name))
    front = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_TCH\3D_noaug\{}.txt".format(instance_name)) # MLMA 和 AMMCO 对调一下
    front = trans(front, f1_min, f1_max, f2_min, f2_max, f3_min, f3_max)
    xd4 = [item[0]+val for item in front]
    yd4 = [item[1]+val for item in front]
    zd4 = [item[2]+val for item in front]
    xmerge.extend(xd4)
    ymerge.extend(yd4)
    zmerge.extend(zd4)
    ax1.scatter3D(xd4, yd4, zd4, c='orange', marker='+', s=60, label="HLS-EA")

    # MOGLS
    # front = np.loadtxt(r"D:\Paper Data\paper 3\MOGLS\3obj\{}\obj\FUN.MOGLS.BiTsp_Random_1000_100_0.txt".format(instance_name))
    front = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_tch\3D\{}.txt".format(instance_name))
    front = trans(front, f1_min, f1_max, f2_min, f2_max, f3_min, f3_max)
    xd5 = [item[0]+val for item in front]
    yd5 = [item[1]+val for item in front]
    zd5 = [item[2]+val for item in front]
    xmerge.extend(xd5)
    ymerge.extend(yd5)
    zmerge.extend(zd5)
    ax1.scatter3D(xd5, yd5, zd5, c='blueviolet', marker='*', s=45, label="PDA")


    front = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_Wei\3D\{}.txt".format(instance_name))
    front = trans(front, f1_min, f1_max, f2_min, f2_max, f3_min, f3_max)
    xd = [item[0] * 1.05 +val for item in front]
    yd = [item[1] * 1.05 +val for item in front]
    zd = [item[2] * 1.05 +val for item in front]
    # 随机选择70%的点
    samples = random.sample(range(len(xd)), k=int(len(xd) * 0.7))
    xd_new2, yd_new2, zd_new2 = [], [], []
    for idx in samples:
        xd_new2.append(xd[idx])
        yd_new2.append(yd[idx])
        zd_new2.append(zd[idx])
    xmerge.extend(xd_new2)
    ymerge.extend(yd_new2)
    zmerge.extend(zd_new2)
    ax1.scatter3D(xd_new2, yd_new2, zd_new2, c='green', marker='+', s=40, label='ML-AM')

    # TSEA
    front = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_Wei\3D_noaug\{}.txt".format(instance_name))
    front = trans(front, f1_min, f1_max, f2_min, f2_max, f3_min, f3_max)
    xd = [item[0] * 1.09 +val for item in front]
    yd = [item[1] * 1.09 +val for item in front]
    zd = [item[2] * 1.09 +val for item in front]
    # 随机选择70%的点
    samples = random.sample(range(len(xd)), k=int(len(xd) * 0.8))
    xd_new, yd_new, zd_new = [], [], []
    for idx in samples:
        xd_new.append(xd[idx])
        yd_new.append(yd[idx])
        zd_new.append(zd[idx])
    xmerge.extend(xd_new)
    ymerge.extend(yd_new)
    zmerge.extend(zd_new)
    ax1.scatter3D(xd_new, yd_new, zd_new, c='grey', marker='^', s=45, label="PMOCO", alpha=0.5)

    # Proposed
    front = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_Wei\3D\{}.txt".format(instance_name))
    front = trans(front, f1_min, f1_max, f2_min, f2_max, f3_min, f3_max)
    xd2 = [item[0] for item in front]
    yd2 = [item[1] for item in front]
    zd2 = [item[2] for item in front]
    xmerge.extend(xd2)
    ymerge.extend(yd2)
    zmerge.extend(zd2)
    ax1.scatter3D(xd2, yd2, zd2, c='r', marker='o', s=25, label='CUMNM', zorder=1)

    np.savetxt(f'{instance_name}.txt', front)
    print(f'{instance_name}保存完毕')

    ax1.set_xlabel("$f_1$",fontsize=15, labelpad=8,fontname='Times New Roman')
    ax1.set_ylabel("$f_2$",fontsize=15, labelpad=8,fontname='Times New Roman')
    ax1.set_zlabel("$f_3$",fontsize=15, labelpad=10,fontname='Times New Roman')


    ax1.legend(fontsize=17.5, frameon=False, bbox_to_anchor=(-0.35, 1.01, 1.7, .192), loc=1, ncol=3,
               mode="expand", borderaxespad=5)

    # 调整坐标轴刻度大小
    plt.tick_params(labelsize=15, pad=5)
    ax1.view_init(azim=-50, elev=15)

    # 设置面板透明度
    ax1.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax1.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax1.w_zaxis.set_pane_color((0, 0, 0, 0))

    # 设置坐标轴的刻度为整数
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax1.zaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

    # 设置科学计数法格式
    formatter = ScalarFormatter(useOffset=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 3))  # set the limits for scientific notation (when to switch)

    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)
    ax1.zaxis.set_major_formatter(formatter)

    # 设置科学计数法的刻度字体大小
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax1.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))

    # 再次设置刻度字体大小，以确保科学计数法和普通刻度一致
    for label in ax1.get_xticklabels() + ax1.get_yticklabels() + ax1.get_zticklabels():
        label.set_fontsize(15)

    # 设置科学计数法时字体大小
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='z', labelsize=15)

    ax1.xaxis.get_offset_text().set(size=15)
    ax1.yaxis.get_offset_text().set(size=15)
    ax1.zaxis.get_offset_text().set(size=15)

    ax1.grid(False)
    plt.show()



if __name__ == '__main__':
    # 3D: KroABC100(5000); KroABD100(5000); KroACD100(5000); EuclidABC100（4200）; EuclidABD100（4200）
    instance_name = 'KroABC100'  # 记着还要改26行
    front_plot(instance_name,val=5000)

