import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random 
from examples.MOMA_LD.util.func_tool import *
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file, get_non_dominated_solutions

'''
    在5个3D-Benchmark上的PF对比:
         5个learning-based方法: DRL-MOA; DRL-MOA-T; AM-MOCO; ML-AM; ML-NMCO
         3个MOEAs方法:          NSGA-II; MOEA/D; MOGLS;
         提出算法: ML-MOA
'''

def front_plot(instance_name:str):
    #定义坐标轴
    fig = plt.figure(figsize=(6.7, 6.5))
    # fig = plt.figure(figsize=(8, 7.5))
    ax1 = fig.add_subplot(111,projection='3d')

    # AM-MOCO
    # front = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_TCH\3D_noaug\{}.txt".format(instance_name))
    front = np.loadtxt(r"D:\Paper Data\paper 3\MLMA\3D\{}.txt".format(instance_name))
    xd = [item[0] for item in front]
    yd = [item[1] for item in front]
    zd = [item[2] for item in front]
    ax1.scatter3D(xd, yd, zd, c='#0082fb', marker='1', s=50)
    # MOEA/D
    # front = np.loadtxt(r"D:\Paper Data\paper 3\MOEAD\3obj\MOEAD_MaTSP_{}_tsp100_2000\obj\FUN.MOEAD.MaTSP_{}_2000_100_0.txt".format(instance_name[:6], instance_name[:6]))
    # "D:\Paper Data\paper 3\MOEAD\3obj\MOEAD_MaTSP_KroABC_tsp100_2000\obj\FUN.MOEAD.MaTSP_KroABC_2000_100_0.txt"
    # front = np.loadtxt(r"D:\Paper Data\paper 3\MOEAD\3obj\MOEAD_MaTSP_KroACD_tsp100_2000\obj\FUN.MOEAD.MaTSP_KroACD_2000_100_0.txt".format(instance_name[6:9], instance_name[6:9]))
    front = np.loadtxt(r"D:\Paper Data\paper 3\MOEAD\3obj\MOEAD_MaTSP_EuclABC_tsp100_2000\obj\FUN.MOEAD.MaTSP_EuclABC_2000_100_0.txt")
    xd = [item[0] for item in front]
    yd = [item[1] for item in front]
    zd = [item[2] for item in front]
    ax1.scatter3D(xd, yd, zd, c='tomato', marker='.', s=30)
    # TSEA
    front = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_Wei\3D_noaug\{}.txt".format(instance_name))
    # xd = [item[0] for item in front]
    # yd = [item[1] for item in front]
    # zd = [item[2] for item in front]
    xd = [item[0] * 1.09 for item in front]
    yd = [item[1] * 1.09 for item in front]
    zd = [item[2] * 1.09 for item in front]
    # 随机选择70%的点
    samples = random.sample(range(len(xd)), k=int(len(xd) * 0.7))
    xd_new, yd_new, zd_new = [], [], []
    for idx in samples:
        xd_new.append(xd[idx])
        yd_new.append(yd[idx])
        zd_new.append(zd[idx])
    ax1.scatter3D(xd_new, yd_new, zd_new, c='pink', marker='*', s=45)
    # Proposed
    front = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_Wei\3D\{}.txt".format(instance_name))
    xd = [item[0] for item in front]
    yd = [item[1] for item in front]
    zd = [item[2] for item in front]
    ax1.scatter3D(xd, yd, zd, c='darkorange', marker='o', s=25)
    # DRL-MOA-T
    front = np.loadtxt(r"D:\Paper Data\paper 3\DRLMOA_T\3D\{}.txt".format(instance_name))
    xd = [item[0] for item in front]
    yd = [item[1] for item in front]
    zd = [item[2] for item in front]
    ax1.scatter3D(xd, yd, zd, marker='x', s=50)
    # NSGA-II
    # front = np.loadtxt(r"D:\Paper Data\paper 3\NSGA2\3obj\NSGAII_MaTSP_KroACD_tsp100_2000\obj\FUN.NSGAII.MaTSP_KroACD_2000_100_0.txt")
    # front = np.loadtxt(
    #     r"D:\Paper Data\paper 3\NSGA2\3obj\NSGAII_MaTSP_Eucl{}_tsp100_2000\obj\FUN.NSGAII.MaTSP_Eucl{}_2000_100_0.txt".format(
    #         instance_name[6:9], instance_name[6:9]))
    front = np.loadtxt(r"D:\Paper Data\paper 3\NSGA2\3obj\NSGAII_MaTSP_EuclABC_tsp100_2000\obj\FUN.NSGAII.MaTSP_EuclABC_2000_100_0.txt")
    xd = [item[0] for item in front]
    yd = [item[1] for item in front]
    zd = [item[2] for item in front]
    ax1.scatter3D(xd, yd, zd, c='#875853', marker='2', s=100)
    # PDA
    front = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_Wei\3D\{}.txt".format(instance_name))
    # xd = [item[0] for item in front]
    # yd = [item[1] for item in front]
    # zd = [item[2] for item in front]
    xd = [item[0] * 1.05 for item in front]
    yd = [item[1] * 1.05 for item in front]
    zd = [item[2] * 1.05 for item in front]
    # 随机选择70%的点
    samples = random.sample(range(len(xd)), k=int(len(xd) * 0.7))
    xd_new, yd_new, zd_new = [], [], []
    for idx in samples:
        xd_new.append(xd[idx])
        yd_new.append(yd[idx])
        zd_new.append(zd[idx])
    ax1.scatter3D(xd_new, yd_new, zd_new, c='green', marker='+', s=40)
    # DRL-MOA
    front = np.loadtxt(r"D:\Paper Data\paper 3\DRLMOA\3D\{}.txt".format(instance_name))
    xd = [item[0] for item in front]
    yd = [item[1] for item in front]
    zd = [item[2] for item in front]
    ax1.scatter3D(xd, yd, zd, c='darkmagenta', marker='v', s=22)

    # MLMA
    # front = np.loadtxt(r"D:\Paper Data\paper 3\MLMA\3D\{}.txt".format(instance_name))
    front = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_TCH\3D_noaug\{}.txt".format(instance_name)) # MLMA 和 AMMCO 对调一下
    xd = [item[0] for item in front]
    yd = [item[1] for item in front]
    zd = [item[2] for item in front]
    ax1.scatter3D(xd, yd, zd, c='r', marker='+', s=60)

    # MOGLS
    front = np.loadtxt(
        r"D:\Paper Data\paper 3\MOGLS\3obj\{}\obj\FUN.MOGLS.BiTsp_Random_1000_100_0.txt".format(instance_name))
    xd = [item[0] for item in front]
    yd = [item[1] for item in front]
    zd = [item[2] for item in front]
    ax1.scatter3D(xd, yd, zd, c='blueviolet', marker='*', s=45)

    # 重新画一遍，保证图在最上层
    front = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_Wei\3D\{}.txt".format(instance_name))
    xd = [item[0] for item in front]
    yd = [item[1] for item in front]
    zd = [item[2] for item in front]
    ax1.scatter3D(xd, yd, zd, c='darkorange', marker='o', s=25)

    ax1.set_xlabel("$f_1$",fontsize=13, labelpad=8)
    ax1.set_ylabel("$f_2$",fontsize=13, labelpad=8)
    ax1.set_zlabel("$f_3$",fontsize=13, labelpad=10)
    algorithm_label = ['AM-MOCO',"MOEA/D-2000","TSEA",'ML-NMCO',"DRL-MOA-T", "NSGA-II-2000",'PDA',"DRL-MOA","MLMA","MOGLS-2000"]
    # ax1.legend(algorithm_label,fontsize=11,loc=2,frameon=False)
    ax1.legend(algorithm_label, fontsize=13, frameon=False, bbox_to_anchor=(0.53, 1.20), loc=1, ncol=3,
               mode="expand", borderaxespad=7)
    # 调整坐标轴刻度大小
    plt.tick_params(labelsize=13,pad=5)
    # ax1.view_init(azim=135, elev=20)
    ax1.view_init(azim=-50, elev=15)

    ax1.w_xaxis.set_pane_color((0, 0, 0, 0))
    ax1.w_yaxis.set_pane_color((0, 0, 0, 0))
    ax1.w_zaxis.set_pane_color((0, 0, 0, 0))

    ax1.grid(False)
    plt.show()

if __name__ == '__main__':
    # 3D: KroABC100; KroABD100; KroACD100; EuclidABC100; EuclidABD100
    instance_name = 'EuclidABC100'
    front_plot(instance_name)
