import matplotlib.pyplot as plt
import pickle
import numpy as np
from plot.Nondominate_sort import Non_donminated_sorting 

'''
    在6个2D-Random上的PF对比:
         5个learning-based方法: DRL-MOA; DRL-AM; ML-AM; P-MOCO(Tch); P-MOCO(Wei)
         3个MOEAs方法:          NSGA-II; MOEA/D; MOGLS;
         提出算法: ML-MOA
'''

def front_plot(instance_name:str):
    plt.rcParams['figure.figsize']=(6,6)

    # DRL-MOA
    data = np.loadtxt(r"E:\Paper Data\paper 3\DRLMOA\2D\{}_1.txt".format(instance_name))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    plt.scatter(data[:,0],data[:,1], marker="v", s=50, color='darkmagenta', label = 'DRL-MOA')

    # LI-Transfer
    data = np.loadtxt(r"E:\Paper Data\paper 3\DRLMOA_T\2D\{}_1.txt".format(instance_name))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    plt.scatter(data[:,0],data[:,1], marker="x", s=80,label = 'DRL-MOA-T')

    # AM-MOCO
    data = np.loadtxt(r"E:\Paper Data\paper 3\PMOCO_WEI\2D_noaug\{}_1.txt".format(instance_name))
    # data = np.loadtxt(r"E:\Paper Data\paper 3\proposed\2D\{}_0.txt".format(instance_name))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    plt.scatter(data[:,0],data[:,1], marker="1", c='#0082fb', s=100,label = 'AM-MOCO')

    # MLMA
    data = np.loadtxt(r"C:\Users\Hangyu\Paper3\Comparison\Metalearning\result\5000_50step\{}_1.txt".format(instance_name))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    plt.scatter(data[:,0],data[:,1], marker="+", s=100, c='r', label = 'MLMA')

    # POMCO-tch
    # data = np.loadtxt(r"E:\Paper Data\paper 3\PMOCO_TCH\2D_noaug\{}_0.txt".format(instance_name))
    # data = np.loadtxt(r"C:\Users\Hangyu\Paper3\PMOCO_2D\MOTSP\POMO\result\wei_aug\Random20_0.txt")
    # data = Non_donminated_sorting(data)
    # data = np.array(data)
    # plt.scatter(data[:,0],data[:,1], marker="+", c='#0082fb', s=100,label = 'PF')

    # # POMCO-wei
    # data = np.loadtxt(r"E:\Paper Data\paper 3\PMOCO_Wei\2D_noaug\{}_0.txt".format(instance_name))
    # data = Non_donminated_sorting(data)
    # data = np.array(data)
    # plt.scatter(data[:,0],data[:,1], marker="+", c='g', s=100, label = 'POMCO_WEI')

    # NSGA-II
    data = np.loadtxt(r"E:\Paper Data\paper 3\NSGA2\2obj\TSP{}\1\obj\FUN.NSGAII.MOTSP_2000_{}_1_9.txt".format(instance_name[6:],instance_name[6:]))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    plt.scatter(data[:,0],data[:,1], marker="2", s=100,c='#875853', label = 'NSGA-II-2000')

    # MOEA/D
    data = np.loadtxt(r"E:\Paper Data\paper 3\MOEAD\2obj\TSP{}\1\obj\FUN.MOEAD.MOTSP_2000_{}_1_0.txt".format(instance_name[6:],instance_name[6:]))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    plt.scatter(data[:,0],data[:,1], marker=".", s=100, c='tomato', label = 'MOEA/D-2000')

    # MOGLS
    data = np.loadtxt(r"E:\Paper Data\paper 3\MOGLS\2obj\TSP{}\1\obj\FUN.MOGLS.BiTsp_Random_2000_100_1_0.txt".format(instance_name[6:]))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    plt.scatter(data[:,0],data[:,1], marker="*", s=50, c='blueviolet', label = 'MOGLS-2000')

    # Proposed
    data = np.loadtxt(r"E:\Paper Data\paper 3\PMOCO_Wei\2D\{}_1.txt".format(instance_name))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    plt.scatter(data[:,0],data[:,1], marker="o", s=50, c='darkorange', label = 'Proposed')

    # 调整坐标轴刻度大小
    plt.legend(fontsize=13,framealpha=0.15,edgecolor='k',loc=1)
    plt.xlabel('$f_1$',fontsize=13)
    plt.ylabel('$f_2$',fontsize=13)
    plt.tick_params(labelsize=14,pad=5)
    # plt.title('KroAB100',fontsize=14)
    plt.show()

if __name__ == '__main__':
    # 2D: Random20; Random50; Random80; Random100; Random150; Random200
    instance_name = 'Random200'
    front_plot(instance_name)
