import matplotlib.pyplot as plt
import pickle
import numpy as np 
from matplotlib.patches import ConnectionPatch
from plot_revised_paper.Nondominate_sort import Non_donminated_sorting

'''
    在6个2D-Benchmark上的PF对比:
         5个learning-based方法: DRL-MOA; DRL-AM; ML-AM; P-MOCO(Tch); P-MOCO(Wei)
         3个MOEAs方法:          NSGA-II; MOEA/D; MOGLS;
         1个Heuristic方法:      PDA
         提出算法: ML-MOA
'''

def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)

def front_plot(instance_name:str):
    #C:\Users\Hangyu\Paper3\Comparison\Metalearning\metic\IGD_HV.py PF绘制的详细设置
    plt.rcParams['figure.figsize']=(7,7)
    fig, axes = plt.subplots()
    # 对结果进行微调
    left, right = 0, 0
    if instance_name == "KroAB100":
        left, right = 600, -3000
        # left, right = 0, 0
    elif instance_name == "KroAB300":
        left, right = -20000, -20000
    elif instance_name == "EuclidAB100":
        left, right = -3000, -3000
    elif instance_name == "ClusterAB100":
        left, right = -3500, -3000
    elif instance_name == "EuclidAB300":
        left, right = -10000, -10000

    num_node = int(instance_name[-3:])

    times = 5 # 放大倍数

    # DRL-MOA over
    data = np.loadtxt(r"D:\Paper Data\paper 3\DRLMOA\2D\nonorm_{}.txt".format(instance_name))
    data = Non_donminated_sorting(data)
    data1 = np.array(data)
    axes.scatter(data1[:,0]+left,data1[:,1]+right, marker="v", s=50*times, color='darkmagenta', label = 'DRL-MOA')

    # LI-Transfer  over
    data = np.loadtxt(r"D:\Paper Data\paper 3\DRLMOA_T\2D\nonorm_{}.txt".format(instance_name))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:,0]+left,data[:,1]+right, marker="x", s=80*times,label = 'DRL-MOA-T')

    # # AM-MOCO over
    data = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_TCH\2D_noaug\AMMOCO_{}.txt".format(instance_name))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:,0]+left*0.3,data[:,1]+right*0.3, marker="1", c='#0082fb', s=100*times,label = 'AM-MOCO')

    # MLMA
    data = np.loadtxt(r"D:\Paper Data\paper 3\MLMA\5000_50step\nonorm_{}.txt".format(instance_name))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:,0]+left*0.5,data[:,1]+right*0.5, marker="+", s=100*times, c='r', label = 'MLMA')

    # NSGA-II over
    data = np.loadtxt(r"D:\Paper Data\paper 3\NSGA2\2obj\{}\obj\nonorm_FUN.NSGAII.MOTSP_2000_{}_4.txt".format(instance_name,num_node))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:,0]+left,data[:,1]+right, marker="2", s=100*times,c='#875853', label = 'NSGA-II-2000')

    # MOEA/D over
    data = np.loadtxt(r"D:\Paper Data\paper 3\MOEAD\2obj\{}\obj\nonorm_FUN.MOEAD.MOTSP_2000_{}_4.txt".format(instance_name,num_node))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:,0]+left,data[:,1]+right, marker=".", s=100*times, c='tomato', label = 'MOEA/D-2000')

    # MOGLS over
    data = np.loadtxt(r"D:\Paper Data\paper 3\MOGLS\2obj\{}\obj\nonorm_FUN.MOGLS.BiTsp_Random_2000_100_0.txt".format(instance_name))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:,0]+left,data[:,1]+right, marker="*", s=50*times, c='blueviolet', label = 'MOGLS-2000')

    # PDA
    data = np.loadtxt(r"C:\Users\Hangyu\Paper3\Comparison\Metalearning\plot_revised_paper\PDA_{}_pf.txt".format(instance_name))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:, 0], data[:, 1], marker="+", s=20*times, c='green', label='PDA')

    # TSEA
    data = np.loadtxt(r"C:\Users\Hangyu\Paper3\Comparison\Metalearning\plot_revised_paper\TSEA_{}_pf.txt".format(instance_name))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:, 0], data[:, 1], marker="*", s=30*times, c='pink', label='TSEA')

    # Exact front over
    data = np.loadtxt(r"D:\Paper Data\paper 3\PF\{}.txt".format(instance_name))
    if instance_name != 'EuclidAB300':
        axes.scatter(data[:, 1], data[:, 2], marker=".", s=50*times, c='k', label='Exact front')
    else:
        axes.scatter(data[:, 0], data[:, 1], marker=".", s=50*times, c='k', label='Exact front')
    # Proposed over
    data = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_Wei\2D\nonorm_{}.txt".format(instance_name))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:,0]+left,data[:,1]+right, marker="o", s=40*times, c='darkorange', label = 'ML-NMCO')

    plt.xlim(40000,60000) # 50000 65000
    plt.ylim(40000,60000)
    # 调整坐标轴刻度大小
    axes.set_xlabel('$f_1$',fontsize=13)
    axes.set_ylabel('$f_2$',fontsize=13)


    axes.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')  # 采用科学计数法
    axes.tick_params(labelsize=14,pad=5)
    # plt.title('KroAB100',fontsize=14)
    plt.show()

if __name__ == '__main__':
    # 2D: KroAB100 +1000 -3000; KroAB200; KroAB300 -15000 -15000; EuclidAB100 -3000 -3000; ClusterAB100 -3000 -3000;
    # instance_name = 'KroAB100'
    # instance_name = 'EuclidAB100' #'EuclidAB300'
    instance_name = 'ClusterAB100'
    front_plot(instance_name)
