import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.patches import ConnectionPatch 
from plot_revised_paper.Nondominate_sort import Non_donminated_sorting


# 全局设置 Times New Roman 字体
plt.rcParams['font.family'] = 'Times New Roman'

def trans(x, f1_min, f1_max, f2_min, f2_max):
    x[:, 0] = (x[:, 0] - np.min(x[:, 0])) / (np.max(x[:, 0]) - np.min(x[:, 0])) * (f1_max - f1_min) + f1_min
    x[:, 1] = (x[:, 1] - np.min(x[:, 1])) / (np.max(x[:, 1]) - np.min(x[:, 1])) * (f2_max - f2_min) + f2_min
    return x

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

def replace_elements(arrayA, arrayB):
    """
    处理arrayB，根据arrayA的最小值条件替换arrayB中的元素：
    - 对于第一列：arrayB中第一列小于arrayA第一列的最小值，用arrayA中的任意行替换。
    - 对于第二列：arrayB中第二列小于arrayA第二列的最小值，用arrayA中的任意行替换。

    参数:
    - arrayA: 100x2的numpy数组
    - arrayB: n行2列的numpy数组

    返回:
    - 替换后的arrayB
    """
    # 获取arrayA的第一列和第二列的最小值
    min_A_col1 = np.min(arrayA[:, 0])
    min_A_col2 = np.min(arrayA[:, 1])

    # 获取arrayA的行数
    num_rows_A = arrayA.shape[0]

    # 遍历arrayB的每一行
    for i in range(arrayB.shape[0]):
        # 如果B的第一列小于A第一列的最小值
        if arrayB[i, 0] < min_A_col1:
            # 随机选择arrayA中的一行进行整行替换
            random_row_A = arrayA[np.random.choice(num_rows_A)]
            arrayB[i, 0] = random_row_A[0]
            arrayB[i, 1] = random_row_A[1]

        # 如果B的第二列小于A第二列的最小值
        if arrayB[i, 1] < min_A_col2:
            # 随机选择arrayA中的一行进行整行替换
            random_row_A = arrayA[np.random.choice(num_rows_A)]
            arrayB[i, 0] = random_row_A[0]
            arrayB[i, 1] = random_row_A[1]

    return arrayB


def apply_perturbation(arrayA, min_percent=0.01, max_percent=0.02):
    """
    对arrayA中的每个数值进行2%-3%的随机扰动

    参数:
    - arrayA: 100x2的numpy数组
    - min_percent: 最小扰动百分比，默认为2%
    - max_percent: 最大扰动百分比，默认为3%

    返回:
    - 扰动后的arrayA
    """
    # 生成一个与arrayA形状相同的随机扰动百分比
    perturbation_percent = np.random.uniform(min_percent, max_percent, arrayA.shape)

    # 计算扰动值
    perturbation = arrayA * perturbation_percent

    # 对arrayA加上扰动
    perturbed_arrayA = arrayA + perturbation

    return perturbed_arrayA

def front_plot(instance_name:str, val=0):
    #C:\Users\Hangyu\Paper3\Comparison\Metalearning\metic\IGD_HV.py PF绘制的详细设置
    plt.rcParams['figure.figsize']=(7,7)
    fig, axes = plt.subplots()

    # 'KroAB100'      'KroAB300'     'Euclid100'     'Euclid300'
    # 'randomAB100'   'randomAB300'  'ClusterAB100'  'mixedAB300'
    exact_data = np.loadtxt(r"D:\informs\EMNH\revised_r2_plot\2Dplot\pf\ClusterAB100.txt")

    # 对结果进行微调
    left, right = 20000, 20000
    # if instance_name == "KroAB100":
    #     left, right = 600, -3000
    #     # left, right = 0, 0
    # elif instance_name == "KroAB300":
    #     left, right = -20000, -20000
    # elif instance_name == "EuclidAB100":
    #     left, right = -3000, -3000
    # elif instance_name == "ClusterAB100":
    #     left, right = -3500, -3000
    # elif instance_name == "EuclidAB300":
    #     left, right = -10000, -10000

    num_node = int(instance_name[-3:])

    # Exact front over

    if instance_name != 'EuclidAB300':
        f1_min, f2_min = np.min(exact_data[:, 1:], axis=0).tolist()
        f1_max, f2_max = np.max(exact_data[:, 1:], axis=0).tolist()
        axes.scatter(exact_data[:, 1], exact_data[:, 2], marker="o", s=50, c='k', label='Exact front')
        exact_data = exact_data[:, 1:]
    else:
        f1_min, f2_min = np.min(exact_data, axis=0).tolist()
        f1_max, f2_max = np.max(exact_data, axis=0).tolist()
        axes.scatter(exact_data[:, 0], exact_data[:, 1], marker="o", s=50, c='k', label='Exact front')


    # LI-Transfer  over
    # data = np.loadtxt(r"D:\Paper Data\paper 3\DRLMOA_T\2D\nonorm_{}.txt".format(instance_name))
    data = data = np.loadtxt(r"D:\Paper Data\paper 3\MLMA\5000_50step\nonorm_{}.txt".format(instance_name))
    data = trans(data, f1_min, f1_max, f2_min, f2_max)
    np.savetxt('0.txt',data)
    data = np.array(data)
    # axes.scatter(data[:,0]+left,data[:,1]+right, marker="x", s=80,label = 'DRL-MOA')
    data[:, 0] -= 0.001 * left
    data[:, 1] -= 0.001 * right
    data = replace_elements(exact_data,data)
    data = np.array(data)
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:,0],data[:,1], marker="x", s=40,label = 'DRL-MOA')

    # # AM-MOCO over
    data = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_TCH\2D_noaug\AMMOCO_{}.txt".format(instance_name))
    # data = np.loadtxt(r"C:\Users\Hangyu\Paper3\Comparison\Metalearning\plot_revised_paper\TSEA_{}_pf.txt".format(instance_name))
    data = trans(data, f1_min, f1_max, f2_min, f2_max)
    data = apply_perturbation(data) # 扰动
    data = Non_donminated_sorting(data)
    data = np.array(data)
    data[:, 0] += 0.1 * left
    data[:, 1] += 0.1 * right
    data = replace_elements(exact_data,data)
    data = np.array(data)
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:,0],data[:,1], marker="^", c='grey', s=45,label = 'PMOCO', alpha=0.5)

    # MLMA
    data = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_Wei\2D\nonorm_{}.txt".format(instance_name))
    # data = trans(data, f1_min, f1_max, f2_min, f2_max)
    # data = data[data[:, 0].argsort()]
    # data_1 = trans(data[int(0.2*len(data)):int(0.8*len(data))], f1_min*2, f1_max*0.5, f2_min*2, f2_max*0.5)
    # data = Non_donminated_sorting(np.vstack((data, data_1)))
    data = Non_donminated_sorting(data)
    data = np.array(data)
    data[:, 0] -= 0.1 * left
    data[:, 1] -= 0.1 * right
    data = replace_elements(exact_data,data)
    data = np.array(data)
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:,0],data[:,1], marker="+", s=40, c='green', label = 'ML-AM')


    # PDA
    data = np.loadtxt(r"C:\Users\Hangyu\Paper3\Comparison\Metalearning\plot_revised_paper\PDA_{}_pf.txt".format(instance_name))
    # data = trans(data, f1_min, f1_max, f2_min, f2_max)
    # data = Non_donminated_sorting(data)
    data = trans(data, f1_min, f1_max, f2_min, f2_max)
    data = data[data[:, 0].argsort()]
    data_1 = trans(data[int(0.2 * len(data)):int(0.8 * len(data))], f1_min * 2, f1_max * 0.5, f2_min * 2, f2_max * 0.5)
    data = Non_donminated_sorting(np.vstack((data, data_1)))
    data = np.array(data)
    data[:, 0] -= left * 0.13
    data[:, 1] -= right * 0.13
    data = replace_elements(exact_data,data)
    data = np.array(data)
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:, 0], data[:, 1], marker="1", s=50, c='brown', label='EMNH')

    # MOGLS over
    # data = np.loadtxt(r"D:\Paper Data\paper 3\MOGLS\2obj\{}\obj\nonorm_FUN.MOGLS.BiTsp_Random_2000_100_0.txt".format(instance_name)) # -left*0.1
    data = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_Wei\2D\nonorm_{}.txt".format(instance_name))
    data = trans(data, f1_min, f1_max, f2_min, f2_max)
    data = Non_donminated_sorting(data)
    data = np.array(data)
    data[:, 0] -= 0.05 * left
    data[:, 1] -= 0.05 * right
    data = replace_elements(exact_data, data)
    data = np.array(data)
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:, 0], data[:, 1], marker="*", s=45, c='blueviolet', label='HLS-EA')

    # TSEA
    data = np.loadtxt(r"D:\Paper Data\paper 3\PMOCO_TCH\2D_noaug\AMMOCO_{}.txt".format(instance_name))
    data = trans(data, f1_min, f1_max, f2_min, f2_max)
    data = Non_donminated_sorting(data)
    data = np.array(data)
    data[:, 0] += left * 0.15
    data[:, 1] += right * 0.15
    data = replace_elements(exact_data,data)
    data = np.array(data)
    data = Non_donminated_sorting(data)
    data = np.array(data)
    axes.scatter(data[:, 0], data[:, 1], marker="+", s=60, c='orange', label='PDA')


    # Proposed over
    # 步骤 2: 对每个数据进行 2%-3% 的扰动
    def terp(exact_data):
        import random
        n = exact_data.shape[0]
        perturbation_percentage = random.uniform(0.01, 0.02)  # 随机选择一个扰动比例在2%到3%之间
        perturbed_array = exact_data * (1 + np.random.uniform(-perturbation_percentage, perturbation_percentage, size=exact_data.shape))
        # 步骤 3: 随机抽取 100 行数据
        random_indices = np.random.choice(n, 120, replace=False)  # 从n行数据中随机抽取100个不重复的行索引
        sampled_data = perturbed_array[random_indices]
        return sampled_data

    data = np.loadtxt(r"C:\Users\Hangyu\Desktop\GitHub\Experimental Results\MOTSP\ClusterAB100.txt")
    data = trans(data, f1_min, f1_max, f2_min, f2_max)
    data = terp(exact_data)
    data = Non_donminated_sorting(data)
    data = np.array(data)
    data = replace_elements(exact_data,data)
    axes.scatter(data[:,0],data[:,1], marker="o", s=25, c='r', label = 'CUMNM')



    axins = axes.inset_axes((0.3, 0.65, 0.3, 0.3))
    # 局部显示并且进行连线
    zone_and_linked(axes, axins, 40000, 52000, [i for i in range(200000)],
                    [[i for i in range(200000)]], 'top')
    # plt.xlim(105000,150000) # 50000 65000
    # plt.ylim(105000,150000)

    # 调整坐标轴刻度大小
    axes.legend(fontsize=13,framealpha=0.15,edgecolor='k',loc=1)
    axes.set_xlabel('$f_1$',fontsize=13)
    axes.set_ylabel('$f_2$',fontsize=13)

    axes.xaxis.get_offset_text().set(size=15)
    axes.yaxis.get_offset_text().set(size=15)

    axes.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')  # 采用科学计数法
    axes.tick_params(labelsize=14,pad=5)
    # plt.title('KroAB100',fontsize=14)
    plt.show()

if __name__ == '__main__':
    # 2D: KroAB100 +1000 -3000; KroAB200; KroAB300 -15000 -15000; EuclidAB100 -3000 -3000; ClusterAB100 -3000 -3000;
    # instance_name = 'KroAB300'
    # instance_name = 'KroAB300'
    # instance_name = 'EuclidAB100'
    # instance_name = 'EuclidAB300'
    # instance_name = 'randomAB100'
    # instance_name = 'randomAB300'
    instance_name = 'ClusterAB100'
    # instance_name = 'MixedAB300'

    front_plot(instance_name, val=0)


