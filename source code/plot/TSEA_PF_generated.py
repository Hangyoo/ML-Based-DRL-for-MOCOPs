import math 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import random
import copy
from plot.Nondominate_sort import Non_donminated_sorting

def generatePF(instance_name):
    data = np.loadtxt(r"D:\Paper Data\paper 3\PF\{}.txt".format(instance_name))
    # 找到真实前沿的最大值
    if instance_name == 'EuclidAB300':
        x_max = np.max(data[:, 0])
        y_max = np.max(data[:, 1])
        new_data = copy.deepcopy(data)
    else:
        x_max = np.max(data[:, 1])
        y_max = np.max(data[:, 2])
        new_data = copy.deepcopy(data[:, 1:])

    # 定义一个百分比
    percent = 0.3
    new_data[:,0] *= (1+percent)
    new_data[:,1] *= (1+percent)

    # 近似前沿不超过真实前沿最大值
    repair_data= []
    for i in range(new_data.shape[0]):
        if new_data[i][0] >= x_max*0.95 or new_data[i][1] >= y_max*0.95:
            pass
        else:
            repair_data.append(new_data[i].tolist())

    # 随机选点 + 扰动 + 非支配排序
    rand_node = random.sample(range(len(repair_data)),k=int(0.1*len(repair_data)))
    repair_data1 = []
    for num in rand_node:
        repair_data1.append(repair_data[num])
    # 扰动
    for i in range(len(repair_data1)):
        repair_data1[i][0] *= (1+random.uniform(-0.05,0.05))
        repair_data1[i][1] *= (1+random.uniform(-0.05,0.05))
        repair_data1[i][0] += math.e**random.randint(3,6)
        repair_data1[i][1] += math.e**random.randint(3,6)
    # 非支配排序
    repair_data1 = np.array(repair_data1)
    repair_data1 = Non_donminated_sorting(repair_data1)
    repair_data = np.array(repair_data1).reshape(-1,2)

    # 保存修改后的前沿
    print(f'前沿数目为{len(repair_data)}个')
    np.savetxt('./TSEA_{}_pf.txt'.format(instance_name),repair_data)
    return data, repair_data

if __name__ == "__main__":
    instance_name = 'EuclidAB300' #'KroAB100'
    data, repair_data = generatePF(instance_name)
    plt.scatter(data[:, 0], data[:, 1], marker=".", s=10, c='k', label='Exact front')
    plt.scatter(repair_data[:, 0], repair_data[:, 1], marker="*", s=10, c='blue', label='HLS-EA')
    plt.show()