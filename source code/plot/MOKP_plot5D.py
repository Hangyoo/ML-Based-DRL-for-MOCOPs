import matplotlib.pyplot as plt
import numpy as np

def Non_donminated_sorting(chroms_obj_record):
    # 非支配排序
    length = len(chroms_obj_record)
    f = np.reshape(chroms_obj_record,(length,len(chroms_obj_record[0]))) 
    Rank = np.zeros(length)  # [0. 2. 1. 1. 1. 0. 0. 0. 2. 1.]
    front = []     # [[0, 5, 6, 7], [2, 3, 4, 9], [1, 8]]
    rank = 0

    n_p = np.zeros(length)
    s_p = []
    for p in range(length):
        a = (f[p, :] - f[:, :] >= 0).all(axis=1)
        b = (~((f[p, :] - f[:, :] == 0).all(axis=1)))
        loc = np.where(a & b)[0].tolist()
        s_p.append(loc)
        temp1 = np.where(((f[p, :] - f[:, :] <= 0).all(axis=1)) & (~((f[p, :] - f[:, :] == 0).all(axis=1))))[0]
        n_p[p] = len(temp1)  # p所支配个数
    # 添加第一前沿
    front.append(list(np.where(n_p == 0)[0]))

    while len(front[rank]) != 0:    # 生成其他前沿
        elementset = front[rank]
        n_p[elementset] = float('inf')
        Rank[elementset] = rank
        rank += 1

        for i in elementset:
            temp = s_p[i]
            n_p[temp] -= 1
        front.append(list(np.where(n_p == 0)[0]))
    front.pop()
    # 第一前沿
    parero_solution_obj = []
    first = front[0]
    for index in first:
        parero_solution_obj.append(chroms_obj_record[index].tolist())

    return parero_solution_obj #np.array(parero_solution_obj)

# 加载数据
# data = np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\mokp\MOEAD\5D.FUN.MOEAD.MOKP_2000_500_4.txt")
# data = np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\mokp\MOGLS\5D.FUN.MOGLS.MOKP_3000_500_2.txt")
# data = np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\mokp\NSGA2\5D.FUN.NSGAII.MOKP_2000_500_4.txt")
# data = np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\mokp\MLMA\5D.FUN.MLMA.MOKP500_wei.txt") # ML-AM
data = np.loadtxt(r"D:\Paper Data\paper 3\Supplementary experiments\obj\5D.FUN.MLNMCO.MOKP500_pbi100.txt") # ML-NMCO
data = Non_donminated_sorting(data)
data = np.array(data)
# 绘制图像
plt.figure(figsize=(7,5.5))
for i in range(len(data)):
    plt.plot(range(1,6),data[i,:]-5)


plt.xlabel("Objective",fontsize=18, labelpad=3)
plt.ylabel("Objective Values",fontsize=18, labelpad=3)
plt.xticks(range(1,6))
plt.grid('on')
plt.ylim(90,140)
plt.xlim(1,5)
plt.tick_params(labelsize=18)
# plt.text(1.75, 135, r'IGD=$4.88e-01$; HV=$4.88e-01$', fontsize=15, bbox={'facecolor':'orange', 'alpha' : 0.45})
plt.show()