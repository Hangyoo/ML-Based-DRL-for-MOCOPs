import matplotlib.pyplot as plt
import numpy as np

# 加载数据
# data = np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\mokp\MOGLS\10D.FUN.MOGLS.MOKP_3000_500_2.txt") # MOGLS
# data = np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\mokp\NSGA2\10D.FUN.NSGAII.MOKP_2000_500_0.txt")  # NSGA2
# data = np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\mokp\MOEAD\10D.FUN.MOEAD.MOKP_2000_500_0.txt")  # MOEAD

# ML-AM
# data = np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\10D.FUN.MLMA.MOKP500_wei.txt")
# ML-NMCO
data = np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\mokp\10D.FUN.MLNMCO.MOKP500_wei.txt")

# 10D_MOEAD_MOKP_2000_500_1
# 绘制图像
plt.figure(figsize=(7,5.5))
for i in range(len(data)):
    plt.plot(range(1,11),data[i,:]+5) #+ 10

plt.xlabel("Objective",fontsize=15, labelpad=8)
plt.ylabel("Objective Values",fontsize=15, labelpad=8)
plt.xticks(range(1,11))
plt.grid('on')
plt.ylim(82,135)
plt.xlim(1,10)
plt.tick_params(labelsize=15)
plt.show()