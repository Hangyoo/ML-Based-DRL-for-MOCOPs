import random
import math
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt 

'''功能: 横轴为算法名字，纵轴为运行时间, 时间越小越好'''


#********** Time **********#

case1 = [50.887171030044556,51.36647653579712,51.46928381919861,105.79,107.79]  # DRL-MOA
case2 = [54.68891716003418,40.879448890686035,58.12209129333496,82.98,81.98]  # DRL-AM  (DRL-MOA-T)
case3 = [33.935723066329956,33.79312562942505,38.12167024612427,97.36,102.36] # AM-MOCO
case4 = [10.34,9.73073935508728,10.20423936843872,32.63,30.63] # ML-AM
case5 = [114.5, 115.0, 115.3, 520.83269834518433, 527.83269834518433]  # NSGA2 时间更新
case6 = [73.8, 66.1, 66.1, 585.3927516460419, 589.3927516460419]     # MOEA/D 时间更新
case7 = [115.1, 192.5, 192.5, 674.016, 676.476]           # MOGLS 时间更新
case8 = [880, 1120, 1080, 2360, 2240]          # PDA
case9 = [805.53, 809.25, 811.4, 1933.9, 2052.3]          # TSEA
case10 = [14.74682903289795,11.45665955543518,11.53190565109253,25.50379729270935,25.50379729270935]
X = [1,2,3,4,5]

font = {'size': 14}
plt.rc('font', **font)
fig,ax = plt.subplots(2,1,figsize=(9,9))
ax1 = ax[0]
ax2 = ax[1]

ins_label = ['KroAB100','ClusterAB100','EuclidAB100','KroAB300','EuclidAB300']

ax1.plot(X,case1,'v-.',linewidth=3,color='darkmagenta',markersize=6)    # DRL-MOA
ax1.plot(X,case2,'x--',linewidth=3,markersize=8)    # DRL-AM  (DRL-MOA-T)
ax1.plot(X,case3,'1-',linewidth=3,color='#0082fb',markersize=12)    # AM-MOCO
ax1.plot(X,case4,'+:',linewidth=3,color='r',markersize=8)    # ML-AM
ax2.plot(X,case5,'2--',linewidth=3,color='#875853',markersize=12,label="NSGA-II")    # NSGA2
ax2.plot(X,case6,'+-.',linewidth=3,color='green',markersize=8,label="PDA")
ax2.plot(X,case6,'.-',linewidth=3,color='tomato',markersize=8,label="MOEA/D")    # MOEA/D
ax2.plot(X,case7,'*-.',linewidth=3,color='pink',markersize=8,label="HLS-EA")
ax2.plot(X,case7,'s-',linewidth=3,color='blueviolet',markersize=8,label="MOGLS")    # MOGLS
ax1.plot(X,case10,'o-.',linewidth=3,color='darkorange',markersize=8)    # MOGLS
algorithm_label1 = ["DRL-MOA",r"DRL-MOA-T",r"AM-MOCO",r"ML-AM",r'ML-NMCO']
# algorithm_label2 = ["NSGA-II","MOEA/D","MOGLS","PDA"]
# ax1.legend(algorithm_label1, fontsize=14,frameon=True,bbox_to_anchor=(0.825, 0.75),  ncol=3,  borderaxespad=0)
# ax2.legend(algorithm_label2, fontsize=14,frameon=True,bbox_to_anchor=(0.78, 0.80),  ncol=3,  borderaxespad=0)
ax1.legend(algorithm_label1, fontsize=14,frameon=True,loc=9,  ncol=3,  borderaxespad=0)
ax2.legend(fontsize=14,frameon=True,loc=9,  ncol=3,  borderaxespad=0)
x_values = [i+1 for i in range(len(ins_label))]
# ax1.set_yticks([1.5,2.5,3.5,5.0])
# ax2.set_yticks([3.0,3.5,4.0,4.5])
ax1.xaxis.set_major_locator(mticker.FixedLocator(x_values))
ax1.xaxis.set_major_formatter(mticker.FixedFormatter(ins_label))
ax2.xaxis.set_major_locator(mticker.FixedLocator(x_values))
ax2.xaxis.set_major_formatter(mticker.FixedFormatter(ins_label))
ax3 = ax2.twinx()
ax3.plot(X,case8,'+-.',linewidth=3,color='green',markersize=8)    # PDA
ax3.plot(X,case9,'*-.',linewidth=3,color='pink',markersize=8)     # TSEA
ax1.set_title('Learning-based Algorithms')
ax2.set_title('Iteration-based Algorithms')
# x轴坐标旋转45度
labels = ax1.get_xticklabels()
ax1.set_ylabel("Running time",fontsize=14)
ax2.set_ylabel("Running time",fontsize=14)
ax3.set_ylabel("Running time for PDA and HLS-EA",fontsize=14)
# algorithm_label3 = ["PDA"]
# ax3.legend(algorithm_label3, fontsize=14,frameon=True,loc=1,  ncol=1,  borderaxespad=0)
# plt.xlabel("Algorithms name",fontsize=14)
plt.setp(labels, rotation=0)
plt.show()

