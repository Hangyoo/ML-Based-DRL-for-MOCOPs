import random
import math
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
 
'''功能: 横轴为算法名字，纵轴为IGD和HV指标下的GAP, GAP越小越好'''


#********** IGD gap **********#
case1 = [math.log(item,10) for item in [86.92244431492092, 95.25904471524602, 94.83401394184453, 86.92244431492092, 93.48468694444571]]
case2 = [math.log(item,10) for item in [50.691535653229984, 89.2383261972882, 82.86414854984622, 50.691535653229984, 79.6099520950419]]
case3 = [math.log(item,10) for item in [37.26491151394533, 41.48281155051022, 51.58889858989623, 37.26491151394533, 60.311269302011354]]
case4 = [math.log(item,10) for item in [25.99862634192053, 38.21812889996734, 33.01498133226386, 25.99862634192053, 39.301310043668124]]
case5 = [math.log(item,10) for item in [91.94888376648005, 93.78207691772126, 94.38848792085724, 91.94888376648005, 95.30272257007233]]
case6 = [math.log(item,10) for item in [92.3298580048132, 92.61489084340765, 94.9981177878123, 92.3298580048132, 95.9799516250887]]
case7 = [math.log(item,10) for item in [69.82037849362652, 80.26585870499534, 70.17122219202199, 69.82037849362652, 83.16068702691288]]
case8 = [math.log(item,10) for item in [24.936712715997437, 8.267852140326656, 47.87403950291486, 24.936712715997437, 42.98950554451024]]
case9 = [math.log(item,10) for item in [25.1555958308273, 30.78491944980476, 58.77348227882275, 25.1555958308273, 42.358589586392306]]

X = [1,2,3,4,5]

font = {'size': 14}
plt.rc('font', **font)
fig,ax = plt.subplots(2,1,figsize=(8,9))
ax1 = ax[0]
ax2 = ax[1]
ins_label = ['KroAB100','ClusterAB100','EuclidAB100','KroAB300','EuclidAB300']

ax1.plot(X,case1,'v-.',linewidth=3,color='darkmagenta',markersize=6)    # DRL-MOA
ax1.plot(X,case2,'x--',linewidth=3,markersize=8)    # DRL-AM  (DRL-MOA-T)
ax1.plot(X,case3,'1-',linewidth=3,color='#0082fb',markersize=12)    # AM-MOCO
ax1.plot(X,case4,'+:',linewidth=3,color='r',markersize=8)    # ML-AM
ax1.set_title('Learning-based Algorithms')
ax2.plot(X,case5,'2--',linewidth=3,color='#875853',markersize=12,label="NSGA-II")    # NSGA2
ax2.plot(X,case8,'+-.',linewidth=3,color='green',markersize=8,label="PDA")    # PDA
ax2.plot(X,case6,'.-',linewidth=3,color='tomato',markersize=8,label="MOEA/D")    # MOEA/D
ax2.plot(X,case9,'*-.',linewidth=3,color='pink',markersize=8,label="HLS-EA")
ax2.plot(X,case7,'s-',linewidth=3,color='blueviolet',markersize=8,label="MOGLS")    # MOGLS
algorithm_label1 = ["DRL-MOA",r"DRL-MOA-T",r"AM-MOCO",r"ML-AM"]
# algorithm_label2 = ["NSGA-II","MOEA/D","MOGLS","PDA"]
ax1.legend(algorithm_label1, fontsize=14,frameon=True,loc=9,  ncol=3,  borderaxespad=0)
ax2.legend(fontsize=14,frameon=True,loc=9,  ncol=3,  borderaxespad=0)
# ax1.legend(algorithm_label, fontsize=14,frameon=True)
ax2.set_title('Iteration-based Algorithms')
x_values = [i+1 for i in range(len(ins_label))]
ax1.set_yticks([1.0,1.5,2.0,2.5])
ax2.set_yticks([0.5,1.0,1.5,2.0,2.5])
ax1.xaxis.set_major_locator(mticker.FixedLocator(x_values))
ax1.xaxis.set_major_formatter(mticker.FixedFormatter(ins_label))
ax2.xaxis.set_major_locator(mticker.FixedLocator(x_values))
ax2.xaxis.set_major_formatter(mticker.FixedFormatter(ins_label))
# x轴坐标旋转45度
labels = ax1.get_xticklabels()
ax1.set_ylabel("IGD metric $log_1$$_0$(gap)",fontsize=14)
ax2.set_ylabel("IGD metric $log_1$$_0$(gap)",fontsize=14)
# plt.xlabel("Algorithms name",fontsize=14)
plt.setp(labels, rotation=0)
plt.show()


#********** HV gap **********#

case1 = [math.log(item,10) for item in [280.1783181959211, 763.515811793677, 388.6081278865422, 280.1783181959211, 377.26477755257605]]
case2 = [math.log(item,10) for item in [12.752843425323308, 52.33898654322258, 24.023227622435826, 12.752843425323308, 25.766584662382787]]
case3 = [math.log(item,10) for item in [6.568295922693449, 5.402065901036173, 5.514062769018317, 6.568295922693449, 9.656077226290824]]
case4 = [math.log(item,10) for item in [1.5796185477781761, 1.451413605799247, 1.9973304523003068, 1.5796185477781761, 1.4234875444839872]]
case5 = [math.log(item,10) for item in [408.4879716277066, 160.104010356123, 146.81303799526245, 408.4879716277066, 411.1236001364968]]
case6 = [math.log(item,10) for item in [533.3816200096605, 117.78674862324509, 193.17977069327276, 533.3816200096605, 725.2578228626206]]
case7 = [math.log(item,10) for item in [27.100847591354654, 21.13315822907042, 11.24235980693575, 27.100847591354654, 31.826098040199106]]
case8 = [math.log(item,10) for item in [2.2570816998118035, 1.4613203924119937, 3.774017606001872, 2.2570816998118035, 3.5320035006010735]]  # PDA
case9 = [math.log(item,10) for item in [2.951672119748005, 3.241131625786749, 6.220888020295254, 2.951672119748005, 3.510895883777243]]  # HLS-EA

X = [1,2,3,4,5]

font = {'size': 14}
plt.rc('font', **font)
fig,ax = plt.subplots(2,1,figsize=(8,9))
ax1 = ax[0]
ax2 = ax[1]

ins_label = ['KroAB100','ClusterAB100','EuclidAB100','KroAB300','EuclidAB300']

ax1.plot(X,case1,'v-.',linewidth=3,color='darkmagenta',markersize=6)    # DRL-MOA
ax1.plot(X,case2,'x--',linewidth=3,markersize=8)    # DRL-AM  (DRL-MOA-T)
ax1.plot(X,case3,'1-',linewidth=3,color='#0082fb',markersize=12)    # AM-MOCO
ax1.plot(X,case4,'+:',linewidth=3,color='r',markersize=8)    # ML-AM
ax1.set_title('Learning-based Algorithms')
ax2.plot(X,case5,'2--',linewidth=3,color='#875853',markersize=12,label="NSGA-II")    # NSGA2
ax2.plot(X,case8,'+-.',linewidth=3,color='green',markersize=8,label="PDA")    # PDA
ax2.plot(X,case6,'.-',linewidth=3,color='tomato',markersize=8,label="MOEA/D")    # MOEA/D
ax2.plot(X,case9,'*-.',linewidth=3,color='pink',markersize=8,label="HLS-EA")    # MOEA/D
ax2.plot(X,case7,'s-',linewidth=3,color='blueviolet',markersize=8,label="MOGLS")    # MOGLS
algorithm_label1 = ["DRL-MOA",r"DRL-MOA-T",r"AM-MOCO",r"ML-AM"]
# algorithm_label2 = ["NSGA-II","MOEA/D","MOGLS","PDA"]
# ax1.legend(algorithm_label1, fontsize=14,frameon=True,bbox_to_anchor=(0.825, 0.75),  ncol=3,  borderaxespad=0)
# ax2.legend(algorithm_label2, fontsize=14,frameon=True,bbox_to_anchor=(0.78, 0.80),  ncol=3,  borderaxespad=0)
ax1.legend(algorithm_label1, fontsize=14,frameon=True,loc=9,  ncol=3,  borderaxespad=0)
ax2.legend(fontsize=14,frameon=True,loc=9,  ncol=3,  borderaxespad=0)
ax2.set_title('Iteration-based Algorithms')
x_values = [i+1 for i in range(len(ins_label))]
ax1.set_yticks([0,0.5,1.0,1.5,2.0,2.5,3.0,3.5])
ax2.set_yticks([0,0.5,1.0,1.5,2.0,2.5,3.0,3.5])
ax1.xaxis.set_major_locator(mticker.FixedLocator(x_values))
ax1.xaxis.set_major_formatter(mticker.FixedFormatter(ins_label))
ax2.xaxis.set_major_locator(mticker.FixedLocator(x_values))
ax2.xaxis.set_major_formatter(mticker.FixedFormatter(ins_label))
# x轴坐标旋转45度
labels = ax1.get_xticklabels()
# ax1.set_ylabel("HV metric gap",fontsize=14)
# ax2.set_ylabel("HV metric gap",fontsize=14)
ax1.set_ylabel("HV metric $log_1$$_0$(gap)",fontsize=14)
ax2.set_ylabel("HV metric $log_1$$_0$(gap)",fontsize=14)
# plt.xlabel("Algorithms name",fontsize=14)
plt.setp(labels, rotation=0)
plt.show()

