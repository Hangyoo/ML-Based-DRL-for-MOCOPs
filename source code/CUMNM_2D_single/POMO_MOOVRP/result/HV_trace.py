# -*- ecoding: utf-8 -*-
# @Project : EMNH
# @Author: Hangyu Lou
# @Time: 2024/9/2 20:58

import matplotlib.pyplot as plt
import numpy as np
from Meta_Transformer_2D_reptile.utils.Nondominate_sort import Non_donminated_sorting
from Meta_Transformer_2D_single.utils.metric import *

reference_front_path = r"D:\informs\EMNH\Meta_Transformer_2D_single\POMO_MOOVRP\result\tch_noaug\cvrp_real\C\C1_epoch_2000.txt"
PF = np.loadtxt(reference_front_path)

HVs = []
# for epoch in range(10,2010,10):
for epoch in range(10,2010,10):
    reference_front_path_ga_1 = rf"D:\informs\EMNH\Meta_Transformer_2D_single\POMO_MOOVRP\result\tch_noaug\cvrp_real\C\C1_epoch_{epoch}.txt"
    reference_front = np.loadtxt(reference_front_path_ga_1)
    IGD_ga_1,HV_ga_1 = calMetric(PF,reference_front)
    HVs.append(HV_ga_1)
    print(f'{epoch} - HV:',IGD_ga_1,HV_ga_1)

plt.plot(range(len(HVs)), HVs)
plt.show()
