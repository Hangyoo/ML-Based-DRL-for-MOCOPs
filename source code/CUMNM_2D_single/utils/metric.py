import numpy as np
import pandas as pd
from jmetal.util.solution import read_solutions, print_function_values_to_file,\
     print_variables_to_file, get_non_dominated_solutions,print_object_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.schedule_util.ranking import Non_donminated_sorting
import copy
from jmetal.core.quality_indicator import HyperVolume, InvertedGenerationalDistance
 
'''
计算IGD和HV, 只需输入PF 和 current PF 即可
'''

def cal_HV(PF,current_PF):
    ture_PF = copy.deepcopy(PF)
    algorithm_PF = copy.deepcopy(current_PF)
    M = len(ture_PF[0])
    # 获取前沿的最大值和最小值
    nadir_point = [float("-inf")] * M
    ideal_point = [float("inf")] * M
    for vector in ture_PF:
        nadir_point = [y if y > x else x for x, y in zip(nadir_point, vector)]
        ideal_point = [y if y < x else x for x, y in zip(ideal_point, vector)]
    # 标准化当前pareto前沿
    for i in range(M):
        algorithm_PF[:, i] = (algorithm_PF[:, i] - ideal_point[i]) / (nadir_point[i] - ideal_point[i])

    hypervolume = HyperVolume([1.0]*M)
    val = hypervolume.compute(algorithm_PF)
    # print(f"Hypervolume: {val}")

    return val

def cal_IGD(PF,current_PF):
    ture_PF = copy.deepcopy(PF)
    algorithm_PF = copy.deepcopy(current_PF)
    M = len(ture_PF[0])
    # 获取前沿的最大值和最小值
    nadir_point = [float("-inf")] * M
    ideal_point = [float("inf")] * M

    for vector in ture_PF:
        nadir_point = [y if y > x else x for x, y in zip(nadir_point, vector)]
        ideal_point = [y if y < x else x for x, y in zip(ideal_point, vector)]

    # 标准化真实pareto前沿
    for i in range(M):
        ture_PF[:, i] = (ture_PF[:, i] - ideal_point[i]) / (nadir_point[i] - ideal_point[i])
    # 标准化当前pareto前沿
    for i in range(M):
        algorithm_PF[:, i] = (algorithm_PF[:, i] - ideal_point[i]) / (nadir_point[i] - ideal_point[i])
    value = InvertedGenerationalDistance(ture_PF).compute(algorithm_PF)
    # print(f"IGD: {val}")

    return value

def calMetric(PF,reference_front):
     reference_front = np.array(Non_donminated_sorting(reference_front))
     IGD = cal_IGD(PF, reference_front)
     HV = cal_HV(PF, reference_front)
     return IGD,HV

# # 合并当前最优解,并筛选当前最好解
# reference_front_path = r"C:\Users\Hangyu\Desktop\data11\referenceFront.csv"
#
# PF = np.loadtxt(reference_front_path)
# reference_front_path_ga_1 = r"C:\Users\Hangyu\Desktop\data11\awGA\Obj1.csv"
#
#
# IGD_ga_1,HV_ga_1 = calMetric(PF,reference_front_path_ga_1)
# print('awGA-Obj1:',IGD_ga_1,HV_ga_1)

