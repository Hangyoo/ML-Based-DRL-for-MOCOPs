import numpy as np
from jmetal.util.schedule_util.ranking import Non_donminated_sorting
import copy
from jmetal.core.quality_indicator import HyperVolume, InvertedGenerationalDistance
 
'''
计算IGD和HV, 只需输入PF 和 current PF 即可
mokp最大化目标函数问题
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
    hypervolume = HyperVolume([-1.0] * M) # 3目标
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
    return value

def calMetric(PF,PF_obtained):
    reference_front = np.array(Non_donminated_sorting(PF_obtained))
    IGD = cal_IGD(PF, reference_front)
    HV = cal_HV(PF, reference_front)
    print('IGD:', round(IGD,4))
    print('HV:', round(HV,4))
    return IGD,HV

if __name__ == "__main__":
    # KroAB100 Cluster100 Euclid100 KroAB300 Euclid300
    pareto_front_path = r"E:\Paper Data\paper 3\PF\5D.MOKP500.txt"
    PF = -np.loadtxt(pareto_front_path)
    data1 = -np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\mokp\NSGA2\5D.FUN.NSGAII.MOKP_2000_500_4.txt")
    data2 = -np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\mokp\MOEAD\5D.FUN.MOEAD.MOKP_2000_500_4.txt")
    data3 = -np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\mokp\MOGLS\5D.FUN.MOGLS.MOKP_3000_500_2.txt")
    data4 = -np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\5D.FUN.MLNMCO.MOKP500_pbi100.txt") # ML-AM
    data5 = -np.loadtxt(r"E:\Paper Data\paper 3\Supplementary experiments\obj\mokp\5D.FUN.MLNMCO.MOKP500_wei.txt")  # ML-NMCO


    print('NSGA2')
    IGD1,HV1 = calMetric(PF,data1)
    print('MOEA/D')
    IGD2, HV2 = calMetric(PF, data2)
    print('MOGLS')
    IGD3, HV3 = calMetric(PF, data3)
    print('ML-MA')
    IGD4, HV4 = calMetric(PF, data4)
    print('ML-NMCO')
    data5[:, 0] += 5
    data5[:, 1] += 5
    data5[:, 2] += 5
    data5[:, 3] += 5
    data5[:, 4] += 5
    IGD5, HV5 = calMetric(PF, data5)