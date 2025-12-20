import numpy as np
import pandas as pd
 
# 将获得的所有最优解写入文件
def print_object_to_file(path,PF):
    if type(PF) is not list:
        PF = [PF]

    with open(path, 'w') as of:
        for point in PF:
            for function_value in point:
                of.write(str(function_value) + ' ')
            of.write('\n')

# 将每一个解写入指定文件
def print_variables_to_file(solutions, filename: str):
    if type(solutions) is not list:
        solutions = [solutions]

    with open(filename, 'w') as of:
        for solution in solutions:
            for variables in solution.variables:
                of.write(str(variables) + " ")
            of.write("\n")

# 将每个个体的目标函数写入指定文件
def print_function_values_to_file(solutions, filename: str):
    if type(solutions) is not list:
        solutions = [solutions]

    with open(filename, 'w') as of:
        for solution in solutions:
            for function_value in solution.objectives:
                of.write(str(function_value) + ' ')
            of.write('\n')

# 将运算过程记录的IGD指标写入指定文件
def print_igd_values_to_file(IGD, filename: str):
    if type(IGD) is not list:
        solutions = [IGD]
    with open(filename, 'w') as of:
        for igd in IGD:
            of.write(str(igd) + ' ')

def name(path):
    idx = path[-6:-4]
    return str(idx)

# 筛选算法输出的非支配集合，使其不重复
def check(result):
    '''
    :param result: 算法运行后得到的前沿
    :return: 不重复的Pareto解
    '''
    front_value = []
    front = []
    for solution in result:
        if solution.objectives not in front_value:
            front.append(solution)
            front_value.append(solution.objectives)
    return front

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
        a = (f[p, :] - f[:, :] <= 0).all(axis=1)
        b = (~((f[p, :] - f[:, :] == 0).all(axis=1)))
        loc = np.where(a & b)[0].tolist()
        s_p.append(loc)
        temp1 = np.where(((f[p, :] - f[:, :] >= 0).all(axis=1)) & (~((f[p, :] - f[:, :] == 0).all(axis=1))))[0]
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
        parero_solution_obj.append(chroms_obj_record[index])
    return parero_solution_obj

def concat_non_dominated_solution(front, idx):
    '''
    筛选算法在执行过程中产生的所有非支配解
    :param front: 算法当前前沿
    :param idx: 问题序号
    :return: PF
    '''
    # 读取已有非支配解
    reference_front_path = r'C:\Users\Hangyu\PycharmProjects\JmetalPlus\examples\MOMA_LD\data\referenceFront LF' + str(
        idx)
    reference_front = np.loadtxt(reference_front_path).tolist()
    # 读取当前算法产生的非支配解
    front_value = [item.objectives for item in front]
    # 合并
    all = front_value + reference_front
    # 非支配排序
    temp_front = Non_donminated_sorting(all)
    # 去冗余(重复的)
    PF = []
    for obj in temp_front:
        if obj not in PF:
            PF.append(obj)
    # 写入文件
    with open(reference_front_path, 'w') as of:
        for objectives in PF:
            for function_value in objectives:
                of.write(str(function_value) + ' ')
            of.write('\n')

# 目标函数值
def min_max_obj(path):
    # path 每个算法在每个算例运行20次后得到的PF
    cup = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
    for j in range(len(cup)):
        data_ma = np.loadtxt(r"C:\Users\Hangyu\Desktop\Data\Fun\front\LF_"+cup[j]+".txt")
        print("LF" + cup[j] + ":")
        print(min(data_ma[:,0]),min(data_ma[:,1]),min(data_ma[:,2]))

#平均pareto个数
def avg_ps_size(path, n=20):
    # path 每个算法在每个算例运行时, 算法每次得到的PF的地址: "C:\Users\Hangyu\Desktop\Data\Fun\FUN. HMOEAD .FJSPLFLF"
    # n 算法独立运行次数

    # 解的唯一性确定
    def Unik(data):
        temp = []
        for item in data:
            if item in temp:
                pass
            else:
                temp.append(item)
        return len(temp)

    cup = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
    for j in range(len(cup)):
        b = 0
        for i in range(n):
            #IGD_data = np.loadtxt(r"C:\Users\Hangyu\Desktop\Data\Fun\FUN. HMOEAD .FJSPLFLF"+cup[problem_idx]+"_"+str(k)+".txt").tolist()
            data = np.loadtxt(path+cup[j]+"_"+str(i)+".txt").tolist()
            length = Unik(data)
            b += length
        print("LF" + cup[j] + ":", end=" ")
        print(b/20)

def pareto_merge():
    # 将不同算法获得的PF进行合并 并写入文件
    instances = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
    for ins in instances:

        reference_front_path1 = r"C:\Users\Hangyu\Desktop\TVCS\CRA\front\TI"+ins+".txt"    # MOEA/D-CRA
        reference_front_path2 = r"C:\Users\Hangyu\Desktop\TVCS\DRA\front\TI"+ins+".txt"    # MOEA/D-DRA
        reference_front_path3 = r"C:\Users\Hangyu\Desktop\TVCS\M2M\front\TI"+ins+".txt"    # MOEA/D-M2M
        reference_front_path4 = r"C:\Users\Hangyu\Desktop\TVCS\NSGA2\front\TI"+ins+".txt"  # NSGA2
        reference_front_path5 = r"C:\Users\Hangyu\Desktop\TVCS\NSGA3\front\TI"+ins+".txt"  # NSGA3
        reference_front_path6 = r"C:\Users\Hangyu\Desktop\TVCS\PD\front\TI"+ins+".txt"     # MOEA/D-PD
        reference_front_path7 = r"C:\Users\Hangyu\Desktop\TVCS\SVM\front\TI"+ins+".txt"  # MOEA/D-SVM

        reference_front1 = np.loadtxt(reference_front_path1).tolist()
        reference_front2 = np.loadtxt(reference_front_path2).tolist()
        reference_front3 = np.loadtxt(reference_front_path3).tolist()
        reference_front4 = np.loadtxt(reference_front_path4).tolist()
        reference_front5 = np.loadtxt(reference_front_path5).tolist()
        reference_front6 = np.loadtxt(reference_front_path6).tolist()
        reference_front7 = np.loadtxt(reference_front_path7).tolist()

        all = reference_front1 + reference_front2 + reference_front3 + reference_front4 + reference_front5 + reference_front6 + reference_front7

        now_reference_front = Non_donminated_sorting(all)

        reference_front_path = r"C:\Users\Hangyu\Desktop\TVCS\PF\PF"+ins+".txt"
        print_object_to_file(reference_front_path, now_reference_front)

def Pareto_merge():
    # 将获得的PF进行合并 并写入文件 (调参时使用)
    instances = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15']
    for ins in instances:
        all = []
        for j in range(20):
            reference_front_path = r"C:\Users\Hangyu\Desktop\data\Momald\FUN.MALD.FJSPLF"+ins+"_"+ str(j)
            reference_front = np.loadtxt(reference_front_path).tolist()
            all += reference_front

        # reference_front_path = r"C:\Users\Hangyu\Desktop\IGD_data\parameters_tunning\referenceFront_LF06.txt"
        # reference_front = np.loadtxt(reference_front_path).tolist()
        # now_path = r"C:\Users\Hangyu\Desktop\PR4\FUN.MALD.FJSPLF06_75"
        # now = np.loadtxt(now_path).tolist()
        # all = reference_front + now

        now_reference_front = Non_donminated_sorting(all)

        reference_front_path = r"C:\Users\Hangyu\Desktop\data\Momald\LF"+ins+".pf"
        print_object_to_file(reference_front_path, now_reference_front)


def cal_RV(path, sub_path):
    # path = r"C:\Users\Hangyu\Desktop\Paper 1\Data\parameters_tunning\front\referenceFront_LF6.txt"
    # sub_path = r"C:\Users\Hangyu\Desktop\Paper 1\Data\parameters_tunning\front\case_"

    # 计算RV值
    RV = []
    pareto_front_path = path
    pareto_front = np.loadtxt(pareto_front_path).tolist()
    size = len(pareto_front)
    print("个数:", size)
    print("-----------------")

    for i in range(16):
        num = 0
        # reference_front_path = r"C:\Users\Hangyu\Desktop\Paper 1\Data\parameters_tunning\front\case_" + str(k) + ".txt"
        reference_front_path = sub_path + str(i) + ".txt"
        reference_front = np.loadtxt(reference_front_path).tolist()
        for obj in reference_front:
            if obj in pareto_front:
                num += 1
        print(f"第{i}个case的贡献度:", num)
        RV.append(round(num/size,4))

    for i in range(16):
        print("贡献度:")
        print(RV[i])


if __name__ == "__main__":
    # path = r"C:\Users\Hangyu\Desktop\parameters_tunning\referenceFront_LF06.txt"
    # sub_path = r"C:\Users\Hangyu\Desktop\parameters_tunning\case"
    # cal_RV(path,sub_path)
    # Pareto_merge()
    pareto_merge()