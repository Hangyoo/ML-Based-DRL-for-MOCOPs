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
def min_max_obj():
    # path 每个算法在每个算例运行20次后得到的PF
    cup = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
    for j in range(len(cup)):
        # data_ma = np.loadtxt(r"D:\Paper Data\TVCS\NSGA3\front\TI"+cup[j]+".pf") # DRA
        data_ma = np.loadtxt(r"D:\Paper Data\TVCS\CRA\front_new\TI"+cup[j]+".pf") # PD
        print("TI" + cup[j] + ":")
        extreme_f1,extreme_f2,extreme_f3 = min(data_ma[:,0]),min(data_ma[:,1]),min(data_ma[:,2])
        print(round(extreme_f1,1),round(extreme_f2,2),round(extreme_f3,3))
        extreme_f1 = np.round(data_ma[data_ma[:,0] == extreme_f1],1).tolist()[0]
        print(extreme_f1[0],',',extreme_f1[1],',',round(extreme_f1[2]/60,1))
        print('********')
        extreme_f2 = np.round(data_ma[data_ma[:,1] == extreme_f2],1).tolist()[0]
        print(extreme_f2[0],',',extreme_f2[1],',',round(extreme_f2[2]/60,1))
        print('********')
        extreme_f3 = np.round(data_ma[data_ma[:,2] == extreme_f3],1).tolist()[0]
        print(extreme_f3[0],',',extreme_f3[1],',',round(extreme_f3[2]/60,1))
        print()

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


# ********** MOEA/D-PD 新写的程序 ********** #
def algorithm_pareto_merge():
    # 将不同算法获得的PF进行合并 并写入文件
    instances = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
    for ins in instances:
        reference_front_path1 = r"C:\Users\Hangyu\Desktop\TVCS\CRA\front\TI"+ins+".pf"    # MOEA/D-CRA
        reference_front_path2 = r"C:\Users\Hangyu\Desktop\TVCS\DRA\front\TI"+ins+".pf"    # MOEA/D-DRA
        reference_front_path3 = r"C:\Users\Hangyu\Desktop\TVCS\M2M\front\TI"+ins+".pf"    # MOEA/D-M2M
        # reference_front_path4 = r"C:\Users\Hangyu\Desktop\TVCS\NSGA2\front\TI"+ins+".pf"  # NSGA2
        reference_front_path5 = r"C:\Users\Hangyu\Desktop\TVCS\NSGA3\front\TI"+ins+".pf"  # NSGA3
        reference_front_path6 = r"C:\Users\Hangyu\Desktop\TVCS\PD\front\TI"+ins+".pf"     # MOEA/D-PD
        reference_front_path7 = r"C:\Users\Hangyu\Desktop\TVCS\SVM\front\TI"+ins+".pf"    # MOEA/D-SVM
        # reference_front_path8 = r"C:\Users\Hangyu\Desktop\TVCS\PD_NL\front\TI"+ins+".pf"  # MOEA/D-PD-NL
        # reference_front_path9 = r"C:\Users\Hangyu\Desktop\TVCS\PD_NP\front\TI"+ins+".pf" # MOEA/D-PD-NP

        reference_front1 = np.loadtxt(reference_front_path1).tolist()
        reference_front2 = np.loadtxt(reference_front_path2).tolist()
        reference_front3 = np.loadtxt(reference_front_path3).tolist()
        # reference_front4 = np.loadtxt(reference_front_path4).tolist()
        reference_front5 = np.loadtxt(reference_front_path5).tolist()
        reference_front6 = np.loadtxt(reference_front_path6).tolist()
        reference_front7 = np.loadtxt(reference_front_path7).tolist()
        # reference_front8 = np.loadtxt(reference_front_path8).tolist()
        # reference_front9 = np.loadtxt(reference_front_path9).tolist()

        all = reference_front1 + reference_front2 + reference_front3 + reference_front5 \
              + reference_front6 + reference_front7 #+ reference_front8 + reference_front9

        now_reference_front = Non_donminated_sorting(all)

        reference_front_path = r"C:\Users\Hangyu\Desktop\TVCS\PF\TI"+ins+".pf"
        print_object_to_file(reference_front_path, now_reference_front)
        print(f"完成instance TI-{ins} 前沿合并！")

def file_pareto_merge():
    # 将每个算法获得的PF进行合并 并写入文件 (调参时使用)
    instances = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15']
    independentRuns = 30 # 算法独立运行的次数
    print(" 0: CRA\n","1: DRA\n","2: M2M\n","3: NSGA2\n","4: NSGA3\n","5: SVM\n","6: MOEA/D-PD\n","7: MOEA/D-PD-NP\n","8: MOEA/D-PD-NL\n")
    idx = int(input("请输入要合并的算法序号:"))
    algorithm = ["CRA","DRA","M2M","NSGA2","NSGA3","SVM","PD","NP","NL"]

    # 判断合并哪种算法的pareto
    if algorithm[idx] == "CRA":
        path = r"D:\Paper Data\TVCS\CRA\FUN.MOEAD-CRA.Cloudcpt"  # MOEA/D-CRA
        path_pront = r"D:\Paper Data\TVCS\CRA\front_new"  # MOEA/D-CRA
    elif algorithm[idx] == "DRA":
        path = r"D:\Paper Data\TVCS\DRA\FUN.MOEAD-DRA.Cloudcpt"  # MOEA/D-DRA
        path_pront = r"D:\Paper Data\TVCS\DRA\front_new"  # MOEA/D-DRA
    elif algorithm[idx] == "M2M":
        path = r"D:\Paper Data\TVCS\M2M\FUN_MOEAD-M2M.Cloudcpt"  # MOEA/D-M2M
        path_pront = r"D:\Paper Data\TVCS\M2M\front_new"  # MOEA/D-M2M
    elif algorithm[idx] == "NSGA2":
        path = r"D:\Paper Data\TVCS\NSGA2\FUN.NSGAII.Cloudcpt"   # NSGA2
        path_pront = r"D:\Paper Data\TVCS\NSGA2\front_new"   # NSGA2
    elif algorithm[idx] == "NSGA3":
        path = r"D:\Paper Data\TVCS\NSGA3\FUN.NSGA3.Cloudcpt"    # NSGA3
        path_pront = r"D:\Paper Data\TVCS\NSGA3\front_new"    # NSGA3
    elif algorithm[idx] == "SVM":
        path = r"D:\Paper Data\TVCS\SVM\FUN.MOEAD-SVM.Cloudcpt"  # MOEA/D-SVM
        path_pront = r"D:\Paper Data\TVCS\SVM\front_new"  # MOEA/D-SVM
    elif algorithm[idx] == "PD":
        path = r"D:\Paper Data\TVCS\PD\FUN.MOEAD-PD.Cloudcpt"    # MOEA/D-PD
        path_pront = r"D:\Paper Data\TVCS\PD\front_new"    # MOEA/D-PD
    elif algorithm[idx] == "NP":
        path = r"D:\Paper Data\TVCS\PD_NP\FUN.MOEAD-PD.Cloudcpt"    # MOEA/D-NP
        path_pront = r"D:\Paper Data\TVCS\NP\front_new"    # MOEA/D-NP
    elif algorithm[idx] == "NL":
        path = r"D:\Paper Data\TVCS\PD_NL\FUN.MOEAD-PD.Cloudcpt"    # MOEA/D-NL
        path_pront = r"D:\Paper Data\TVCS\NL\front_new"    # MOEA/D-NL
    else:
        print('算法名称输入有误')

    for i in range(15):
        ins = instances[i]
        all = []
        for j in range(independentRuns):
            reference_front_path = path + str(i) + "_" + str(j) + ".txt"
            reference_front = np.loadtxt(reference_front_path).tolist()
            all += reference_front
        print(f"完成instance TI-{i+1} 前沿合并！")

        now_reference_front = Non_donminated_sorting(all)

        reference_front_path = path_pront + r"\TI"+ins+".pf"
        print_object_to_file(reference_front_path, now_reference_front)

def experiment_pareto_merge():
    # 在experimental实验中,将不同参数组合下得的PF进行合并 并写入文件
    # THETA = [0.6, 0.7, 0.8, 0.9, 1.0]  # 第2个参数敏感性分析变量取值
    # L = [2, 4, 6, 8, 10]  # 第1个参数敏感性分析变量取值
    case0_list = ['1', '2', '3', '4', '5']  # θ=0.6;θ=0.7;θ=0.8;θ=0.9;θ=1.0
    case1_list = ['1', '2', '3', '4', '5']  # L=2; L=4; L=6; L=8; L=10
    instances = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
    file_path = r'C:/Users/Hangyu/Desktop/TVCS/Sensitive Analysis/data/case'
    PF = [[] for _ in range(len(instances))] # 存放所有问题的PF前沿

    for case0 in case0_list:
        for case1 in case1_list:
            for problem_idx in range(len(instances)):
                for run in range(10):
                    path = file_path + case0 + '/case' + case0 + case1 + '/TI' + instances[problem_idx] + '/FUN.' + str(run) + '.tsv'
                    reference_front = np.loadtxt(path).tolist()
                    PF[problem_idx] += reference_front

    # 对experimental中问题的PF进行汇总
    for problem_idx in range(len(instances)):
        _pf = PF[problem_idx]
        pf = Non_donminated_sorting(_pf)
        reference_front_path = r"D:\Paper Data\TVCS\Sensitive Analysis\PF\TI" + instances[problem_idx] + ".pf"
        print_object_to_file(reference_front_path, pf)
        print(f"完成instance TI-{instances[problem_idx]} 前沿合并！")

    # 将experimental中问题的PF与其余对比算法获得的PF进行汇总
    for problem_idx in range(len(instances)):
        _pf = PF[problem_idx]
        path = r"D:\Paper Data\TVCS\PF\TI"+ instances[problem_idx] + ".pf"
        pf = Non_donminated_sorting(_pf + np.loadtxt(path).tolist())
        reference_front_path = r"D:\Paper Data\TVCS\PF\TI"+ instances[problem_idx] + ".pf"
        print_object_to_file(reference_front_path, pf)
        print(f"完成instance TI-{instances[problem_idx]} 前沿合并！")

def DOE_analysis():
    problemNames = ['TI03','TI05','TI06','TI10','TI13']
    runtime = 10  # 独立运行10次
    combinations = 16 # 16种参数组合
    RVs = np.zeros((len(problemNames),combinations))
    path_front = r"D:\Paper Data\TVCS\DOE"
    for k in range(len(problemNames)): # 对问题循环
        problemName = problemNames[k]
        all0 = []
        for i in range(combinations): # 对组合循环
            all1 = []
            for j in range(runtime):  # 对运行次数循环
                path = r"D:\Paper Data\TVCS\DOE\data\FUN.MOEAD-PD.Cloudcpt"+problemName+"_case_"+str(i)+"_"+str(j)+".txt"
                reference_front = np.loadtxt(path).tolist()
                all1 += reference_front
            print(f"完成{problemName}-case{i+1}前沿合并！")
            now_reference_front = Non_donminated_sorting(all1)
            reference_front_path_case = path_front + r"/" + problemName +"_case"+str(i)+ ".pf"
            print_object_to_file(reference_front_path_case, now_reference_front)
            all0 += now_reference_front
        now_reference_front = Non_donminated_sorting(all0)
        reference_front_path = path_front + r"/" + problemName + ".pf"
        print_object_to_file(reference_front_path, now_reference_front)
        print(f"完成{problemName}所有case合并！")

        path = reference_front_path
        sub_path = path_front + r"/" + problemName +"_case"
        RV = cal_RV(path,sub_path,combinations)
        RVs[k,:] = RV

    # 对RVs按列求均值
    print(RVs)
    print(RVs.mean(axis=0))

def cal_RV(path, sub_path, combinations):
    # path = r"C:\Users\Hangyu\Desktop\Paper 1\Data\parameters_tunning\front\referenceFront_LF6.txt"
    # sub_path = r"C:\Users\Hangyu\Desktop\Paper 1\Data\parameters_tunning\front\case_"

    # 计算RV值
    RV = []
    pareto_front_path = path
    pareto_front = np.loadtxt(pareto_front_path).tolist()
    size = len(pareto_front)
    # print("个数:", size)
    # print("-----------------")

    for i in range(combinations):
        num = 0
        # reference_front_path = r"C:\Users\Hangyu\Desktop\Paper 1\Data\parameters_tunning\front\case_" + str(k) + ".txt"
        reference_front_path = sub_path + str(i) + ".pf"
        reference_front = np.loadtxt(reference_front_path).tolist()
        for obj in reference_front:
            if obj in pareto_front:
                num += 1
        # print(f"第{i}个case的贡献度:", num)
        RV.append(round(num / size, 4))

    # for i in range(combinations):
    #     print("贡献度:")
    #     print(RV[i])

    return RV






if __name__ == "__main__":
    # ["CRA", "DRA", "M2M", "NSGA2", "NSGA3", "PD", "SVM"]
    # file_pareto_merge()
    # algorithm_pareto_merge()
    # experiment_pareto_merge()

    # 获取算法的极端点
    min_max_obj()

    # DOE_analysis()

    # path = r"D:\Paper Data\TVCS\PF\TI14.pf"
    # pf = np.loadtxt(path).tolist()
    # temp = []
    # for item in pf:
    #     if item[2] < 100 * 60 :
    #         temp.append(item)
    #     else:
    #         print(item)
    # print_object_to_file(path, temp)