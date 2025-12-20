# -*- ecoding: utf-8 -*-
# @Project : EMNH
# @Author: Hangyu Lou


import torch
import numpy as np
import gzip
import random 

###-----
# if problem_type is 'unified', it is trained on
# 25% CVRP   属性为坐标、depot坐标 和 点需求
# 25% MOKP   属性为重量 和 价值
# 25% MOTSP  属性为坐标
# 25% VRPTW  属性为坐标、最早窗口、最晚窗口、服务时间
###----


def get_random_problems_mixed(batch_size, problem_size, problem_type):  # CVRP  VRPTW  KP  TSP
    ########### ----- MOCVRP ----- ############
    depot_xy = torch.zeros(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.zeros(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    node_demand = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    ########### ----- MOVRPTW ----- ############
    node_serviceTime = torch.zeros(size=(batch_size, problem_size)) # 服务时间
    # shape: (batch, problem)

    node_lengthTW = torch.zeros(size=(batch_size, problem_size)) # 持续时间
    # shape: (batch, problem)

    node_earlyTW = torch.zeros(size=(batch_size, problem_size)) # 最早时间
    # shape: (batch, problem)
    # default velocity = 1.0

    node_lateTW = node_earlyTW + node_lengthTW # 最晚时间
    # shape: (batch, problem)

    ########### ----- MOTSP ----- ############
    tsp_problems = torch.zeros(size=(batch_size, problem_size, 4))
    # problems.shape: (batch, problem, 4)

    ########### ----- MOKP ----- ############
    kp_problems = torch.zeros(size=(batch_size, problem_size, 3))
    # shape: (batch, problem, 3)

    type = None # 问题的类型

    seed = np.random.rand() # 产生随机数
    # < 0.25 是为CVRP问题
    if ((problem_type == 'unified' and seed < 0.25) or 'C' in problem_type):  # MOCVRP

        depot_xy = torch.rand(size=(batch_size, 1, 2))
        # shape: (batch, 1, 2)

        node_xy = torch.rand(size=(batch_size, problem_size, 2))
        # shape: (batch, problem, 2)

        # if size > 50, demand_scaler = 30 + size/5
        if problem_size == 20:
            demand_scaler = 30
        elif problem_size == 50:
            demand_scaler = 40
        elif problem_size == 100:
            demand_scaler = 50
        elif problem_size == 200:
            demand_scaler = 70
        elif problem_size == 500:
            demand_scaler = 130
        elif problem_size == 1000:
            demand_scaler = 230
        else:
            raise NotImplementedError

        node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
        type = 'CVRP'
        # shape: (batch, problem)
        return

    if ((problem_type == 'unified' and seed>=0.25 and seed <0.5) or 'TW' in problem_type): #VRPTW problem_type is 'unified' or there is 'TW' in the problem_type
        print('TW调用')

        # if size > 50, demand_scaler = 30 + size/5
        if problem_size == 20:
            demand_scaler = 30
        elif problem_size == 50:
            demand_scaler = 40
        elif problem_size == 100:
            demand_scaler = 50
        elif problem_size == 200:
            demand_scaler = 70
        elif problem_size == 500:
            demand_scaler = 130
        elif problem_size == 1000:
            demand_scaler = 230
        else:
            raise NotImplementedError

        node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)

        depot_xy = torch.rand(size=(batch_size, 1, 2))
        # shape: (batch, 1, 2)

        node_xy = torch.rand(size=(batch_size, problem_size, 2))
        # shape: (batch, problem, 2)

        node_serviceTime = torch.rand(size=(batch_size, problem_size)) * 0.05 +0.15
        # shape: (batch, problem)
        # range: (0.15,0.2) for T=4.6

        node_lengthTW = torch.rand(size=(batch_size, problem_size)) * 0.05 +0.15
        # shape: (batch, problem)
        # range: (0.15,0.2) for T=4.6

        d0i = ((node_xy - depot_xy.expand(size=(batch_size,problem_size,2)))**2).sum(2).sqrt()
        # shape: (batch, problem)

        ei = torch.rand(size=(batch_size, problem_size)).mul((torch.div((4.6*torch.ones(size=(batch_size, problem_size)) - node_serviceTime - node_lengthTW),d0i) - 1)-1)+1
        # shape: (batch, problem)
        # default velocity = 1.0

        node_earlyTW = ei.mul(d0i)
        # shape: (batch, problem)
        # default velocity = 1.0

        node_lateTW = node_earlyTW + node_lengthTW
        type = 'VRPTW'
        # shape: (batch, problem)

        return depot_xy, node_xy, node_demand, node_serviceTime, node_earlyTW, node_lateTW

    if ((problem_type == 'unified' and seed>=0.5 and seed <0.75) or 'TSP' in problem_type):  # MOTSP
        print('TSP调用')
        tsp_problems = torch.rand(size=(batch_size, problem_size, 4))
        type = 'TSP'
        return tsp_problems

    if ((problem_type == 'unified' and seed>=0.75) or 'KP' in problem_type): #MOKP
        print('KP调用')
        if problem_size == 20:
            demand_scaler = 5 # 设置背包容量
        elif problem_size == 50:
            demand_scaler = 12.5 # 设置背包容量
        elif problem_size == 100:
            demand_scaler = 25
        elif problem_size == 200:
            demand_scaler = 25
        elif problem_size == 500:
            demand_scaler = 50
        else:
            raise NotImplementedError
        kp_problems = torch.rand(size=(batch_size, problem_size, 3)) # 3 = 2目标 + 物品重量
        kp_problems[:, :, 0] /= float(demand_scaler) # 只对重量归一化, 不对值归一化
        # problems.shape: (batch, problem, 3)
        type = 'KP'
        return kp_problems

    return depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW, node_serviceTime, tsp_problems, kp_problems, type


if __name__ == "__main__":
    # TSP
    # problem_size = 100
    # file_path = fr'C:\Users\Hangyu\Desktop\GitHub\Test Data\Random Test Set\MOTSP/MOTSP_{problem_size}_random_instance'
    # batch_size = 30
    # # 创建张量
    # tensor = torch.rand(size=(batch_size, problem_size, 4))
    # # 转换为 NumPy 数组
    # numpy_array = tensor.numpy()
    # # 打开文件进行写入
    # for i in range(numpy_array.shape[0]):
    #     with open(file_path + f'{i + 1}.txt', 'w') as f:
    #         f.write(f'MOTSP INSTANCE {i + 1}' + '\n')
    #         f.write(' '.join('X1 Y1 X2 Y2') + '\n')
    #         # 将每个 (problem_size, 4) 的数组写入文件
    #         np.savetxt(f, numpy_array[i], fmt='%.6f', delimiter=' ')
    #         f.write('\n')

    # KP
    # problem_size = 100
    # file_path =  fr'C:\Users\Hangyu\Desktop\GitHub\Test Data\Random Test Set\MOKP/MOKP_{problem_size}_random_instance'
    # batch_size = 30
    # kp_problems = get_random_problems_mixed(batch_size=batch_size, problem_size=problem_size, problem_type='KP')
    # numpy_array = kp_problems.numpy()
    # # 打开文件进行写入
    # for i in range(numpy_array.shape[0]):
    #     with open(file_path + f'{i + 1}.txt', 'w') as f:
    #         f.write(f'MOKP INSTANCE {i + 1}' + '\n')
    #         f.write(' '.join('WEIGHT VALUE1 VALUE2') + '\n')
    #         # 将每个 (problem_size, 3) 的数组写入文件
    #         np.savetxt(f, numpy_array[i], fmt='%.6f', delimiter=' ')
    #         f.write('\n')

    # CVRP
    # problem_size = 100
    # file_path =  fr'C:\Users\Hangyu\Desktop\GitHub\Test Data\Random Test Set\MOCVRP/MOCVRP_{problem_size}_random_instance'
    # batch_size = 30
    # if problem_size == 20:
    #     demand_scaler = 30
    # elif problem_size == 50:
    #     demand_scaler = 40
    # elif problem_size == 100:
    #     demand_scaler = 50
    # node_xy = torch.rand(size=(batch_size, problem_size+1, 2))
    # # shape: (batch, problem, 2)
    # node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # depot_node_demand = torch.cat((torch.zeros((batch_size, 1), dtype=node_demand.dtype), node_demand), dim=1)
    # depot_node_demand = depot_node_demand.unsqueeze(-1)
    # numpy_array = torch.cat((node_xy, depot_node_demand), dim=2)
    # numpy_array = numpy_array.numpy()
    # # 打开文件进行写入
    # for i in range(numpy_array.shape[0]):
    #     with open(file_path + f'{i + 1}.txt', 'w') as f:
    #         f.write(f'NAME : MOCVRP INSTANCE {i + 1}' + '\n')
    #         f.write(f'DEPOT_SECTION 1' + '\n')
    #         f.write(f'CAPACITY : {1.0}' + '\n')
    #         f.write(f'COORD_DEMAND_SECTION' + '\n')
    #         # 将每个 (problem_size, 3) 的数组写入文件
    #         np.savetxt(f, numpy_array[i], fmt='%.6f', delimiter=' ')


    # # VRPTW
    problem_size = 100
    file_path = rf'C:\Users\Hangyu\Desktop\GitHub\0/MOVRPTW_{problem_size}_random_instance'
    batch_size = 56
    depot_xy, node_xy, node_demand, node_serviceTime, node_earlyTW, node_lateTW = get_random_problems_mixed(batch_size=batch_size, problem_size=problem_size, problem_type='VRPTW')
    depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
    depot_node_demand = torch.cat((torch.zeros((batch_size, 1), dtype=node_demand.dtype), node_demand), dim=1)
    depot_node_demand = depot_node_demand.unsqueeze(-1)
    node_earlyTW = torch.cat((torch.zeros((batch_size, 1), dtype=node_earlyTW.dtype), node_earlyTW), dim=1)
    node_earlyTW = node_earlyTW.unsqueeze(-1)

    # 计算每行的最大值并乘以 1.1
    max_values = node_lateTW.max(dim=1, keepdim=True).values
    new_values = max_values * 1.1
    # 将新的列值与原张量拼接
    node_lateTW = torch.cat([new_values, node_lateTW], dim=1)
    node_lateTW = node_lateTW.unsqueeze(-1)
    node_serviceTime =torch.cat((torch.zeros((batch_size, 1), dtype=node_serviceTime.dtype), node_serviceTime), dim=1)
    node_serviceTime = node_serviceTime.unsqueeze(-1)
    numpy_array = torch.cat((depot_node_xy, depot_node_demand,node_earlyTW,node_lateTW,node_serviceTime), dim=2)
    numpy_array = numpy_array.numpy()
    # 打开文件进行写入

    for i in range(numpy_array.shape[0]):
        with open(file_path + f'{i+1}.txt', 'w') as f:
            f.write(f'MOVRPTW INSTANCE {i + 1}' + '\n')
            f.write(f'VEHICLE NUMBER {25}    CAPACITY 1.0' + '\n')
            f.write(f'DEPOT 1' + '\n')
            f.write(f'CUSTOMER {problem_size}' + '\n')
            f.write('XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME' + '\n')
            # 将每个 (problem_size, 3) 的数组写入文件
            np.savetxt(f, numpy_array[i], fmt='%.6f', delimiter=' ')




