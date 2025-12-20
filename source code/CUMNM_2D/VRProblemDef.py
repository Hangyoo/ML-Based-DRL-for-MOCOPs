
import torch
import numpy as np
 

###-----
# if problem_type is 'unified', it is trained on
# 25% CVRP   属性为坐标、depot坐标 和 点需求
# 25% MOKP   属性为重量 和 价值
# 25% MOTSP  属性为坐标
# 25% VRPTW  属性为坐标、最早窗口、最晚窗口、服务时间
###----

def get_random_problems_mixed(batch_size, problem_size, problem_type):
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

    if ((problem_type == 'unified' and seed>=0.25 and seed <0.5) or 'TW' in problem_type): #VRPTW problem_type is 'unified' or there is 'TW' in the problem_type

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

    if ((problem_type == 'unified' and seed>=0.5 and seed <0.75) or 'TSP' in problem_type):  # MOTSP
        tsp_problems = torch.rand(size=(batch_size, problem_size, 4))
        type = 'TSP'

    if ((problem_type == 'unified' and seed>=0.75) or 'KP' in problem_type): #MOKP
        if problem_size == 50:
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

    return depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW, node_serviceTime, tsp_problems, kp_problems, type
    # return depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW, node_serviceTime, route_open, route_length_limit



def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data

def augment_xy_data_by_64_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x1 = xy_data[:, :, [0]]
    y1 = xy_data[:, :, [1]]
    x2 = xy_data[:, :, [2]]
    y2 = xy_data[:, :, [3]]

    dat1 = {}
    dat2 = {}

    dat_aug = []

    # 产生8种变体
    dat1[0] = torch.cat((x1, y1), dim=2)
    dat1[1] = torch.cat((1 - x1, y1), dim=2)
    dat1[2] = torch.cat((x1, 1 - y1), dim=2)
    dat1[3] = torch.cat((1 - x1, 1 - y1), dim=2)
    dat1[4] = torch.cat((y1, x1), dim=2)
    dat1[5] = torch.cat((1 - y1, x1), dim=2)
    dat1[6] = torch.cat((y1, 1 - x1), dim=2)
    dat1[7] = torch.cat((1 - y1, 1 - x1), dim=2)

    # 产生8种变体
    dat2[0] = torch.cat((x2, y2), dim=2)
    dat2[1] = torch.cat((1 - x2, y2), dim=2)
    dat2[2] = torch.cat((x2, 1 - y2), dim=2)
    dat2[3] = torch.cat((1 - x2, 1 - y2), dim=2)
    dat2[4] = torch.cat((y2, x2), dim=2)
    dat2[5] = torch.cat((1 - y2, x2), dim=2)
    dat2[6] = torch.cat((y2, 1 - x2), dim=2)
    dat2[7] = torch.cat((1 - y2, 1 - x2), dim=2)

    # 产生 8 * 8 = 64 种变体
    for i in range(8):
        for j in range(8):
            dat = torch.cat((dat1[i], dat2[j]), dim=2)
            dat_aug.append(dat)

    aug_problems = torch.cat(dat_aug, dim=0)

    return aug_problems

if __name__ == "__main__":
    get_random_problems_mixed(16, 50, 'unified')