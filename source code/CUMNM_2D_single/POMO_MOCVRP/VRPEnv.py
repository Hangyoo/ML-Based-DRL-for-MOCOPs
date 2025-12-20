import copy
from dataclasses import dataclass
 
import pandas as pd
import torch
import numpy as np

from Meta_Transformer_2D_single.VRProblemDef import get_random_problems_mixed, augment_xy_data_by_8_fold, augment_xy_data_by_64_fold, augment_xy_data_by_8_fold_tsp, augment_xy_data_by_64_fold_tsp


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_earlyTW: torch.Tensor = None
    # shape: (batch, problem)
    node_lateTW: torch.Tensor = None
    # shape: (batch, problem)
    # route_open: torch.Tensor = None
    # # shape: (batch, problem)
    # length: torch.Tensor = None
    # # shape: (batch, problem)

    # 多模态下自己加的
    tsp_problems: torch.Tensor = None
    # shape: (batch, problem, 4)
    kp_problems: torch.Tensor = None
    # shape: (batch, problem, 3)
    attribute_c: bool = None
    attribute_tw: bool = None
    attribute_kp: bool = None
    attribute_tsp: bool = None
    attribute_mixedtsp: bool = None
    attribute_ovrp: bool = None

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    time: torch.Tensor = None
    # shape: (batch, pomo)
    route_open: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)
    
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)

    attribute_c: bool = None
    attribute_tw: bool = None
    attribute_kp: bool = None
    attribute_tsp: bool = None
    attribute_mixedtsp: bool = None
    attribute_ovrp: bool = None

class VRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.problem_type = env_params['problem_type']
        self.benchmark_instance = env_params['benchmark_instance']
        self.instance_type = env_params['instance_type']
        self.instance_name = env_params['instance_name']

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_earlyTW = None
        # shape: (batch, problem+1)
        self.depot_node_lateTW = None
        # shape: (batch, problem+1)
        self.depot_node_servicetime = None
        # shape: (batch, problem+1)
        self.length = None
        # shape: (batch, pomo)

        self.tsp_problems = None
        # shape: (batch, problem, 4)
        self.kp_problems = None
        # shape: (batch, problem, 3)

        ##################################
        self.attribute_c = False
        self.attribute_tw = False
        self.attribute_kp = False
        self.attribute_tsp = False
        self.attribute_mixedtsp = False
        self.attribute_ovrp = False
        # self.attribute_o = False
        # self.attribute_b = False  # currently regard as CVRP with negative demand
        # self.attribute_l = False


        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.time = None
        # shape: (batch, pomo)
        self.route_open= None
        # shape: (batch, pomo)
        self.length= None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_node_earlyTW = loaded_dict['node_earlyTW']
        self.saved_node_lateTW = loaded_dict['node_lateTW']
        self.saved_node_servicetime = loaded_dict['node_serviceTime']
        self.saved_route_open = loaded_dict['route_open']
        self.saved_route_length = loaded_dict['route_length_limit']
        self.saved_index = 0

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size
        if not self.FLAG__use_saved_problems and (self.benchmark_instance == False):
            # 随机产生算例
            depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW, node_servicetime, tsp_problems, kp_problems, type = get_random_problems_mixed(batch_size, self.problem_size, self.problem_type)
        elif self.benchmark_instance:
            # 使用benchmark算例
            self.batch_size = 1
            depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW, node_servicetime, tsp_problems, kp_problems, type = get_random_problems_mixed(1, self.problem_size, self.problem_type)
            if self.problem_type == 'MOTSP':
                # 用benchmark算例替换原有tsp_problems
                tsp_data = read_benchmark_tsp(self.instance_type, self.problem_size)
                tsp_problems.copy_(tsp_data)
                print(self.batch_size, self.problem_size, self.problem_type)
            elif self.problem_type == 'MOMIXTSP':
                # 用benchmark算例替换原有tsp_problems
                # tsp_data = read_benchmark_tsp(self.instance_type, self.problem_size)
                # tsp_problems.copy_(tsp_data)
                print(self.batch_size, self.problem_size, self.problem_type)
            elif self.problem_type == 'MOCVRP':
                # 用benchmark算例替换原有depot_xy, node_xy, node_demand
                depot_xy_benchmark, node_xy_benchmark, node_demand_benchmark = read_benchmark_cvrp(self.instance_type, self.instance_name, random=True) # Random正常是false
                depot_xy.copy_(depot_xy_benchmark)
                node_xy.copy_(node_xy_benchmark)
                node_demand.copy_(node_demand_benchmark)
                print(self.batch_size, self.instance_name, self.problem_type)
            elif self.problem_type == 'MOOVRP':
                # 用benchmark算例替换原有depot_xy, node_xy, node_demand
                depot_xy_benchmark, node_xy_benchmark, node_demand_benchmark = read_benchmark_ovrp(self.instance_type, self.instance_name)
                depot_xy.copy_(depot_xy_benchmark)
                node_xy.copy_(node_xy_benchmark)
                node_demand.copy_(node_demand_benchmark)
                print(self.batch_size, self.instance_name, self.problem_type)
            elif self.problem_type == 'MOKP':
                # 用benchmark算例替换原有tsp_problems
                kp_problems_benchmark = read_benchmark_kp(self.instance_type, self.instance_name)
                kp_problems.copy_(kp_problems_benchmark)
                print(self.batch_size, self.problem_size, self.problem_type)
            elif self.problem_type == 'MOVRPTW':
                # 用benchmark算例替换原有depot_xy, node_xy, node_demand, node_serviceTime, node_earlyTW, node_lateTW
                depot_xy, node_xy, node_demand, node_serviceTime, node_earlyTW, node_lateTW = read_benchmark_vrptwp(self.instance_type, self.instance_name)
                depot_xy.copy_(depot_xy)
                node_xy.copy_(node_xy)
                node_demand.copy_(node_demand)
                node_serviceTime.copy_(node_serviceTime)
                node_earlyTW.copy_(node_earlyTW)
                node_lateTW.copy_(node_lateTW)
                print(self.batch_size, self.problem_size, self.problem_type)
            else:
                raise NotImplementedError

        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            node_earlyTW = self.saved_node_earlyTW[self.saved_index:self.saved_index+batch_size]
            node_lateTW = self.saved_node_lateTW[self.saved_index:self.saved_index+batch_size]
            node_servicetime = self.saved_node_servicetime[self.saved_index:self.saved_index+batch_size]
            route_open = self.saved_route_open[self.saved_index:self.saved_index+batch_size]
            route_length_limit = self.saved_route_length[self.saved_index:self.saved_index+batch_size]
            tsp_problems = None
            kp_problems = None
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                tsp_problems = augment_xy_data_by_8_fold_tsp(tsp_problems)
                node_demand = node_demand.repeat(8, 1)
                node_earlyTW = node_earlyTW.repeat(8, 1)
                node_lateTW = node_lateTW.repeat(8, 1)
                kp_problems = kp_problems.repeat(8, 1, 1)
                node_servicetime = node_servicetime.repeat(8, 1)
                # route_open = route_open.repeat(8,1)
                # route_length_limit = route_length_limit.repeat(8,1)
            elif aug_factor == 64:
                self.batch_size = self.batch_size * 64
                depot_xy = augment_xy_data_by_64_fold(depot_xy)
                node_xy = augment_xy_data_by_64_fold(node_xy)
                tsp_problems = augment_xy_data_by_64_fold_tsp(tsp_problems)
                node_demand = node_demand.repeat(64, 1)  # 简单重复并没有增强
                node_earlyTW = node_earlyTW.repeat(64, 1)
                node_lateTW = node_lateTW.repeat(64, 1)
                kp_problems = kp_problems.repeat(64, 1, 1)
                node_servicetime = node_servicetime.repeat(64, 1)
                # route_open = route_open.repeat(64, 1)
                # route_length_limit = route_length_limit.repeat(64, 1)
            else:
                raise NotImplementedError
        
        # self.route_open = route_open
        # # shape: (batch,pomo)
        # self.length = route_length_limit
        # # shape: (batch,pomo)

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1) # [batch, 1, 2] + [batch, problem, 2] 包含仓库和城市坐标信息
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1) # [batch,1] + [batch, problem]
        # shape: (batch, problem+1)

        depot_earlyTW = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        depot_lateTW = 4.6*torch.ones(size=(self.batch_size, 1)) # the lenght of time windows should be normalized into [0,1] not 4.6
        # shape: (batch, 1)
        depot_servicetime = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_earlyTW = torch.cat((depot_earlyTW, node_earlyTW), dim=1) # [batch, 1] + [batch, problem] 包含仓库和城市时间信息
        # shape: (batch, problem+1)
        self.depot_node_lateTW = torch.cat((depot_lateTW, node_lateTW), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_servicetime = torch.cat((depot_servicetime, node_servicetime), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_earlyTW = node_earlyTW
        self.reset_state.node_lateTW = node_lateTW

        self.tsp_problems = tsp_problems  # (batch, problem, 4)
        self.kp_problems = kp_problems  # (batch, problem, 3)
        self.reset_state.tsp_problems = tsp_problems
        self.reset_state.kp_problems = kp_problems

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

        self.attribute_c = None
        self.attribute_tw = None
        self.attribute_kp = None
        self.attribute_tsp = None
        self.attribute_mixedtsp = None
        self.attribute_ovrp = None

        # if (node_demand.sum()>0): # 判定为MOCVRP
        if (type == 'CVRP'): # 判定为MOCVRP
            self.attribute_c = True
            self.reset_state.attribute_c = True
            self.step_state.attribute_c = True
        else:
            self.attribute_c = False
        # if (node_lateTW.sum()>0): # 判定为MOVRPTW
        if (type == 'VRPTW'): # 判定为MOVRPTW
            self.attribute_tw = True
            self.reset_state.attribute_tw = True
            self.step_state.attribute_tw = True
        else:
            self.attribute_tw = False
        if (type == 'OVRP'): # 判定为MOVRPTW
            self.attribute_ovrp = True
            self.reset_state.attribute_ovrp = True
            self.step_state.attribute_ovrp = True
        else:
            self.attribute_ovrp = False
        # if (kp_problems.sum()>0): # 判定为MOKP
        if (type == 'KP'): # 判定为MOKP
            self.attribute_kp = True
            self.reset_state.attribute_kp = True
            self.step_state.attribute_kp = True
        else:
            self.attribute_kp = False
        # if (tsp_problems.sum()>0): # 判定为MOTSP
        if (type == 'TSP'): # 判定为MOTSP
            self.attribute_tsp = True
            self.reset_state.attribute_tsp = True
            self.step_state.attribute_tsp = True
        else:
            self.attribute_tsp = False
        if (type == 'MIXTSP'): # 判定为MOTSP
            self.attribute_mixedtsp = True
            self.reset_state.attribute_mixedtsp = True
            self.step_state.attribute_mixedtsp = True
        else:
            self.attribute_mixedtsp = False

        # print(self.attribute_c, self.attribute_tw, self.attribute_kp,self.attribute_tsp)
        # if (route_open.sum()>0): # 判定为OVRP
        #     self.attribute_o = True
        # else:
        #     self.attribute_o = False
        # if (route_length_limit.sum()>0): # 判定为VRPL
        #     self.attribute_l = True
        # else:
        #     self.attribute_l = False


    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.time = torch.zeros(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.length = 3.0*torch.ones(size=(self.batch_size, self.pomo_size))
        # # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.capacity = torch.Tensor(np.ones((self.batch_size, self.pomo_size)))

        if self.attribute_kp:
            # MOKP
            if self.benchmark_instance:
                self.items_and_a_dummy = torch.Tensor(np.zeros((self.batch_size, self.problem_size + 1, 3)))  # (batch, problem+1, 3)
                self.items_and_a_dummy[:, :self.problem_size,:] = self.kp_problems  # (batch, problem, 3) 前problem行为problem, 最后一行为0

            else:
                kp_problems = read_benchmark_kp(None,self.instance_name)
                self.items_and_a_dummy.copy_(kp_problems)

            self.item_data = self.items_and_a_dummy[:, :self.problem_size,:]  # (batch, problem, 3) item_data 就是 problem_size
            self.accumulated_value_obj1 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))  # (batch, problem)
            self.accumulated_value_obj2 = torch.Tensor(np.zeros((self.batch_size, self.pomo_size)))  # (batch, problem)
            self.ninf_mask_w_dummy = torch.zeros(self.batch_size, self.pomo_size,self.problem_size + 1)  # (batch, pomo, problem+1)
            self.ninf_mask = self.ninf_mask_w_dummy[:,:,:self.problem_size]
            self.fit_ninf_mask = None
        elif self.attribute_tsp or self.attribute_mixedtsp:
            self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size))
        else:
            self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.time = self.time
        self.step_state.route_open = self.route_open
        self.step_state.length = self.length.clone()
        self.step_state.capacity = self.capacity # MOKP

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # MOTSP
        if self.attribute_tsp:
            self.step_state.attribute_tsp = True
            self.step_state.current_node = self.current_node
            # shape: (batch, pomo)
            self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')  # 将当前选择的点mask
            # shape: (batch, pomo, problem)
            self.step_state.selected_count = self.selected_count

            # returning values
            done = (self.selected_count == self.problem_size)  # 当选择的点和问题规模一致的时候，判断是否done

        elif self.attribute_mixedtsp:
            self.step_state.attribute_mixedtsp = True
            self.step_state.current_node = self.current_node
            # shape: (batch, pomo)
            self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')  # 将当前选择的点mask
            # shape: (batch, pomo, problem)
            self.step_state.selected_count = self.selected_count

            # returning values
            done = (self.selected_count == self.problem_size)  # 当选择的点和问题规模一致的时候，判断是否done
        # MOKP
        elif self.attribute_kp:
            self.step_state.attribute_kp = True
            items_mat = self.items_and_a_dummy[:, None, :, :].expand(self.batch_size, self.pomo_size,self.problem_size + 1, 3)
            gathering_index = selected[:, :, None, None].expand(self.batch_size, self.pomo_size, 1, 3)
            selected_item = items_mat.gather(dim=2, index=gathering_index).squeeze(dim=2)  # (64,50,3)

            self.accumulated_value_obj1 += selected_item[:, :, 1]
            self.accumulated_value_obj2 += selected_item[:, :, 2]
            self.capacity -= selected_item[:, :, 0]

            batch_idx_mat = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
            group_idx_mat = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
            self.ninf_mask_w_dummy[batch_idx_mat, group_idx_mat, selected] = -np.inf

            unfit_bool = (self.capacity[:, :, None] - self.item_data[:, None, :, 0]) < 0  # 判断是否过载
            self.fit_ninf_mask = self.ninf_mask.clone()
            self.fit_ninf_mask[unfit_bool] = -np.inf  # mask

            self.finished = (self.fit_ninf_mask == -np.inf).all(dim=2)

            self.fit_ninf_mask[self.finished[:, :, None].expand(self.batch_size, self.pomo_size, self.problem_size)] = 0

            done = self.step_state.finished.all()

            self.step_state.current_node = self.current_node
            self.step_state.ninf_mask = self.fit_ninf_mask
            self.step_state.capacity = self.capacity
            self.step_state.finished = self.finished
            self.step_state.selected_count = self.selected_count

        # MOCVRP, MOOVRP 和 MOVRPTW (需求量大于剩余容量，以及到达城市时迟于时间窗的点都被mask)
        else:
            # Dynamic-2
            ####################################
            if self.attribute_c:
                self.step_state.attribute_c = True

            self.at_the_depot = (selected == 0) # [32, 50] 全True or 全False

            #### update load information ###

            demand_list = self.depot_node_demand[:, None, :].expand(-1, self.pomo_size, -1) # [batch, problem+1]
            # shape: (batch, pomo, problem+1)
            gathering_index = selected[:, :, None]
            # shape: (batch, pomo, 1)
            selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2) # [32, 50, 1] --> [32, 50]
            # shape: (batch, pomo)

            self.load -= selected_demand # 更新剩余容量
            self.load[self.at_the_depot] = 1 # refill loaded at the depot (回到depot则容量为1)


            #### mask nodes if load exceed ###

            self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
            # shape: (batch, pomo, problem+1) 算了depot
            self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

            self.ninf_mask = self.visited_ninf_flag.clone() # (batch, pomo, problem+1)
            round_error_epsilon = 0.000001
            demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list # 剩余资源不够demand
            # shape: (batch, pomo, problem+1)
            self.ninf_mask[demand_too_large] = float('-inf') # mark掉这些需求无法覆盖的点
            # shape: (batch, pomo, problem+1)

            #### update time&distance information ###

            servicetime_list = self.depot_node_servicetime[:, None, :].expand(-1, self.pomo_size, -1)
            # shape: (batch, pomo, problem+1)
            selected_servicetime = servicetime_list.gather(dim=2,index=gathering_index).squeeze(dim=2) # 服务时间
            # shape: (batch, pomo)

            earlyTW_list = self.depot_node_earlyTW[:, None, :].expand(-1, self.pomo_size, -1)
            # shape: (batch, pomo, problem+1)
            selected_earlyTW = earlyTW_list.gather(dim=2,index=gathering_index).squeeze(dim=2) # 最早时间窗
            # shape: (batch, pomo)

            xy_list = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1,-1)
            # shape: (batch, pomo, problem+1, 2)
            gathering_index = selected[:, :, None,None].expand(-1,-1,-1,2)
            # shape: (batch, pomo, 1, 2)
            selected_xy = xy_list.gather(dim=2, index=gathering_index).squeeze(dim=2) # depot点
            # shape: (batch, pomo, 2)

            if self.selected_node_list.size()[2] == 1: # 第一步，只选择了depot点
                gathering_index_last = self.selected_node_list[:, :, -1][:,:,None,None].expand(-1,-1,-1,2)
                # shape: (batch, pomo, 1,2)
            else: # 最近访问的点的上一个点
                gathering_index_last = self.selected_node_list[:, :, -2][:,:,None,None].expand(-1,-1,-1,2)
                # shape: (batch, pomo, 1,2)
            last_xy = xy_list.gather(dim=2, index=gathering_index_last).squeeze(dim=2)
            # shape: (batch, pomo, 2)
            selected_time = ((selected_xy - last_xy)**2).sum(dim=2).sqrt()
            # shape: (batch, pomo)

            # update time window attribute if it is used
            if (self.attribute_tw):
                self.step_state.attribute_tw = True
                #print(selected_time)
                #selected_time += selected_servicetime
                self.time = torch.max((self.time + selected_time), selected_earlyTW)
                self.time += selected_servicetime
                # shape: (batch, pomo)
                self.time[self.at_the_depot] = 0 # refill time at the depot

                time_to_next = ((selected_xy[:,:,None,:].expand(-1,-1,self.problem_size+1,-1) - xy_list)**2).sum(dim=3).sqrt()
                # shape: (batch, pomo, problem+1)
                # time_to_depot = ((xy_list[:,:,0,:].expand(-1,-1,self.problem_size+1,-1)  - xy_list)**2).sum(dim=3).sqrt()
                # shape: (batch, pomo, problem+1)
                time_too_late = self.time[:, :, None] + time_to_next > self.depot_node_lateTW[:, None, :].expand(-1, self.pomo_size, -1)
                # shape: (batch, pomo, problem+1)
                time_too_late[self.depot_node_lateTW[:, None, :].expand(-1, self.pomo_size, -1) == 0]= 0
                # unmask the the zero late TW

                self.ninf_mask[time_too_late] = float('-inf')
                # shape: (batch, pomo, problem+1)

            newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
            # shape: (batch, pomo)
            self.finished = self.finished + newly_finished
            # shape: (batch, pomo)

            # do not mask depot for finished episode. 不mask depot点
            self.ninf_mask[:, :, 0][self.finished] = 0

            self.step_state.selected_count = self.selected_count
            self.step_state.load = self.load
            self.step_state.current_node = self.current_node
            self.step_state.ninf_mask = self.ninf_mask
            self.step_state.finished = self.finished

            # returning values
            done = self.finished.all() # 所有元素都为不为0,返回True; 否则返回False

        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        if self.attribute_tsp:
            gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 4)
            # shape: (batch, pomo, problem, 4)
            if self.benchmark_instance:
                # 加载真实值
                # 用benchmark算例替换原有tsp_problems
                tsp_problems = copy.deepcopy(self.tsp_problems)
                tsp_data = read_benchmark_tsp(self.instance_type, self.problem_size, normalize = False)
                tsp_problems.copy_(tsp_data)
                seq_expanded = tsp_problems[:, None, :, :].expand(self.batch_size, self.pomo_size,self.problem_size, 4)
            else:
                seq_expanded = self.tsp_problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 4)

            ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
            # shape: (batch, pomo, problem, 2)
            rolled_seq = ordered_seq.roll(dims=2, shifts=-1)

            segment_lengths_obj1 = ((ordered_seq[:, :, :, :2] - rolled_seq[:, :, :, :2]) ** 2).sum(3).sqrt()
            segment_lengths_obj2 = ((ordered_seq[:, :, :, 2:] - rolled_seq[:, :, :, 2:]) ** 2).sum(3).sqrt()

            travel_distances_obj1 = segment_lengths_obj1.sum(2)  # (64,20)
            travel_distances_obj2 = segment_lengths_obj2.sum(2)  # (64,20)

            objs = torch.stack([travel_distances_obj1, travel_distances_obj2], axis=2)  # (64,20,2)

        if self.attribute_mixedtsp:
            gathering_index = self.selected_node_list.unsqueeze(3).expand(self.batch_size, -1, self.problem_size, 4)
            # shape: (batch, pomo, problem, 3)
            seq_expanded = self.tsp_problems[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size, 4)

            ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
            # shape: (batch, pomo, problem, 3)
            rolled_seq = ordered_seq.roll(dims=2, shifts=-1)

            segment_lengths_obj1 = ((ordered_seq[:, :, :, :2] - rolled_seq[:, :, :, :2]) ** 2).sum(3).sqrt()
            segment_lengths_obj2 = (torch.abs(ordered_seq[:, :, :, 2] - rolled_seq[:, :, :, 2]))

            travel_distances_obj1 = segment_lengths_obj1.sum(2)  # (64,20)
            travel_distances_obj2 = segment_lengths_obj2.sum(2)  # (64,20)

            objs = torch.stack([travel_distances_obj1, travel_distances_obj2], axis=2)  # (64,20,2)

        if self.attribute_c:
            gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
            # shape: (batch, pomo, selected_list_length, 2)
            all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
            # shape: (batch, pomo, problem+1, 2)

            if self.benchmark_instance:
                # 加载真实值
                # 用benchmark算例替换原有tsp_problems
                depot_node_xy = copy.deepcopy(self.depot_node_xy)
                depot_xy, node_xy, node_demand = read_benchmark_cvrp(self.instance_type, self.instance_name, normalize = False)
                temp = torch.cat((depot_xy, node_xy), dim=1)
                depot_node_xy.copy_(temp)
                all_xy = depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
            else:
                all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
                # shape: (batch, pomo, problem+1, 2)

            # obj1: travel_distances
            ordered_seq = all_xy.gather(dim=2, index=gathering_index)
            # shape: (batch, pomo, selected_list_length, 2)

            rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
            segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
            # shape: (batch, pomo, selected_list_length)

            # print(segment_lengths[0][0])
            travel_distances = segment_lengths.sum(2)
            # shape: (batch, pomo)

            # obj2: makespans
            not_idx = (gathering_index[:, :, :, 0] > 0)
            cum_lengths = torch.cumsum(segment_lengths, dim=2)

            cum_lengths[not_idx] = 0
            sorted_cum_lengths, _ = cum_lengths.sort(axis=2)

            rolled_sorted_cum_lengths = sorted_cum_lengths.roll(dims=2, shifts=1)
            diff_mat = sorted_cum_lengths - rolled_sorted_cum_lengths
            diff_mat[diff_mat < 0] = 0

            makespans, _ = torch.max(diff_mat, dim=2)

            objs = torch.stack([travel_distances, makespans], axis=2)


        if self.attribute_kp:

            objs = torch.stack([-self.accumulated_value_obj1,-self.accumulated_value_obj2], axis=2)

        if self.attribute_tw:
            gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
            # shape: (batch, pomo, selected_list_length, 2)

            if self.benchmark_instance:
                # 加载真实值
                # 用benchmark算例替换原有tsp_problems
                depot_node_xy = copy.deepcopy(self.depot_node_xy)
                depot_xy, node_xy, node_demand = read_benchmark_vrptwp(self.instance_type, self.instance_name, normalize = False)
                temp = torch.cat((depot_xy, node_xy), dim=1)
                depot_node_xy.copy_(temp)
                all_xy = depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
            else:
                all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
                # shape: (batch, pomo, problem+1, 2)

            # obj1: travel_distances
            ordered_seq = all_xy.gather(dim=2, index=gathering_index)
            # shape: (batch, pomo, selected_list_length, 2)

            rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
            segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
            # shape: (batch, pomo, selected_list_length)

            # print(segment_lengths[0][0])
            travel_distances = segment_lengths.sum(2)
            # shape: (batch, pomo)

            # obj2: makespans
            not_idx = (gathering_index[:, :, :, 0] > 0)
            cum_lengths = torch.cumsum(segment_lengths, dim=2)

            cum_lengths[not_idx] = 0
            sorted_cum_lengths, _ = cum_lengths.sort(axis=2)

            rolled_sorted_cum_lengths = sorted_cum_lengths.roll(dims=2, shifts=1)
            diff_mat = sorted_cum_lengths - rolled_sorted_cum_lengths
            diff_mat[diff_mat < 0] = 0

            makespans, _ = torch.max(diff_mat, dim=2)

            objs = torch.stack([travel_distances, makespans], axis=2)

        if self.attribute_ovrp:
            gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
            # shape: (batch, pomo, selected_list_length, 2)

            if self.benchmark_instance:
                # 加载真实值
                # 用benchmark算例替换原有tsp_problems
                depot_node_xy = copy.deepcopy(self.depot_node_xy)
                depot_xy, node_xy, node_demand = read_benchmark_ovrp(self.instance_type, self.instance_name, normalize = False)
                temp = torch.cat((depot_xy, node_xy), dim=1)
                depot_node_xy.copy_(temp)
                all_xy = depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
            else:
                all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
                # shape: (batch, pomo, problem+1, 2)

            # obj1: travel_distances
            ordered_seq = all_xy.gather(dim=2, index=gathering_index)
            # shape: (batch, pomo, selected_list_length, 2)

            rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
            segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
            # shape: (batch, pomo, selected_list_length)

            segment_lengths[self.selected_node_list.roll(dims=2, shifts=-1) == 0] = 0

            # print(segment_lengths[0][0])
            travel_distances = segment_lengths.sum(2)
            # shape: (batch, pomo)

            # obj2: makespans
            not_idx = (gathering_index[:, :, :, 0] > 0)
            cum_lengths = torch.cumsum(segment_lengths, dim=2)

            cum_lengths[not_idx] = 0
            sorted_cum_lengths, _ = cum_lengths.sort(axis=2)

            rolled_sorted_cum_lengths = sorted_cum_lengths.roll(dims=2, shifts=1)
            diff_mat = sorted_cum_lengths - rolled_sorted_cum_lengths
            diff_mat[diff_mat < 0] = 0

            makespans, _ = torch.max(diff_mat, dim=2)

            objs = torch.stack([travel_distances, makespans], axis=2)


        return objs

    def get_node_seq(self):
        '没有用到'
        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        return gathering_index,ordered_seq

def read_benchmark_tsp(NAME, num_nodes, normalize=True,random=False):
    '''
    NAME = 'Random'
    NAME = 'KroAB'
    '''
    if not random:
        path = r'D:\Paper Data\paper 3\instance\TSP_Bechmark'
        if NAME[:-2] == "Kro":
            x1 = np.loadtxt(path + f'/kro{NAME[-2]}{num_nodes}.tsp', skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
            x2 = np.loadtxt(path + f'/kro{NAME[-1]}{num_nodes}.tsp', skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
        elif NAME[:-2] == 'Cluster':
            x1 = np.loadtxt(path + f'/Cluster{NAME[-2]}{num_nodes}.tsp', skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
            x2 = np.loadtxt(path + f'/Cluster{NAME[-1]}{num_nodes}.tsp', skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
        elif NAME[:-2] == 'Euclid':
            x1 = np.loadtxt(path + f'/euclid{NAME[-2]}{num_nodes}.tsp', skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
            x2 = np.loadtxt(path + f'/euclid{NAME[-1]}{num_nodes}.tsp', skiprows=6, usecols=(1, 2), delimiter=' ', dtype=float)
        elif NAME[:-2] == 'rd':
            x1 = np.loadtxt(path + f'/random{NAME[-2]}{num_nodes}.tsp', skiprows=6, usecols=(1, 2), delimiter=' ',dtype=float)
            x2 = np.loadtxt(path + f'/random{NAME[-1]}{num_nodes}.tsp', skiprows=6, usecols=(1, 2), delimiter=' ',dtype=float)
        if normalize:
            x1 = x1 / (np.max(x1, 0))  # 数据归一化
            x2 = x2 / (np.max(x2, 0))  # 数据归一化
    else:
        path = r'C:\Users\Hangyu\Desktop\GitHub\Test Data\Random Test Set\MOTSP/'
        data = np.loadtxt(path + NAME, skiprows=3)
        x1 = data[:,:2]
        x2 = data[:,2:]

    # elif NAME == 'Random':
    #     path = r'D:\Paper Data\paper 3\instance\TSP_Random'
    #     x1 = np.loadtxt(path + f'/Tsp{num_nodes}_{idx}_1.txt', delimiter=' ', dtype=float)  # usecols=(1, 2) 指定读取1,2列
    #     x2 = np.loadtxt(path + f'/Tsp{num_nodes}_{idx}_2.txt', delimiter=' ', dtype=float)  # usecols=(1, 2) 指定读取1,2列
    tsp_data = np.concatenate((x1, x2), axis=1)  # (100,2) --> (100,4) 改变了第1维度, 即axis=1
    tsp_data = tsp_data.reshape(1, num_nodes, 4)
    tsp_data = torch.tensor(tsp_data, device='cuda')

    return tsp_data

def read_benchmark_cvrp(NAME, instance_name, normalize=True, random=False):
    '''
    NAME = 'A', 'P', 'B', 'E', 'F', 'M', 'X'
    '''

    if not random:
        path = r'D:\Paper Data\paper 3\instance\CVRP_Benchmark'
        if normalize:
            data = np.array(pd.read_csv(path + f'/{NAME}/{instance_name}',header=None))
        else:
            a, b = instance_name.split('.')
            new_instance_name = a + '_real.' + b
            data = np.array(pd.read_csv(path + f'/{NAME}/{new_instance_name}', header=None))
    else:
        def read_cvrp_data(file_path):
            with open(file_path, 'r') as f:
                # 跳过前四行，读取数据并转换为 np.array
                return np.array([list(map(float, line.split())) for line in f.readlines()[4:] if
                                 line.strip() and len(line.split()) == 3])

        path = r'C:\Users\Hangyu\Desktop\GitHub\Test Data\Random Test Set\MOCVRP/'
        data = read_cvrp_data(path + instance_name)

    depot_xy = data[0, :2].reshape(1, 1, 2)
    node_xy = data[1:, :2].reshape(1, -1, 2)
    node_demand = data[1:, 2].reshape(1, -1)

    depot_xy = torch.tensor(depot_xy, device='cuda')
    node_xy = torch.tensor(node_xy, device='cuda')
    node_demand = torch.tensor(node_demand, device='cuda')

    return depot_xy, node_xy, node_demand

def read_benchmark_ovrp(NAME, instance_name, normalize=True):
    '''
    NAME = 'C', 'F'
    '''

    path = r'D:\Paper Data\paper 3\instance\OVRP_Benchmark'
    NAME = 'C1-C8' if (NAME=='C' or NAME=='c') else 'F10-F12'
    if normalize:
        data = np.array(pd.read_csv(path + f'/{NAME}/{instance_name}',header=None))
    else:
        a, b = instance_name.split('.')
        new_instance_name = a + '_real.' + b
        data = np.array(pd.read_csv(path + f'/{NAME}/{new_instance_name}', header=None))

    depot_xy = data[0, :2].reshape(1, 1, 2)
    node_xy = data[1:, :2].reshape(1, -1, 2)
    node_demand = data[1:, 2].reshape(1, -1)

    depot_xy = torch.tensor(depot_xy, device='cuda')
    node_xy = torch.tensor(node_xy, device='cuda')
    node_demand = torch.tensor(node_demand, device='cuda')

    return depot_xy, node_xy, node_demand

def read_benchmark_kp(NAME, instance_name):

    path = r'D:\Paper Data\paper 3\instance\KP_benchmark'

    data = np.loadtxt(path + f'/{instance_name}')
    # data = np.loadtxt(r"D:\Paper Data\paper 3\instance\KP_Random\kp100_2obj.txt")
    # data[:,0] /= 25
    kp_problems = data.reshape(1, -1, 3)

    kp_problems = torch.tensor(kp_problems, device='cuda')

    return kp_problems

def read_benchmark_vrptwp(NAME, instance_name, normalize=True):
    path = r'D:\Paper Data\paper 3\instance\VRPTW_Benchmark'
    import pickle

    if normalize:
        # 需要对depot_xy, node_xy, node_demand进行归一化, 其他都不用动
        with open(path + f"/{instance_name}.pkl", 'rb') as f:
            [depot_xy, node_xy, node_demand, node_serviceTime, node_earlyTW, node_lateTW] = pickle.load(f)
    else:
        # 需要对node_demand进行归一化, depot_xy, node_xy都不用动
        with open(path + f"/{instance_name}_real.pkl", 'rb') as f:
            [depot_xy, node_xy, node_demand, node_serviceTime, node_earlyTW, node_lateTW] = pickle.load(f)

    depot_xy = depot_xy.reshape(1, 1, 2)
    node_xy = node_xy.reshape(1, -1, 2)
    node_demand = node_demand.reshape(1, -1)
    node_serviceTime = node_serviceTime.reshape(1, -1)
    node_earlyTW = node_earlyTW.reshape(1, -1)
    node_lateTW = node_lateTW.reshape(1, -1)


    depot_xy = torch.tensor(depot_xy, device='cuda', dtype=torch.float32)
    node_xy = torch.tensor(node_xy, device='cuda', dtype=torch.float32)
    node_demand = torch.tensor(node_demand, device='cuda', dtype=torch.float32)
    node_serviceTime = torch.tensor(node_serviceTime, device='cuda', dtype=torch.float32)
    node_earlyTW = torch.tensor(node_earlyTW, device='cuda', dtype=torch.float32)
    node_lateTW = torch.tensor(node_lateTW, device='cuda', dtype=torch.float32)

    return depot_xy, node_xy, node_demand, node_serviceTime, node_earlyTW, node_lateTW