import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        # encoder 共用
        self.encoder = VRP_Encoder(**model_params)
        # MOCVRP 的解码器
        self.decoder_CVRP = VRP_Decoder(**model_params)
        # MOVRPTW 的解码器
        self.decoder_VRPTW = VRP_Decoder(**model_params)
        # MOKP 的解码器
        self.decoder_KP = KP_Decoder(**model_params)
        # MOTSP 的解码器
        self.decoder_TSP = VRP_Decoder(**model_params)

        self.encoded_nodes = None # 对MOTSP, MOVRPTW, MOCVRP
        # shape: (batch, problem+1, EMBEDDING_DIM)
        self.encoded_graph = None # 对KP问题

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_earlyTW = reset_state.node_earlyTW
        # shape: (batch, problem)
        node_lateTW = reset_state.node_lateTW
        # shape: (batch, problem)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)  # 城市坐标 + 需求 [batch,problem,2] + [batch,problem,1]
        # shape: (batch, problem, 3)
        node_TW = torch.cat((node_earlyTW[:, :, None],node_lateTW[:, :, None]),dim=2) # 时间窗 [batch, problem, 1] + [batch, problem, 1]
        # shape: (batch, problem, 2)
        node_xy_demand_TW = torch.cat((node_xy_demand,node_TW),dim=2)
        # shape: (batch, problem, 3+2=5)

        tsp_problems = reset_state.tsp_problems
        # shape: (batch, problem, 4)
        kp_problems = reset_state.kp_problems
        # shape: (batch, problem, 3)

        # tsp的编码方案
        if reset_state.attribute_tsp:
            self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_TW, tsp_problems,kp_problems,reset_state)
            self.decoder_TSP.set_kv(self.encoded_nodes)
        # kp的编码方案
        elif reset_state.attribute_kp:
            batch_size = reset_state.kp_problems.size(0)
            problem_size = reset_state.kp_problems.size(1)
            self.encoded_nodes_and_dummy = torch.Tensor(np.zeros((batch_size, problem_size + 1, self.model_params['embedding_dim'])))
            self.encoded_nodes_and_dummy[:, :problem_size, :] = self.encoder(depot_xy, node_xy_demand_TW, tsp_problems,kp_problems,reset_state)
            self.encoded_nodes = self.encoded_nodes_and_dummy[:, :problem_size, :]
            self.encoded_graph = self.encoded_nodes.mean(dim=1, keepdim=True)
            self.decoder_KP.set_kv(self.encoded_nodes)
        # CVRP
        elif reset_state.attribute_c:
            self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_TW, tsp_problems,kp_problems,reset_state)  # self.encoded_nodes (batch, 151, embedding)
            self.decoder_CVRP.set_kv(self.encoded_nodes)
        # VRPTW
        elif reset_state.attribute_tw:
            self.encoded_nodes = self.encoder(depot_xy, node_xy_demand_TW, tsp_problems,kp_problems,reset_state)
            self.decoder_VRPTW.set_kv(self.encoded_nodes)
        else:
            raise NotImplementedError


    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.attribute_kp:
            # shape: (batch, pomo, embedding)
            # probs = self.decoder(self.encoded_nodes, state.load, state.capacity, state.time, ninf_mask=state.ninf_mask)
            probs = self.decoder_KP(self.encoded_graph, state.capacity, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)

            if self.training or self.model_params['eval_type'] == 'softmax':
                # while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                #     with torch.no_grad():
                #         selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                #             .squeeze(dim=1).reshape(batch_size, pomo_size) # 根据概率选择物品序号
                #     # shape: (batch, pomo)
                #     prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                #     # shape: (batch, pomo)
                #     if (prob != 0).all():
                #         break

                with torch.no_grad():
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)  # 根据概率选择物品序号
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

        elif state.attribute_tsp:
            if state.selected_count == 0:
                selected = torch.arange(pomo_size)[None, :].expand(batch_size,pomo_size)  # (64,20) 从1-POMO,每个点都做为一个初始化点
                prob = torch.ones(size=(batch_size, pomo_size))  # 全1向量 (64,20)

                # # 对选择的第一个点进行编码
                # encoded_first_node = _get_encoding(self.encoded_nodes, selected)
                # # shape: (batch, pomo, embedding)
                # self.decoder.set_q1(encoded_first_node)  # 计算会得到 self.q_first
            else:
                encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
                # shape: (batch, pomo, embedding)
                # tensor = torch.full((state.ninf_mask.size()[0], state.ninf_mask.size()[1], 101), float('-inf'))
                # new_ninf_mask = torch.cat([state.ninf_mask, tensor], dim=2)
                probs = self.decoder_TSP(encoded_last_node, ninf_mask=state.ninf_mask)  # decoder 返回每个点的概率值，已经选过的prob值为0
                # shape: (batch:64, pomo:20, problem:20); ninf_mask.shape: (batch, pomo, problem)
                if self.training or self.model_params['eval_type'] == 'softmax':
                    while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                        with torch.no_grad():
                            # print(probs.reshape(batch_size * pomo_size, -1))
                            # print("current time = ",state.time," mask= ",state.ninf_mask)
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                                .squeeze(dim=1).reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)
                        prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)
                        if (prob != 0).all():
                            break

                else:
                    selected = probs.argmax(dim=2)
                    # shape: (batch, pomo)
                    prob = None  # value not needed. Can be anything.

        elif state.attribute_c:
            if state.selected_count == 0:  # First Move, depot
                selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long) #  (首先从depot开始)
                prob = torch.ones(size=(batch_size, pomo_size))

                # # Use Averaged encoded nodes for decoder input_1
                # encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True) # 使用contex均值
                # # shape: (batch, 1, embedding)
                # self.decoder.set_q1(encoded_nodes_mean)

                # # Use encoded_depot for decoder input_2
                # encoded_first_node = self.encoded_nodes[:, [0], :] # 使用node1
                # # shape: (batch, 1, embedding)
                # self.decoder.set_q2(encoded_first_node)

            elif state.selected_count == 1:  # Second Move, POMO
                selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)  # [batch_size, pomo_size]
                prob = torch.ones(size=(batch_size, pomo_size))

            else:
                encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node) # 获取选择点的embedding self.encoded_nodes (batch, 151, embedding); state.current_node (batch, problem)
                # shape: (batch, pomo, embedding)

                # 论文中的(xt, ct, tt, lt 和 ot)
                # probs = self.decoder(encoded_last_node, state.load, state.time,state.length,state.route_open, ninf_mask=state.ninf_mask) # 原始程序
                probs = self.decoder_CVRP(encoded_last_node, ninf_mask=state.ninf_mask) # 原始程序 state.ninf_mask
                # shape: (batch, pomo, problem+1)
                #print(probs.shape)

                if self.training or self.model_params['eval_type'] == 'softmax':
                    while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                        with torch.no_grad():
                            # print(probs.reshape(batch_size * pomo_size, -1))
                            # print("current time = ",state.time," mask= ",state.ninf_mask)
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                                .squeeze(dim=1).reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)
                        prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)
                        if (prob != 0).all():
                            break

                else:
                    selected = probs.argmax(dim=2)
                    # shape: (batch, pomo)
                    prob = None  # value not needed. Can be anything.

        elif state.attribute_tw:
            if state.selected_count == 0:  # First Move, depot
                selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long) #  (首先从depot开始)
                prob = torch.ones(size=(batch_size, pomo_size))

                # # Use Averaged encoded nodes for decoder input_1
                # encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True) # 使用contex均值
                # # shape: (batch, 1, embedding)
                # self.decoder.set_q1(encoded_nodes_mean)

                # # Use encoded_depot for decoder input_2
                # encoded_first_node = self.encoded_nodes[:, [0], :] # 使用node1
                # # shape: (batch, 1, embedding)
                # self.decoder.set_q2(encoded_first_node)

            elif state.selected_count == 1:  # Second Move, POMO
                selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)  # [batch_size, pomo_size]
                prob = torch.ones(size=(batch_size, pomo_size))

            else:
                encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node) # 获取选择点的embedding self.encoded_nodes (batch, 151, embedding); state.current_node (batch, problem)
                # shape: (batch, pomo, embedding)

                # 论文中的(xt, ct, tt, lt 和 ot)
                # probs = self.decoder(encoded_last_node, state.load, state.time,state.length,state.route_open, ninf_mask=state.ninf_mask) # 原始程序
                probs = self.decoder_VRPTW(encoded_last_node, ninf_mask=state.ninf_mask) # 原始程序 state.ninf_mask
                # shape: (batch, pomo, problem+1)
                #print(probs.shape)

                if self.training or self.model_params['eval_type'] == 'softmax':
                    while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                        with torch.no_grad():
                            # print(probs.reshape(batch_size * pomo_size, -1))
                            # print("current time = ",state.time," mask= ",state.ninf_mask)
                            selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                                .squeeze(dim=1).reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)
                        prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                        # shape: (batch, pomo)
                        if (prob != 0).all():
                            break

                else:
                    selected = probs.argmax(dim=2)
                    # shape: (batch, pomo)
                    prob = None  # value not needed. Can be anything.

        else:
            raise NotImplementedError
        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0) # 32
    pomo_size = node_index_to_pick.size(1)  # 50
    embedding_dim = encoded_nodes.size(2)   # 128

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class VRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        # self.embedding_depot = nn.Linear(2, embedding_dim) # depote坐标
        # self.embedding_node = nn.Linear(5, embedding_dim)  # 点坐标、需求、时间窗

        self.embedding_vrptw_depot = nn.Linear(2, embedding_dim) # depote坐标
        self.embedding_vrptw_node = nn.Linear(5, embedding_dim)  # 点坐标、需求、时间窗

        self.embedding_cvrp_depot = nn.Linear(2, embedding_dim) # depote坐标
        self.embedding_cvrp_node = nn.Linear(3, embedding_dim)  # 点坐标、需求

        self.embedding_kp = nn.Linear(3, embedding_dim)   # MOkP embedding 层 3 --> 128
        self.embedding_tsp = nn.Linear(4, embedding_dim)  # MOTSP embedding 层 4 --> 128
        
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand_TW, tsp_problems, kp_problems, reset_state):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand_TW.shape: (batch, problem, 5)
        # tsp_problems.shape: (batch, problem, 4)
        # kp_problems.shape: (batch, problem, 3)

        if reset_state.attribute_tsp:
            out = self.embedding_tsp(tsp_problems)
            # out.shape: (batch, problem, embedding)
        elif reset_state.attribute_kp:
            out = self.embedding_kp(kp_problems)
            # out.shape: (batch, problem, embedding)
        elif reset_state.attribute_c:
            embedded_cvrp_depot = self.embedding_cvrp_depot(depot_xy)
            # embedded_depot.shape: (batch, 1, embedding)
            embedded_cvrp_node = self.embedding_cvrp_node(node_xy_demand_TW[:,:,:3]) # 只要前3列数据
            # embedded_node.shape: (batch, problem, embedding)

            # 6 features are: x_coord, y_coord, demands, earlyTW, lateTW
            # embedded_node shape: (batch, problem, embedding)
            out = torch.cat((embedded_cvrp_depot, embedded_cvrp_node), dim=1)
            # shape: (batch, problem+1, embedding)

        elif reset_state.attribute_tw:
            embedded_vrptw_depot = self.embedding_vrptw_depot(depot_xy)
            # embedded_depot.shape: (batch, 1, embedding)
            embedded_vrptw_node = self.embedding_vrptw_node(node_xy_demand_TW)
            # embedded_node.shape: (batch, problem, embedding)

            # 6 features are: x_coord, y_coord, demands, earlyTW, lateTW
            # embedded_node shape: (batch, problem, embedding)
            out = torch.cat((embedded_vrptw_depot, embedded_vrptw_node), dim=1)
            # shape: (batch, problem+1, embedding)

        else:
            raise NotImplementedError

        for layer in self.layers: # 6层
            out = layer(out)
        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):  # 6个layer中每一个的网络结构
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)
        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class VRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim'] # embedding_dim: 128
        head_num = self.model_params['head_num'] # head_num: 8
        qkv_dim = self.model_params['qkv_dim'] # qkv_dim: 16

        hyper_input_dim = 2 # 权重向量的维度 (如0.4,0.6)
        hyper_hidden_embd_dim = 256 # 论文中写的是128
        self.embd_dim = 2 # 权重向量
        self.hyper_output_dim = 4 * self.embd_dim # 根据权重向量扩展出的网络权重
        # 处理权重向量使用的网络 2 - 256 - 256 - 10
        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)  # 2 --> 256
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True)  # 256 --> 256
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True)  # 256 --> 10

        # decoder使用的网络(都是带权重的 下面5个 每个分配上面神经网络输出的2个)
        self.hyper_Wq_last = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)  # 2 --> 16384  # 原始程序 +4
        self.hyper_Wk = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)  # 2 --> 16384
        self.hyper_Wv = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)  # 2 --> 16384
        self.hyper_multi_head_combine = nn.Linear(self.embd_dim, head_num * qkv_dim * embedding_dim,bias=False)  # 2 --> 16384

        self.Wq_last_para = None
        self.multi_head_combine_para = None

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def assign(self, pref): # pref: tensor([0.772,0.228])
        '''将权重向量嵌入神经网络当中'''
        embedding_dim = self.model_params['embedding_dim'] # 128
        head_num = self.model_params['head_num'] # 8
        qkv_dim = self.model_params['qkv_dim'] # 16

        # 通过神经网络将2维度映射到10维度 2 - 256 - 256 - 10
        hyper_embd = self.hyper_fc1(pref)
        hyper_embd = self.hyper_fc2(hyper_embd)
        mid_embd = self.hyper_fc3(hyper_embd)
        # [0.1249, -0.0571, 0.0289, 0.2872, -0.2496, 0.4080, -0.2827, -0.4525, 0.2236, 0.1886]

        # hyper_Wq_first取1,2; Wq_last_para取3,4; Wk_para取5,6; Wv_para取7,8; multi_head_combine_para取9,10
        # 将weight权重 变成相应尺寸的权重（权重嵌入到参数中了）
        self.Wq_last_para = self.hyper_Wq_last(mid_embd[:1 * self.embd_dim]).reshape(head_num * qkv_dim, embedding_dim) # 2 --> 128*128
        self.Wk_para = self.hyper_Wk(mid_embd[1 * self.embd_dim: 2 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim) # 2 --> 128*128
        self.Wv_para = self.hyper_Wv(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim) # 2 --> 128*128
        self.multi_head_combine_para = self.hyper_multi_head_combine(mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(head_num * qkv_dim, embedding_dim) # 128*128

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch:64, problem:20, embedding:128)
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(F.linear(encoded_nodes, self.Wk_para), head_num=head_num)
        self.v = reshape_by_heads(F.linear(encoded_nodes, self.Wv_para), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(F.linear(encoded_q1, self.Wq_first_para), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(F.linear(encoded_q2, self.Wq_first_para), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, ninf_mask): # encoded_last_node, load, time,length, route_open, ninf_mask
        '''
            :param encoded_last_node: (batch:64, pomo:20, embedding:128)
            :param ninf_mask: (batch, pomo, problem) 当前mask的状态, mask 的点会将选择的点标记为 -inf
            :return: 输入每个点可能的概率值，即概率分布函数
        '''
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)

        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        # input_cat = torch.cat((encoded_last_node, load[:, :, None], capacity[:, :, None], time[:, :, None], length[:, :, None], route_open[:, :, None]), dim=2)  # 原始程序
        # if encoded_last_node.size(1) != capacity.size(1): # MOKP 没有aug, 需要维度对齐
        #     encoded_last_node = encoded_last_node.expand(-1, capacity.size(1), embedding_dim)
        # input_cat = torch.cat((encoded_last_node, load[:, :, None], capacity[:, :, None], time[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+4)

        input_cat = encoded_last_node

        # F.linear(input,weight) 其实就是矩阵相乘 input * weight^T, 应用给定权重矩阵的线性变换结果
        q_last = reshape_by_heads(F.linear(input_cat, self.Wq_last_para), head_num=head_num) # input_cat.shape = (batch,problem,131)
        # q_last shape: (batch, head_num, pomo, qkv_dim)

        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)
        #print("ninf_mask",ninf_mask)
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask) # 多头注意力的输出
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = F.linear(out_concat, self.multi_head_combine_para)  # 对out_concat进行整合
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)  # 根据多头输入，通过单头计算概率值, score= q*k
        # shape: (batch, pomo, problem)
        #print("score",score)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim'] # (dk)**0.5
        logit_clipping = self.model_params['logit_clipping'] # C = 10

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)
        #print("score_scaled",score_scaled)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs

class KP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim'] # embedding_dim: 128
        head_num = self.model_params['head_num'] # head_num: 8
        qkv_dim = self.model_params['qkv_dim'] # qkv_dim: 16

        hyper_input_dim = 2 # 权重向量的维度 (如0.4,0.6)
        hyper_hidden_embd_dim = 256 # 论文中写的是128
        self.embd_dim = 2 # 权重向量
        self.hyper_output_dim = 4 * self.embd_dim # 根据权重向量扩展出的网络权重
        # 处理权重向量使用的网络 2 - 256 - 256 - 10
        self.hyper_fc1 = nn.Linear(hyper_input_dim, hyper_hidden_embd_dim, bias=True)  # 2 --> 256
        self.hyper_fc2 = nn.Linear(hyper_hidden_embd_dim, hyper_hidden_embd_dim, bias=True)  # 256 --> 256
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, self.hyper_output_dim, bias=True)  # 256 --> 10

        # decoder使用的网络(都是带权重的 下面5个 每个分配上面神经网络输出的2个)
        self.hyper_Wq_last = nn.Linear(self.embd_dim, (1 + embedding_dim) * head_num * qkv_dim, bias=False)  # 2 --> 16384  # 原始程序 +4
        self.hyper_Wk = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)  # 2 --> 16384
        self.hyper_Wv = nn.Linear(self.embd_dim, embedding_dim * head_num * qkv_dim, bias=False)  # 2 --> 16384
        self.hyper_multi_head_combine = nn.Linear(self.embd_dim, head_num * qkv_dim * embedding_dim,bias=False)  # 2 --> 16384

        self.Wq_last_para = None
        self.multi_head_combine_para = None

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def assign(self, pref): # pref: tensor([0.772,0.228])
        '''将权重向量嵌入神经网络当中'''
        embedding_dim = self.model_params['embedding_dim'] # 128
        head_num = self.model_params['head_num'] # 8
        qkv_dim = self.model_params['qkv_dim'] # 16

        # 通过神经网络将2维度映射到10维度 2 - 256 - 256 - 10
        hyper_embd = self.hyper_fc1(pref)
        hyper_embd = self.hyper_fc2(hyper_embd)
        mid_embd = self.hyper_fc3(hyper_embd)
        # [0.1249, -0.0571, 0.0289, 0.2872, -0.2496, 0.4080, -0.2827, -0.4525, 0.2236, 0.1886]

        # hyper_Wq_first取1,2; Wq_last_para取3,4; Wk_para取5,6; Wv_para取7,8; multi_head_combine_para取9,10
        # 将weight权重 变成相应尺寸的权重（权重嵌入到参数中了）
        self.Wq_last_para = self.hyper_Wq_last(mid_embd[:1 * self.embd_dim]).reshape(head_num * qkv_dim, (1 + embedding_dim)) # 2 --> 128*128
        self.Wk_para = self.hyper_Wk(mid_embd[1 * self.embd_dim: 2 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim) # 2 --> 128*128
        self.Wv_para = self.hyper_Wv(mid_embd[2 * self.embd_dim: 3 * self.embd_dim]).reshape(embedding_dim, head_num * qkv_dim) # 2 --> 128*128
        self.multi_head_combine_para = self.hyper_multi_head_combine(mid_embd[3 * self.embd_dim: 4 * self.embd_dim]).reshape(head_num * qkv_dim, embedding_dim) # 128*128

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch:64, problem:20, embedding:128)
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(F.linear(encoded_nodes, self.Wk_para), head_num=head_num)
        self.v = reshape_by_heads(F.linear(encoded_nodes, self.Wv_para), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(F.linear(encoded_q1, self.Wq_first_para), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(F.linear(encoded_q2, self.Wq_first_para), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, capacity, ninf_mask): # encoded_last_node, load, time,length, route_open, ninf_mask
        '''
            :param encoded_last_node: (batch:64, pomo:20, embedding:128)
            :param ninf_mask: (batch, pomo, problem) 当前mask的状态, mask 的点会将选择的点标记为 -inf
            :return: 输入每个点可能的概率值，即概率分布函数
        '''
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)

        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']

        batch_size = capacity.size(0)
        group_size = capacity.size(1)

        #  Multi-Head Attention
        #######################################################
        # input_cat = torch.cat((encoded_last_node, load[:, :, None], capacity[:, :, None], time[:, :, None], length[:, :, None], route_open[:, :, None]), dim=2)  # 原始程序
        # if encoded_last_node.size(1) != capacity.size(1): # MOKP 没有aug, 需要维度对齐
        #     encoded_last_node = encoded_last_node.expand(-1, capacity.size(1), embedding_dim)
        # input_cat = torch.cat((encoded_last_node, load[:, :, None], capacity[:, :, None], time[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+4)

        input1 = encoded_last_node.expand(batch_size, group_size, embedding_dim)
        input2 = capacity[:, :, None]
        input_cat = torch.cat((input1, input2), dim=2)

        # F.linear(input,weight) 其实就是矩阵相乘 input * weight^T, 应用给定权重矩阵的线性变换结果
        q_last = reshape_by_heads(F.linear(input_cat, self.Wq_last_para), head_num=head_num) # input_cat.shape = (batch,problem,131)
        # q_last shape: (batch, head_num, pomo, qkv_dim)

        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)
        #print("ninf_mask",ninf_mask)
        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask) # 多头注意力的输出
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = F.linear(out_concat, self.multi_head_combine_para)  # 对out_concat进行整合
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)  # 根据多头输入，通过单头计算概率值, score= q*k
        # shape: (batch, pomo, problem)
        #print("score",score)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim'] # (dk)**0.5
        logit_clipping = self.model_params['logit_clipping'] # C = 10

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)
        #print("score_scaled",score_scaled)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs

########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))# 计算u值，以计算权重
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        # print(q.shape, score.shape, rank3_ninf_mask.shape)
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans

class FeedForward(nn.Module):
    '''transfomer encoder 中的 FF 层；128 - 512 - 128'''
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))