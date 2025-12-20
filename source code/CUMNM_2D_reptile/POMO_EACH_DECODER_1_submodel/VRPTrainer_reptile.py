import copy
import torch
from logging import getLogger

from VRPEnv import VRPEnv as Env
from VRPModel import VRPModel as Model
 
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.nn import DataParallel

from Meta_Transformer_2D_reptile.utils.utils import *

def get_inner_model(model): # 判断是否为多GPU运行，并返回真正的内部模型
    return model.module if isinstance(model, DataParallel) else model

class VRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            # if torch.cuda.device_count() > 1:
            #     os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
            #     gpus = [0, 1]
            #     torch.cuda.set_device('cuda:{}'.format(gpus[0]))
            #     device = torch.device('cuda', gpus[0])
            #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
            # else:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.env = Env(**self.env_params)

        # Meta-learning
        self.meta_lr = 1.0  # 元学习率
        self.task_num = 1  # 子任务的个数
        self.model = Model(**self.model_params) # 待更新的主模型参数
        self.sub_model = copy.deepcopy(self.model) # 子模型参数
        self.optimizer = Optimizer(self.sub_model.parameters(), **self.optimizer_params['optimizer']) # 更新子模型参数, 通过子模型更新主模型参数
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        total_params = sum(param.numel() for param in self.model.parameters())  # 计算神经网络模型中所有参数的总数量。
        print('Total Params Num -- embedding和decoder独立, encoder共享:', total_params)  # 1989536

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            get_inner_model(self.model).load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

        #Parallel
        # if USE_CUDA and torch.cuda.device_count() > 1:
        #     self.sub_model = torch.nn.DataParallel(self.sub_model.to(device), device_ids=gpus, output_device=gpus[0])

    def meta_run(self):
        """
            Meta Learner
        """
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()
            # Train
            model_new_weights = []
            meta_lr = self.meta_lr * (1. - epoch / float(self.trainer_params['epochs'] + self.start_epoch))  # META_LR 从1递减到0
            sub_model_weights_original = copy.deepcopy(get_inner_model(self.model).state_dict())  # 复制主网络参数
            train_score_obj1_sum = 0
            train_score_obj2_sum = 0
            train_loss_sum = 0
            for task_id in range(self.task_num):
                get_inner_model(self.sub_model).load_state_dict(sub_model_weights_original)  # 子模型加载参数
                train_score_obj1, train_score_obj2, train_loss = self._train_one_epoch(epoch)
                train_score_obj1_sum += train_score_obj1
                train_score_obj2_sum += train_score_obj2
                train_loss_sum += train_loss
                model_new_weights.append(copy.deepcopy(get_inner_model(self.sub_model).state_dict()))  # 更新了100次之后, 将子模型分别加入

            # reptile 元学习更新
            k_fw = {name: model_new_weights[0][name] / float(self.task_num) for name in model_new_weights[0]}  # 权重参数取平均
            for i in range(1, self.task_num):
                for name in model_new_weights[i]:
                    k_fw[name] += model_new_weights[i][name] / float(self.task_num)

            # 用子模型来更新主模型
            main_model_weights_original = copy.deepcopy(get_inner_model(self.model).state_dict())  # 复制主网络参数
            new_weight = {name:main_model_weights_original[name] + (k_fw[name] - main_model_weights_original[name]) * meta_lr for name in main_model_weights_original}
            get_inner_model(self.model).load_state_dict(new_weight) # 更新主网络参数

            self.result_log.append('train_score_obj1_reptile_avg', epoch, train_score_obj1_sum/self.task_num)
            self.result_log.append('train_score_obj2_reptile_avg', epoch, train_score_obj2_sum/self.task_num)
            self.result_log.append('train_loss_reptile_avg', epoch, train_loss_sum/self.task_num)

            # Log Once, for each epoch
            self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Obj1 Score: {:.4f},  Obj2 Score: {:.4f},  Loss: {:.4f}'
                             .format(epoch, 100. * epoch / self.trainer_params['train_episodes'],
                train_score_obj1_sum/self.task_num, train_score_obj2_sum/self.task_num, train_loss_sum/self.task_num))

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # Save latest images, every epoch
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_f1'],
                                               self.result_log, labels=['train_score_obj1_reptile_avg'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_f2'],
                                               self.result_log, labels=['train_score_obj2_reptile_avg'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss_reptile_avg'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': get_inner_model(self.model).state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # Save Image
            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_f1'],
                                               self.result_log, labels=['train_score_obj1_reptile_avg'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_f2'],
                                               self.result_log, labels=['train_score_obj2_reptile_avg'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss_reptile_avg'])

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM_obj1 = AverageMeter()
        score_AM_obj2 = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0

        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            avg_score_obj1, avg_score_obj2, avg_loss = self._train_one_batch(batch_size)
            score_AM_obj1.update(avg_score_obj1, batch_size)
            score_AM_obj2.update(avg_score_obj2, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f},  Loss: {:.4f}'
                        .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg))

        return score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.env = Env(**self.env_params) # 应该重新初始化, 不然有东西没初始化干净
        # self.model.train()
        self.sub_model.train()
        self.env.load_problems(batch_size)

        pref = torch.rand([2])
        pref = pref / torch.sum(pref)

        reset_state, _, _ = self.env.reset()

        if self.env.attribute_c:
            self.sub_model.decoder_CVRP.assign(pref)
        elif self.env.attribute_tw:
            self.sub_model.decoder_VRPTW.assign(pref)
        elif self.env.attribute_kp:
            self.sub_model.decoder_KP.assign(pref)
        elif self.env.attribute_tsp:
            self.sub_model.decoder_TSP.assign(pref)
        elif self.env.attribute_mixedtsp:
            self.sub_model.decoder_MIXTSP.assign(pref)
        else:
            raise NotImplementedError

        self.sub_model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            if self.env.attribute_kp:
                selected, prob = self.sub_model(state)
                # shape: (batch, pomo)
                action_w_finisehd = selected.clone()
                action_w_finisehd[state.finished] = self.env_params['problem_size']  # dummy item
                state, reward, done = self.env.step(action_w_finisehd)

                # shape = (batch, group)
                chosen_action_prob = prob
                chosen_action_prob[state.finished] = 1  # done episode will gain no more probability
                prob_list = torch.cat((prob_list, chosen_action_prob[:, :, None]), dim=2)
            else:
                selected, prob = self.sub_model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)


        # Loss
        ###############################################
        # reward_mean = reward.float().mean(dim=1, keepdims=True)
        # advantage = torch.div((reward - reward_mean),-reward_mean) # normalize different probelms have different level of rewards

        if self.env_params['aggregate_function'] == 'TCH':
            reward = - reward
            z = torch.ones(reward.shape).cuda() * 0.0
            tch_reward = pref * (reward - z)
            tch_reward , _ = tch_reward.max(dim = 2)
            # set back reward and group_reward to positive and to maximize
            reward = -reward
            tch_reward = -tch_reward
        elif self.env_params['aggregate_function'] == 'Weighted':
            reward = - reward  # (batch,pomo,2)
            tch_reward = (pref * reward).sum(dim=2)  # Weighted-sum (batch:64,pomo:20) 求和
            # set back reward and group_reward to positive and to maximize
            reward = -reward
            tch_reward = -tch_reward
        elif self.env_params['aggregate_function'] == 'PBI':
            theta = 0.9
            reward = - reward
            z = torch.ones(reward.shape).cuda() * 0.0  # 设定一个理想点(64*64,20,2)
            d1 = torch.norm((reward - z) * pref, dim=2) / torch.norm(pref)
            d2 = torch.norm((reward - z - (d1.unsqueeze(-1) * pref / torch.norm(pref))), dim=2)
            pbi_reward = d1 + theta * d2

            # set back reward and group_reward to positive and to maximize
            reward = -reward
            tch_reward = -pbi_reward
        else:
            sys.exit("请检查聚合函数")

        advantage = tch_reward - tch_reward.mean(dim=1, keepdim=True)

        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        _, max_idx = tch_reward.max(dim=1)
        max_idx = max_idx.reshape(max_idx.shape[0], 1)
        max_reward_obj1 = reward[:, :, 0].gather(1, max_idx)
        max_reward_obj2 = reward[:, :, 1].gather(1, max_idx)

        if self.env.attribute_kp:
            score_mean_obj1 = max_reward_obj1.float().mean()
            score_mean_obj2 = max_reward_obj2.float().mean()
        else:
            score_mean_obj1 = - max_reward_obj1.float().mean()
            score_mean_obj2 = - max_reward_obj2.float().mean()

        # Step & Return
        ###############################################
        self.sub_model.zero_grad() # 和self.optimizer.step()等效
        loss_mean.backward()
        self.optimizer.step()

        return score_mean_obj1.item(), score_mean_obj2.item(), loss_mean.item()

    def _finetune_one_batch(self, pref, fine_model, batch_size, finetuneType):
        # Prep
        ###############################################
        self.env = Env(**self.env_params) # 应该重新初始化, 不然有东西没初始化干净
        # self.model.train()
        fine_model.train()
        fine_optimizer = Optimizer(fine_model.parameters(),**self.optimizer_params['optimizer'])  # 更新子模型参数, 通过子模型更新主模型参数
        self.env.problem_type = 'MOTSP'
        self.env.load_problems(batch_size)

        # pref = torch.tensor(pref)
        pref = torch.rand([2])
        pref = pref / torch.sum(pref)
        # print("weight:{}, {}".format(pref[0].item(), pref[1].item()))

        reset_state, _, _ = self.env.reset()

        if finetuneType == 'cvrp':
            fine_model.decoder_CVRP.assign(pref)
        elif finetuneType == 'vrptw':
            fine_model.decoder_VRPTW.assign(pref)
        elif finetuneType == 'kp':
            fine_model.decoder_KP.assign(pref)
        elif finetuneType == 'tsp':
            fine_model.decoder_TSP.assign(pref)
        elif finetuneType == 'mixedtsp':
            fine_model.decoder_MIXTSP.assign(pref)
        else:
            raise NotImplementedError

        fine_model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            if finetuneType == 'kp':
                selected, prob = fine_model(state)
                # shape: (batch, pomo)
                action_w_finisehd = selected.clone()
                action_w_finisehd[state.finished] = self.env_params['problem_size']  # dummy item
                state, reward, done = self.env.step(action_w_finisehd)

                # shape = (batch, group)
                chosen_action_prob = prob
                chosen_action_prob[state.finished] = 1  # done episode will gain no more probability
                prob_list = torch.cat((prob_list, chosen_action_prob[:, :, None]), dim=2)
            else:
                selected, prob = fine_model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)


        # Loss
        ###############################################

        if self.env_params['aggregate_function'] == 'TCH':
            reward = - reward
            z = torch.ones(reward.shape).cuda() * 0.0
            tch_reward = pref * (reward - z)
            tch_reward , _ = tch_reward.max(dim = 2)
            # set back reward and group_reward to positive and to maximize
            reward = -reward
            tch_reward = -tch_reward
        elif self.env_params['aggregate_function'] == 'Weighted':
            reward = - reward  # (batch,pomo,2)
            tch_reward = (pref * reward).sum(dim=2)  # Weighted-sum (batch:64,pomo:20) 求和
            # set back reward and group_reward to positive and to maximize
            reward = -reward
            tch_reward = -tch_reward
        elif self.env_params['aggregate_function'] == 'PBI':
            theta = 0.9
            reward = - reward
            z = torch.ones(reward.shape).cuda() * 0.0  # 设定一个理想点(64*64,20,2)
            d1 = torch.norm((reward - z) * pref, dim=2) / torch.norm(pref)
            d2 = torch.norm((reward - z - (d1.unsqueeze(-1) * pref / torch.norm(pref))), dim=2)
            pbi_reward = d1 + theta * d2

            # set back reward and group_reward to positive and to maximize
            reward = -reward
            tch_reward = -pbi_reward
        else:
            sys.exit("请检查聚合函数")

        advantage = tch_reward - tch_reward.mean(dim=1, keepdim=True)

        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        _, max_idx = tch_reward.max(dim=1)
        max_idx = max_idx.reshape(max_idx.shape[0], 1)
        max_reward_obj1 = reward[:, :, 0].gather(1, max_idx)
        max_reward_obj2 = reward[:, :, 1].gather(1, max_idx)

        if finetuneType == 'kp':
            score_mean_obj1 = max_reward_obj1.float().mean()
            score_mean_obj2 = max_reward_obj2.float().mean()
        else:
            score_mean_obj1 = - max_reward_obj1.float().mean()
            score_mean_obj2 = - max_reward_obj2.float().mean()

        # Step & Return
        ###############################################
        fine_optimizer.zero_grad() # 和self.optimizer.step()等效
        loss_mean.backward()
        fine_optimizer.step()

        return score_mean_obj1.item(), score_mean_obj2.item(), loss_mean.item()

    def _validate_one_batch(self, pref, valid_model, batch_size, finetuneType):

        # Prep
        ###############################################
        self.env = Env(**self.env_params) # 应该重新初始化, 不然有东西没初始化干净
        # self.model.train()
        valid_model.eval()
        fine_optimizer = Optimizer(valid_model.parameters(),**self.optimizer_params['optimizer'])  # 更新子模型参数, 通过子模型更新主模型参数
        self.env.load_problems(batch_size)

        pref = torch.rand(pref)

        reset_state, _, _ = self.env.reset()

        if finetuneType == 'cvrp':
            valid_model.decoder_CVRP.assign(pref)
        elif finetuneType == 'vrptw':
            valid_model.decoder_VRPTW.assign(pref)
        elif finetuneType == 'kp':
            valid_model.decoder_KP.assign(pref)
        elif finetuneType == 'tsp':
            valid_model.decoder_TSP.assign(pref)
        elif finetuneType == 'mixedtsp':
            valid_model.decoder_MIXTSP.assign(pref)
        else:
            raise NotImplementedError

        valid_model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            if finetuneType == 'kp':
                selected, prob = valid_model(state)
                # shape: (batch, pomo)
                action_w_finisehd = selected.clone()
                action_w_finisehd[state.finished] = self.env_params['problem_size']  # dummy item
                state, reward, done = self.env.step(action_w_finisehd)

                # shape = (batch, group)
                chosen_action_prob = prob
                chosen_action_prob[state.finished] = 1  # done episode will gain no more probability
                prob_list = torch.cat((prob_list, chosen_action_prob[:, :, None]), dim=2)
            else:
                selected, prob = valid_model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)


        # Loss
        ###############################################

        if self.env_params['aggregate_function'] == 'TCH':
            reward = - reward
            z = torch.ones(reward.shape).cuda() * 0.0
            tch_reward = pref * (reward - z)
            tch_reward , _ = tch_reward.max(dim = 2)
            # set back reward and group_reward to positive and to maximize
            reward = -reward
            tch_reward = -tch_reward
        elif self.env_params['aggregate_function'] == 'Weighted':
            reward = - reward  # (batch,pomo,2)
            tch_reward = (pref * reward).sum(dim=2)  # Weighted-sum (batch:64,pomo:20) 求和
            # set back reward and group_reward to positive and to maximize
            reward = -reward
            tch_reward = -tch_reward
        elif self.env_params['aggregate_function'] == 'PBI':
            theta = 0.9
            reward = - reward
            z = torch.ones(reward.shape).cuda() * 0.0  # 设定一个理想点(64*64,20,2)
            d1 = torch.norm((reward - z) * pref, dim=2) / torch.norm(pref)
            d2 = torch.norm((reward - z - (d1.unsqueeze(-1) * pref / torch.norm(pref))), dim=2)
            pbi_reward = d1 + theta * d2

            # set back reward and group_reward to positive and to maximize
            reward = -reward
            tch_reward = -pbi_reward
        else:
            sys.exit("请检查聚合函数")


        # Score
        ###############################################
        _, max_idx = tch_reward.max(dim=1)
        max_idx = max_idx.reshape(max_idx.shape[0], 1)
        max_reward_obj1 = reward[:, :, 0].gather(1, max_idx)
        max_reward_obj2 = reward[:, :, 1].gather(1, max_idx)

        if finetuneType == 'kp':
            score_mean_obj1 = max_reward_obj1.float().mean()
            score_mean_obj2 = max_reward_obj2.float().mean()
        else:
            score_mean_obj1 = - max_reward_obj1.float().mean()
            score_mean_obj2 = - max_reward_obj2.float().mean()

        return score_mean_obj1.item(), score_mean_obj2.item()