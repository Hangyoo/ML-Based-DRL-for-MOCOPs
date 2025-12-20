##########################################################################################
# Machine Environment Config

DEBUG_MODE = False 
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys
import torch
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from Meta_Transformer_2D_reptile.utils.utils import create_logger, copy_all_src

from VRPTester import VRPTester as Tester
import time
import os

##########################################################################################
# parameters
# problem_type can be CVRP, OVRP, VRPB, VRPTW, VRPL and their any combinations, e.g., OVRPBTW
# 2-obj 3-obj 4-obj 5-obj 6-obj (5种)

instance_name = 'MOKP_p-2_n-200_1.txt'

PATH = r'D:\Paper Data\paper 3\instance\KP_benchmark'
for instance_name in [f for f in os.listdir(PATH) if f.endswith('.txt')]:
    print(f'当前执行{instance_name}')
    problem_size = int(instance_name.split('_')[2].split('-')[1])
    # problem_size = 100
    env_params = {
        'problem_type': "MOKP", # test problem type
        'problem_size': problem_size,
        'pomo_size': problem_size,
        'aggregate_function': 'TCH',
        'benchmark_instance': True,
        'instance_type': None,
        'instance_name': instance_name
    }

    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
    }


    tester_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'model_load': {
            'path': 'Trained_models/50',  # directory path of pre-trained model and log files saved.
            'epoch': 10000,  # epoch version of pre-trained model to laod.
        },
        'test_episodes': 1,
        'test_batch_size': 1,
        'augmentation_enable': True,
        'aug_factor': 1,
        'aug_batch_size': 1,
        'test_data_load': {
            'enable': False,
            'filename': '../../Test_instances/data_'+env_params['problem_type']+'_'+str(env_params['pomo_size'])\
                +'_'+str(5000)+'.pt'
        },
    }
    if tester_params['augmentation_enable']:
        tester_params['test_batch_size'] = tester_params['aug_batch_size']


    logger_params = {
        'log_file': {
            'desc': 'test_'+env_params['problem_type']+'_n'+str(env_params['problem_size'])+'_with_instNorm',
            'filename': 'run_log'
        }
    }


    ##########################################################################################
    # main

    def main(n_sols = 101,idx=None):
        if DEBUG_MODE:
            _set_debug_mode()

        create_logger(**logger_params)
        _print_config()

        timer_start = time.time()
        tester = Tester(env_params=env_params,
                          model_params=model_params,
                          tester_params=tester_params)

        copy_all_src(tester.result_folder)

        sols = np.zeros([n_sols, 2])

        for i in range(n_sols):
            pref = torch.zeros(2).cuda()
            pref[0] = 1 - 0.01 * i
            pref[1] = 0.01 * i
            pref = pref / torch.sum(pref)
            # 目标函数 <-- [问题, 权重向量]
            aug_score = tester.run(pref)
            sols[i] = np.array(aug_score)  # 记录目标函数

        timer_end = time.time()

        total_time = timer_end - timer_start

        # 保存结果到.txt格式
        idx = 0
        subfile = 'tch_aug' if env_params['aggregate_function'] == 'TCH' else 'wei_aug'
        if env_params['instance_type'] == 'Random':
            np.savetxt(f"result/{subfile}/kp/{env_params['instance_type']}{problem_size}_{idx}.txt", sols)
            with open(f"result/{subfile}/kp/TIME_{env_params['instance_type']}{problem_size}_{idx}.txt", 'w') as f:
                f.write(str(total_time))
        else:
            np.savetxt(f"result/{subfile}/kp/{instance_name}_each_decoder.txt", sols)
            with open(f"result/{subfile}/kp/TIME_{instance_name}_each_decoder.txt", 'w') as f:
                f.write(str(total_time))

        print('Run Time(s): {:.4f}'.format(total_time))

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(sols[:, 0], sols[:, 1], "ro", label='DRL-MOA')
        plt.savefig(f'result/{subfile}/kp/{instance_name}_each_decoder.png')
        plt.clf()


    def _set_debug_mode():
        global tester_params
        tester_params['test_episodes'] = 10


    def _print_config():
        logger = logging.getLogger('root')
        logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
        logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
        [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

    main()

##########################################################################################

# if __name__ == "__main__":
#     main()

