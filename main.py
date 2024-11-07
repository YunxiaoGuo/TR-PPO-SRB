from argument import parameters
from trainer import train
from test import pytest
from visualization.plot_curves import learning_curve
import numpy as np
import os
import multiprocessing

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main_multiprocessing(args,seeds):
    processes = []
    for seed in seeds:
        args.seed = int(seed)
        process = multiprocessing.Process(target=train,args=(args,))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()

if __name__ == '__main__':
    env_list = ["Swimmer-v2","Reacher-v2","Hopper-v2","HalfCheetah-v2","Walker2d-v2","Humanoid-v2",
                 'MountainCarContinuous-v0', "LunarLanderContinuous-v2", "BipedalWalker-v3"]
    algorithm_list = ['PPO-KL','PPO-Clip','PPO-S','TR-PPO','TR-PPO-SRB','TR-PPO-RB']
    args = parameters.get_paras()
    if args.plot == True:
        learning_curve(args,algorithm_list)
    elif args.evaluate == True:
        pytest(args)
    else:
        seeds = np.random.randint(1,1000,args.num_para)
        main_multiprocessing(args,seeds)
            
