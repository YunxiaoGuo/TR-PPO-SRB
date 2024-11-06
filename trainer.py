import gym
import torch
from torch import Tensor
from networks.networks import ActorCritic
from utils.utils import *
from algorithms import PPO_Clip,PPO_KL,PPO_S,TR_PPO,TR_PPO_RB,TR_PPO_SRB
from tqdm import tqdm
import os

def train(args):
    device = 'cuda:0' if args.is_GPU == 'True' else 'cpu'
    print("###############Training algorithm: "+args.algorithm+"##################")
    print("###############Training environment: " + args.env_name + "#############")
    print("###############Training device: " + device + "#############")
    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    network = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], layer_norm=args.layer_norm).to(device)
    running_state = ZFilter((env.observation_space.shape[0],), clip=5.0)
    reward_record = []
    global_steps = 0
    for i_episode in tqdm(range(args.num_episode)):
        memory = Memory()
        num_steps = 0
        reward_list = []
        len_list = []
        while num_steps < args.batch_size:
            state = env.reset()
            if args.state_norm:
                state = running_state(state)
            reward_sum = 0
            for t in range(args.max_step_per_round):
                state_ = Tensor(np.array(state)).unsqueeze(0).to(device)
                action_mean, action_logstd, value = network(state_)
                action, logproba = network.select_action(action_mean, action_logstd)
                action = action.detach().cpu().numpy()[0]#.detach().cpu()[0]#.data.numpy()[0]#
                logproba = logproba.detach().cpu().numpy()[0]#.detach().cpu()[0]#
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward
                if args.state_norm:
                    next_state = running_state(next_state)
                mask = 0 if done else 1
                memory.push(state, value, action, logproba, mask, next_state, reward)
                if done:
                    break
                state = next_state
            num_steps += (t + 1)
            global_steps += (t + 1)
            reward_list.append(reward_sum)
            len_list.append(t + 1)
        reward_record.append(np.mean(reward_list))
        print(np.mean(reward_list))
        for i_epoch in range(int(args.num_epoch * len(memory) / args.minibatch_size)):
            method = {
                'PPO-Clip': PPO_Clip,
                'PPO-KL': PPO_KL,
                'PPO-S': PPO_S,
                'TR-PPO': TR_PPO,
                'TR-PPO-RB': TR_PPO_RB,
                'TR-PPO-SRB': TR_PPO_SRB
            }
            method[args.algorithm].algorithm(args, memory, network)
    data_path = os.path.join(args.data_path,args.env_name)
    data_path = os.path.join(data_path,args.algorithm)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    np.save(os.path.join(data_path,+args.algorithm+'_'+'episode_reward_'+str(args.seed)+'.npy'),np.array(reward_record))




















