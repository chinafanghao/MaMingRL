import sys

if "../" not in sys.path:
  sys.path.append("../")

from utils.util import BufferReplay,train_off_policy_agent,moving_average
from Solver.SACSolver import SACContinuous
from utils.plot_func import plot_return_list
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random
import gym
from tqdm import tqdm
from collections import deque

env_name='Pendulum-v1'
env=gym.make(env_name)

gamma=0.99
tau=0.005
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.shape[0]
hidden_dim=128
actor_lr=3e-4
critic_lr=3e-3
alpha_lr=3e-4
target_entropy=-env.action_space.shape[0]
action_bound=env.action_space.high[0]
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

agent=SACContinuous(gamma,tau,state_dim,action_dim,hidden_dim,actor_lr,critic_lr,alpha_lr,target_entropy,action_bound,device)

max_buffer_size=100000
minimal_buffer_size=1000
batch_size=64
num_episode=500

buffer=BufferReplay(max_buffer_size)
return_list=train_off_policy_agent(env,agent,num_episode,buffer,minimal_buffer_size,batch_size)
plot_return_list(return_list)
plot_return_list(moving_average(return_list,9))