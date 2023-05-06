import sys

if "../" not in sys.path:
  sys.path.append("../")

from utils.util import BufferReplay,train_off_policy_agent,moving_average
from Solver.DDPGSolver import DDPG
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

gamma=0.98
tau=0.005
sigma=0.01
state_dim=env.observation_space.shape[0]
hidden_dim=64
action_dim=env.action_space.shape[0]
actor_lr=5e-4
critic_lr=5e-3
discrete=False
action_bound=env.action_space.high[0]
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

agent=DDPG(gamma,tau,sigma,state_dim,action_dim,state_dim+action_dim,hidden_dim,actor_lr,critic_lr,discrete,action_bound,device)

buffer_size=10000
batch_size=64
minimal_size=1000
buffer=BufferReplay(buffer_size)
num_episode=2000

return_list=train_off_policy_agent(env,agent,num_episode,buffer,minimal_size,batch_size)
plot_return_list(return_list,'DDPG on Pendulumn')
plot_return_list(moving_average(return_list,9),'DDPG on Pendulumn with moving average')
