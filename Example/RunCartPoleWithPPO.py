import sys

if "../" not in sys.path:
  sys.path.append("../")

from utils.util import BufferReplay,moving_average
from Solver.PPOSolver import PPO

#from utils.util import BufferReplay

import torch,random
import matplotlib.pyplot as plt
import numpy as np
import gym
from tqdm import tqdm
from utils.plot_func import plot_return_list
from utils.util import train_on_policy_agent,moving_average

env=gym.make('CartPole-v0')

gamma=0.98
lmbda=0.95
a_lr=1e-2
c_lr=1e-2
epsilon=0.2
epochs=10
input_dim=env.observation_space.shape[0]
hidden_dim=128
output_dim=env.action_space.n
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_episodes=500

agent=PPO(gamma,lmbda,a_lr,c_lr,epsilon,epochs,input_dim,hidden_dim,output_dim,device)
return_list=train_on_policy_agent(env,agent,num_episodes)
plot_return_list(return_list,'PPO on CartPole')
plot_return_list(moving_average(return_list,9),'PPO on CartPole with moving average')