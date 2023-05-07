import sys
import gym
if "../" not in sys.path:
  sys.path.append("../")

import torch
from utils.util import BufferReplay,train_off_policy_agent,moving_average
from Solver.SACSolver import SACDiscrete
from utils.plot_func import plot_return_list

env_name='CartPole-v0'
env=gym.make(env_name)
gamma=0.98
tau=0.005
state_dim=env.observation_space.shape[0]
hidden_dim=128
action_dim=env.action_space.n
actor_lr=1e-3
critic_lr=1e-2
alpha_lr=1e-2
target_entropy=-1
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

buffer_size=10000
minimal_size=500
batch_size=64
buffer=BufferReplay(buffer_size)

num_episode=200
agent=SACDiscrete(gamma,tau,state_dim,hidden_dim,action_dim,actor_lr,critic_lr,alpha_lr,target_entropy,device)
return_list=train_off_policy_agent(env,agent,buffer,num_episode,minimal_size,batch_size)

plot_return_list(return_list)
plot_return_list(moving_average(return_list,9))