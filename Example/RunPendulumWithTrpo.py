import sys

if "../" not in sys.path:
  sys.path.append("../")

import gym,torch
from Solver.TRPOContinuousSolver import TRPOContinuous
from utils.util import train_on_policy_agent,moving_average
from utils.plot_func import plot_return_list
env_name='Pendulum-v1'
env=gym.make(env_name)
print(env.reset())
print(env.action_space)

gamma=0.9
lr=1e-2
lmbda=0.9
alpha=0.5
input_dim=env.observation_space.shape[0]
hidden_dim=128
output_dim=env.action_space.shape[0]
kl_constraints=0.0005
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_episode=2000

agent=TRPOContinuous(gamma,lr,lmbda,alpha,input_dim,hidden_dim,output_dim,kl_constraints,device)
return_list=train_on_policy_agent(env,agent,num_episode)
plot_return_list(return_list,'TRPO on Pendulum')
plot_return_list(moving_average(return_list,9),'TRPO on Pendulum with moving average 9')
