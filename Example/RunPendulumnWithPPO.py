import sys

if "../" not in sys.path:
  sys.path.append("../")

import gym,torch
from Solver.PPOSolver import PPOContinuous
from utils.util import train_on_policy_agent,moving_average
from utils.plot_func import plot_return_list
env_name='Pendulum-v1'
env=gym.make(env_name)

gamma=0.9
lmbda=0.9
a_lr=1e-4
c_lr=5e-3
epsilon=0.2
epochs=10
input_dim=env.observation_space.shape[0]
hidden_dim=128
output_dim=env.action_space.shape[0]
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_episodes=2000

agent=PPOContinuous(gamma,lmbda,a_lr,c_lr,epsilon,epochs,input_dim,hidden_dim,output_dim,device)
return_list=train_on_policy_agent(env,agent,num_episodes)
plot_return_list(return_list,'PPO on Pendulumn')
plot_return_list(moving_average(return_list,9),'PPO on Pendulumn with moving average')