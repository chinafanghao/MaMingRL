import sys

if "../" not in sys.path:
  sys.path.append("../")

import gym,torch
from Solver.TRPOSolver import TRPO
from utils.plot_func import plot_return_list
from utils.util import train_on_policy_agent,moving_average
env_name='CartPole-v0'
env=gym.make(env_name)

print(env.reset())
print(env.action_space.n)
print(env.step(0))

gamma=0.98
lmbda=0.95
input_dim=env.observation_space.shape[0]
hidden_dim=128
action_space=env.action_space.n
lr=1e-2
alpha=0.5
kl_constraint=0.0005
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_episodes=500

agent=TRPO(gamma,lmbda,input_dim,hidden_dim,action_space,lr,alpha,kl_constraint,device)
return_list=train_on_policy_agent(env,agent,num_episodes)

plot_return_list(return_list,'TRPO')
plot_return_list(moving_average(return_list,9),'TRPO moving average')
