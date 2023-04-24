import sys

if "../" not in sys.path:
  sys.path.append("../")

from utils.util import BufferReplay
from Solver.DQNSolver import DQNs

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

epsilon=1e-2
gamma=0.98
learning_rate=2e-3
input_dim=env.observation_space.shape[0]
hidden_dim=128
output_dim=11
target_update=10
minimal_size=500
batch_size=64
capacity=10000
num_episode=500
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dqn_types=['Valli_dqn','double_dqn','dueling_dqn']

buffer=BufferReplay(capacity)

def dis_to_continue(a):
  low=env.action_space.low[0]
  high=env.action_space.high[0]
  return low+(high-low)/(output_dim-1)*a

return_list=[[] for _ in range(3)]
max_values=[[] for _ in range(3)]
for idx,dqn_type in enumerate(dqn_types):
  agent=DQNs(epsilon,gamma,learning_rate,input_dim,hidden_dim,output_dim,target_update,device,dqn_type)
  max_value=0
  for i in range(10):
    with tqdm(total=int(num_episode/10),desc='Iteration: %d'%(i+1)) as pbar:
      for i_episode in range(int(num_episode/10)):
        state=env.reset()
        done=False
        episode_return=0
        while not done:
          action=agent.take_action(state)
          action_cont=dis_to_continue(action)
          next_state,reward,done,_=env.step([action_cont])
          episode_return+=reward
          max_value=0.995*max_value+agent.max_q(state)*0.005
          max_values[idx].append(max_value)
          buffer.add((state,action,reward,next_state,done))
          state=next_state
          if len(buffer)>minimal_size:
            b_s,b_a,b_r,b_ns,b_d=buffer.sample(batch_size)
            transition={
                'states':b_s,
                'actions':b_a,
                'rewards':b_r,
                'next_states':b_ns,
                'dones':b_d
            }
            agent.update(transition)
        return_list[idx].append(episode_return)
        if (i_episode+1)%10==0:
          pbar.set_postfix({'episode':'%d'%(num_episode//10*i+i_episode+1),'return':'%.3f'%np.mean(return_list[idx][-10:])})
        pbar.update(1)

def moving_average(data,wind_size):
  data=np.array(data)
  middle=(data[wind_size:]-data[wind_size:])/wind_size
  r=np.arange(1,wind_size)
  front=data[:wind_size-1]/r
  ends=data[-wind_size+1:]/r[::-1]
  return np.concatenate((front,middle,ends))

for idx,dqn_type in enumerate(dqn_types):
  m=moving_average(return_list[idx],9)[:20]
  plt.plot(range(len(m)),m,label=dqn_type)
plt.title('DQNs on pendulum')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Return')
plt.show()

plt.figure(figsize=(18, 16))
for idx,dqn_type in enumerate(dqn_types):
  m=max_values[idx]
  plt.plot(range(len(m)),m,label=dqn_type)
plt.title('DQNs on pendulum')
plt.axhline(0,c='black',ls='--')
plt.axhline(10,c='pink',ls='-.')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('max Q')
plt.show()