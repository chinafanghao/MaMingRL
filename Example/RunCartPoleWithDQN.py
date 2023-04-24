import sys

if "../" not in sys.path:
  sys.path.append("../")

from utils.util import BufferReplay,moving_average
from Solver.DQNSolver import DQNs

#from utils.util import BufferReplay

import torch,random
import matplotlib.pyplot as plt
import numpy as np
import gym
from tqdm import tqdm
env=gym.make('CartPole-v0')


lr=2e-3
num_episodes=500
hidden_dim=128
gamma=0.98
epsilon=0.01
target_update=10
buffer_size=10000
minimal_size=500
batch_size=64
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

replay_buffer=BufferReplay(buffer_size)

state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n
agent=DQN(env,epsilon,gamma,lr,[state_dim,hidden_dim,action_dim],target_update,device)

return_list=[]

for i in range(10):
  with tqdm(total=int(num_episodes/10),desc='Iteration %d'%(i+1)) as pbar:
    for i_episode in range(int(num_episodes/10)):
      episode_return=0
      state=env.reset()[0]
      done=False
      while not done:
        action=agent.take_action(state)
        next_state,reward,done,truncated,_=env.step(action)
        replay_buffer.add((state,action,reward,next_state,done,truncated))
        state=next_state
        episode_return+=reward
        if len(replay_buffer)>minimal_size:
          b_s,b_a,b_r,b_ns,b_d,b_t=replay_buffer.sample(batch_size)
          transition_dict={'states':b_s,'actions':b_a,'next_states':b_ns,'rewards':b_r,'dones':b_d,'truncateds':b_t}
          agent.update(transition_dict)
        if truncated:
          break
      return_list.append(episode_return)
      if (i_episode+1)%10==0:
        pbar.set_postfix({'episode':'%d'%(num_episodes/10*i+i_episode+1),'return':'%.3f'%np.mean(return_list[-10:])})
      pbar.update(1)

plt.plot(range(len(return_list)),return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on CartPole-v0')
plt.show()

mv_return=moving_average(return_list,9)
plt.plot(range(len(mv_return)),mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on CartPole-v0')
plt.show()