import sys

if "../" not in sys.path:
  sys.path.append("../")

from utils.util import moving_average
from Solver.reinforce import REINFORCE,REINFOCEWithBaseline
import matplotlib.pyplot as plt
import torch
import numpy as np
import gym
from tqdm import tqdm

env_name='CartPole-v1'
env=gym.make(env_name)
print(env.reset())
print(env.step(1))

gamma=0.98
learning_rate=1e-3
hidden_dim=128
param_list=[env.observation_space.shape[0],hidden_dim,env.action_space.n]
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_episodes=1000


agent1 = REINFORCE(gamma, learning_rate, param_list, device)
agent2=REINFOCEWithBaseline(gamma,learning_rate,param_list,device)
def run(agent):

    return_list=[]
    for i in range(10):
        with tqdm(total=(int(num_episodes/10)),desc='iteration : %d'%(i+1)) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return=0
                state=env.reset()[0]
                done=False
                transition={'states':[],'actions':[],'rewards':[],'next_states':[],'dones':[]}
                while not done:
                    action=agent.take_action(state)
                    next_state,reward,done,truncated,_=env.step(action)
                    episode_return+=reward
                    transition['states'].append(state)
                    transition['actions'].append(action)
                    transition['rewards'].append(reward)
                    transition['next_states'].append(next_state)
                    transition['dones'].append(done)
                    state=next_state
                agent.update(transition)
                return_list.append(episode_return)

                if i_episode%10==0:
                    pbar.set_postfix({'episode':'%d'% i_episode,'value':'%.3f'%(np.mean(return_list[-10:]))})
                pbar.update(1)

    plt.plot(range(len(return_list)),return_list)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('REINFORCE on CartPole')
    plt.show()

    m=moving_average(return_list,9)
    plt.plot(range(len(m)),m)
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('REINFORCE on CartPole')
    plt.show()

run(agent2)