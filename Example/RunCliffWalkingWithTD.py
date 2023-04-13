import sys
if "../" not in sys.path:
  sys.path.append("../")

import gym
import numpy as np
from enviroment import CliffWalking
from utils.TD_method import SARSA_one_step,SARSA_n_step,Qlearning
nrow=4
ncol=12
ends=[nrow*ncol-1]
disaster=[(nrow-1)*ncol+i for i in range(1,ncol)]

env=CliffWalking.CliffWalking(nrow,ncol)
np.random.seed(0)
gamma=0.9
lr=0.1
epsilon=0.1
action_space=list(range(env.action_space.n))


def run_one_step_SARSA(env, gamma, lr, epsilon, action_space,episode_num):
  solver = SARSA_one_step(env, gamma, lr, epsilon, action_space)
  solver.run(episode_num)
  solver.plot_score('SARSA one step')
  solver.plot_agent(disaster, ends)

def run_n_step_SARSA(env,n_step, gamma, lr, epsilon, action_space,episode_num):
  solver = SARSA_n_step(env,n_step, gamma, lr, epsilon, action_space)
  solver.run(episode_num)
  solver.plot_score('SARSA n step')
  solver.plot_agent(disaster, ends)

def run_Qlearning(env,gamma,lr,epsilon,action_space,episode_num):
  solver = Qlearning(env, gamma, lr, epsilon, action_space)
  solver.run(episode_num)
  solver.plot_score('Qlearning one step')
  solver.plot_agent(disaster, ends)
#run_one_step_SARSA(env,gamma,lr,epsilon,action_space,1000)
#run_n_step_SARSA(env,5,gamma,lr,epsilon,action_space,500)
run_Qlearning(env,gamma,lr,epsilon,action_space,1000)