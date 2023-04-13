import sys
if "../" not in sys.path:
  sys.path.append("../")

import gym
import numpy as np
from enviroment import CliffWalking
from utils.TD_method import SARSA_one_step
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

solver=SARSA_one_step(env,gamma,lr,epsilon,action_space)
solver.run(1000)
solver.plot_score('SARSA one step')
solver.plot_agent(disaster,ends)