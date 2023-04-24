import sys
if "../" not in sys.path:
  sys.path.append("../")

import gym
import numpy as np
from enviroment import CliffWalking
from utils.dynaQ import DynaQ
from tqdm import tqdm
import matplotlib.pyplot as plt

nrow=4
ncol=12
ends=[nrow*ncol-1]
disaster=[(nrow-1)*ncol+i for i in range(1,ncol)]

epsilon=0.01
alpha=0.1
gamma=0.9
num_episode=300
n_action=4
env=CliffWalking.CliffWalking(nrow,ncol)

def DynaQ_CliffWalking(env,nrow,ncol,epsilon,gamma,alpha,n_planning,n_action,num_episode):
    agent=DynaQ(env,nrow,ncol,epsilon,gamma,alpha,n_planning,n_action)
    return agent.run(num_episode)

n_plannings=[0,2,20]
np.random.seed(0)
for n_planning in n_plannings:
    return_list=DynaQ_CliffWalking(env,nrow,ncol,epsilon,gamma,alpha,n_planning,n_action,num_episode)
    plt.plot(range(len(return_list)),return_list,label=str(n_planning)+' planning step')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Returns')
plt.show()



