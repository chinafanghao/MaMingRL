import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class CliffWalking(gym.Env):

    def __init__(self,nrow,ncol,):
        self.action_space=spaces.Discrete(4)
        self.action=[[0,1],[0,-1],[1,0],[-1,0]]
        self.action_meaning=['v','^','>','<']
        self.nrow=nrow
        self.ncol=ncol
        self.observation_space=spaces.Discrete(self.nrow*self.ncol)
        self.reset()

    def reset(self):
        self.x=0
        self.y=self.nrow-1
        return self.y*self.ncol+self.x

    def step(self,action):
        self.x=min(max(self.x+self.action[action][0],0),self.ncol-1)
        self.y=min(max(self.y+self.action[action][1],0),self.nrow-1)
        reward=-1
        done=False
        if self.y==self.nrow-1 and self.x>0:
            done=True
            if self.x!=self.ncol-1:
                reward=-100
        return self.y*self.ncol+self.x,reward,done,{}