import numpy as np
import copy

class PolicyIteration:
    def __init__(self,env,theta,gamma):
        self.env=env
        self.theta=theta
        self.gamma=gamma

        self.v=[0]*self.env.ncol*self.env.nrow
        self.pi=[[0.25,0.25,0.25,0.25] for _ in range(self.env.ncol*self.env.nrow)]

    def policy_evaluation(self):


    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            old_pi=copy.deepcopy(self.pi)
            new_pi=self.policy_improvement()
            if old_pi==new_pi:
                break