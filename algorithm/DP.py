import sys
import copy
import numpy as np

if '../' not in sys.path:
    sys.path.append('..')

class PolicyIterationDP:
    '''
    Implemention of dynamic programming on policy iteration
    '''
    def __init__(self,env,theta,gamma):
        self.env=env
        self.space_dim=env.action_directions.n
        self.action_dim=env.action_space.n
        self.V=np.zeros([self.space_dim])
        self.pi=np.ones([self.space_dim,self.action_dim])/4
        self.theta=theta
        self.gamma=gamma

    def policy_evaluation(self):
        cnt=0
        while True:
            cnt+=1
            max_delta=0
            new_V=np.zeros_like(self.V)
            for i in range(self.space_dim):
                for j in range(self.action_dim):
                    next_state,reward,done,_=self.env.step_from_state(i,j)
                    new_V[i]+=(reward+self.gamma*self.V[next_state]*(1-done))*self.pi[i][j]
                max_delta=max(max_delta,abs(new_V[i]-self.V[i]))
            self.V=new_V
            if max_delta<self.theta:
                break
        print(f'Finished after {cnt} epoch(s) iteration!')

    def policy_improvement(self):
        for i in range(self.space_dim):
            qsa_list=[]
            for j in range(self.action_dim):
                next_state, reward, done, _ = self.env.step_from_state(i, j)
                qsa_list.append(reward+self.gamma*self.V[next_state]*(1-done))
            max_V=np.max(qsa_list)
            max_cnt=qsa_list.count(max_V)
            self.pi[i]=[1.0/max_cnt if qsa_list[j]==max_V else 0 for j in range(self.action_dim)]
        print(f'policy improvement finished!')
        return self.pi

    def run(self,min_delta=1e-3):
        while True:
            self.policy_evaluation()
            old_pi=copy.deepcopy(self.pi)
            new_pi=self.policy_improvement()
            if old_pi==new_pi:break
