import numpy as np
import copy

class PolicyIteration:
    def __init__(self,env,theta,gamma):
        self.env=env
        self.theta=theta
        self.gamma=gamma

        self.v=[0]*self.env.ncol*self.env.nrow
        self.pi=[[0.25,0.25,0.25,0.25] for _ in range(self.env.ncol*self.env.nrow)]

        self.action_meaning=['<','v','>','^']

    def policy_evaluation(self):
        cnt=1
        while True:
            new_v=[0]*self.env.ncol*self.env.nrow
            max_diff = 0
            for s in range(self.env.nrow*self.env.ncol):
                qsa_list=[]
                for a in range(4):
                    qsa=0
                    for res in self.env.P[s][a]:
                        p,next_state,r,done=res
                        qsa+=p*(r+self.gamma*self.v[next_state]*(1-done))
                    qsa_list.append(self.pi[s][a]*qsa)
                new_v[s]=np.sum(qsa_list)
                max_diff=max(max_diff,abs(new_v[s]-self.v[s]))
            self.v=new_v
            if max_diff<self.theta:
                break;
            cnt+=1
        print("Policy evaluation finished after {} iterations".format(cnt))
    def policy_improvement(self):
        for s in range(self.env.nrow*self.env.ncol):
            qsa_list=[]
            for a in range(4):
                qsa=0
                for res in self.env.P[s][a]:
                    p,next_state,r,done=res
                    qsa+=p*(r+self.gamma*self.v[next_state]*(1-done))
                qsa_list.append(qsa)
            max_v=np.max(qsa_list)
            max_count=qsa_list.count(max_v)
            self.pi[s]=[1./max_count if qsa_list[i]==max_v else 0. for i in range(4)]
        print('policy improvement finished')
        return self.pi
    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            old_pi=copy.deepcopy(self.pi)
            new_pi=self.policy_improvement()
            if old_pi==new_pi:
                break

    def plot_agent(self,disaster=[],end=[]):
        print('state value:')
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                print('%6.6s'% ('%.3f' % self.v[i*self.env.ncol+j]),end=' ')
            print()

        print('policy:')
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                if (i*self.env.ncol+j) in disaster:
                    print('****',end=' ')
                elif (i*self.env.ncol+j) in end:
                    print('EEEE',end=' ')
                else:
                    a=self.pi[i*self.env.ncol+j]
                    pi_str=''
                    for k in range(len(self.action_meaning)):
                        pi_str+=self.action_meaning[k] if a[k]>0 else 'o'
                    print(pi_str,end=' ')
            print()