import numpy as np
import matplotlib.pyplot as plt
from utils.explore_discrete_action import  naive_epsilon_greedy,step_len_greedy,UCB_action

class Solver:
    def __init__(self,bandit):
        self.bandit=bandit
        self.counts=np.zeros(self.bandit.K)
        self.regret=0  #后悔值,regret value
        self.actions=[]
        self.regrets=[]

    def update_regret(self,k):
        self.regret+=self.bandit.best_prob-self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError

    def run(self,num_steps):
        for _ in range(num_steps):
            k=self.run_one_step()
            self.counts[k]+=1
            self.actions.append(k)
            self.update_regret(k)

    def plot_regret(self,name=''):
        plt.plot(range(len(self.regrets)),self.regrets,label=name)
        plt.xlabel('Time step')
        plt.ylabel('Cumulative regrets')
        plt.title('{}-armed bandit'.format(self.bandit.K))
        plt.legend()
        plt.show()

class EpsilonGreedy(Solver):
    def __init__(self,bandit,epsilon=0.01,init_prob=1.0):
        super(EpsilonGreedy,self).__init__(bandit)
        self.epsilon=epsilon
        self.estimates=np.array([init_prob]*self.bandit.K)

    def run_one_step(self):
        '''
        if np.random.random()<self.epsilon:
            k=np.random.randint(0,self.bandit.K)
        else:
            k=np.argmax(self.estimates)
        '''
        k=naive_epsilon_greedy(self.estimates,self.epsilon)
        r=self.bandit.step(k)
        self.estimates[k]+=1./(self.counts[k]+1)*(r-self.estimates[k])
        return k

class DecayingEpsilonGreedy(Solver):
    def __init__(self,bandit,init_prob=1.0):
        super(DecayingEpsilonGreedy,self).__init__(bandit)
        self.total_counts=0
        self.estimates=np.array([init_prob]*self.bandit.K)
        self.regret=0
        self.actions=[]
        self.regrets=[]
    def run_one_step(self):
        self.total_counts+=1
        k=step_len_greedy(self.estimates,self.total_counts)
        r=self.bandit.step(k)
        self.estimates[k]+=1./(self.counts[k]+1)*(r-self.estimates[k])
        return k

class UCBaction(Solver):
    def __init__(self,bandit,coef,init_pro=1.0):
        super(UCBaction,self).__init__(bandit)
        self.total_counts=0
        self.estimates=np.array([init_pro]*self.bandit.K)
        self.coef=coef
    def run_one_step(self):
        self.total_counts+=1
        k=UCB_action(self.estimates,self.counts,self.total_counts,self.coef)
        r=self.bandit.step(k)
        self.estimates[k]+=1./(self.counts[k]+1)*(r-self.estimates[k])
        return k

class ThompsonSampling(Solver):
    def __init__(self,bandit):
        super(ThompsonSampling,self).__init__(bandit)
        self._a=np.ones(self.bandit.K)
        self._b=np.ones(self.bandit.K)

    def run_one_step(self):
        samples=np.random.beta(self._a,self._b)
        k=np.argmax(samples)
        r=self.bandit.step(k)

        self._a[k]+=r
        self._b[k]+=1-r
        return k


