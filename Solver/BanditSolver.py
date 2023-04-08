import numpy as np
import matplotlib.pyplot as plt
from utils.explore_discrete_action import  naive_epsilon_greedy

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

    def plot_regret(self):
        plt.plot(range(len(self.regrets)),self.regrets)
        plt.xlabel('Time step')
        plt.ylabel('Cumulative regrets')
        plt.title('{}-armed bandit'.format(self.bandit.K))
        plt.show()

class EpsilonGreedy(Solver):
    def __init__(self,bandit,epsilon=1e-2,init_prob=1.0):
        super(EpsilonGreedy,self).__init__(bandit)
        self.epsilon=epsilon
        self.estimates=np.array([init_prob]*self.bandit.K)

    def run_one_step(self):
        '''
        if np.random.rand()<self.epsilon:
            k=np.random.randint(0,self.bandit.K)
        else:
            k=np.argmax(self.estimates)
        '''
        k=naive_epsilon_greedy(self.estimates,self.epsilon)
        r=self.bandit.step(k)
        self.estimates[k]=1./(self.counts[k]+1)*(r-self.estimates[k])
        return k

