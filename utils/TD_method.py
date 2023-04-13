import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plot

class RL_brain:
    def __init__(self,env,gamma,learning_rate,action_space):
        self.env=env
        self.gamma=gamma
        self.learning_rate=learning_rate
        self.action_space=action_space
        self.Q_table=np.zeros([self.env.ncol*self.env.nrow,self.env.action_space.n])
        self.score=[]
    '''
    def check_state_exist(self,state):
        if state not in self.Q_table.index:
            self.Q_table=self.Q_table.append(
                pd.Series(
                    [0] * len(self.action_space),
                    index=self.Q_table.columns,
                    name=state
                )
            )
    '''

    def reset_score(self):
        self.score=[]

    def get_policy(self,state):
        Q_max=np.max(self.Q_table[state])
        a=[1 if self.Q_table[state][i]==Q_max else 0 for i in range(self.env.action_space.n)]
        return a

    def run_one_episode(self):
        raise NotImplementedError

    def run(self,num_episode):
        for i in tqdm(range(num_episode)):
            self.run_one_episode()

    def chose_action(self,state):
        if np.random.rand()<self.epsilon:
            return np.random.choice(list(range(self.env.action_space.n)))
        else:
            return np.argmax(self.Q_table[state])

    def plot_score(self,title):
        plt.plot(range(len(self.score)),self.score)
        plt.title(title)
        plt.xlabel('train episode')
        plt.ylabel('return')
        plt.show()

    def plot_agent(self,disaster,end):

        print('policy:')
        for i in range(self.env.nrow):
            for j in range(self.env.ncol):
                if (i * self.env.ncol + j) in disaster:
                    print('****', end=' ')
                elif (i * self.env.ncol + j) in end:
                    print('EEEE', end=' ')
                else:
                    a = self.get_policy(i * self.env.ncol + j)
                    pi_str = ''
                    for k in range(len(self.env.action_meaning)):
                        pi_str += self.env.action_meaning[k] if a[k] > 0 else 'o'
                    print(pi_str, end=' ')
            print()

class SARSA_one_step(RL_brain):
    def __init__(self,env,gamma,learning_rate,epsilon,action_space):
        super(SARSA_one_step,self).__init__(env,gamma,learning_rate,action_space)
        self.epsilon=epsilon

    def run_one_episode(self):
        state=self.env.reset()
        action = self.chose_action(state)
        done=False
        score=0
        while not done:
            next_state,reward,done,_=self.env.step(action)
            #self.check_state_exist(next_state)
            next_action = self.chose_action(next_state)
            score+=reward
            TD_error=reward+self.gamma*self.Q_table[next_state][next_action]-self.Q_table[state][action]
            self.Q_table[state][action]+=self.learning_rate*TD_error
            state=next_state
            action=next_action
        self.score.append(score)
