import numpy as np

class DynaQ:
    def __init__(self,env,nrow,ncol,epsilon,gamma,alpha,n_planning,n_action=4):
        self.env=env
        self.Q_table=np.zeros([nrow*ncol,n_action])
        self.nrow=nrow
        self.ncol=ncol
        self.epsilon=epsilon
        self.gamma=gamma
        self.alpha=alpha
        self.n_planning=n_planning
        self.n_action=n_action

        self.model=dict()

    def take_action(self,state):
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.n_action)
        else:
            action=np.argmax(self.Q_table[state])
        return action

    def Q_learning(self,s0,a0,r,s1,done):
        TD_error=r+self.gamma*np.max(self.Q_table[s1])*(1-done)-self.Q_table[s0][a0]
        self.Q_table[s0][a0]+=self.alpha*TD_error

    def update(self,s0,a0,r,s1,done):
        self.Q_learning(s0,a0,r,s1,done)
        self.model[(s0,a0)]=(r,s1,done)
        for _ in range(self.n_planning):
            index=np.random.choice(len(self.model.items()))
            (s,a),(r,s_,done_)=list(self.model.items())[index]
            self.Q_learning(s,a,r,s_,done_)

    def run(self,episode_num):
        return_list=[]
        for _ in range(episode_num):
            episode_reward=0
            state=self.env.reset()
            done=False
            while not done:
                action=self.take_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward+=reward
                self.update(state,action,reward,next_state,done)
                if done:
                    break
                state=next_state
            return_list.append(episode_reward)
        return return_list

