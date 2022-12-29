import numpy as np
import gym

class GridWorld(gym.Env):
    '''
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    At first an agent is on an MxN grid(position at start_state which is user given)
    and it's goal is to reach the terminal state(end_state which is given by user or random).

    For example, a 5x5 grid looks as follows:

    o  o  o  o  o
    o  o  o  T  o   # end_state == 8
    o  o  o  o  o
    o  x  o  o  o
    o  o  o  o  o

    x is start_state and T are the end states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    receive a reward of -1 at each step until the agent reach a terminal state.

    '''
    def __init__(self,shape=[5,5],start_state=None,end_state=None):
        assert  isinstance(shape, (tuple, list)) or len(shape) == 2, f'input shape must be 2 dims like [5,5]'
        self.shape = shape
        self.observation_space = gym.spaces.Discrete(shape[0] * shape[1])

        if end_state is None:
            self.end_state=self.generate_end_state()
        else:
            self.end_state = end_state

        if start_state is None:
            self.start_state=self.generate_start_state()
        else:
            self.start_state = start_state

        self.action_space = gym.spaces.Discrete(4)
        self.actions={0:'up',1:'right',2:'down',3:'left'}
        self.action_directions=[[-1,0],[0,1],[1,0],[0,-1]]

    def generate_end_state(self):
        return np.random.randint(self.observation_space.n)

    def generate_start_state(self):
        state=np.random.randint(self.observation_space.n)
        while state==self.end_state:
            state = np.random.randint(self.observation_space.n)
        return state

    def step(self, action):
        now_state=[self.state//self.shape[1],self.state%self.shape[0]]
        if now_state==self.end_state:
            print('Warning: End state has been reached,please reset the environment!')
        now_state[0]=max(min(self.shape[0]-1,now_state[0]+self.action_directions[action][0]),0)
        now_state[1]=max(min(self.shape[1]-1,now_state[1]+self.action_directions[action][1]),0)
        self.state=now_state[0]*self.shape[1]+now_state[1]
        if self.state == self.end_state:
            reward = 0
            done=True
        else:
            reward = -1
            done=False
        return self.state, reward, done, {}

    def step_from_state(self,state,action):
        now_state = [state // self.shape[1], state % self.shape[0]]
        if now_state == self.end_state:
            print('Warning: End state has been reached,please reset the environment!')
        now_state[0] = max(min(self.shape[0] - 1, now_state[0] + self.action_directions[action][0]), 0)
        now_state[1] = max(min(self.shape[1] - 1, now_state[1] + self.action_directions[action][1]), 0)
        state = now_state[0] * self.shape[1] + now_state[1]
        if state == self.end_state:
            reward = 0
            done = True
        else:
            reward = -1
            done = False
        return state, reward, done, {}

    def reset(self):
        self.state = self.start_state
        return self.state,None








