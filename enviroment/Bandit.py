import numpy as np

'''
多臂老虎机
    __init__():
        @param: 
            K:老虎机拉杆的数量
    step():
        @param:
            k:要拉动的杆的杆号
        @return:
            成功返回1，否则返回0 

Multi-Armed Bandit
    init():
        @param:
            K: number of levers in the bandit
    step():
        @param:
            k: the number of the lever to pull
        @return:
            1 if successful, 0 otherwise
'''
class MulitiBandit:
    def __init__(self,K):
        self.K=K
        self.prob=np.random.uniform(K)
        self.best_id=np.argmax(self.prob)
        self.best_prob=np.max(self.prob)

    def step(self,k):
        if np.random.rand()<=self.prob[k]:
            return 1
        else:
            return 0