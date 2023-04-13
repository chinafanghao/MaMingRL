This is a reinforcement learning library.

The library will contain these RL algorithms:

| 算法algorithm    | 算法版本 algorithm version                                   | 实现的环境 env                                | 进度 status | 位置position |
| ---------------- | ------------------------------------------------------------ | --------------------------------------------- | ---- | ---------------- |
| 蒙特卡罗评估 MC  | on-policy version                                            | 十点半 half ten                               | √ | /utils/MC_method |
|                  | off-policy version                                           |                                               | √ | /utils/MC_method |
|                  | weighted importance off-policy version         |                                               | √ | /utils/MC_method |
| 时序差分TD       | 1-step SARSA                                                | CliffWalking | √ | /utils/TD_method |
|                  | Q-learning                                                   |                                               |      |  |
|                  | n-step SARSA                                                 |                                               |      |  |
|                  | n-step Q-learning                                            |                                               |      |  |
| 资格迹 eligible trace         |forward TD(λ)                                                | 暂定清华大学出版社《强化学习》第6章里的风格子 |      |  |
|                  | backward TD(λ)                                           |                                               |      |  |
|                  | forward SARSA(λ)                                         |                                               |      |  |
|                  | backward SARSA(λ)                                        |                                               |      |  |
|                  | forward Watkins‘s Q(λ)                                   |                                               |      |  |
|                  | backward Watkins's Q(λ)                                  |                                               |      |  |
|                  | Peng's Q(λ)                                                  |                                               |      |  |
| 值函数线性逼近   | 多项式基函数-增量法-MC参数逼近-SARSA实现                     |                                               |      |  |
|                  | 多项式基函数-增量法-MC参数逼近-Q实现                         |                                               |      |  |
|                  | 多项式基函数-增量法-TD参数逼近-SARSA实现                     |                                               |      |  |
|                  | 多项式基函数-增量法-TD参数逼近-Q实现                         |                                               |      |  |
|                  | 多项式基函数-增量法-前向TD(λ)参数逼近-SARSA实现              |                                               |      |  |
|                  | 多项式基函数-增量法-后向TD(λ)参数逼近-Q实现                  |                                               |      |  |
|                  | 多项式基函数-批量法-MC参数逼近-SARSA实现                     |                                               |      |  |
|                  | 多项式基函数-批量法-MC参数逼近-Q实现                         |                                               |      |  |
|                  | 多项式基函数-批量法-TD参数逼近-SARSA实现                     |                                               |      |  |
|                  | 多项式基函数-批量法-TD参数逼近-Q实现                         |                                               |      |  |
|                  | 多项式基函数-批量法-前向TD(λ)参数逼近-SARSA实现              |                                               |      |  |
|                  | 多项式基函数-批量法-后向TD(λ)参数逼近-Q实现                  |                                               |      |  |
|                  | 傅里叶基函数                                                 |                                               |      |  |
|                  | 径向基函数                                                   |                                               |      |  |
| 值函数非线性逼近 | DQN                                                          | 飞翔的小鸟 breakout                           |      |  |
|                  | Double DQN                                                   |                                               |      |  |
|                  | Dueling DQN                                                  |                                               |      |  |
| 策略梯度policy gradient | REINFORCE                                                    | MountainCar                                   |      |  |
|                  | REINFORCE with baseline                                      |                                               |      |  |
|                  | TRPO                                                         |                                               |      |  |
|                  | PPO                                                          |                                               |      |  |
| Actor-Critic     | on-policy AC(Critic use TD(0),SARSA,TD(lambda),Q-learning,n-step) | Pendulum                                      |      |  |
|                  | off-policy AC(Critic use TD(0),SARSA,TD(lambda),Q-learning,n-step) |                                               |      |  |
|                  | off-policy AC(Critic use TD with Gradient Correction Term)   |                                               |      |  |
|                  | A2C                                                          |                                               |      |  |
| 异步方法         | 异步Q-learning                                               |                                               |      |  |
|                  | 异步SARSA                                                    |                                               |      |  |
|                  | 异步n-step Q-learning                                        |                                               |      |  |
|                  | A3C                                                          |                                               |      |  |
| 确定性策略DPG    | on-policy 确定性AC                                           | Pendulum                                      |      |  |
|                  | off-policy 确定性AC                                          |                                               |      |  |
|                  | DDPG                                                         |                                               |      |  |
| 学习与规划<br />learning and plan | Dyna-Q                                                       | CliffWalking                                  |      |  |
|                  | Dyna-Q+                                                      |                                               |      |  |
|                  | 优先级扫描的Dyna-Q                                           |                                               |      |  |
|                  | Dyna-2                                                       |                                               |      |  |
| 探索与利用 <br />exploration vs exploitation | epsilon-greedy                                           | 多臂老虎机                                    | √ | utils/explore_discrete_action |
|                  | linear decaying epsilon greedy     |                                               | √           | utils/explore_discrete_action |
|                                              | UCB1                                                         |                                               | √ | utils/explore_discrete_action |
|                                              | Thompson Sampling                                            |                                               | √ | Solver/BanditSolver |
| DP                                           | policy iteration                                             | frozenLake                                    | √ | Solver/DPIteration |
|  | value iteration |  | √ | Solver/DPIteration |
| 博弈强化学习 | AlphaGo Zero | 五子棋 | |  |
|  |  |  | |  |






​			
​			
​			
​			
​			
​			
​			
