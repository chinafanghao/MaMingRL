This is a reinforcement learning library.

The library will contain these RL algorithms:

| 算法             | 算法版本                                                     | 实现的环境                                    | 进度 |
| ---------------- | ------------------------------------------------------------ | --------------------------------------------- | ---- |
| 蒙特卡罗评估 MC  | on-policy version                                            | 十点半 half ten                               |      |
|                  | off-policy version                                           |                                               |      |
|                  | 加权重要性off-policy版本                                     |                                               |      |
| 时序差分TD       | SARSA                                                        | 动手学强化学习里提到的FreezeLake              |      |
|                  | Q-learning                                                   |                                               |      |
|                  | n-step SARSA                                                 |                                               |      |
|                  | n-step Q-learning                                            |                                               |      |
| 资格迹 eligible trace         |前向算法TD(λ)                                                | 暂定清华大学出版社《强化学习》第6章里的风格子 |      |
|                  | 后向TD(λ)                                                    |                                               |      |
|                  | 前向SARSA(λ)                                                 |                                               |      |
|                  | 后向SARSA(λ)                                                 |                                               |      |
|                  | 前向Watkins‘s Q(λ)                                           |                                               |      |
|                  | 后向Watkins's Q(λ)                                           |                                               |      |
|                  | Peng's Q(λ)                                                  |                                               |      |
| 值函数线性逼近   | 多项式基函数-增量法-MC参数逼近-SARSA实现                     |                                               |      |
|                  | 多项式基函数-增量法-MC参数逼近-Q实现                         |                                               |      |
|                  | 多项式基函数-增量法-TD参数逼近-SARSA实现                     |                                               |      |
|                  | 多项式基函数-增量法-TD参数逼近-Q实现                         |                                               |      |
|                  | 多项式基函数-增量法-前向TD(λ)参数逼近-SARSA实现              |                                               |      |
|                  | 多项式基函数-增量法-后向TD(λ)参数逼近-Q实现                  |                                               |      |
|                  | 多项式基函数-批量法-MC参数逼近-SARSA实现                     |                                               |      |
|                  | 多项式基函数-批量法-MC参数逼近-Q实现                         |                                               |      |
|                  | 多项式基函数-批量法-TD参数逼近-SARSA实现                     |                                               |      |
|                  | 多项式基函数-批量法-TD参数逼近-Q实现                         |                                               |      |
|                  | 多项式基函数-批量法-前向TD(λ)参数逼近-SARSA实现              |                                               |      |
|                  | 多项式基函数-批量法-后向TD(λ)参数逼近-Q实现                  |                                               |      |
|                  | 傅里叶基函数                                                 |                                               |      |
|                  | 径向基函数                                                   |                                               |      |
| 值函数非线性逼近 | DQN                                                          | 飞翔的小鸟 breakout                           |      |
|                  | Double DQN                                                   |                                               |      |
|                  | Dueling DQN                                                  |                                               |      |
| 策略梯度         | REINFORCE                                                    | MountainCar                                   |      |
|                  | REINFORCE with baseline                                      |                                               |      |
|                  | TRPO                                                         |                                               |      |
|                  | PPO                                                          |                                               |      |
| Actor-Critic     | on-policy AC(Critic use TD(0),SARSA,TD(lambda),Q-learning,n-step) | Pendulum                                      |      |
|                  | off-policy AC(Critic use TD(0),SARSA,TD(lambda),Q-learning,n-step) |                                               |      |
|                  | off-policy AC(Critic use TD with Gradient Correction Term)   |                                               |      |
|                  | A2C                                                          |                                               |      |
| 异步方法         | 异步Q-learning                                               |                                               |      |
|                  | 异步SARSA                                                    |                                               |      |
|                  | 异步n-step Q-learning                                        |                                               |      |
|                  | A3C                                                          |                                               |      |
| 确定性策略       | on-policy 确定性AC                                           | Pendulum                                      |      |
|                  | off-policy 确定性AC                                          |                                               |      |
|                  | DDPG                                                         |                                               |      |
| 学习与规划       | Dyna-Q                                                       | CliffWalking                                  |      |
|                  | Dyna-Q+                                                      |                                               |      |
|                  | 优先级扫描的Dyna-Q                                           |                                               |      |
|                  | Dyna-2                                                       |                                               |      |
| 探索与利用       | UCB1                                                         | 多臂老虎机                                    |      |
|                  | Thompson Sampling                                            |                                               |      |
| 博弈强化学习     | AlphaGo Zero                                                 | 五子棋                                        |      |




​			
​			
​			
​			
​			
​			
​			
