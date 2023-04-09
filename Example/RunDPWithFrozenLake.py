import gym
from Solver.DPIteration import PolicyIteration
from utils.plot_func import  plot_agent
env=gym.make('FrozenLake-v1')
env=env.unwrapped

holes=set()
ends=set()

for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2]==1.0:
                ends.add(s_[1])
            if s_[3]==True:
                holes.add(s_[1])
holes=holes-ends

actions=['<','v','>','^']

theta=1e-5
gamma=0.9
agent=PolicyIteration(env,theta,gamma)
agent.policy_iteration()
agent.plot_agent(holes,ends)
plot_agent(agent,actions,holes,ends)