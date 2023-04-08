from enviroment import Bandit
from Solver import BanditSolver
from utils import plot_func
import numpy as np
import matplotlib.pyplot as plt

func=[BanditSolver.EpsilonGreedy,BanditSolver.DecayingEpsilonGreedy,BanditSolver.UCBaction,BanditSolver.ThompsonSampling]
coefs=[{'epsilon':0.01},{},{'coef':1},{}]
func_name=['EpsilonGreedy','DecayingEpsilongGreedy','UCB action','ThompsonSampling']
np.random.seed(1)
K=10
bandit=Bandit.MulitiBandit(K)
'''
for id,f in enumerate(func):
    np.random.seed(1)
    solver=f(bandit,**coefs[id])
    solver.run(5000)
    print('{}\'s regret value is {}'.format(func_name[id],solver.regret))
    solver.plot_regret(func_name[id])
'''
epsilon_greedy_solver_list=[f(bandit,**coefs[id]) for id,f in enumerate(func)]
for id,solver in enumerate(epsilon_greedy_solver_list):
    solver.run(5000)
    print('{}\'s regret value is {}'.format(func_name[id], solver.regret))
plot_func.plot_regret(epsilon_greedy_solver_list,func_name)

np.random.seed(0)
epsilons=[1e-4,1e-2,0.1,0.25,0.5]
epsilon_greedy_solver_list=[BanditSolver.EpsilonGreedy(bandit,e) for e in epsilons]
epsilon_greedy_solver_names=['epsilon={}'.format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)
plot_func.plot_regret(epsilon_greedy_solver_list,epsilon_greedy_solver_names)