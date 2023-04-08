from enviroment import Bandit
from Solver import BanditSolver
from utils import plot_func
import numpy as np

np.random.seed(1)
K=10
bandit=Bandit.MulitiBandit(K)
solver=BanditSolver.EpsilonGreedy(bandit)
solver.run(5000)
solver.plot_regret()

np.random.seed(0)
epsilons=[1e-4,1e-2,0.1,0.25,0.5]
epsilon_greedy_solver_list=[BanditSolver.EpsilonGreedy(bandit,e) for e in epsilons]
epsilon_greedy_solver_names=['epsilon={}'.format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)
plot_func.plot_regret(epsilon_greedy_solver_list,epsilon_greedy_solver_names)