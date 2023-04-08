import matplotlib.pyplot as plt

def plot_regret(solvers,solvers_name):
    if len(solvers)==0:
        return
    for idx,solver in enumerate(solvers):
        l=range(len(solver.regrets))
        plt.plot(l,solver.regrets,label=solvers_name[idx])
    plt.xlabel('Time step')
    plt.ylabel('Cumulative regrets')
    plt.title('{}-armed bandit'.format(solvers[0].bandit.K))
    plt.legend()
    plt.show()