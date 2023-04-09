import matplotlib.pyplot as plt

def plot_regret(solvers,solvers_name):
    #if len(solvers)==0:
    #    return
    for idx,solver in enumerate(solvers):
        l=range(len(solver.regrets))
        plt.plot(l,solver.regrets,label=solvers_name[idx])
    plt.xlabel('Time step')
    plt.ylabel('Cumulative regrets')
    plt.title('{}-armed bandit'.format(solvers[0].bandit.K))
    plt.legend()
    plt.show()

def plot_agent(agent,action_meaning,disaster=[],end=[]):
    print('value state function:')
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s' % ('%.3f' % agent.v[i*agent.env.ncol+j]),end=' ')
        print()

    print('policy:')
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i*agent.env.ncol+j) in disaster:
                print('****',end=' ')
            elif (i*agent.env.ncol+j) in end:
                print('EEEE',end=' ')
            else:
                pi_str=''
                a=agent.pi[i*agent.env.ncol+j]
                for k in range(len(action_meaning)):
                    pi_str+=action_meaning[k] if a[k]>0 else 'o'
                print(pi_str,end=' ')
        print()