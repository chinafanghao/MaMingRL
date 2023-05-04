import matplotlib.pyplot as plt
from matplotlib.ticker import  MultipleLocator,FormatStrFormatter
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

def plot_3D_HalfTen(datas,tile,zlabel='return'):

    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['font.family']='sans-serif'
    plt.rcParams['axes.unicode_minus']=False

    xmajorLocator=MultipleLocator(0.5)
    xmajorFormatter=FormatStrFormatter('%1.1f')

    ymajorLocator=MultipleLocator(1)
    ymajorFormatter=FormatStrFormatter('%d')

    zmajorLocator=MultipleLocator(1)
    zmajorFormatter=FormatStrFormatter('%d')

    fig=plt.figure()
    fig.suptitle(tile)
    fig.set_size_inches(18.5,10.5)

    ax=fig.add_subplot(111,projection='3d')
    axisX=[]
    axisY=[]
    axisZ=[]

    ax.set_xlim(0.5,10.5)
    ax.set_ylim(1,5)
    ax.set_zlim(0,1)

    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)

    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)

    ax.zaxis.set_major_locator(zmajorLocator)
    ax.zaxis.set_major_formatter(zmajorFormatter)

    for data in datas:
        axisX.append(data['x'])
        axisY.append(data['y'])
        axisZ.append(data['z'])

    ax.scatter(axisX,axisY,axisZ)
    ax.set_xlabel("player's score")
    ax.set_ylabel("player's cards number")
    ax.set_zlabel(zlabel)

def plot_return_list(return_list,agent_name):
    plt.plot(range(1,len(return_list)+1),return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(str(agent_name))
    plt.show()