import numpy as np
from collections import defaultdict
import sys

def make_episilon_greedy_policy(Q,epsilon,nA):

    def policy_fn(observation):
        A=np.ones(nA,dtype=float)*epsilon/nA
        A_maxid=np.argmax(Q[observation])
        A[A_maxid]+=1.-epsilon
        return A

    return policy_fn

def mc_control_epsilon_greedy(env,num_episodes,gamma=1.0,epsilon=0.1,alpha=0.1,max_episodes_len=1000):

    Q=defaultdict(lambda:np.zeros(env.action_space.n))

    policy=make_episilon_greedy_policy(Q,epsilon,env.action_space.n)

    for i in range(1,num_episodes+1):
        if i%5000==0:
            print('\nEpisodes: {}/{}'.format(i,num_episodes),end=' ')
            sys.stdout.flush()

        episode=[]
        state=env.reset()

        for t in range(max_episodes_len):
            probs=policy(state)
            action=np.random.choice(np.arange(len(probs)),p=probs)
            next_state,reward,done,_=env.step(action)

            episode.append((state,action,reward))
            if done:
                break
            state=next_state
        sa_in_episode=[(tuple(x[0]),x[1]) for x in episode]

        for state,action in sa_in_episode:
            first_occurence_id=next(i for i,x in enumerate(episode) if x[0]==state and x[1]==action)

            G=sum([x[2]*(gamma**i) for i,x in enumerate(episode[first_occurence_id:])])
            Q[state][action]+=alpha*(G-Q[state][action])

    return Q,policy