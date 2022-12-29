import numpy as np

def naive_epsilon_greedy(q_a,epsilon=1e-2):
    if np.random.uniform()<epsilon:
        return np.random.choice(len(q_a))
    else:
        return np.argmax(q_a)

def step_len_greedy(q_a,step_len):
    if np.random.uniform()<1.0/step_len:
        return np.random.choice(len(q_a))
    else:
        return np.argmax(q_a)

def bolzman_action(q_a):
    exp_q_a=np.exp(q_a-np.max(q_a))
    exp_q_a/=np.sum(exp_q_a)
    return np.random.choice(list(range(len(q_a))),p=exp_q_a)