import numpy as np
from collections import defaultdict
import sys
from tqdm import tqdm


def make_episilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        A_maxid = np.argmax(Q[observation])
        A[A_maxid] += 1. - epsilon
        return A

    return policy_fn


def mc_control_epsilon_greedy(env, num_episodes, gamma=1.0, epsilon=0.1, alpha=0.1, max_episodes_len=1000):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = make_episilon_greedy_policy(Q, epsilon, env.action_space.n)

    for i in tqdm(range(1, num_episodes + 1)):
        # if i%5000==0:
        #    print('\nEpisodes: {}/{}'.format(i,num_episodes),end=' ')
        #    sys.stdout.flush()

        episode = []
        state = env.reset()

        for t in range(max_episodes_len):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)

            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        sa_in_episode = set([(tuple(x[0]), x[1]) for x in episode])

        for state, action in sa_in_episode:
            # First Visit MC
            first_occurence_id = next(i for i, x in enumerate(episode) if x[0] == state and x[1] == action)

            G = sum([x[2] * (gamma ** i) for i, x in enumerate(episode[first_occurence_id:])])
            Q[state][action] += alpha * (G - Q[state][action])

    return Q, policy


def make_random_policy(nA):
    A = np.ones(nA, dtype=float) / nA

    def policy_fn(observation):
        return A

    return policy_fn


def create_greedy_policy(Q):
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        A[np.argmax(Q[state])] = 1.0
        return A

    return policy_fn


def mc_control_unweigthed_importance_sampling(env, num_episodes, gamma,epsilon=1e-2, max_episode_len=100):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    behavior_policy=make_episilon_greedy_policy(Q,epsilon,env.action_space.n)
    target_policy = create_greedy_policy(Q)

    for i_episode in tqdm(range(num_episodes)):
        episode = []
        state = env.reset()

        for t in range(max_episode_len):
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)

            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        episode_len = len(episode)

        if episode_len < 4:
            continue
        G = 0.0
        W = 1.0
        for t in range(episode_len)[::-1]:
            state, action, reward = episode[t]
            G = gamma * G + reward
            C[state][action] += 1
            Q[state][action] += 1. / C[state][action] * (G - Q[state][action])

            if action != np.argmax(target_policy(state)):
                break

            W *= 1. / behavior_policy(state)[action]
    return Q, target_policy


def mc_control_weighted_importance_sampling(env, behavior_policy, num_episodes, gamma, max_episode_len=100):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    C = defaultdict(lambda: np.zeros(env.action_space.n))

    target_policy = create_greedy_policy(Q)

    for i_episode in tqdm(range(num_episodes)):
        # if i_episode%5000==0:
        #    print('\rEpisode:{}/{}'.format(i_episode,num_episodes))
        #    sys.stdout.flush()

        episode = []
        state = env.reset()

        for t in range(max_episode_len):
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        episode_len = len(episode)

        if episode_len < 4:
            continue

        G = 0.0
        W = 1.0

        for t in range(episode_len)[::-1]:
            state, action, reward = episode[t]
            G = gamma * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

            if action != np.argmax(target_policy(state)):
                break

            W = W * 1. / behavior_policy(state)[action]
    return Q, target_policy
