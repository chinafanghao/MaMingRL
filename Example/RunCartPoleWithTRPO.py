import gym

env_name='CartPole-v1'
env=gym.make(env_name)

print(env.reset())
print(env.action_space.n)
print(env.step(0))