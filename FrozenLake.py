import random

import gym
import numpy as np
import sys, itertools, math

env = gym.make('FrozenLake-v0')

MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1


def select_action(state, explore_rate, episode):
    if random.random() < explore_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state])
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t + 1) / 25)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t + 1) / 25)))


# print("env.observation_space.n", env.observation_space.n)
# print("env.action_space.n", env.action_space.n)
Q = np.zeros([env.observation_space.n, env.action_space.n])

lr = .8
y = .95

num_episodes = 2000
iterations = 100

rList = []
# learning_rate = get_learning_rate(0)
explore_rate = get_explore_rate(0)
for episode in range(num_episodes):
    if (episode + 1) % 100 == 0:
        print("\r Episode {}/{}.".format(episode + 1, num_episodes), end="")
        # print("learning Rate=",learning_rate,"explore_rate=",explore_rate)
        sys.stdout.flush()

    state = env.reset()
    rAll = 0
    d = False
    for t in itertools.count(iterations):

        action = select_action(state, explore_rate,episode)

        # # # another qAction Update using noise instead of random action
        # action = np.argmax(Q[state] + np.random.randn(1, env.action_space.n) * (1. / (episode + 1)))

        state1, reward, d, _ = env.step(action)
        Q[state, action] = Q[state, action] + lr * (reward + y * np.max(Q[state1, :]) - Q[state, action])
        rAll += reward
        state = state1
        # env.render()
        if d == True:
            break
    explore_rate = get_explore_rate(episode)
    # learning_rate = get_learning_rate(episode)
    rList.append(rAll)

print("Score over time: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
