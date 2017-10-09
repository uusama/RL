import random

import gym
import copy
import Qvalue, agent, memory
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

epslons = np.linspace(.9, .1, 10000)


def run(env,
        batch_size, agent, memory, discount, steps=300, episode_i=0, eps=.9):
    state = env.reset()
    done = False
    acc_reward = 0.0
    loss = 0.0
    for i in range(steps):
        if done:
            break
        # eps should decay overtime
        action = agent.move(state, eps=.9)
        next_state, reward, done, _ = env.step(action)
        acc_reward += reward
        memory.add((state, action, next_state, reward, done))
        if episode_i > 900000:
            env.render()

        if len(memory.memory) > batch_size:
            state_m, action_m, next_state_m, reward_m, done_m = zip(*memory.sample(batch_size))
            state_m = np.array(state_m)
            action_m = np.array(action_m)
            next_state_m = np.array(next_state_m)
            reward_m = np.array(reward_m)
            done_m = np.array(done_m)

            q_m = agent.predict(next_state_m)

            actual_target_m = reward_m + (1. - done_m) * discount * np.amax(q_m, axis=1)

            targets = agent.predict(state_m)

            # assign the actual reward to the taken action
            for i, action in enumerate(action_m):
                targets[i, action] = actual_target_m[i]
            loss = agent.train(states=state_m, targets=targets)
            state = copy.copy(next_state)

    print("acc_reward:", acc_reward)
    return acc_reward, i, loss


env = gym.make("MountainCar-v0")

n_actions = env.action_space.n
state_dim = env.observation_space.high.shape[0]
print("n_actions:", n_actions, "state_dim", state_dim)
batch_size = 64
qvalue_model = Qvalue.Qvalue(state_dim=state_dim, n_actions=n_actions, batch_size=64, h1_n=512, h2_n=256)
agent = agent.Agent(actions=n_actions, q_value_model=qvalue_model)
memory = memory.RandomMemory(max_size=1024)

discount = .95
rewards = []
episodes_end = []
losses = []
eps = .9
reward, episode_end, loss = 0., 0., 0.
for episode_i in range(1000000):
    print("episode_i:", episode_i)
    if episode_i % 1000 == 0:
        eps = epslons[int(episode_i / 10000)]
        print("epslons:", eps)
        reward, episode_end, loss = run(env,
                                        batch_size, agent, memory, discount, steps=200, episode_i=episode_i, eps=eps)
    rewards.append(reward)
    episodes_end.append(episode_end)
    losses.append(loss)

plt.plot(rewards)
plt.xlabel("episode:")
plt.ylabel("reward")
plt.show()

plt.plot(episodes_end)
plt.xlabel("episode:")
plt.ylabel("episodes_end")
plt.show()

plt.plot(losses)
plt.xlabel("episode:")
plt.ylabel("loss")
plt.show()
