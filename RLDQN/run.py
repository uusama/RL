import random

import gym
import copy, sys

import sklearn
import sklearn.pipeline
from sklearn.kernel_approximation import RBFSampler

import Qvalue, agent, memory
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

epslons = np.linspace(.9, .0001, 10)
env = gym.make("MountainCar-v0")

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))


def featurize_state(state):
    """
    Returns the featurized representation for a state.
    """
    scaled = scaler.transform([state])
    # print("scaled state:", scaled.shape, scaled)
    # featurized = featurizer.transform(scaled)
    # print("featurized state:", featurized.shape)

    return scaled[0]


def run(env,
        batch_size, agent, memory, discount, steps=300, episode_i=0, eps=.9, render=False, normalize=False):
    state = env.reset()
    done = False
    acc_reward = 0.0
    loss = 0.0
    for i in range(steps):
        if done:
            break
        # eps should decay overtime
        action = agent.move(state, eps=.9)
        # print("state:",state.shape,state)
        if normalize:
            state = featurize_state(state)

        next_state, reward, done, _ = env.step(action)
        acc_reward += reward
        memory.add((state, action, next_state, reward, done))
        if render:
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

    # print("acc_reward:", acc_reward)
    return acc_reward, i, loss


n_actions = env.action_space.n
state_dim = env.observation_space.high.shape[0]
print("n_actions:", n_actions, "state_dim", state_dim)
batch_size = 64
checkpoint_path = "/tmp/my_dqn.ckpt"
qvalue_model = Qvalue.Qvalue(state_dim=state_dim, n_actions=n_actions, batch_size=64, h1_n=512, h2_n=256,
                             checkpoint_path=checkpoint_path)
agent = agent.Agent(actions=n_actions, q_value_model=qvalue_model)
memory = memory.RandomMemory(max_size=1024)

discount = .95
rewards = []
episodes_end = []
losses = []
eps = .9
reward, episode_end, loss = 0., 0., 0.
render = False
print(reward)
while reward < 20.:
    for episode_i in range(1000):
        # print("episode_i:", episode_i)
        if episode_i % 100 == 0:
            eps = epslons[int(episode_i / 100)]
            print("\rEpisode/eps: reward {}/{} : {}.".format(episode_i, eps, reward), end="")
            sys.stdout.flush()

            if episode_i != 0:
                qvalue_model.saver.save(qvalue_model.session, checkpoint_path)

        # if episode_i > 9000:
        if 0 > reward > -100:
            render = True
        reward, episode_end, loss = run(env,
                                        batch_size, agent, memory, discount, steps=200, episode_i=episode_i, eps=eps,
                                        render=render, normalize=True)
        rewards.append(reward)
        episodes_end.append(episode_end)
        losses.append(loss)
        sys.stdout.flush()
    print("reward:", reward)

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
